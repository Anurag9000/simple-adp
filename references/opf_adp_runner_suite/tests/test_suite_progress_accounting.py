import dataclasses
import json
import io
import os
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import hashlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from utils.opf_massive_suite_runner import (
    JobRuntime,
    JobSpec,
    RecoveredProcessHandle,
    _close_runtime_log_handle,
    _manifest_equivalent_under_interpreter_aliases,
    _requeue_recovered_runtime_after_exit,
    _resume_runtime_from_pause,
    _suspend_runtime_for_pressure,
    _terminate_process_tree,
    summarize_suite_telemetry,
    launch_job,
    load_existing_runtime,
    is_terminal_inexact_resume_failure,
    should_retry,
    summarize_suite_runtime_progress,
)
from utils import opf_resume_bundle
from utils.opf_resume_bundle import (
    _safe_extract_tar,
    export_results_bundle,
    rehydrate_sharded_telemetry,
    restore_results_bundle,
)
from utils import pressure_aware_wrapper
from scripts.kill_all_runners import _is_owned_command
from Dyn_DNN4OPF.models.checkpoint_utils import checkpoint_architecture_matches
from Dyn_DNN4OPF.utils.run_telemetry import TrainingTelemetry
from Dyn_DNN4OPF.models.checkpoint_utils import restore_stage_architecture
from Dyn_DNN4OPF.utils.resume_state import capture_training_state


def _spec(name: str, results_dir: Path) -> JobSpec:
    return JobSpec(
        job_id=name,
        phase=name,
        family=name,
        case_name="pglib_opf_case14_ieee",
        repeat_index=1,
        depth=1,
        width=1,
        seed=42,
        results_dir=str(results_dir),
        command=(),
    )


def _write_state(run_root: Path, spec: JobSpec, status: str, **values: object) -> None:
    state_path = run_root / "jobs" / spec.job_id / "job_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"status": status, **values}), encoding="utf-8")


class SuiteProgressAccountingTest(unittest.TestCase):
    def test_restart_manifest_accepts_only_same_interpreter_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            python3 = Path(temporary) / "python3"
            python = Path(temporary) / "python"
            python3.symlink_to(sys.executable)
            python.symlink_to(sys.executable)
            existing = {
                "signature": "old-signature",
                "jobs": [{"command": [str(python3), "-u", "train.py"]}],
            }
            new = {
                "signature": "new-signature",
                "jobs": [{"command": [str(python), "-u", "train.py"]}],
            }
            self.assertTrue(_manifest_equivalent_under_interpreter_aliases(existing, new))
            changed_plan = {
                "signature": "new-signature",
                "jobs": [{"command": [str(python), "-u", "different.py"]}],
            }
            self.assertFalse(_manifest_equivalent_under_interpreter_aliases(existing, changed_plan))

    def test_inexact_resume_failure_stays_terminal_across_restarts(self) -> None:
        self.assertTrue(is_terminal_inexact_resume_failure({"status": "failed", "reason": "inexact_checkpoint_resume"}))
        self.assertFalse(is_terminal_inexact_resume_failure({"status": "failed", "reason": "cuda_oom"}))
        self.assertFalse(is_terminal_inexact_resume_failure({"status": "queued", "reason": "inexact_checkpoint_resume"}))

    def test_legacy_cuda_oom_flag_is_ignored_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_root"
            spec = _spec("legacy-oom-device", Path(temporary) / "results")
            _write_state(
                root,
                spec,
                "queued",
                device_mode="gpu",
                force_cpu_after_cuda_oom=True,
            )
            runtime = load_existing_runtime(root, spec)
            self.assertEqual(runtime.device_mode, "gpu")
            self.assertFalse(runtime.force_cpu_after_cuda_oom)

    @unittest.skipIf(os.name == "nt", "POSIX PID liveness is required")
    def test_recovered_child_exit_is_requeued_without_fake_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _spec("recovered-exit", root / "results")
            runtime = JobRuntime(
                spec=spec,
                state_path=root / "run_root" / "jobs" / spec.job_id / "job_state.json",
                log_path=root / "run_root" / "jobs" / spec.job_id / "job.log",
                proc=RecoveredProcessHandle(999999999),
                attempt=4,
                retry_count=1,
            )
            active = [runtime]
            queue = deque()

            self.assertTrue(_requeue_recovered_runtime_after_exit(runtime, active, queue))
            self.assertEqual(list(queue), [runtime])
            self.assertEqual(active, [])
            self.assertIsNone(runtime.proc)
            self.assertIsNone(runtime.last_exit_code)
            state = json.loads(runtime.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "queued")
            self.assertEqual(state["reason"], "recovered_child_exit_status_unknown")

    def test_offline_telemetry_excludes_terminal_final_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "results"
            telemetry_dir = result_dir / "telemetry" / "model" / "case" / "task_0000"
            telemetry_dir.mkdir(parents=True)
            (telemetry_dir / "telemetry.csv").write_text(
                "record_type,kind\n"
                "batch,batch\n"
                "epoch,epoch\n"
                "epoch,final\n",
                encoding="utf-8",
            )
            spec = _spec("offline-summary", result_dir)
            summary = summarize_suite_telemetry([spec])
            self.assertEqual(summary["batches_done"], 1)
            self.assertEqual(summary["epochs_done"], 1)

    def test_checkpoint_architecture_matching_distinguishes_stale_growth(self) -> None:
        current = {"width": 2, "depth": 2, "hidden_widths": [2, 1]}
        self.assertTrue(checkpoint_architecture_matches(current, {"hidden_widths": [2, 1]}))
        self.assertFalse(checkpoint_architecture_matches(current, {"hidden_widths": [1, 1]}))
        self.assertTrue(checkpoint_architecture_matches({"width": 2, "depth": 2}, {"width": 2, "depth": 2}))
        self.assertFalse(checkpoint_architecture_matches({"width": 2, "depth": 2}, {"width": 1, "depth": 2}))
        self.assertFalse(checkpoint_architecture_matches(current, None))

    def test_lg_nd_completed_stage_resume_returns_saved_best_value(self) -> None:
        from Dyn_DNN4OPF.models.adp_lg_nd_2_head_width_only import ADP_LG_ND_2Head
        from Dyn_DNN4OPF.models.adp_lg_nd_4_head_width_only import ADP_LG_ND_4Head
        from Dyn_DNN4OPF.models.adp_lg_nd_width_only import ADP_LG_ND

        config = SimpleNamespace(
            dims=[2, 1, 1],
            n_gen=1,
            n_bus=1,
            max_epochs=1,
            patience_es=1,
            max_width=2,
            max_neurons=100,
        )
        cases = (
            (ADP_LG_ND, "Dyn_DNN4OPF.models.adp_lg_nd_width_only", "single-head LG-ND"),
            (ADP_LG_ND_2Head, "Dyn_DNN4OPF.models.adp_lg_nd_2_head_width_only", "two-head LG-ND"),
            (ADP_LG_ND_4Head, "Dyn_DNN4OPF.models.adp_lg_nd_4_head_width_only", "four-head LG-ND"),
        )
        for model_class, module_name, _ in cases:
            with patch(f"{module_name}._build_telemetry", return_value=SimpleNamespace(
                load_stage_state=lambda: {"completed": True, "best_val": 0.25}
            )), patch(f"{module_name}.restore_stage_checkpoint", return_value=True), patch(
                "Dyn_DNN4OPF.models.adp_den_width_only.torch.cuda.is_available", return_value=False
            ):
                model = model_class(config)
                self.assertEqual(model.train_task([], []), 0.25)

    def test_round_robin_growth_reaches_each_target_width_for_all_head_counts(self) -> None:
        from Dyn_DNN4OPF.models.adp_den_width_only import ADP_DEN
        from Dyn_DNN4OPF.models.adp_den_2_head_width_only import ADP_DEN_2Head
        from Dyn_DNN4OPF.models.adp_den_4_head_width_only import ADP_DEN_4_head_width_only

        config = SimpleNamespace(
            dims=[2, 1, 1, 1, 1],
            n_gen=1,
            n_bus=1,
            width_growth_strategy="round_robin",
            max_width=2,
            max_neurons=100,
            max_epochs=0,
            patience_es=1,
        )
        with patch("Dyn_DNN4OPF.models.adp_den_width_only.torch.cuda.is_available", return_value=False), patch(
            "Dyn_DNN4OPF.models.adp_den_2_head_width_only.torch.cuda.is_available", return_value=False
        ), patch("Dyn_DNN4OPF.models.adp_den_4_head_width_only.torch.cuda.is_available", return_value=False):
            models = (ADP_DEN(config), ADP_DEN_2Head(config), ADP_DEN_4_head_width_only(config))
            for model in models:
                initial_widths = model.hidden_widths()
                self.assertGreaterEqual(len(initial_widths), 3)
                self.assertTrue(all(width == 1 for width in initial_widths))
                model.expand_width(1)
                self.assertEqual(model.hidden_widths(), [2, *([1] * (len(initial_widths) - 1))])
                self.assertEqual(model.width, 1)
                for _ in range(len(initial_widths) - 1):
                    model.expand_width(1)
                self.assertEqual(model.hidden_widths(), [2] * len(initial_widths))
                self.assertEqual(model.width, 2)

    def test_width_growth_never_exceeds_aggregate_neuron_cap(self) -> None:
        from Dyn_DNN4OPF.models.adp_den_width_only import ADP_DEN
        from Dyn_DNN4OPF.models.adp_den_2_head_width_only import ADP_DEN_2Head
        from Dyn_DNN4OPF.models.adp_den_4_head_width_only import ADP_DEN_4_head_width_only
        from Dyn_DNN4OPF.models.adp_den_depth_only import ADP_DEN as ADP_DEPTH_ONLY
        from Dyn_DNN4OPF.models.adp_den_2head_depth_only import ADP_DEN_2head_DepthOnly
        from Dyn_DNN4OPF.models.adp_den_4head_depth_only import ADP_DEN_4head_DepthOnly

        common = dict(
            dims=[2, 1, 1, 1],
            n_gen=1,
            n_bus=1,
            max_width=8,
            max_depth=5,
            max_epochs=0,
            patience_es=1,
        )
        constructors = (
            (ADP_DEN, 4),
            (ADP_DEN_2Head, 7),
            (ADP_DEN_4_head_width_only, 7),
        )
        with patch("Dyn_DNN4OPF.models.adp_den_width_only.torch.cuda.is_available", return_value=False), patch(
            "Dyn_DNN4OPF.models.adp_den_2_head_width_only.torch.cuda.is_available", return_value=False
        ), patch("Dyn_DNN4OPF.models.adp_den_4_head_width_only.torch.cuda.is_available", return_value=False):
            for constructor, cap in constructors:
                config = SimpleNamespace(**common, max_neurons=cap, width_growth_strategy="all_layers")
                model = constructor(config)
                initial_total = model.total_neurons()
                model.expand_width(1)
                self.assertEqual(model.total_neurons(), initial_total, constructor.__name__)
                self.assertLessEqual(model.total_neurons(), cap, constructor.__name__)

                model.width_growth_strategy = "round_robin"
                while model.can_grow_width(1, all_layers=False):
                    model.expand_width(1)
                self.assertLessEqual(model.total_neurons(), cap, constructor.__name__)
                self.assertFalse(model.can_grow_width(1, all_layers=False), constructor.__name__)

        depth_models = (
            (ADP_DEPTH_ONLY, 4),
            (ADP_DEN_2head_DepthOnly, 7),
            (ADP_DEN_4head_DepthOnly, 7),
        )
        with patch("Dyn_DNN4OPF.models.adp_den_depth_only.torch.cuda.is_available", return_value=False), patch(
            "Dyn_DNN4OPF.models.adp_den_2head_depth_only.torch.cuda.is_available", return_value=False
        ), patch("Dyn_DNN4OPF.models.adp_den_4head_depth_only.torch.cuda.is_available", return_value=False):
            for constructor, cap in depth_models:
                config = SimpleNamespace(**common, max_neurons=cap, width_growth_strategy="all_layers")
                model = constructor(config)
                initial_total = model.total_neurons()
                model.grow_width_all_layers(1)
                self.assertEqual(model.total_neurons(), initial_total, constructor.__name__)
                self.assertLessEqual(model.total_neurons(), cap, constructor.__name__)

                model.width_growth_strategy = "round_robin"
                while model.can_grow_width(1, all_layers=False):
                    model.grow_width_round_robin(model.width + 1, 1)
                self.assertLessEqual(model.total_neurons(), cap, constructor.__name__)
                self.assertFalse(model.can_grow_width(1, all_layers=False), constructor.__name__)

            for constructor in (ADP_DEPTH_ONLY, ADP_DEN_2head_DepthOnly, ADP_DEN_4head_DepthOnly):
                config = SimpleNamespace(
                    dims=[2, 3, 3, 1],
                    n_gen=1,
                    n_bus=1,
                    max_width=8,
                    max_depth=5,
                    max_neurons=100,
                    max_epochs=0,
                    patience_es=1,
                    seed_new_depth_to_current_width=True,
                )
                model = constructor(config)
                initial_total = model.total_neurons()
                model.max_neurons = initial_total + model.width - 1
                self.assertFalse(model.can_add_depth_stage(), constructor.__name__)
                model.max_neurons = initial_total + model.width
                self.assertTrue(model.can_add_depth_stage(), constructor.__name__)
                model.expand_depth(1)
                self.assertEqual(model.total_neurons(), initial_total + 3, constructor.__name__)

    def test_four_head_depth_growth_honors_seed_and_width_strategy(self) -> None:
        from Dyn_DNN4OPF.models.adp_den_4head_depth_only import ADP_DEN_4head_DepthOnly

        base = dict(
            dims=[2, 3, 3, 1],
            n_gen=1,
            n_bus=1,
            max_width=8,
            max_depth=5,
            max_neurons=100,
            max_epochs=0,
            patience_es=1,
        )
        with patch("Dyn_DNN4OPF.models.adp_den_4head_depth_only.torch.cuda.is_available", return_value=False):
            current = ADP_DEN_4head_DepthOnly(SimpleNamespace(**base, seed_new_depth_to_current_width=True))
            current.expand_depth(1)
            self.assertEqual(current.hidden_widths(), [3, 3, 3])

            round_robin = ADP_DEN_4head_DepthOnly(
                SimpleNamespace(**base, seed_new_depth_to_current_width=False, width_growth_strategy="round_robin")
            )
            round_robin.expand_depth(1)
            round_robin.grow_width_round_robin(3, 1)
            self.assertEqual(round_robin.hidden_widths(), [3, 3, 2])

            all_layers = ADP_DEN_4head_DepthOnly(
                SimpleNamespace(**base, seed_new_depth_to_current_width=False, width_growth_strategy="all_layers")
            )
            all_layers.expand_depth(1)
            all_layers.grow_width_all_layers(1)
            self.assertEqual(all_layers.hidden_widths(), [4, 4, 2])

    def test_stale_stage_checkpoint_cannot_restore_pre_growth_architecture(self) -> None:
        from Dyn_DNN4OPF.models.adp_den_width_only import ADP_DEN

        with patch("Dyn_DNN4OPF.models.adp_den_width_only.torch.cuda.is_available", return_value=False):
            model = ADP_DEN(
                SimpleNamespace(
                    dims=[2, 1, 1],
                    width_growth_strategy="all_layers",
                    max_width=3,
                    max_neurons=100,
                    max_epochs=0,
                    patience_es=1,
                )
            )
        stale_snapshot = model.snapshot_arch_and_state()
        model.expand_width(1)
        grown_width = model.width
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            torch.save(
                {
                    "model_state": stale_snapshot["state_dict"],
                    "best_snap": stale_snapshot,
                    "best_batch_snap": stale_snapshot,
                    "best_val": 1.0,
                    "best_batch_loss": 1.0,
                    "epoch": 4,
                    "global_batch": 0,
                    "patience": 1,
                },
                checkpoint_dir / "checkpoint_last.pt",
            )
            model.train_with_early_stopping([], [], checkpoint_dir=str(checkpoint_dir))
        self.assertEqual(model.width, grown_width)

    def test_pressure_wrapper_reopens_from_ram_or_gpu_release_without_swap_gate(self) -> None:
        self.assertTrue(
            pressure_aware_wrapper._external_drop_reopens_gates(
                host_ready=False,
                gpu_ready=True,
                host_drop_mib=500.0,
                gpu_drop_mib=0.0,
                threshold_mib=500.0,
            )
        )
        self.assertFalse(
            pressure_aware_wrapper._external_drop_reopens_gates(
                host_ready=False,
                gpu_ready=True,
                host_drop_mib=500.0,
                gpu_drop_mib=0.0,
                threshold_mib=0.0,
            )
        )
        self.assertFalse(
            pressure_aware_wrapper._external_drop_reopens_gates(
                host_ready=True,
                gpu_ready=False,
                host_drop_mib=500.0,
                gpu_drop_mib=0.0,
                threshold_mib=500.0,
            )
        )
        self.assertTrue(
            pressure_aware_wrapper._external_drop_reopens_gates(
                host_ready=True,
                gpu_ready=False,
                host_drop_mib=0.0,
                gpu_drop_mib=500.0,
                threshold_mib=500.0,
            )
        )

    @unittest.skipIf(os.name == "nt", "POSIX process termination is required")
    def test_massive_runner_pressure_path_persists_and_relaunches_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run_root"
            results_dir = root / "results"
            spec = JobSpec(
                job_id="pressure-recovery",
                phase="pressure-recovery",
                family="pressure-recovery",
                case_name="pglib_opf_case14_ieee",
                repeat_index=1,
                depth=1,
                width=1,
                seed=42,
                results_dir=str(results_dir),
                command=(sys.executable, "-c", "import time; time.sleep(30)"),
            )
            runtime = JobRuntime(
                spec=spec,
                state_path=run_root / "jobs" / spec.job_id / "job_state.json",
                log_path=run_root / "jobs" / spec.job_id / "job.log",
            )
            args = SimpleNamespace(gpu_device_index=0)
            suspended = []
            active = []
            try:
                launch_job(args, runtime, concurrency=1, device_mode="cpu", run_root=run_root)
                active.append(runtime)
                first_proc = runtime.proc
                first_pid = first_proc.pid
                time.sleep(0.1)
                self.assertTrue(
                    _suspend_runtime_for_pressure(
                        runtime,
                        reason="host_ram_pressure",
                        suspended=suspended,
                        active=active,
                        run_root=run_root,
                    )
                )
                self.assertTrue(runtime.paused)
                self.assertNotIn(runtime, active)
                paused_state = json.loads(runtime.state_path.read_text(encoding="utf-8"))
                self.assertEqual(paused_state["status"], "paused")
                self.assertEqual(paused_state["pause_reason"], "host_ram_pressure")
                self.assertIsNone(runtime.proc)
                self.assertIsNotNone(first_proc.poll())
                first_proc.wait(timeout=3)
                self.assertFalse(Path(f"/proc/{first_pid}").exists())

                _resume_runtime_from_pause(
                    runtime,
                    args,
                    concurrency=1,
                    run_root=run_root,
                    device_mode="cpu",
                    active=active,
                    launch_context={"resumed_from_pause": True},
                )
                self.assertIsNotNone(runtime.proc)
                self.assertIn(runtime, active)
                self.assertNotEqual(runtime.proc.pid, first_pid)
                self.assertIsNone(runtime.proc.poll())
                resumed_state = json.loads(runtime.state_path.read_text(encoding="utf-8"))
                self.assertEqual(resumed_state["status"], "running")
                self.assertEqual(resumed_state["attempt"], 2)
            finally:
                last_proc = runtime.proc
                if last_proc is not None and last_proc.poll() is None:
                    _terminate_process_tree(last_proc)
                if last_proc is not None:
                    last_proc.wait(timeout=3)
                _close_runtime_log_handle(runtime)

    def test_legacy_checkpoint_is_rejected_before_model_restore(self) -> None:
        from Dyn_DNN4OPF.models.adp_den_width_only import ADP_DEN

        with patch("Dyn_DNN4OPF.models.adp_den_width_only.torch.cuda.is_available", return_value=False):
            model = ADP_DEN(
                SimpleNamespace(
                    dims=[2, 2, 1],
                    case_name="pglib_opf_case14_ieee",
                    max_epochs=0,
                    patience_es=1,
                    max_width=2,
                    max_neurons=100,
                )
            )
        latest_state = {key: torch.full_like(value, 3.0) for key, value in model.state_dict().items()}
        checkpoint = {
            **capture_training_state([], []),
            "model_state": latest_state,
            "best_snap": model.snapshot_arch_and_state(),
            "best_batch_snap": model.snapshot_arch_and_state(),
            "best_val": 1.0,
            "best_batch_loss": 1.0,
            "epoch": 4,
            "global_batch": 0,
            "patience": 1,
        }

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "stages" / "000_initial"
            checkpoint_dir.mkdir(parents=True)
            torch.save(checkpoint, checkpoint_dir / "checkpoint_last.pt")
            observed_before_restore = []
            original_restore = model.restore_arch_and_state

            def capture_before_restore(snapshot):
                observed_before_restore.append(
                    {key: value.detach().clone() for key, value in model.state_dict().items()}
                )
                return original_restore(snapshot)

            with patch.object(model, "restore_arch_and_state", side_effect=capture_before_restore):
                with self.assertRaisesRegex(ValueError, "exact-resume failure:"):
                    model.train_with_early_stopping([], [], checkpoint_dir=str(checkpoint_dir))

        self.assertEqual(observed_before_restore, [])

    def test_telemetry_stage_state_decodes_boolean_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_json = Path(temporary) / "case.json"
            case_json.write_text(json.dumps({"grid": {}}), encoding="utf-8")
            telemetry = TrainingTelemetry(
                results_dir=Path(temporary),
                model_name="test-model",
                case_name="pglib_opf_case14_ieee",
                phase="train",
                task_id=0,
                num_gens=1,
                num_buses=1,
                case_json_path=case_json,
                constraints={},
            )
            telemetry.save_stage_state({"stage_index": 0, "completed": False, "best_val": 1.0})
            state = telemetry.load_stage_state()
            self.assertIsNotNone(state)
            self.assertIs(state["completed"], False)

            telemetry.save_stage_state({"stage_index": 1, "completed": True, "best_val": 0.5})
            state = telemetry.load_stage_state()
            self.assertIs(state["completed"], True)

    def test_active_legacy_stage_state_uses_exact_training_checkpoint_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_json = root / "case.json"
            case_json.write_text(json.dumps({"grid": {}}), encoding="utf-8")
            telemetry = TrainingTelemetry(
                results_dir=root, model_name="test-model", case_name="pglib_opf_case14_ieee",
                phase="train", task_id=0, num_gens=1, num_buses=1,
                case_json_path=case_json, constraints={},
            )
            stage_dir = root / "stages" / "002_expand_002"
            stage_dir.mkdir(parents=True)
            checkpoint_path = stage_dir / "checkpoint_last.pt"
            torch.save({
                "checkpoint_schema_version": 2,
                "model_state": {}, "opt_state": {}, "sched_state": {}, "epoch": 5,
                "global_batch": 1, "train_sampler_state": {"order": [0, 1], "position": 1},
                "rng_state": {"python": (3, (1, 2), None), "numpy": {}, "torch_cpu": torch.get_rng_state()},
                "best_val": 1.0, "best_batch_loss": 1.0, "best_batch_snap": {}, "patience": 3,
                "loader_contract": {}, "train_loader_generator_state": torch.Generator().get_state(),
                "val_loader_generator_state": torch.Generator().get_state(), "backend_state": {},
                "best_snap": {"hidden_widths": [4, 4]},
            }, checkpoint_path)
            telemetry.stage_state_path.write_text(json.dumps({
                "stage_index": 2, "stage_name": "expand_002", "active_stage": True,
                "completed": False, "stage_checkpoint": str(checkpoint_path),
                "current_hidden_widths": [4, 4],
            }), encoding="utf-8")

            state = telemetry.load_stage_state()

            self.assertTrue(state["active_stage"])
            self.assertTrue(telemetry._stage_rng_deferred_to_checkpoint)

    def test_inactive_controller_rng_is_restored_after_architecture_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_json = root / "case.json"
            case_json.write_text(json.dumps({"grid": {}}), encoding="utf-8")
            telemetry = TrainingTelemetry(
                results_dir=root, model_name="test-model", case_name="pglib_opf_case14_ieee",
                phase="train", task_id=0, num_gens=1, num_buses=1,
                case_json_path=case_json, constraints={},
            )
            stage_dir = root / "stages" / "001_growth"
            stage_dir.mkdir(parents=True)
            checkpoint_path = stage_dir / "checkpoint_last.pt"
            snapshot = {"hidden_widths": [4], "state_dict": {}}
            torch.save({"best_snap": snapshot}, checkpoint_path)
            torch.manual_seed(73)
            telemetry.save_stage_state({
                "stage_index": 1, "stage_name": "growth", "completed": False,
                "active_stage": False, "current_hidden_widths": [4],
                "stage_checkpoint": str(checkpoint_path),
            })
            sidecar = json.loads(telemetry.stage_state_path.read_text(encoding="utf-8"))
            saved_rng = torch.load(telemetry.stage_state_path.parent / sidecar["rng_checkpoint"], map_location="cpu", weights_only=False)
            torch.manual_seed(991)
            before_load = torch.get_rng_state().clone()

            loaded = telemetry.load_stage_state()

            self.assertTrue(telemetry._stage_rng_deferred_to_checkpoint)
            self.assertTrue(torch.equal(torch.get_rng_state(), before_load))

            class RebuildModel:
                def restore_arch_and_state(self, saved_snapshot):
                    self.widths = list(saved_snapshot["hidden_widths"])
                    torch.rand(17)

            self.assertTrue(restore_stage_architecture(RebuildModel(), loaded))
            self.assertTrue(torch.equal(torch.get_rng_state(), saved_rng["torch_cpu"]))

    def test_active_legacy_stage_state_rejects_checkpoint_with_wrong_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_json = root / "case.json"
            case_json.write_text(json.dumps({"grid": {}}), encoding="utf-8")
            telemetry = TrainingTelemetry(
                results_dir=root, model_name="test-model", case_name="pglib_opf_case14_ieee",
                phase="train", task_id=0, num_gens=1, num_buses=1,
                case_json_path=case_json, constraints={},
            )
            stage_dir = root / "stages" / "002_expand_002"
            stage_dir.mkdir(parents=True)
            checkpoint_path = stage_dir / "checkpoint_last.pt"
            torch.save({
                "checkpoint_schema_version": 2,
                "model_state": {}, "opt_state": {}, "sched_state": {}, "epoch": 5,
                "global_batch": 1, "train_sampler_state": {"order": [0, 1], "position": 1},
                "rng_state": {"python": (3, (1, 2), None), "numpy": {}, "torch_cpu": torch.get_rng_state()},
                "best_val": 1.0, "best_batch_loss": 1.0, "best_batch_snap": {}, "patience": 3,
                "loader_contract": {}, "train_loader_generator_state": torch.Generator().get_state(),
                "val_loader_generator_state": torch.Generator().get_state(), "backend_state": {},
                "best_snap": {"hidden_widths": [8, 8]},
            }, checkpoint_path)
            telemetry.stage_state_path.write_text(json.dumps({
                "stage_index": 2, "stage_name": "expand_002", "active_stage": True,
                "completed": False, "stage_checkpoint": str(checkpoint_path),
                "current_hidden_widths": [4, 4],
            }), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "exact-resume failure"):
                telemetry.load_stage_state()

    @unittest.skipIf(os.name == "nt", "POSIX process termination is required")
    def test_failed_resume_relaunch_keeps_job_paused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = dataclasses.replace(_spec("resume-failure", root / "results"), command=(str(root / "does-not-exist"),))
            runtime = JobRuntime(
                spec=spec,
                state_path=root / "run_root" / "jobs" / spec.job_id / "job_state.json",
                log_path=root / "run_root" / "jobs" / spec.job_id / "job.log",
                paused=True,
                pause_reason="host_ram_pressure",
            )
            args = SimpleNamespace(gpu_device_index=0)
            with self.assertRaises((FileNotFoundError, OSError)):
                _resume_runtime_from_pause(runtime, args, 1, root / "run_root", device_mode="cpu")
            self.assertTrue(runtime.paused)
            self.assertEqual(runtime.pause_reason, "host_ram_pressure")
            state = json.loads(runtime.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "paused")
            self.assertEqual(state["reason"], "resume_launch_error")

    def test_native_adaptive_engines_use_stage_checkpoint_mirrors(self) -> None:
        model_root = Path(__file__).resolve().parents[1] / "DNN" / "VANILLA" / "Dyn_DNN4OPF" / "models"
        engines = (
            "adp_den_width_only.py",
            "adp_den_2_head_width_only.py",
            "adp_den_4_head_width_only.py",
            "adp_den_depth_only.py",
            "adp_den_2head_depth_only.py",
            "adp_den_4head_depth_only.py",
        )
        for filename in engines:
            source = (model_root / filename).read_text(encoding="utf-8")
            checkpoint_calls = re.findall(
                r"telemetry\.save_(?:last|best|best_batch)_checkpoint\(([^\n]+)",
                source,
            )
            self.assertTrue(checkpoint_calls, filename)
            self.assertTrue(all("mirror_path=" in call for call in checkpoint_calls), filename)
            load_body = source.split("def train_with_early_stopping", 1)[1].split(
                "train_batches =", 1
            )[0]
            self.assertNotIn("restore_arch_and_state(best_snap)", load_body, filename)

    def test_telemetry_checkpoint_mirror_supports_stage_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            telemetry = object.__new__(TrainingTelemetry)
            telemetry.last_checkpoint_path = Path(temporary) / "telemetry" / "checkpoint_last.pt"
            telemetry.best_checkpoint_path = Path(temporary) / "telemetry" / "checkpoint_best.pt"
            telemetry.best_batch_checkpoint_path = Path(temporary) / "telemetry" / "checkpoint_best_batch.pt"
            stage_last = Path(temporary) / "stages" / "000_initial" / "checkpoint_last.pt"
            stage_best = Path(temporary) / "stages" / "000_initial" / "checkpoint_best.pt"
            stage_best_batch = Path(temporary) / "stages" / "000_initial" / "checkpoint_best_batch.pt"
            payload = {"epoch": 7, "global_batch": 3, "model_state": {"weight": 1}}

            telemetry.save_last_checkpoint(payload, mirror_path=stage_last)
            telemetry.save_best_checkpoint(payload, mirror_path=stage_best)
            telemetry.save_best_batch_checkpoint(payload, mirror_path=stage_best_batch)

            self.assertEqual(telemetry.last_checkpoint_path.read_bytes(), stage_last.read_bytes())
            self.assertEqual(telemetry.best_checkpoint_path.read_bytes(), stage_best.read_bytes())
            self.assertEqual(telemetry.best_batch_checkpoint_path.read_bytes(), stage_best_batch.read_bytes())

    def test_scopes_progress_and_ignores_state_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            running_results = root / "running_results"
            failed_results = root / "failed_results"
            paused_results = root / "paused_results"
            stale_results = root / "stale_results"
            running = _spec("running", running_results)
            failed = _spec("failed", failed_results)
            paused = _spec("paused", paused_results)
            stale = _spec("stale", stale_results)

            _write_state(root, running, "running")
            running_progress = running_results / "telemetry" / "task" / "progress_state.json"
            running_progress.parent.mkdir(parents=True, exist_ok=True)
            running_progress.write_text(json.dumps({"history_rows": 10, "final_epoch": 2}), encoding="utf-8")

            _write_state(root, failed, "failed")
            failed_progress = failed_results / "telemetry" / "task" / "progress_state.json"
            failed_progress.parent.mkdir(parents=True, exist_ok=True)
            failed_progress.write_text(json.dumps({"history_rows": 100, "final_epoch": 50}), encoding="utf-8")

            _write_state(root, paused, "paused")
            paused_log = root / "jobs" / paused.job_id / "job.log"
            paused_log.parent.mkdir(parents=True, exist_ok=True)
            paused_log.write_text("batch=1/1 done\nepoch=1 val=1.0\n", encoding="utf-8")

            _write_state(root, stale, "failed")

            summary = summarize_suite_runtime_progress(root, [running, failed, paused, stale])

        self.assertEqual(summary["batches_done"], 111)
        self.assertEqual(summary["epochs_done"], 53)
        self.assertEqual(summary["live_batches_done"], 11)
        self.assertEqual(summary["live_epochs_done"], 3)
        self.assertEqual(summary["telemetry_jobs"], 3)
        self.assertEqual(summary["telemetry_dirs"], 3)

    def test_bundle_rejects_traversal_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "results"
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                member = tarfile.TarInfo("results/../outside.txt")
                member.size = 4
                archive.addfile(member, io.BytesIO(b"oops"))
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r") as archive:
                with self.assertRaises(ValueError):
                    _safe_extract_tar(archive, destination.parent, destination.name)
            self.assertFalse((Path(temporary) / "outside.txt").exists())

    def test_bundle_round_trip_excludes_telemetry_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            (results / "checkpoint.pt").write_bytes(b"checkpoint")
            (results / "telemetry.csv").write_text("should not be bundled", encoding="utf-8")
            bundle = root / "bundle.tar.gz"
            self.assertEqual(export_results_bundle(results, bundle), bundle)
            (results / "checkpoint.pt").unlink()
            (results / "telemetry.csv").unlink()
            results.rmdir()
            self.assertTrue(restore_results_bundle(results, bundle))
            self.assertEqual((results / "checkpoint.pt").read_bytes(), b"checkpoint")
            self.assertFalse((results / "telemetry.csv").exists())

    def test_chunked_bundle_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            (results / "checkpoint.pt").write_bytes(b"x" * 4096)
            bundle = root / "bundle.tar.gz"
            with patch.object(opf_resume_bundle, "BUNDLE_PART_SIZE_BYTES", 64):
                self.assertEqual(export_results_bundle(results, bundle), bundle)
                self.assertFalse(bundle.exists())
                self.assertGreater(len(list(root.glob("bundle.tar.gz.part*"))), 1)
                (results / "checkpoint.pt").unlink()
                results.rmdir()
                self.assertTrue(restore_results_bundle(results, bundle))
            self.assertEqual((results / "checkpoint.pt").read_bytes(), b"x" * 4096)

    def test_sharded_telemetry_rehydrates_and_rejects_bad_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            shard_dir = results / "telemetry" / "task"
            shard_dir.mkdir(parents=True)
            payload = b"epoch,batch\n1,2\n"
            part = shard_dir / "telemetry.csv.part000"
            part.write_bytes(payload)
            manifest = shard_dir / "telemetry.csv.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "original_path": "results/telemetry/task/telemetry.csv",
                        "parts": [part.name],
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(opf_resume_bundle, "REPO_ROOT", root):
                self.assertEqual(rehydrate_sharded_telemetry(results), 1)
            self.assertEqual((shard_dir / "telemetry.csv").read_bytes(), payload)

            (shard_dir / "telemetry.csv").unlink()
            manifest.write_text(
                json.dumps(
                    {
                        "original_path": "results/telemetry/task/telemetry.csv",
                        "parts": [part.name],
                        "sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(opf_resume_bundle, "REPO_ROOT", root):
                with self.assertRaises(ValueError):
                    rehydrate_sharded_telemetry(results)
            self.assertFalse((shard_dir / "telemetry.csv").exists())

    def test_retry_budget_zero_is_unbounded_and_separate_from_pause_attempts(self) -> None:
        spec = _spec("retry", Path("/tmp/retry-results"))
        runtime = JobRuntime(
            spec=spec,
            state_path=Path("/tmp/retry-state.json"),
            log_path=Path("/tmp/retry.log"),
        )
        runtime.attempt = 20
        self.assertTrue(should_retry(runtime, max_retries=1))
        runtime.retry_count = 1
        self.assertFalse(should_retry(runtime, max_retries=1))
        self.assertTrue(should_retry(runtime, max_retries=0))

    def test_cleanup_does_not_match_runner_filename_in_pycompile_input(self) -> None:
        self.assertFalse(
            _is_owned_command(
                [
                    ".venv_opf_adp/bin/python",
                    "-m",
                    "py_compile",
                    "utils/opf_massive_suite_runner.py",
                ]
            )
        )
        self.assertTrue(
            _is_owned_command(
                [
                    ".venv_opf_adp/bin/python",
                    "utils/opf_massive_suite_runner.py",
                    "--mode",
                    "case",
                ]
            )
        )

    @unittest.skipIf(os.name == "nt", "POSIX process groups provide the tested suspend contract")
    def test_pressure_wrapper_suspends_and_resumes_process_group(self) -> None:
        process = subprocess.Popen(["sleep", "10"], start_new_session=True)
        pressure_aware_wrapper._ACTIVE_CHILD_PROCESS = process
        try:
            pressure_aware_wrapper._pause_active_child_tree()
            time.sleep(0.1)
            self.assertEqual(process.poll(), None)
            _, status = os.waitpid(process.pid, os.WUNTRACED | os.WNOHANG)
            self.assertTrue(os.WIFSTOPPED(status))
            self.assertEqual(os.WSTOPSIG(status), signal.SIGSTOP)
            pressure_aware_wrapper._resume_active_child_tree()
            time.sleep(0.1)
            self.assertEqual(process.poll(), None)
        finally:
            pressure_aware_wrapper._ACTIVE_CHILD_PROCESS = process
            pressure_aware_wrapper._resume_active_child_tree()
            process.terminate()
            process.wait(timeout=3)
            pressure_aware_wrapper._ACTIVE_CHILD_PROCESS = None


if __name__ == "__main__":
    unittest.main()
