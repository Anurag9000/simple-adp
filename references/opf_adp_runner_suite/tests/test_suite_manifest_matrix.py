import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils.opf_massive_suite_runner import (
    GpuMemorySample,
    JobRuntime,
    JobSpec,
    MemorySample,
    binary_search_width,
    compute_opf_params,
    pressure_pause_reopened,
    select_launch_device,
)
from utils.opf_massive_suite_runner import validate_args

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "utils" / "opf_massive_suite_runner.py"
CASES = (
    "pglib_opf_case14_ieee",
    "pglib_opf_case30_ieee",
    "pglib_opf_case57_ieee",
    "pglib_opf_case118_ieee",
    "pglib_opf_case500_goc",
)


def build_manifest(*extra: str) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="opf_manifest_test_") as temp_dir:
        root = Path(temp_dir)
        command = [
            sys.executable,
            str(RUNNER),
            "--mode",
            "massive",
            "--run-root",
            str(root / "run_root"),
            "--results-dir",
            str(root / "results"),
            "--cases",
            ",".join(CASES),
            "--repeat-count",
            "1",
            "--max-retries",
            "0",
            "--seed",
            "42",
            "--dry-run",
            *extra,
        ]
        environment = os.environ.copy()
        environment.pop("INCLUDE_VANILLA", None)
        subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
        return json.loads((root / "run_root" / "dry_run_manifest.json").read_text(encoding="utf-8"))


class SuiteManifestMatrixTests(unittest.TestCase):
    def test_epoch_limit_inf_is_unlimited_and_defaults_are_uncapped(self) -> None:
        from itertools import islice

        from utils.opf_shared_defaults import epoch_indices, parse_epoch_limit
        from Dyn_DNN4OPF.utils.run_defaults import RUNNER_DEFAULTS

        self.assertEqual(parse_epoch_limit("inf"), 0)
        self.assertEqual(parse_epoch_limit("unlimited"), 0)
        self.assertEqual(parse_epoch_limit("0"), 0)
        self.assertEqual(list(islice(epoch_indices(0, 7), 3)), [7, 8, 9])
        self.assertEqual(RUNNER_DEFAULTS["max_epochs_vanilla"], 0)
        self.assertEqual(RUNNER_DEFAULTS["max_epochs_adp"], 0)

    def test_explicit_zero_override_is_not_replaced_by_environment(self) -> None:
        from utils.opf_suite_launcher import _arg_or_env

        with patch.dict(os.environ, {"TEST_REPEAT_OVERRIDE": "5"}, clear=False):
            self.assertEqual(_arg_or_env(0, "TEST_REPEAT_OVERRIDE", 1), 0)
            self.assertEqual(_arg_or_env(None, "TEST_REPEAT_OVERRIDE", 1), "5")

    def test_explicit_seed_overrides_global_seed_environment(self) -> None:
        from utils.ml_backends import resolve_seed

        with patch.dict(os.environ, {"OPF_ADP_GLOBAL_SEED": "42"}, clear=False):
            self.assertEqual(resolve_seed(123), 123)
            self.assertEqual(resolve_seed(None), 42)

    def test_active_model_registry_contains_only_importable_entry_points(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import importlib, main; "
                "assert main.MODEL_MODULES; "
                "[importlib.import_module(module) for module in main.MODEL_MODULES.values()]; "
                "assert 'vanilla_mlp' in main.MODEL_MODULES"
            ),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "DNN" / "VANILLA"), environment.get("PYTHONPATH", "")]
        )
        subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True)

    def test_all_cases_include_the_same_complete_manifest(self) -> None:
        jobs = build_manifest("--include-vanilla")
        self.assertEqual(len(jobs), 895)
        self.assertEqual(Counter(job["case_name"] for job in jobs), Counter({case: 179 for case in CASES}))
        self.assertEqual(len({job["job_id"] for job in jobs}), len(jobs))
        self.assertEqual(len({job["seed"] for job in jobs}), len(jobs))

        per_case = [job for job in jobs if job["case_name"] == CASES[0]]
        families = Counter(job["family"] for job in per_case)
        self.assertEqual(sum(families.values()), 179)
        self.assertEqual(sum(value for key, value in families.items() if key != "vanilla"), 129)
        self.assertEqual(families["vanilla"], 50)
        self.assertEqual(Counter(job["head_variant"] for job in per_case), Counter({"single_head": 93, "two_heads": 43, "four_heads": 43}))
        for job in per_case:
            command = job["command"]
            self.assertIn("max_neurons=1000000000", command)
            self.assertIn("patience=10", command)

    def test_vanilla_is_enabled_by_default_for_direct_runner(self) -> None:
        jobs = build_manifest()
        self.assertEqual(len(jobs), 895)
        self.assertEqual(sum(job["family"] == "vanilla" for job in jobs), 250)

    def test_vanilla_opt_out_and_head_task_filters_are_exact(self) -> None:
        self.assertEqual(len(build_manifest("--no-include-vanilla")), 645)
        self.assertEqual(
            len(build_manifest("--no-include-vanilla", "--head-variants", "single")),
            215,
        )
        width_only = build_manifest(
            "--no-include-vanilla",
            "--head-variants",
            "single",
            "--tasks",
            "width_only",
        )
        self.assertEqual(len(width_only), 75)
        self.assertEqual({job["phase"] for job in width_only}, {
            "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement",
            "patience_based_width_growth_round_robin_layer_wise",
            "patience_based_width_growth_all_layers_together",
        })

    def test_seed_derivation_is_reproducible_and_repeat_specific(self) -> None:
        first = build_manifest("--include-vanilla", "--repeat-count", "2")
        second = build_manifest("--include-vanilla", "--repeat-count", "2")
        first_seeds = {(job["job_id"], job["seed"]) for job in first}
        second_seeds = {(job["job_id"], job["seed"]) for job in second}
        self.assertEqual(first_seeds, second_seeds)
        repeat_seeds = {
            job["seed"]
            for job in first
            if job["case_name"] == CASES[0]
            and job["family"] == "vanilla"
            and job["depth"] == 1
            and job["width"] == 1
        }
        self.assertEqual(len(repeat_seeds), 2)

    def test_argument_validation_rejects_unsafe_pressure_and_retry_values(self) -> None:
        args = SimpleNamespace(
            concurrency=0,
            gpu_device_index=0,
            pressure_poll_interval_sec=1.0,
            post_launch_sample_delay_sec=0.0,
            host_ram_resume_avail_mib=1280,
            host_ram_stop_avail_mib=1024,
            gpu_resume_avail_mib=50,
            gpu_stop_avail_mib=100,
            batch_backoff_factor=0.5,
            repeat_count=0,
            max_retries=1,
            prediction_repeats=1,
            missing_present_task_repeats=1,
            max_active_jobs=0,
            max_active_gpu_jobs=0,
            launch_host_reserve_mib=0,
            launch_gpu_reserve_mib=0,
            gpu_child_gate_close_buffer_mib=0,
            host_child_gate_close_buffer_mib=0,
            external_memory_drop_reopen_mib=500,
            width_to_depth_width=1,
            width_to_depth_depth=1,
            width_only_width=1,
            max_width=512,
            max_depth=5,
            max_neurons=1000000000,
            max_epochs_vanilla=100,
            max_epochs_adp=100,
            width_only_depths="1",
            depth_only_widths="1",
            width_to_depth_widths="1",
            vanilla_depths="1",
            vanilla_widths="1",
        )
        with self.assertRaises(SystemExit):
            validate_args(args)
        args.gpu_resume_avail_mib = 500
        args.max_retries = -1
        with self.assertRaises(SystemExit):
            validate_args(args)

    def test_architecture_and_neuron_caps_are_propagated_and_validated(self) -> None:
        jobs = build_manifest("--include-vanilla", "--max-neurons", "12345")
        self.assertTrue(jobs)
        self.assertTrue(all("max_neurons=12345" in job["command"] for job in jobs))
        with self.assertRaises(subprocess.CalledProcessError):
            build_manifest("--depth-only-widths", "32", "--max-width", "16")

    def test_scheduler_selection_and_parameter_caps(self) -> None:
        args = SimpleNamespace(
            cpu_only=False,
            scheduler="gpu_first",
            gpu_resume_avail_mib=500,
            host_ram_resume_avail_mib=1280,
            max_active_gpu_jobs=1,
            max_active_jobs=2,
        )
        host = MemorySample(16000, 4000, 75.0, 4000, 0, 0.0)
        gpu = GpuMemorySample(6144, 1000, 5144, 16.0)
        spec = build_manifest("--no-include-vanilla", "--head-variants", "single")[0]
        job = JobRuntime(
            spec=JobSpec(
                job_id=spec["job_id"],
                phase=spec["phase"],
                family=spec["family"],
                case_name=spec["case_name"],
                repeat_index=spec["repeat_index"],
                depth=spec["depth"],
                width=spec["width"],
                seed=spec["seed"],
                results_dir=spec["results_dir"],
                command=tuple(spec["command"]),
                head_variant=spec["head_variant"],
            ),
            state_path=Path("/tmp/selection-state.json"),
            log_path=Path("/tmp/selection.log"),
        )
        self.assertEqual(select_launch_device(job, args, host, gpu, 0, 0), "gpu")
        self.assertEqual(select_launch_device(job, args, host, gpu, 1, 0), "cpu")
        self.assertEqual(select_launch_device(job, args, host, None, 0, 0), "cpu")
        host_low = MemorySample(16000, 1000, 94.0, 4000, 0, 0.0)
        self.assertIsNone(select_launch_device(job, args, host_low, None, 0, 0))
        self.assertGreaterEqual(compute_opf_params(5, 512), compute_opf_params(1, 1))
        target = compute_opf_params(2, 64)
        self.assertGreaterEqual(binary_search_width(2, target, max_width=512), 16)

    def test_scheduler_selection_ignores_legacy_oom_lane_flags(self) -> None:
        args = SimpleNamespace(
            cpu_only=False,
            scheduler="gpu_first",
            gpu_resume_avail_mib=500,
            host_ram_resume_avail_mib=1280,
            max_active_gpu_jobs=0,
            max_active_jobs=0,
        )
        host = MemorySample(16000, 4000, 75.0, 4000, 0, 0.0)
        gpu = GpuMemorySample(6144, 1000, 5144, 16.0)
        spec = build_manifest("--no-include-vanilla", "--head-variants", "single")[0]
        job = JobRuntime(
            spec=JobSpec(
                job_id=spec["job_id"], phase=spec["phase"], family=spec["family"],
                case_name=spec["case_name"], repeat_index=spec["repeat_index"],
                depth=spec["depth"], width=spec["width"], seed=spec["seed"],
                results_dir=spec["results_dir"], command=tuple(spec["command"]),
                head_variant=spec["head_variant"],
            ),
            state_path=Path("/tmp/selection-state-fallback.json"),
            log_path=Path("/tmp/selection-fallback.log"),
            force_cpu_after_cuda_oom=True,
        )
        self.assertEqual(
            select_launch_device(job, args, host, gpu, 0, 0),
            "gpu",
        )
        self.assertEqual(
            select_launch_device(
                job, args, host, gpu, 0, 0,
                gpu_admission_enabled=False,
            ),
            "cpu",
        )
        self.assertEqual(
            select_launch_device(
                job, args, host, gpu, 0, 0,
                gpu_admission_enabled=False,
            ),
            "cpu",
        )
        self.assertIsNone(
            select_launch_device(
                job, args, host, gpu, 0, 0,
                host_admission_enabled=False,
                gpu_admission_enabled=False,
            )
        )

    def test_combined_pressure_requires_both_domains_to_reopen(self) -> None:
        common = dict(
            pause_reason="gpu_memory_pressure+host_ram_pressure",
            device_mode="gpu",
            host_used_mib=14500,
            gpu_used_mib=5800,
            host_resume_avail_mib=1280,
            gpu_resume_avail_mib=500,
            host_peak_used_mib=14500,
            gpu_peak_used_mib=5800,
            external_drop_threshold_mib=100,
        )
        self.assertFalse(pressure_pause_reopened(host_available_mib=900, gpu_free_mib=1000, **common))
        self.assertFalse(pressure_pause_reopened(host_available_mib=2000, gpu_free_mib=100, **common))
        self.assertTrue(pressure_pause_reopened(host_available_mib=2000, gpu_free_mib=1000, **common))

    def test_external_drop_can_reopen_the_matching_pressure_domain(self) -> None:
        common = dict(
            pause_reason="host_ram_pressure",
            device_mode="gpu",
            host_available_mib=900,
            gpu_free_mib=1000,
            host_used_mib=15000,
            gpu_used_mib=5000,
            host_resume_avail_mib=1280,
            gpu_resume_avail_mib=500,
            host_peak_used_mib=15000,
            gpu_peak_used_mib=5000,
            external_drop_threshold_mib=100,
        )
        self.assertFalse(pressure_pause_reopened(**common))
        self.assertTrue(pressure_pause_reopened(external_host_drop_mib=100, **common))

    def test_concurrency_file_rejects_invalid_values(self) -> None:
        from utils.opf_massive_suite_runner import resolve_concurrency

        args = SimpleNamespace(concurrency=0, concurrency_file="/tmp/opf-invalid-concurrency")
        Path(args.concurrency_file).write_text("-1\n", encoding="utf-8")
        try:
            with self.assertRaises(SystemExit):
                resolve_concurrency(args)
        finally:
            Path(args.concurrency_file).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
