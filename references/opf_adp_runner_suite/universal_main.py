
import os
import sys
import argparse
import subprocess
import signal
import glob
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import datetime
import time
import importlib.util
import torch
from utils.opf_shared_defaults import parse_epoch_limit

try:
    from utils.runtime_tuning import bootstrap_runtime
    bootstrap_runtime("universal_main")
except ImportError as e:
    print(f"Warning: runtime_tuning not found: {e}")

try:
    from utils.ml_backends import enable_optional_gpu_accelerators

    enable_optional_gpu_accelerators()
except ImportError:
    pass

# Add DNN/VANILLA to path to access legacy config/main if needed
VANILLA_PATH = Path(__file__).parent / "DNN" / "VANILLA"
sys.path.append(str(VANILLA_PATH))

ADP_UNBOUNDED_DEFAULTS: Dict[str, Any] = {
    "max_depth": 1_000_000_000,
    "max_width": 1_000_000_000,
    "max_neurons": 1_000_000_000,
    "max_epochs": 0,
    "ex_k": 1,
    "ex_k_width": 1,
    "ex_k_depth": 1,
    "patience": 10,
    "patience_es": 10,
    "patience_width": 10,
    "patience_depth": 10,
    "patience_width_exp": 10,
    "patience_depth_exp": 10,
    "trials_depth": 10,
    "trials_width": 10,
}

torch.set_default_dtype(torch.float64)

_ACTIVE_CHILD_PROCESS: Optional[subprocess.Popen] = None


def _terminate_active_child_tree(timeout_sec: float = 10.0) -> None:
    global _ACTIVE_CHILD_PROCESS
    proc = _ACTIVE_CHILD_PROCESS
    if proc is None:
        return
    if proc.poll() is not None:
        _ACTIVE_CHILD_PROCESS = None
        return

    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    else:
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.time() + max(0.1, float(timeout_sec))
    while time.time() < deadline:
        if proc.poll() is not None:
            _ACTIVE_CHILD_PROCESS = None
            return
        time.sleep(0.1)

    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            proc.kill()
        except Exception:
            pass
    _ACTIVE_CHILD_PROCESS = None

# --- LOGGING UTILS ---
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

# --- MODEL DISCOVERY ---
class ModelRegistry:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.models: Dict[str, Path] = {}
        self.types: Dict[str, str] = {} # 'ported' or 'native'
        self.scan()

    def _register_alias(self, alias: str, target: str) -> None:
        if alias in self.models or target not in self.models:
            return
        self.models[alias] = self.models[target]
        self.types[alias] = self.types[target]

    def scan(self):
        print("Scanning Dyn_DNN4OPF for non-penalty models...")
        # Prefer the authoritative model registry from DNN/VANILLA/main.py so
        # the canonical suite names and their aliases resolve exactly the same
        # way the launcher does.
        try:
            from DNN.VANILLA.main import MODEL_MODULES as canonical_model_modules
        except Exception:
            canonical_model_modules = {}

        def register(name: str, path: Path, kind: str) -> None:
            if "penalty" in name.lower():
                return
            if name in self.models:
                return
            self.models[name] = path
            self.types[name] = kind

        for model_name, module_path in canonical_model_modules.items():
            if "penalty" in model_name.lower():
                continue
            try:
                spec = importlib.util.find_spec(module_path)
            except Exception:
                spec = None
            if spec is None or spec.origin is None:
                continue
            module_file = Path(spec.origin)
            if not module_file.exists():
                continue
            register(model_name, module_file, "native")
            stem = module_file.stem
            if stem.startswith("run_"):
                register(stem.replace("run_", "", 1), module_file, "native")
                register(stem, module_file, "native")

        # Keep a lightweight fallback scan for any additional Dyn_DNN4OPF
        # entry points that may exist outside the canonical registry.
        dyndnn_root = self.root / "DNN" / "VANILLA" / "Dyn_DNN4OPF"
        native_dir = dyndnn_root / "examples"
        if native_dir.exists():
            for path in native_dir.glob("run_*.py"):
                name = path.name.replace("run_", "").replace(".py", "")
                register(name, path, "native")
        for path in dyndnn_root.rglob("*_adp_width_to_depth.py"):
            name = path.name.replace("_adp_width_to_depth.py", "")
            register(name, path, "ported")

        canonical_aliases = {
            "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement_single_head": "adp_lg_nd_width_only",
            "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement_two_heads": "adp_lg_nd_2_head_width_only",
            "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement_four_heads": "adp_lg_nd_4_head_width_only",
            "patience_based_width_growth_round_robin_layer_wise_single_head": "adp_den_width_only_round_robin",
            "patience_based_width_growth_round_robin_layer_wise_two_heads": "adp_2head_width_only",
            "patience_based_width_growth_round_robin_layer_wise_four_heads": "adp_4head_width_only",
            "patience_based_width_growth_all_layers_together_single_head": "adp_width_only",
            "patience_based_width_growth_all_layers_together_two_heads": "adp_2head_width_only",
            "patience_based_width_growth_all_layers_together_four_heads": "adp_4head_width_only",
            "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1_single_head": "adp_depth_only",
            "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1_two_heads": "adp_2head_depth_only",
            "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1_four_heads": "adp_4head_depth_only",
            "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width_single_head": "adp_depth_only",
            "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width_two_heads": "adp_2head_depth_only",
            "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width_four_heads": "adp_4head_depth_only",
            "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width_single_head": "adp_depth_only",
            "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width_two_heads": "adp_2head_depth_only",
            "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width_four_heads": "adp_4head_depth_only",
            "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1_single_head": "adp_depth_only",
            "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1_two_heads": "adp_2head_depth_only",
            "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1_four_heads": "adp_4head_depth_only",
            "depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1_single_head": "adp_depth_only",
            "depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1_two_heads": "adp_2head_depth_only",
            "depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1_four_heads": "adp_4head_depth_only",
            "depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width_single_head": "adp_depth_only",
            "depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width_two_heads": "adp_2head_depth_only",
            "depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width_four_heads": "adp_4head_depth_only",
            "depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width_single_head": "adp_depth_only",
            "depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width_two_heads": "adp_2head_depth_only",
            "depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width_four_heads": "adp_4head_depth_only",
            "depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1_single_head": "adp_depth_only",
            "depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1_two_heads": "adp_2head_depth_only",
            "depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1_four_heads": "adp_4head_depth_only",
            "stagewise_depth_growth_start_new_stage_at_width_1_single_head": "adp_depth_only",
            "stagewise_depth_growth_start_new_stage_at_width_1_two_heads": "adp_2head_depth_only",
            "stagewise_depth_growth_start_new_stage_at_width_1_four_heads": "adp_4head_depth_only",
            "stagewise_depth_growth_start_new_stage_at_current_width_single_head": "adp_depth_only",
            "stagewise_depth_growth_start_new_stage_at_current_width_two_heads": "adp_2head_depth_only",
            "stagewise_depth_growth_start_new_stage_at_current_width_four_heads": "adp_4head_depth_only",
            "fixed_mlp_one_shot_grid": "vanilla_mlp",
        }
        for alias, target in canonical_aliases.items():
            self._register_alias(alias, target)
        
        print(f"Found {len(self.models)} non-penalty models in Dyn_DNN4OPF context.")



    def get_info(self, model_name: str) -> Optional[Tuple[Path, str]]:
        if model_name in self.models:
            return self.models[model_name], self.types[model_name]
        return None

    def list_models(self):
        return sorted(self.models.keys())


def is_adp_model_name(model_name: str) -> bool:
    return "adp" in model_name.lower()


def parse_unknown_kv_args(unknown_args: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in unknown_args:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        if v.isdigit():
            parsed[k] = int(v)
        elif v.replace(".", "", 1).isdigit():
            parsed[k] = float(v)
        else:
            parsed[k] = v
    return parsed


def _resolve_seed(args: argparse.Namespace) -> int:
    if getattr(args, "seed", None) is not None:
        return int(args.seed)
    raw = os.environ.get("OPF_ADP_GLOBAL_SEED")
    if raw is not None and str(raw).strip():
        try:
            return int(raw)
        except Exception:
            pass
    return 42

# --- NATIVE EXECUTION ---
def run_native_model(model_path: Path, args: argparse.Namespace, unknown_args: List[str]):
    print(f"\n[Universal Main] Launching NATIVE model: {model_path.name}")
    # Native entry points in Dyn_DNN4OPF historically import sibling
    # packages both as ``Dyn_DNN4OPF.*`` and as top-level ``training`` /
    # ``config``.  Loading the file by path does not establish the latter
    # import root, so add the package directory before executing it.
    native_package_root = model_path.parent.parent
    for import_root in (native_package_root, model_path.parent):
        import_root_text = str(import_root)
        if import_root_text not in sys.path:
            sys.path.insert(0, import_root_text)
    module_name = model_path.stem
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load spec for {model_path}")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    seed = _resolve_seed(args)
    
    # Inject env var for logging
    if args.results_dir:
        os.environ["OPF_ADP_RESULTS_DIR"] = args.results_dir
    else:
        out_path = Path("Results") / f"{args.model}_{args.case_name}"
        os.environ["OPF_ADP_RESULTS_DIR"] = str(out_path)
    os.environ["OPF_ADP_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_pipeline"):
        print(f"Error: Native script {module_name} does not have run_pipeline()")
        sys.exit(1)
    cfg = {}
    if args.case_name: cfg['case_name'] = args.case_name
    if args.max_epochs: cfg['max_epochs'] = args.max_epochs
    if args.batch_size: cfg['batch_size'] = args.batch_size
    if args.lambda_eq: cfg['lambda_eq'] = args.lambda_eq
    if args.lambda_ineq: cfg['lambda_ineq'] = args.lambda_ineq
    if args.lambda_loss: cfg['lambda_loss'] = args.lambda_loss
    cfg["seed"] = seed
    
    for item in unknown_args:
        if "=" in item:
            k, v = item.split("=", 1)
            if v.isdigit(): v = int(v)
            elif v.replace(".", "", 1).isdigit(): v = float(v)
            cfg[k] = v
        else:
            print(f"Warning: Ignored unknown arg '{item}' for native model")
    if is_adp_model_name(args.model):
        for k, v in ADP_UNBOUNDED_DEFAULTS.items():
            cfg.setdefault(k, v)
    cfg['model'] = args.model
    print(f"Config: {cfg}")
    mod.run_pipeline(cfg)

# --- PORTED EXECUTION ---
def run_ported_model(model_path: Path, args: argparse.Namespace, unknown_args: List[str]):
    print(f"\n[Universal Main] Launching PORTED model: {model_path.name}")
    
    cmd = [sys.executable, "-u", str(model_path)]
    seed = _resolve_seed(args)
    
    if args.case_name: cmd.extend(["--case-name", args.case_name])
    if args.max_epochs: cmd.extend(["--max-epochs", str(args.max_epochs)])
    if args.results_dir: cmd.extend(["--results-dir", args.results_dir])
    else:
        out_dir = Path("Results") / f"{args.model}_{args.case_name}"
        cmd.extend(["--results-dir", str(out_dir)])
    if args.lambda_eq: cmd.extend(["--lambda-eq", str(args.lambda_eq)])
    if args.lambda_ineq: cmd.extend(["--lambda-ineq", str(args.lambda_ineq)])
    if args.lambda_loss: cmd.extend(["--lambda-loss", str(args.lambda_loss)])
    if args.adp_mode: cmd.extend(["--adp-mode", args.adp_mode])
    cmd.extend(["--seed", str(seed)])
    cmd.extend(unknown_args)
    parsed_unknown = parse_unknown_kv_args(unknown_args)
    if is_adp_model_name(args.model):
        if args.max_epochs is None and "max_epochs" not in parsed_unknown:
            cmd.extend(["--max-epochs", str(ADP_UNBOUNDED_DEFAULTS["max_epochs"])])
        for k, v in ADP_UNBOUNDED_DEFAULTS.items():
            if k == "max_epochs" or k in parsed_unknown:
                continue
            cmd.append(f"{k}={v}")
    
    print(f"Command: {' '.join(cmd)}")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{args.model}_{timestamp}.log"
    print(f"Logging to {log_file}")
    
    # Export results dir for child processes 
    env = os.environ.copy()
    if args.results_dir:
        env["OPF_ADP_RESULTS_DIR"] = args.results_dir
    else:
        env["OPF_ADP_RESULTS_DIR"] = str(Path("Results") / f"{args.model}_{args.case_name}")
    env["OPF_ADP_GLOBAL_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)

    global _ACTIVE_CHILD_PROCESS
    popen_kwargs: Dict[str, Any] = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "universal_newlines": True, "bufsize": 1, "env": env}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    with open(log_file, 'w') as f_log:
        process = subprocess.Popen(cmd, **popen_kwargs)
        _ACTIVE_CHILD_PROCESS = process
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None: break
            if output:
                sys.stdout.write(output)
                sys.stdout.flush()
                f_log.write(output)
                f_log.flush()
        _ACTIVE_CHILD_PROCESS = None
    if process.poll() != 0:
        print(f"Failed with exit code {process.poll()}")
        sys.exit(process.poll())

def main():
    parser = argparse.ArgumentParser(description="Universal Launcher for OPF ADP")
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--list", action="store_true", help="List models")
    parser.add_argument("--case-name", "-k", type=str, default="pglib_opf_case14_ieee")
    parser.add_argument("--max-epochs", type=parse_epoch_limit)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--results-dir", type=str)
    parser.add_argument("--lambda-eq", type=float)
    parser.add_argument("--lambda-ineq", type=float)
    parser.add_argument("--lambda-loss", type=float)
    parser.add_argument("--adp-mode", type=str)
    parser.add_argument("--seed", type=int, default=None)
    args, unknown = parser.parse_known_args()
    args.seed = _resolve_seed(args)
    os.environ["OPF_ADP_GLOBAL_SEED"] = str(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    registry = ModelRegistry(os.getcwd())
    if args.list:
        print("\nAvailable Models:")
        for m in registry.list_models():
             t = registry.types[m]
             print(f" - {m:<40} [{t.upper()}]")
        return
    if not args.model:
        parser.print_help()
        sys.exit(1)
    info = registry.get_info(args.model)
    if not info:
        print(f"Error: Model '{args.model}' not found.")
        sys.exit(1)
    path, mtype = info
    if mtype == 'native': run_native_model(path, args, unknown)
    else: run_ported_model(path, args, unknown)

if __name__ == "__main__":
    def _handle_shutdown(signum, frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"\n[INTERRUPT] Caught {signame}. Terminating active child process tree...")
        try:
            _terminate_active_child_tree()
        except Exception:
            pass
        sys.exit(130)

    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(_sig, _handle_shutdown)
        except Exception:
            pass

    try:
        main()
    except KeyboardInterrupt:
        _handle_shutdown(signal.SIGINT, None)
