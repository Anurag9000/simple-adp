"""
Unified CLI launcher for Dyn_DNN4OPF: dispatches to the correct run_*.py pipeline.

Overview
-------------------------------------------------------------------------------
This script exposes the model pipelines that are present and importable in this checkout using a unified interface.
All training, evaluation, data loading, case selection, and hyperparameter configuration can be
controlled via command-line arguments or optional config files.

Usage Examples
-------------------------------------------------------------------------------

▶ Basic usage (defaults to the vanilla one-shot pipeline):
    $ python main.py

▶ Run a specific active pipeline:
    $ python main.py --model vanilla_mlp --case_name pglib_opf_case14_ieee

▶ Run an adaptive family/head variant:
    $ python main.py --model loss_guided_neural_densification_width_growth_stop_on_first_non_improvement_single_head

Data Selection Options
-------------------------------------------------------------------------------

• case_name
    - Format: full PGLib OPF name, e.g., "pglib_opf_case14_ieee"
    - Default: "pglib_opf_case14_ieee"

• train_samples, val_samples, test_samples
    - Integer values specifying how many examples to load from each split
    - If None (default), loads:
        → All batches for training (∼270,000 points)
        → 1 full batch (∼15,000 points) for val/test

• batches
    - List[int] format (quoted Python list), e.g. "[1, 3, 7]"
    - Selects specific batches from training split only

Masking & Bounded Output Control
-------------------------------------------------------------------------------

• use_bounds
    - Enables bounded ReLU output using specified or default mask

• mask
    - Custom list of 0s and 1s to apply bounding mask
    - Default: dynamically inferred based on case (all zeros of correct size)
    - Order: [Pg] * ngen + [Qg] * ngen + [Va] * nbus + [Vm] * nbus

Training Parameters
-------------------------------------------------------------------------------

• epochs, batch_size, lr, patience, hidden_dim
    - Standard training loop hyperparameters

Constraint Loss Weights
-------------------------------------------------------------------------------

• lambda_vu, lambda_vl, lambda_pl, lambda_ql, lambda_th
• lambda_real, lambda_imag

Model-Specific Keys
-------------------------------------------------------------------------------

• lambda_ewc         (for EWC)
• alpha_threshold    (for progressive)
• prune_adapters     (for progressive)

Evaluation / Tuning
-------------------------------------------------------------------------------

• evaluate_only       = bool   (skip training and evaluate only)
• optuna_trials       = int    (if tuning enabled)
• optuna_sampler      = str    (e.g., "TPESampler")
• optuna_metric       = str    (e.g., "gen_loss", "val_loss")

Overrides and Config File Merging
-------------------------------------------------------------------------------

• CLI inline overrides can specify any key=value (e.g., batch_size=65536)
• Quoted Python expressions are supported (e.g., mask="[1]*10 + [0]*28")
• Final config resolution order:
      CLI overrides > config file > run_* default config

Available Models
-------------------------------------------------------------------------------

• stl
• loss_guided_neural_densification_width_growth_stop_on_first_non_improvement
• patience_based_width_growth_round_robin_layer_wise
• patience_based_width_growth_all_layers_together
• width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1
• width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width
• width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width
• width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1
• depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1
• depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width
• depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width
• depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1
• stagewise_depth_growth_start_new_stage_at_width_1
• stagewise_depth_growth_start_new_stage_at_current_width
• fixed_mlp_one_shot_grid

"""

from __future__ import annotations
import argparse, importlib, pathlib, sys, os, csv
from typing import Dict, Any
from pathlib import Path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
try:
    from utils.ml_backends import enable_optional_gpu_accelerators

    enable_optional_gpu_accelerators()
except Exception:
    pass
import ast
from Dyn_DNN4OPF.utils.repro import derive_seed, resolve_seed, set_determinism
from Dyn_DNN4OPF.utils.config import infer_case_sizes
import time
import logging
import re

BASE_SEED = resolve_seed()
set_determinism(BASE_SEED)

MODEL_MODULES: dict[str, str] = {
    # Historical entries are retained below as source documentation, then
    # unavailable modules are removed before argparse exposes this registry.

    # ----------------- Dyn_DNN4OPF -----------------
    # STL
    "stl":                          "Dyn_DNN4OPF.examples.run_dnn_stl",

    # DNN (den variants)
    "dnn_den_4head":                "Dyn_DNN4OPF.examples.run_dnn_den_4head",
    "dnn_den_2head":                "Dyn_DNN4OPF.examples.run_dnn_den_2head",
    "dnn_den":                      "Dyn_DNN4OPF.examples.run_dnn_den",

    # ADP (width_only)
    "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement_single_head":
                                    "Dyn_DNN4OPF.examples.run_adp_lg_nd_width_only",
    "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement_two_heads":
                                    "Dyn_DNN4OPF.examples.run_adp_lg_nd_2_head_width_only",
    "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement_four_heads":
                                    "Dyn_DNN4OPF.examples.run_adp_lg_nd_4_head_width_only",
    "patience_based_width_growth_round_robin_layer_wise_single_head":
                                    "Dyn_DNN4OPF.examples.run_adp_den_width_only_round_robin",
    "patience_based_width_growth_round_robin_layer_wise_two_heads":
                                    "Dyn_DNN4OPF.examples.run_adp_den_2_head_width_only",
    "patience_based_width_growth_round_robin_layer_wise_four_heads":
                                    "Dyn_DNN4OPF.examples.run_adp_den_4_head_width_only",
    "patience_based_width_growth_all_layers_together_single_head":
                                    "Dyn_DNN4OPF.examples.run_adp_den_width_only",
    "patience_based_width_growth_all_layers_together_two_heads":
                                    "Dyn_DNN4OPF.examples.run_adp_den_2_head_width_only",
    "patience_based_width_growth_all_layers_together_four_heads":
                                    "Dyn_DNN4OPF.examples.run_adp_den_4_head_width_only",

    # ADP (alt — alternative architecture variants)
    "adp_alt_depth":                "Dyn_DNN4OPF.examples.run_adp_alt_depth",
    "adp_alt_depth_2head":          "Dyn_DNN4OPF.examples.run_adp_alt_depth_2_head",
    "adp_alt_depth_4head":          "Dyn_DNN4OPF.examples.run_adp_alt_depth_4_head",
    "adp_alt_width":                "Dyn_DNN4OPF.examples.run_adp_alt_width",
    "adp_alt_width_2head":          "Dyn_DNN4OPF.examples.run_adp_alt_width_2_head",
    "adp_alt_width_4head":          "Dyn_DNN4OPF.examples.run_adp_alt_width_4_head",

    # ADP (depth sweep)
    "stagewise_depth_growth_start_new_stage_at_width_1_single_head":
                                    "Dyn_DNN4OPF.examples.run_adp_den_depth_only",
    "stagewise_depth_growth_start_new_stage_at_width_1_four_heads":
                                    "Dyn_DNN4OPF.examples.run_adp_den_4head_depth_only",
    "stagewise_depth_growth_start_new_stage_at_width_1_two_heads":
                                    "Dyn_DNN4OPF.examples.run_adp_den_2head_depth_only",

    # DNN (other heads/losses/regularizers)
    "dnn_mtl_4head":                "Dyn_DNN4OPF.examples.run_dnn_mtl_4head",
    "dnn_mtl_2head":                "Dyn_DNN4OPF.examples.run_dnn_mtl_2head",
    "dnn_mtl":                      "Dyn_DNN4OPF.examples.run_dnn_mtl",
    "dnn_l1":                       "Dyn_DNN4OPF.examples.run_dnn_l1",
    "dnn_l2":                       "Dyn_DNN4OPF.examples.run_dnn_l2",
    "dnn_ldf":                      "Dyn_DNN4OPF.examples.run_dnn_ldf",
    "dnn_elastic":                  "Dyn_DNN4OPF.examples.run_dnn_elastic",
    "dnn_mae":                      "Dyn_DNN4OPF.examples.run_dnn_mae",
    "dnn_fsnet":                    "Dyn_DNN4OPF.examples.run_dnn_fsnet",
    "dnn_pdl":                      "Dyn_DNN4OPF.examples.run_dnn_pdl",
    "dnn_dc3":                      "Dyn_DNN4OPF.examples.run_dnn_dc3",

    # DNN (progressive)
    "dnn_progressive_4head":        "Dyn_DNN4OPF.examples.run_dnn_progressive_4head",
    "dnn_progressive_2head":        "Dyn_DNN4OPF.examples.run_dnn_progressive_2head",
    "dnn_progressive":              "Dyn_DNN4OPF.examples.run_dnn_progressive",

    # DNN (EWC)
    "dnn_ewc_4head":                "Dyn_DNN4OPF.examples.run_dnn_ewc_4head",
    "dnn_ewc_2head":                "Dyn_DNN4OPF.examples.run_dnn_ewc_2head",
    "dnn_ewc":                      "Dyn_DNN4OPF.examples.run_dnn_ewc",

    # ----------------- penalty_nn -----------------
    # STL
    "penalty_stl":                  "penalty_nn.examples.run_penalty_stl",

    # penalty ADP (den)
    "penalty_adp_4head":            "penalty_nn.examples.run_penalty_adp_4head",
    "penalty_adp_2head":            "penalty_nn.examples.run_penalty_adp_2head",
    "penalty_adp":                  "penalty_nn.examples.run_penalty_adp_den",

    # penalty ADP (depth)
    "penalty_adp_depth":            "penalty_nn.examples.run_penalty_adp_depth",
    "penalty_adp_depth_4head":      "penalty_nn.examples.run_penalty_adp_depth_4head",
    "penalty_adp_depth_2head":      "penalty_nn.examples.run_penalty_adp_depth_2head",

    # penalty ADP (depth_only)
    "penalty_adp_depth_only":       "penalty_nn.examples.run_penalty_adp_depth_only",
    "penalty_adp_depth_only_4head": "penalty_nn.examples.run_penalty_adp_depth_only_4head",
    "penalty_adp_2head_depth_only": "penalty_nn.examples.run_penalty_adp_2head_depth_only",

    # penalty ADP (width_only)
    "penalty_adp_width_only":       "penalty_nn.examples.run_penalty_adp_width_only",
    "penalty_adp_width_only_4head": "penalty_nn.examples.run_penalty_adp_width_only_4head",
    "penalty_adp_2head_width_only": "penalty_nn.examples.run_penalty_adp_2head_width_only",

    # penalty ADP (width)
    "penalty_adp_width":            "penalty_nn.examples.run_penalty_adp_width",
    "penalty_adp_width_4head":      "penalty_nn.examples.run_penalty_adp_width_4head",
    "penalty_adp_width_2head":      "penalty_nn.examples.run_penalty_adp_width_2head",

    # penalty DEN
    "penalty_den_4head":            "penalty_nn.examples.run_penalty_den_4head",
    "penalty_den_2head":            "penalty_nn.examples.run_penalty_den_2head",
    "penalty_den":                  "penalty_nn.examples.run_penalty_den",

    # penalty ADP (alt — alternative architecture variants)
    "penalty_adp_alt_depth":        "penalty_nn.examples.run_penalty_adp_alt_depth",
    "penalty_adp_alt_depth_2head":  "penalty_nn.examples.run_penalty_adp_alt_depth_2_head",
    "penalty_adp_alt_depth_4head":  "penalty_nn.examples.run_penalty_adp_alt_depth_4_head",
    "penalty_adp_alt_width":        "penalty_nn.examples.run_penalty_adp_alt_width",
    "penalty_adp_alt_width_2head":  "penalty_nn.examples.run_penalty_adp_alt_width_2_head",
    "penalty_adp_alt_width_4head":  "penalty_nn.examples.run_penalty_adp_alt_width_4_head",

    # penalty (other heads/losses/regularizers)
    "penalty_ewc_4head":            "penalty_nn.examples.run_penalty_ewc_4head",
    "penalty_ewc_2head":            "penalty_nn.examples.run_penalty_ewc_2head",
    "penalty_ewc":                  "penalty_nn.examples.run_penalty_ewc",
    "penalty_mtl_4head":            "penalty_nn.examples.run_penalty_mtl_4head",
    "penalty_mtl_2head":            "penalty_nn.examples.run_penalty_mtl_2head",
    "penalty_mtl":                  "penalty_nn.examples.run_penalty_mtl",
    "penalty_progressive_4head":    "penalty_nn.examples.run_penalty_progressive_4head",
    "penalty_progressive_2head":    "penalty_nn.examples.run_penalty_progressive_2head",
    "penalty_progressive":          "penalty_nn.examples.run_penalty_progressive",
    "penalty_l1":                   "penalty_nn.examples.run_penalty_l1",
    "penalty_l2":                   "penalty_nn.examples.run_penalty_l2",
    "penalty_ldf":                  "penalty_nn.examples.run_penalty_ldf",
    "penalty_elastic":              "penalty_nn.examples.run_penalty_elastic",
    "penalty_mae":                  "penalty_nn.examples.run_penalty_mae",
    "penalty_fsnet":                "penalty_nn.examples.run_penalty_fsnet",
    "penalty_pdl":                  "penalty_nn.examples.run_penalty_pdl",
    "penalty_dc3":                  "penalty_nn.examples.run_penalty_dc3",
}


def _register_canonical_model_aliases() -> None:
    def add_family(base: str, single_module: str, two_module: str, four_module: str) -> None:
        MODEL_MODULES[f"{base}_single_head"] = single_module
        MODEL_MODULES[f"{base}_two_heads"] = two_module
        MODEL_MODULES[f"{base}_four_heads"] = four_module

    add_family(
        "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement",
        "Dyn_DNN4OPF.examples.run_adp_lg_nd_width_only",
        "Dyn_DNN4OPF.examples.run_adp_lg_nd_2_head_width_only",
        "Dyn_DNN4OPF.examples.run_adp_lg_nd_4_head_width_only",
    )
    add_family(
        "patience_based_width_growth_round_robin_layer_wise",
        "Dyn_DNN4OPF.examples.run_adp_den_width_only_round_robin",
        "Dyn_DNN4OPF.examples.run_adp_den_2_head_width_only",
        "Dyn_DNN4OPF.examples.run_adp_den_4_head_width_only",
    )
    add_family(
        "patience_based_width_growth_all_layers_together",
        "Dyn_DNN4OPF.examples.run_adp_den_width_only",
        "Dyn_DNN4OPF.examples.run_adp_den_2_head_width_only",
        "Dyn_DNN4OPF.examples.run_adp_den_4_head_width_only",
    )
    for base in (
        "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1",
        "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width",
        "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width",
        "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1",
        "depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1",
        "depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width",
        "depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width",
        "depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1",
        "stagewise_depth_growth_start_new_stage_at_width_1",
        "stagewise_depth_growth_start_new_stage_at_current_width",
    ):
        add_family(
            base,
            "Dyn_DNN4OPF.examples.run_adp_den_depth_only",
            "Dyn_DNN4OPF.examples.run_adp_den_2head_depth_only",
            "Dyn_DNN4OPF.examples.run_adp_den_4head_depth_only",
        )

    MODEL_MODULES["fixed_mlp_one_shot_grid"] = "Dyn_DNN4OPF.examples.run_vanilla_mlp"
    MODEL_MODULES["vanilla_mlp"] = "Dyn_DNN4OPF.examples.run_vanilla_mlp"


_register_canonical_model_aliases()


def _prune_unavailable_model_modules() -> None:
    """Expose only entry points that exist in this checkout.

    The old registry included pipelines from an optional/removed penalty
    package and legacy example files that are not part of this repository.
    Leaving those names in argparse made the advertised registry unusable and
    caused universal_main to silently skip them. Do not replace missing
    algorithms with a different implementation; remove them from the active
    registry until their source is restored.
    """
    package_root = pathlib.Path(__file__).resolve().parent
    unavailable = []
    for name, module_name in MODEL_MODULES.items():
        if module_name.startswith("Dyn_DNN4OPF."):
            source_path = package_root / (module_name.replace(".", "/") + ".py")
        else:
            source_path = package_root.parent.parent / (module_name.replace(".", "/") + ".py")
        if not source_path.is_file():
            unavailable.append(name)
    for name in unavailable:
        del MODEL_MODULES[name]


_prune_unavailable_model_modules()


# Right now all neurons are unmasked
# To mask a neuron put the value 1
# Or one can manually specicify an entire list containing 38 collections of ones and zeros
# To find the most optimal solution maybe try with different permutes to be precise 38 factorial permutes of ones and zeroes
# The masking tensor should be filled in the order [5*Pg, 5*Qg, 14*Va, 14*Vm] in a contigous array

class Tee:
    def __init__(self, log_file):
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.terminal = sys.__stdout__  # raw terminal
        self.log = open(log_file, "w", encoding="utf-8")  # "w" to overwrite each run

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_default_mask(case_name: str) -> list[int]:
    """Build a 0-mask dynamically for Pg, Qg, Va, Vm based on case_name."""
    sizes = infer_case_sizes(case_name)
    n_gen, n_bus = sizes["n_gen"], sizes["n_bus"]
    return [0] * (2 * n_gen + 2 * n_bus)

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Dyn_DNN4OPF launcher",
        description="Run any model pipeline from a single entry-point.")
    
    # Required model selection
    p.add_argument("-m", "--model", default="vanilla_mlp", choices=MODEL_MODULES,
                   help="Which model pipeline to invoke (default: vanilla_mlp)")
    
    p.add_argument("--hidden_dims",   type=int, default=10,
                   help="Number of hidden neurons in each layer")
    
    p.add_argument("--num_runs",      type=int, default=5,
                                    help="How many times to repeat each run")

    # Optional config file
    p.add_argument("-c", "--config", type=str,
                   help="Optional JSON/YAML file with parameter overrides")
    
    # Inline overrides (key=value)
    p.add_argument("overrides", nargs="*",
                   help="Extra KEY=VALUE pairs that override defaults")
    
    # Case selection
    p.add_argument("--case_name", "-k",
                   default="pglib_opf_case14_ieee",
                   help="Any PGLib OPF case (e.g. pglib_opf_case14_ieee)")

    # Sample sizes and batch selection
    p.add_argument("--train_samples", type=int, default=2000,
                   help="Number of training samples to load (default: 20000)")
    p.add_argument("--val_samples", type=int, default=None,
                   help="Number of validation samples to load (default:1500 )")
    p.add_argument("--test_samples", type=int, default=1500,
                   help="Number of test samples to load (default: 1500)")
    p.add_argument("--batches", type=str, default=None,
                   help='List of batches to load, e.g., "[1,2,3]"')
    p.add_argument("--seed", type=int, default=None,
                   help="Base seed for the launcher; each repeat derives a distinct deterministic seed from it")
    
    return p

def _decode_value(v: str) -> Any:
    """Best-effort cast from CLI string to bool / int / float / str"""
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:                      # try int/float first
        return int(v) if v.isdigit() else float(v)
    except ValueError:
        pass
    if v.strip().startswith(("[", "(")):
        return ast.literal_eval(v)
    return v

def _parse_keyvals(pairs: list[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' is not of form KEY=VALUE")
        k, v = pair.split("=", 1)
        cfg[k] = _decode_value(v)
    return cfg

def main() -> None:
    args = _build_parser().parse_args()
    base_seed = resolve_seed(args.seed)
    
    num_runs = args.num_runs

    if args.val_samples is None:
        args.val_samples = args.train_samples // 18

    if args.test_samples is None:
        args.test_samples = 1500
        
    # ——— dynamic logging per run configuration ———
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{args.model}_{args.case_name}_" \
            f"{args.train_samples}samples_{args.hidden_dims}neurons_{num_runs}runs.txt"
    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout
    # Interactive input prompts if required arguments are not provided
    models = list(MODEL_MODULES.keys())
    print("\nAvailable models:")
    for i, name in enumerate(models, 1):
        print(f"  {i:2d}. {name}")
    default_model = models[0]  # expected 'stl'
    while True:
        choice = input(f"Select model by number (default: 1 = '{default_model}'): ").strip()
        if choice == "":
            args.model = default_model
            break
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                args.model = models[idx - 1]
                break
            else:
                print(f"Invalid number. Enter a value between 1 and {len(models)}.")
        except ValueError:
            print("Please enter a valid integer.")

    args.case_name = input("Enter the OPF case name (default: pglib_opf_case14_ieee): ") or "pglib_opf_case14_ieee"
    args.hidden_dims = int(input("Enter the no of hidden dimensions (default: 10): ") or 10 )
    args.train_samples = int(input("Enter the number of training samples (max value : 270000 and def : 20000): ") or 20000)
    args.val_samples = args.train_samples // 18
    args.test_samples = int(input("Enter the number of test samples (max value : 15000 and def: 1500): ") or 1500)
    num_runs = int(input("Enter how many times to run the model (default: 5): ") or 5)

    # # Interactive threshold inputs
    # loss_thr = input("Enter MSE loss threshold for expansion (default: 1e-3): ") or "1e-3"
    # dp_thr   = input("Enter ΔP mean-violation threshold (default: 0.005): ") or "0.005"
    # dq_thr   = input("Enter ΔQ mean-violation threshold (default: 0.003): ") or "0.003"
    # pg_thr   = input("Enter PG mean-violation threshold (default: 0.002): ") or "0.002"
    # qg_thr   = input("Enter QG mean-violation threshold (default: 0.0015): ") or "0.0015"
    # vm_thr   = input("Enter VM mean-violation threshold (default: 0.025): ") or "0.025"

    mod_name = MODEL_MODULES[args.model]
    mod = importlib.import_module(mod_name)

    # 2. start from that script’s DEFAULT_CONFIG 
    try:
        config = mod.DEFAULT_CONFIG
    except AttributeError as e:
        sys.exit(f"[FATAL] {mod_name} must expose DEFAULT_CONFIG dict – {e}")

    # Update the DEFAULT_CONFIG with user inputs
    config["case_name"] = args.case_name
    config["train_samples"] = args.train_samples
    config["val_samples"] = args.val_samples
    config["test_samples"] = args.test_samples

    config['hidden_dim'] = args.hidden_dims
    config['h1_dim'] = args.hidden_dims
    config['h2_dim'] = args.hidden_dims
    config["base_seed"] = base_seed
    
    # #Assign to config
    # config["loss_thr"] = float(loss_thr)
    # config["dp_thr"]   = float(dp_thr)
    # config["dq_thr"]   = float(dq_thr)
    # config["pg_thr"]   = float(pg_thr)
    # config["qg_thr"]   = float(qg_thr)
    # config["vm_thr"]   = float(vm_thr)

    for run_idx in range(num_runs):
        run_seed = derive_seed(base_seed, args.model, args.case_name, run_idx + 1)
        os.environ["OPF_ADP_GLOBAL_SEED"] = str(run_seed)
        os.environ["PYTHONHASHSEED"] = str(run_seed)
        set_determinism(run_seed)

        # ─── log model‐size & sample‐size ───────────────────────────────────────────
        neurons = config['hidden_dim']  # you could also compute total neurons if you have more layers
        logging.info(
            f"🚀 Starting runs: model={args.model!r}, hidden_dim={neurons}, "
            f"train_samples={config['train_samples']}, "
            f"val_samples={config['val_samples']}, "
            f"test_samples={config['test_samples']}"
        )
        print(
            f"🔧 Configuration: model={args.model}, hidden_dim={neurons}, "
            f"train={config['train_samples']}, val={config['val_samples']}, "
            f"test={config['test_samples']}"
        )
        print(f"🔐 Seed for run {run_idx + 1}: {run_seed}")

        print(f"\n🔁 Running model iteration {run_idx + 1}/{num_runs}")

        start_time = time.time()
        config["seed"] = run_seed
        config["repeat_index"] = run_idx + 1
        mod.run_pipeline(config)
        end_time = time.time()
        elapsed = end_time - start_time

        # Log the elapsed time
        logging.info(f"✅ Run {run_idx + 1}: Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

        summary_csv_path = pathlib.Path("Results/SUMMARY.csv")

        if summary_csv_path.exists():
            # Format as 3m 2.45s
            minutes = int(elapsed // 60)
            seconds = round(elapsed % 60, 2)
            format_time = f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"

            with summary_csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            if rows and "Time Taken" in rows[-1]:
                rows[-1]["Time Taken"] = format_time
                with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                logging.info(f"✅ Time Taken updated in last row of {summary_csv_path}")
            else:
                logging.warning(f"⚠️ 'Time Taken' column missing in SUMMARY.csv. Not updated.")
        else:
            logging.warning(f"⚠️ SUMMARY.csv not found — couldn't update with time.")

    # def process_summary_data(summary_csv_path: str):
    #     if not Path(summary_csv_path).exists():
    #         print(f"SUMMARY.csv not found at {summary_csv_path}")
    #         return

    #     # Load the SUMMARY.csv file
    #     summary_data = pd.read_csv(summary_csv_path)

    #     # Define paths for min_data and mean_data
    #     min_data_path = summary_csv_path.replace("SUMMARY.csv", "min_data.csv")
    #     mean_data_path = summary_csv_path.replace("SUMMARY.csv", "mean_data.csv")

    #     # Process Min Data
    #     min_data = summary_data.loc[
    #         summary_data.groupby(['Model Name', 'Case Name', 'Train Samples','Test Samples','Total_Neurons'])['Mean MSE'].idxmin()
    #     ]
    #     if Path(min_data_path).exists():
    #         existing_min_data = pd.read_csv(min_data_path)
    #         min_data = pd.concat([existing_min_data, min_data]).drop_duplicates(
    #             subset=['Model Name', 'Case Name', 'Train Samples', 'Test Samples','Total_Neurons']
    #         )
    #     min_data.to_csv(min_data_path, index=False)
    #     print(f"Min Data saved to {min_data_path}")

    #     # Process Mean Data
    #     mean_data = summary_data.groupby(
    #         ['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons'], as_index=False
    #     ).mean(numeric_only=True)

    #     # Retain all non-numerical columns
    #     non_numeric_data = summary_data[
    #         ['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons']
    #     ].drop_duplicates()
    #     mean_data = pd.merge(non_numeric_data, mean_data, on=['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons'], how='inner')

    #     if Path(mean_data_path).exists():
    #         existing_mean_data = pd.read_csv(mean_data_path)
    #         mean_data = pd.concat([existing_mean_data, mean_data]).drop_duplicates(
    #             subset=['Model Name', 'Case Name', 'Train Samples', 'Test Samples','Total_Neurons']
    #         )
    #     mean_data.to_csv(mean_data_path, index=False)
    #     print(f"Mean Data saved to {mean_data_path}")

    # ── helpers ─────────────────────────────────────────────────────────────────────
    _time_pat = re.compile(r'(?:(\d+)m\s*)?([\d.]+)s')

    def _is_missing(value: object) -> bool:
        return value is None or (isinstance(value, float) and math.isnan(value))

    def _time_to_secs(t: str | float | int | None) -> float | None:
        """
        Convert '3m 2.45s' → 182.45 (float seconds).
        Returns None if the value is missing or cannot be parsed.
        """
        if _is_missing(t):
            return None
        m = _time_pat.match(str(t).strip())
        if not m:
            return None
        mins = int(m.group(1)) if m.group(1) else 0
        secs = float(m.group(2))
        return mins * 60 + secs


    def _secs_to_hms(elapsed: float | None) -> str:
        """
        Convert seconds back to the original string format.
        182.45 → '3m 2.45s', 45.32 → '45.32s'
        """
        if _is_missing(elapsed):
            return ""
        minutes = int(elapsed // 60)
        seconds = round(elapsed % 60, 2)
        return f"{minutes}m {seconds:.2f}s" if minutes else f"{seconds:.2f}s"


    def process_summary_data(summary_csv_path: str) -> None:
        path = Path(summary_csv_path)

        if not path.exists():
            logging.warning(f"SUMMARY.csv not found at {summary_csv_path}")
            return

        with path.open(newline="", encoding="utf-8") as f:
            summary_data = list(csv.DictReader(f))

        if summary_data and "Time Taken" not in summary_data[0]:
            logging.warning("'Time Taken' column missing in SUMMARY.csv – skipping time aggregation.")
            for row in summary_data:
                row["Time Taken"] = ""

        for row in summary_data:
            row["Time Taken (s)"] = _time_to_secs(row.get("Time Taken"))

        def _read_rows(file_path: Path) -> list[dict[str, str]]:
            if not file_path.exists():
                return []
            with file_path.open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))

        def _write_rows(file_path: Path, rows: list[dict[str, object]]) -> None:
            if not rows:
                return
            fieldnames = list(rows[0].keys())
            for row in rows[1:]:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with file_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        def _to_float(value: object) -> float:
            try:
                return float(value)
            except Exception:
                return float("nan")

        def _dedupe(rows: list[dict[str, object]], subset: list[str]) -> list[dict[str, object]]:
            seen: dict[tuple[object, ...], dict[str, object]] = {}
            for row in rows:
                key = tuple(row.get(col) for col in subset)
                seen[key] = row
            return list(seen.values())

        def _group_min(rows: list[dict[str, str]], key_cols: list[str]) -> list[dict[str, object]]:
            best: dict[tuple[str, ...], dict[str, object]] = {}
            for row in rows:
                key = tuple(row.get(col, "") for col in key_cols)
                mse = _to_float(row.get("Mean MSE", ""))
                current = best.get(key)
                if current is None or mse < _to_float(current.get("Mean MSE", float("inf"))):
                    best[key] = dict(row)
            return list(best.values())

        def _group_mean(rows: list[dict[str, str]], key_cols: list[str]) -> list[dict[str, object]]:
            groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
            for row in rows:
                key = tuple(row.get(col, "") for col in key_cols)
                groups.setdefault(key, []).append(row)

            output: list[dict[str, object]] = []
            for key, group_rows in groups.items():
                row_out: dict[str, object] = {col: val for col, val in zip(key_cols, key)}
                columns = set().union(*(row.keys() for row in group_rows))
                for column in columns:
                    if column in key_cols or column == "Time Taken":
                        continue
                    numeric_values = [_to_float(row.get(column, "")) for row in group_rows]
                    numeric_values = [value for value in numeric_values if value == value]
                    if numeric_values:
                        row_out[column] = sum(numeric_values) / len(numeric_values)
                time_values = [_time_to_secs(row.get("Time Taken")) for row in group_rows]
                time_values = [value for value in time_values if value is not None]
                if time_values:
                    row_out["Time Taken"] = _secs_to_hms(sum(time_values) / len(time_values))
                output.append(row_out)
            return output

        adp_data = [row for row in summary_data if row.get("Model Name") == "ADP-DEN"]
        other_data = [row for row in summary_data if row.get("Model Name") != "ADP-DEN"]
        adp_depth_data = [row for row in summary_data if row.get("Model Name") == "ADP-DEPTH"]
        other_depth_data = [row for row in summary_data if row.get("Model Name") != "ADP-DEPTH"]
        adp_width_data = [row for row in summary_data if row.get("Model Name") == "ADP-WIDTH"]
        other_width_data = [row for row in summary_data if row.get("Model Name") != "ADP-WIDTH"]

        min_data = _group_min(adp_data, ["Model Name", "Case Name", "Train Samples", "Test Samples"]) + _group_min(
            other_data, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
        )
        min_data_depth = _group_min(adp_depth_data, ["Model Name", "Case Name", "Train Samples", "Test Samples"]) + _group_min(
            other_depth_data, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
        )
        min_data_width = _group_min(adp_width_data, ["Model Name", "Case Name", "Train Samples", "Test Samples"]) + _group_min(
            other_width_data, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
        )

        min_path = path.with_name("min_data.csv")
        if min_path.exists():
            min_data = _dedupe(_read_rows(min_path) + min_data, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"])
        _write_rows(min_path, min_data)
        logging.info(f"Min Data saved to {min_path}")

        min_path_depth = path.with_name("min_data_depth.csv")
        if min_path_depth.exists():
            min_data_depth = _dedupe(_read_rows(min_path_depth) + min_data_depth, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"])
        _write_rows(min_path_depth, min_data_depth)
        logging.info(f"Min Depth Data saved to {min_path_depth}")

        min_path_width = path.with_name("min_data_width.csv")
        if min_path_width.exists():
            min_data_width = _dedupe(_read_rows(min_path_width) + min_data_width, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"])
        _write_rows(min_path_width, min_data_width)
        logging.info(f"Min Width Data saved to {min_path_width}")

        mean_data = _group_mean(adp_data, ["Model Name", "Case Name", "Train Samples", "Test Samples"]) + _group_mean(
            other_data, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
        )
        mean_path = path.with_name("mean_data.csv")
        if mean_path.exists():
            mean_data = _dedupe(_read_rows(mean_path) + mean_data, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"])
        _write_rows(mean_path, mean_data)
        logging.info(f"Mean Data saved to {mean_path}")

        mean_data_depth = _group_mean(adp_depth_data, ["Model Name", "Case Name", "Train Samples", "Test Samples"]) + _group_mean(
            other_depth_data, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
        )
        mean_path_depth = path.with_name("mean_data_depth.csv")
        if mean_path_depth.exists():
            mean_data_depth = _dedupe(_read_rows(mean_path_depth) + mean_data_depth, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"])
        _write_rows(mean_path_depth, mean_data_depth)
        logging.info(f"Mean Depth Data saved to {mean_path_depth}")

        mean_data_width = _group_mean(adp_width_data, ["Model Name", "Case Name", "Train Samples", "Test Samples"]) + _group_mean(
            other_width_data, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
        )
        mean_path_width = path.with_name("mean_data_width.csv")
        if mean_path_width.exists():
            mean_data_width = _dedupe(_read_rows(mean_path_width) + mean_data_width, ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"])
        _write_rows(mean_path_width, mean_data_width)
        logging.info(f"Mean Width Data saved to {mean_path_width}")

        # Run the function
        summary_csv_path = "Results/SUMMARY.csv"
        process_summary_data(summary_csv_path)

if __name__ == "__main__":
    main()
