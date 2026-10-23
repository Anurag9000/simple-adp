from __future__ import annotations

import csv
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from Dyn_DNN4OPF.data.opf_loader import load_cost_coeff
from Dyn_DNN4OPF.utils.constraint_losses import (
    gap_objective,
    objective,
    power_balance_residuals,
)
from Dyn_DNN4OPF.utils.logger_plotter import (
    compute_branch_thermal_violation_per_sample,
    compute_inequality_violation_per_sample,
)

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_slug(value: str) -> str:
    cleaned = [
        ch.lower() if ch.isalnum() else "_"
        for ch in str(value).strip()
    ]
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _csv_cell(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def append_csv_row(path: Path, payload: Dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: _csv_cell(payload.get(key, "")) for key in fieldnames})


def _flatten_payload(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_payload(value, name))
        else:
            flat[name] = _jsonable(value)
    return flat


def _vector_summary_columns(
    tensor: torch.Tensor,
    *,
    prefix: str,
    labels: list[str] | None = None,
) -> Dict[str, Any]:
    values = tensor.detach().to(torch.float64).cpu()
    if values.numel() == 0:
        return {}
    if values.ndim == 0:
        scalar = float(values.item())
        return {
            f"{prefix}.mean": scalar,
            f"{prefix}.abs_mean": abs(scalar),
            f"{prefix}.abs_max": abs(scalar),
            f"{prefix}.l2": abs(scalar),
        }

    if values.ndim == 1:
        values = values.reshape(-1, 1)
        inferred_labels = ["value"]
    else:
        values = values.reshape(values.shape[0], -1)
        inferred_labels = [f"{idx:04d}" for idx in range(values.shape[1])]

    if labels is None:
        labels = inferred_labels
    else:
        labels = list(labels)

    abs_values = values.abs()
    mean = values.mean(dim=0)
    abs_mean = abs_values.mean(dim=0)
    abs_max = abs_values.max(dim=0).values
    l2 = torch.linalg.vector_norm(values, ord=2, dim=0)
    summary: Dict[str, Any] = {}
    for index, label in enumerate(labels):
        if index >= values.shape[1]:
            break
        summary[f"{prefix}.mean.{label}"] = float(mean[index].item())
        summary[f"{prefix}.abs_mean.{label}"] = float(abs_mean[index].item())
        summary[f"{prefix}.abs_max.{label}"] = float(abs_max[index].item())
        summary[f"{prefix}.l2.{label}"] = float(l2[index].item())
    summary[f"{prefix}.shape"] = "x".join(str(dim) for dim in values.shape)
    return summary


def _selected_tensor_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    tensor_prefixes = [
        "xb",
        "yb",
        "pred",
        "output_error",
        "residual_real",
        "residual_imag",
        "pg_violation",
        "qg_violation",
        "vm_violation",
        "thermal_violation",
        "thermal_ratio",
        "objective_pred",
        "objective_true",
        "objective_gap_pct",
    ]
    for key in tensor_prefixes:
        value = raw.get(key)
        if isinstance(value, torch.Tensor):
            summary.update(_vector_summary_columns(value, prefix=f"raw.{key}"))
    return summary


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _split_pred(flat: torch.Tensor, num_gens: int, num_buses: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pg = flat[:, :num_gens]
    qg = flat[:, num_gens : 2 * num_gens]
    va = flat[:, 2 * num_gens : 2 * num_gens + num_buses]
    vm = flat[:, 2 * num_gens + num_buses : 2 * num_gens + 2 * num_buses]
    return pg, qg, va, vm


def _split_input(flat: torch.Tensor, num_buses: int) -> tuple[torch.Tensor, torch.Tensor]:
    pd = flat[:, :num_buses]
    qd = flat[:, num_buses : 2 * num_buses]
    return pd, qd


def _compute_inequality_raw_vectors(
    pred: torch.Tensor,
    bounds: Dict[str, Any],
    num_gens: int,
    num_buses: int,
) -> Dict[str, torch.Tensor]:
    pred = pred.detach().to(torch.float64).cpu()
    p_max = bounds["p_max"].detach().to(torch.float64).cpu()
    p_min = bounds["p_min"].detach().to(torch.float64).cpu()
    q_max = bounds["q_max"].detach().to(torch.float64).cpu()
    q_min = bounds["q_min"].detach().to(torch.float64).cpu()
    v_max = bounds["v_max"].detach().to(torch.float64).cpu()
    v_min = bounds["v_min"].detach().to(torch.float64).cpu()

    pg = pred[:, :num_gens]
    qg = pred[:, num_gens : 2 * num_gens]
    vm = pred[:, 2 * num_gens + num_buses : 2 * num_gens + 2 * num_buses]

    def violation(values: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values - hi, min=0) + torch.clamp(lo - values, min=0)

    return {
        "pg": violation(pg, p_min, p_max),
        "qg": violation(qg, q_min, q_max),
        "vm": violation(vm, v_min, v_max),
    }


def _extract_branch_specs(
    opf_json: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], float]:
    grid = opf_json.get("grid", {})
    context = grid.get("context", [[1.0]])
    try:
        mva_base = float(context[0][0][0])
    except Exception:
        mva_base = 1.0
    if mva_base <= 0:
        mva_base = 1.0

    branch_specs = []
    for kind, r_idx, x_idx, b_idx, rating_idx, tap_idx, shift_idx in (
        ("ac_line", 4, 5, 2, 6, None, None),
        ("transformer", 2, 3, 9, 4, 7, 8),
    ):
        branch = grid.get("edges", {}).get(kind, {})
        senders = list(branch.get("senders", []))
        receivers = list(branch.get("receivers", []))
        features = list(branch.get("features", []))
        for idx, (src, dst) in enumerate(zip(senders, receivers)):
            if idx >= len(features):
                continue
            feat = features[idx]
            try:
                rating = float(feat[rating_idx]) / mva_base
            except Exception:
                rating = 0.0
            if rating <= 0:
                continue
            try:
                r = float(feat[r_idx])
                x = float(feat[x_idx])
                b = float(feat[b_idx])
                tau = float(feat[tap_idx]) if tap_idx is not None else 1.0
                shift = float(feat[shift_idx]) if shift_idx is not None else 0.0
            except Exception:
                continue
            branch_specs.append(
                {
                    "kind": kind,
                    "index": len(branch_specs),
                    "src": int(src),
                    "dst": int(dst),
                    "r": float(r),
                    "x": float(x),
                    "b": float(b),
                    "tau": float(tau),
                    "shift": float(shift),
                    "rating": float(rating),
                }
            )
    return branch_specs, mva_base


def _branch_thermal_vectors(
    opf_json: Dict[str, Any],
    y_pred: torch.Tensor,
    num_gens: int,
    num_buses: int,
) -> Dict[str, Any]:
    y_pred = y_pred.detach().cpu().to(torch.float64)
    va = y_pred[:, 2 * num_gens : 2 * num_gens + num_buses]
    vm = y_pred[:, 2 * num_gens + num_buses : 2 * num_gens + 2 * num_buses]
    v = vm * torch.exp(1j * va)

    branch_specs, _ = _extract_branch_specs(opf_json)

    n_samples = int(v.shape[0])
    n_branches = len(branch_specs)
    thermal_violation = torch.zeros((n_samples, n_branches), dtype=torch.float64)
    thermal_ratio = torch.zeros((n_samples, n_branches), dtype=torch.float64)
    for branch_idx, spec in enumerate(branch_specs):
        eps = 1e-6
        r = spec["r"]
        x = spec["x"]
        b = spec["b"]
        tau = spec["tau"]
        shift = spec["shift"]
        rating = spec["rating"]
        if abs(r) < eps and abs(x) < eps:
            r = eps
        y = 1 / complex(r, x)
        lc_fr = 1j * b
        lc_to = 1j * b
        tr = tau * math.cos(math.radians(shift))
        ti = tau * math.sin(math.radians(shift))
        t = complex(tr, ti)
        if abs(t) < eps:
            t = complex(1.0, 0.0)
        a = (y + lc_fr) / (abs(t) ** 2)
        bmat = -y / complex(t.real, -t.imag)
        cmat = -y / t
        d = y + lc_to
        vf = v[:, spec["src"]]
        vt = v[:, spec["dst"]]
        if_sr = a * vf + bmat * vt
        it_sr = cmat * vf + d * vt
        s_from = vf * torch.conj(if_sr)
        s_to = vt * torch.conj(it_sr)
        flow_mag = torch.maximum(s_from.abs(), s_to.abs())
        thermal_violation[:, branch_idx] = torch.clamp(flow_mag - rating, min=0)
        thermal_ratio[:, branch_idx] = flow_mag / max(rating, 1e-12)

    return {
        "branch_specs": branch_specs,
        "thermal_violation": thermal_violation,
        "thermal_ratio": thermal_ratio,
    }


@lru_cache(maxsize=16)
def _load_case_json_cached(case_json_path: str) -> Dict[str, Any]:
    return json.loads(Path(case_json_path).read_text(encoding="utf-8"))


def _metric_tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(torch.float64).reshape(-1)
    if values.numel() == 0:
        return {"mean": 0.0, "max": 0.0, "l2": 0.0}
    return {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "l2": float(torch.linalg.vector_norm(values, ord=2).item()),
    }


def compute_batch_metrics(
    xb: torch.Tensor,
    yb: torch.Tensor,
    pred: torch.Tensor,
    *,
    constraints: Dict[str, Any],
    case_json_path: str,
    num_gens: int,
    num_buses: int,
    split: str,
) -> Dict[str, float]:
    metrics, _ = compute_batch_artifacts(
        xb,
        yb,
        pred,
        constraints=constraints,
        case_json_path=case_json_path,
        num_gens=num_gens,
        num_buses=num_buses,
        split=split,
    )
    return metrics


def compute_batch_artifacts(
    xb: torch.Tensor,
    yb: torch.Tensor,
    pred: torch.Tensor,
    *,
    constraints: Dict[str, Any],
    case_json_path: str,
    num_gens: int,
    num_buses: int,
    split: str,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    case_json = _load_case_json_cached(str(case_json_path))
    cost = load_cost_coeff(case_json)
    device = pred.device

    pred = pred.detach()
    yb = yb.detach().to(device)
    xb = xb.detach().to(device)
    pg_pred, qg_pred, va_pred, vm_pred = _split_pred(pred, num_gens, num_buses)
    pg_true, qg_true, va_true, vm_true = _split_pred(yb, num_gens, num_buses)
    pd, qd = _split_input(xb, num_buses)

    mse = F.mse_loss(pred, yb, reduction="mean")
    mae = F.l1_loss(pred, yb, reduction="mean")
    output_l2 = torch.linalg.vector_norm((pred - yb).detach().to(torch.float64), ord=2, dim=1)
    res_real, res_imag = power_balance_residuals(
        pg=pg_pred,
        qg=qg_pred,
        pd=pd,
        qd=qd,
        vm=vm_pred,
        va=va_pred,
        y_bus=constraints["y_bus"],
        gen_bus_idx=constraints["gen_bus_idx"],
        load_bus_idx=constraints["load_bus_idx"],
        n_bus=num_buses,
    )
    res_real = res_real.detach().to(torch.float64)
    res_imag = res_imag.detach().to(torch.float64)
    eq_real = res_real.abs().mean(dim=1)
    eq_imag = res_imag.abs().mean(dim=1)
    eq_real_max = res_real.abs().max(dim=1).values
    eq_imag_max = res_imag.abs().max(dim=1).values
    eq_l2 = torch.linalg.vector_norm(torch.cat([res_real, res_imag], dim=1), ord=2, dim=1)

    bounds = constraints["ineq"]
    pg_mean, qg_mean, vm_mean, pg_max, qg_max, vm_max = compute_inequality_violation_per_sample(
        pred.detach().cpu(),
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in bounds.items()},
        num_gens,
        num_buses,
    )
    thermal_detail = _branch_thermal_vectors(
        case_json,
        pred.detach().cpu(),
        num_gens,
        num_buses,
    )
    thermal_violation = thermal_detail["thermal_violation"]
    thermal_ratio = thermal_detail["thermal_ratio"]
    if thermal_violation.numel() == 0:
        thermal_mean = torch.zeros(pred.shape[0], dtype=torch.float64)
        thermal_l2 = torch.zeros(pred.shape[0], dtype=torch.float64)
        thermal_max = torch.zeros(pred.shape[0], dtype=torch.float64)
        thermal_cnt = torch.zeros(pred.shape[0], dtype=torch.float64)
        thermal_ratio_mean = torch.zeros(pred.shape[0], dtype=torch.float64)
        thermal_ratio_max = torch.zeros(pred.shape[0], dtype=torch.float64)
    else:
        thermal_mean = thermal_violation.mean(dim=1)
        thermal_l2 = torch.linalg.vector_norm(thermal_violation, ord=2, dim=1)
        thermal_max = thermal_violation.max(dim=1).values
        thermal_cnt = (thermal_violation > 0).sum(dim=1).to(torch.float64)
        thermal_ratio_mean = thermal_ratio.mean(dim=1)
        thermal_ratio_max = thermal_ratio.max(dim=1).values

    objective_pred = objective(pg_pred, cost["cost_q"], cost["cost_l"], cost["cost_c"]).detach().to(torch.float64).reshape(-1)
    objective_true = objective(pg_true, cost["cost_q"], cost["cost_l"], cost["cost_c"]).detach().to(torch.float64).reshape(-1)
    gap_pct = gap_objective(
        objective=objective_pred.unsqueeze(1),
        true_objective=objective_true.unsqueeze(1),
    ).detach().to(torch.float64).reshape(-1)

    raw_ineq = _compute_inequality_raw_vectors(pred, bounds, num_gens, num_buses)

    rows = {
        "split": split,
        "batch_size": int(yb.shape[0]),
        "loss_mse": float(mse.item()),
        "loss_mae": float(mae.item()),
        "output_l2_mean": float(output_l2.mean().item()),
        "output_l2_max": float(output_l2.max().item()),
        "eq_real_mean": float(eq_real.mean().item()),
        "eq_real_l2": float(eq_real.detach().pow(2).mean().sqrt().item()),
        "eq_real_max": float(eq_real_max.max().item()),
        "eq_imag_mean": float(eq_imag.mean().item()),
        "eq_imag_l2": float(eq_imag.detach().pow(2).mean().sqrt().item()),
        "eq_imag_max": float(eq_imag_max.max().item()),
        "eq_l2_mean": float(eq_l2.mean().item()),
        "eq_l2_max": float(eq_l2.max().item()),
        "ineq_pg_mean": float(pg_mean.mean().item()),
        "ineq_pg_l2": float(pg_mean.detach().pow(2).mean().sqrt().item()),
        "ineq_pg_max": float(pg_max.max().item()),
        "ineq_qg_mean": float(qg_mean.mean().item()),
        "ineq_qg_l2": float(qg_mean.detach().pow(2).mean().sqrt().item()),
        "ineq_qg_max": float(qg_max.max().item()),
        "ineq_vm_mean": float(vm_mean.mean().item()),
        "ineq_vm_l2": float(vm_mean.detach().pow(2).mean().sqrt().item()),
        "ineq_vm_max": float(vm_max.max().item()),
        "thermal_mean": float(thermal_mean.mean().item()),
        "thermal_l2": float(thermal_l2.mean().item()),
        "thermal_max": float(thermal_max.max().item()),
        "thermal_count_mean": float(thermal_cnt.mean().item()),
        "thermal_ratio_mean": float(thermal_ratio_mean.mean().item()),
        "thermal_ratio_max": float(thermal_ratio_max.max().item()),
        "objective_pred_mean": float(objective_pred.mean().item()),
        "objective_true_mean": float(objective_true.mean().item()),
        "objective_gap_mean": float(gap_pct.mean().item()),
        "objective_gap_l2": float(torch.linalg.vector_norm(gap_pct, ord=2).item()),
        "objective_gap_max": float(gap_pct.max().item()),
    }
    raw = {
        "split": split,
        "batch_size": int(yb.shape[0]),
        "xb": xb.detach().cpu(),
        "yb": yb.detach().cpu(),
        "pred": pred.detach().cpu(),
        "output_error": (pred - yb).detach().cpu(),
        "residual_real": res_real.detach().cpu(),
        "residual_imag": res_imag.detach().cpu(),
        "pg_violation": raw_ineq["pg"],
        "qg_violation": raw_ineq["qg"],
        "vm_violation": raw_ineq["vm"],
        "thermal_violation": thermal_violation.detach().cpu(),
        "thermal_ratio": thermal_ratio.detach().cpu(),
        "objective_pred": objective_pred.detach().cpu(),
        "objective_true": objective_true.detach().cpu(),
        "objective_gap_pct": gap_pct.detach().cpu(),
        "branch_specs": thermal_detail["branch_specs"],
        "gen_count": int(num_gens),
        "bus_count": int(num_buses),
    }
    return rows, raw


def aggregate_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    totals: Dict[str, float] = {}
    max_keys = {
        "output_l2_max",
        "eq_real_max",
        "eq_imag_max",
        "eq_l2_max",
        "ineq_pg_max",
        "ineq_qg_max",
        "ineq_vm_max",
        "thermal_max",
        "thermal_ratio_max",
        "objective_gap_max",
    }
    weighted_keys = {
        key
        for key in rows[0]
        if key
        not in {
            "split",
            "batch_size",
            "batch_index",
            "global_batch",
            "epoch",
            "patience",
            "stage_name",
            "kind",
            "model_name",
            "case_name",
            "phase",
            "task_id",
            "checkpoint_last",
            "checkpoint_best",
            "checkpoint_best_batch",
            "raw_artifact",
            "raw_summary",
        }
        and key not in max_keys
    }
    total_weight = 0.0
    for row in rows:
        weight = float(row.get("batch_size", 1.0))
        total_weight += weight
        for key in weighted_keys:
            totals[key] = totals.get(key, 0.0) + float(row.get(key, 0.0)) * weight
        for key in max_keys:
            totals[key] = max(totals.get(key, float("-inf")), float(row.get(key, 0.0)))
    out = {key: val / max(total_weight, 1.0) for key, val in totals.items() if key in weighted_keys}
    for key in max_keys:
        out[key] = totals.get(key, 0.0)
    out["batch_size"] = int(total_weight)
    return out


def compute_loader_metrics(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    constraints: Dict[str, Any],
    case_json_path: str,
    num_gens: int,
    num_buses: int,
    split: str,
    device: torch.device,
) -> Dict[str, float]:
    metrics, _ = compute_loader_artifacts(
        model,
        loader,
        constraints=constraints,
        case_json_path=case_json_path,
        num_gens=num_gens,
        num_buses=num_buses,
        split=split,
        device=device,
    )
    return metrics


def compute_loader_artifacts(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    constraints: Dict[str, Any],
    case_json_path: str,
    num_gens: int,
    num_buses: int,
    split: str,
    device: torch.device,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    model.eval()
    rows: list[Dict[str, float]] = []
    batches: list[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            xb, yb = batch[0], batch[1]
            xb = xb.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
            yb = yb.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
            pred = model(xb)
            batch_metrics, batch_raw = compute_batch_artifacts(
                xb,
                yb,
                pred,
                constraints=constraints,
                case_json_path=case_json_path,
                num_gens=num_gens,
                num_buses=num_buses,
                split=split,
            )
            batch_metrics = dict(batch_metrics)
            batch_metrics.setdefault("batch_index", int(batch_index))
            batch_metrics.setdefault("global_batch", int(batch_index))
            rows.append(batch_metrics)
            batches.append(
                {
                    "batch_index": int(batch_index),
                    "metrics": batch_metrics,
                    "raw": batch_raw,
                }
            )
    return aggregate_rows(rows), {
        "split": split,
        "batch_count": len(batches),
        "batches": batches,
    }


@dataclass
class TrainingTelemetry:
    results_dir: Path
    model_name: str
    case_name: str
    phase: str
    task_id: int
    num_gens: int
    num_buses: int
    case_json_path: Path
    constraints: Dict[str, Any]
    telemetry_csv_path: Path = field(init=False)
    progress_state_path: Path = field(init=False)
    stage_state_path: Path = field(init=False)
    last_checkpoint_path: Path = field(init=False)
    best_checkpoint_path: Path = field(init=False)
    best_batch_checkpoint_path: Path = field(init=False)
    csv_columns: list[str] = field(init=False)
    branch_count: int = field(init=False)
    input_dim: int = field(init=False)
    output_dim: int = field(init=False)
    model_slug: str = field(init=False)
    case_slug: str = field(init=False)
    batches_done: int = field(init=False)
    epochs_done: int = field(init=False)
    _stage_rng_deferred_to_checkpoint: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.model_slug = _safe_slug(self.model_name)
        self.case_slug = _safe_slug(self.case_name)
        telemetry_dir = self.results_dir / "telemetry" / self.model_slug / self.case_slug / f"task_{int(self.task_id):04d}"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_csv_path = telemetry_dir / "telemetry.csv"
        self.progress_state_path = telemetry_dir / "progress_state.json"
        self.stage_state_path = telemetry_dir / "stage_state.json"
        self.last_checkpoint_path = telemetry_dir / "checkpoint_last.pt"
        self.best_checkpoint_path = telemetry_dir / "checkpoint_best.pt"
        self.best_batch_checkpoint_path = telemetry_dir / "checkpoint_best_batch.pt"
        self.input_dim = 2 * self.num_buses
        self.output_dim = 2 * self.num_gens + 2 * self.num_buses
        self.branch_count = self._infer_branch_count()
        self.csv_columns = self._build_csv_columns()
        self.batches_done, self.epochs_done = self._load_progress_counts()

    def _infer_branch_count(self) -> int:
        case_json = _load_case_json_cached(str(self.case_json_path))
        dummy = torch.zeros((1, 2 * self.num_gens + 2 * self.num_buses), dtype=torch.float64)
        detail = _branch_thermal_vectors(case_json, dummy, self.num_gens, self.num_buses)
        return len(detail["branch_specs"])

    @staticmethod
    def _scalar_metric_keys() -> list[str]:
        return [
            "loss_mse",
            "loss_mae",
            "output_l2_mean",
            "output_l2_max",
            "eq_real_mean",
            "eq_real_l2",
            "eq_real_max",
            "eq_imag_mean",
            "eq_imag_l2",
            "eq_imag_max",
            "eq_l2_mean",
            "eq_l2_max",
            "ineq_pg_mean",
            "ineq_pg_l2",
            "ineq_pg_max",
            "ineq_qg_mean",
            "ineq_qg_l2",
            "ineq_qg_max",
            "ineq_vm_mean",
            "ineq_vm_l2",
            "ineq_vm_max",
            "thermal_mean",
            "thermal_l2",
            "thermal_max",
            "thermal_count_mean",
            "thermal_ratio_mean",
            "thermal_ratio_max",
            "objective_pred_mean",
            "objective_true_mean",
            "objective_gap_mean",
            "objective_gap_l2",
            "objective_gap_max",
        ]

    @staticmethod
    def _vector_stats_columns(prefix: str, feature_count: int) -> list[str]:
        if feature_count <= 1:
            return [
                f"{prefix}.mean",
                f"{prefix}.abs_mean",
                f"{prefix}.abs_max",
                f"{prefix}.l2",
                f"{prefix}.shape",
            ]
        columns: list[str] = []
        for stat in ("mean", "abs_mean", "abs_max", "l2"):
            columns.extend(f"{prefix}.{stat}.{index:04d}" for index in range(feature_count))
        columns.append(f"{prefix}.shape")
        return columns

    def _build_csv_columns(self) -> list[str]:
        scalar_keys = self._scalar_metric_keys()
        columns: list[str] = [
            "record_type",
            "kind",
            "state_kind",
            "model_name",
            "case_name",
            "phase",
            "task_id",
            "stage_name",
            "split",
            "epoch",
            "batch_index",
            "global_batch",
            "batch_size",
            "batch_count",
            "patience",
            "width",
            "depth",
            "best_val",
            "val_loss",
            "best_batch_loss",
            "stage_index",
            "best_width",
            "best_depth",
            "width_fail",
            "completed",
            "checkpoint_last",
            "checkpoint_best",
            "checkpoint_best_batch",
            "best_checkpoint",
            "batch_raw_artifact",
            "epoch_raw_artifact",
            "raw.kind",
            "raw.split",
            "raw.batch_size",
            "raw.batch_count",
            "raw.gen_count",
            "raw.bus_count",
            "raw.branch_count",
            "raw.keys",
            "raw.tensor_keys",
            "raw.branch_specs_json",
        ]
        columns.extend(scalar_keys)
        columns.extend(f"batch_metrics.{key}" for key in scalar_keys)
        columns.extend(f"train_epoch_metrics.{key}" for key in scalar_keys)
        columns.extend(f"val_epoch_metrics.{key}" for key in scalar_keys)
        columns.extend(self._vector_stats_columns("raw.xb", self.input_dim))
        columns.extend(self._vector_stats_columns("raw.yb", self.output_dim))
        columns.extend(self._vector_stats_columns("raw.pred", self.output_dim))
        columns.extend(self._vector_stats_columns("raw.output_error", self.output_dim))
        columns.extend(self._vector_stats_columns("raw.residual_real", self.num_buses))
        columns.extend(self._vector_stats_columns("raw.residual_imag", self.num_buses))
        columns.extend(self._vector_stats_columns("raw.pg_violation", self.num_gens))
        columns.extend(self._vector_stats_columns("raw.qg_violation", self.num_gens))
        columns.extend(self._vector_stats_columns("raw.vm_violation", self.num_buses))
        columns.extend(self._vector_stats_columns("raw.thermal_violation", self.branch_count))
        columns.extend(self._vector_stats_columns("raw.thermal_ratio", self.branch_count))
        columns.extend(self._vector_stats_columns("raw.objective_pred", 1))
        columns.extend(self._vector_stats_columns("raw.objective_true", 1))
        columns.extend(self._vector_stats_columns("raw.objective_gap_pct", 1))
        return columns

    def _append_row(self, row: Dict[str, Any]) -> None:
        append_csv_row(self.telemetry_csv_path, row, self.csv_columns)

    def _load_progress_counts(self) -> tuple[int, int]:
        if self.progress_state_path.exists():
            try:
                state = json.loads(self.progress_state_path.read_text(encoding="utf-8"))
                if isinstance(state, dict):
                    return int(state.get("history_rows") or 0), int(state.get("final_epoch") or 0)
            except Exception:
                pass
        batch_rows = 0
        epoch_rows = 0
        if not self.telemetry_csv_path.exists():
            return batch_rows, epoch_rows
        try:
            with self.telemetry_csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    kind = str(row.get("record_type") or row.get("kind") or "").strip().lower()
                    if kind == "batch":
                        batch_rows += 1
                    elif kind == "epoch":
                        epoch_rows += 1
        except Exception:
            return 0, 0
        return batch_rows, epoch_rows

    def _write_progress_state(self, payload: Optional[Dict[str, Any]] = None, *, kind: Optional[str] = None) -> None:
        state: Dict[str, Any] = {
            "schema": 1,
            "model_name": self.model_name,
            "case_name": self.case_name,
            "phase": self.phase,
            "task_id": int(self.task_id),
            "history_rows": int(self.batches_done),
            "final_epoch": int(self.epochs_done),
            "updated_at": _now_stamp(),
        }
        if payload is not None:
            state.update(
                {
                    "kind": kind or payload.get("kind"),
                    "stage_name": payload.get("stage_name"),
                    "split": payload.get("split"),
                    "epoch": payload.get("epoch"),
                    "batch_index": payload.get("batch_index"),
                    "global_batch": payload.get("global_batch"),
                    "batch_size": payload.get("batch_size"),
                    "batch_count": payload.get("batch_count"),
                    "patience": payload.get("patience"),
                    "width": payload.get("width"),
                    "depth": payload.get("depth"),
                    "best_val": payload.get("best_val"),
                    "val_loss": payload.get("val_loss"),
                    "best_batch_loss": payload.get("best_batch_loss"),
                    "checkpoint_last": payload.get("checkpoint_last"),
                    "checkpoint_best": payload.get("checkpoint_best"),
                    "checkpoint_best_batch": payload.get("checkpoint_best_batch"),
                    "active_stage": payload.get("active_stage"),
                    "current_hidden_widths": payload.get("current_hidden_widths"),
                    "current_width_growth_cursor": payload.get("current_width_growth_cursor"),
                    "stage_checkpoint": payload.get("stage_checkpoint"),
                    "batch_raw_artifact": payload.get("batch_raw_artifact"),
                    "epoch_raw_artifact": payload.get("epoch_raw_artifact"),
                }
            )
        atomic_write_json(self.progress_state_path, state)

    def _raw_row(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "raw.kind": raw.get("kind"),
            "raw.split": raw.get("split"),
            "raw.batch_size": raw.get("batch_size"),
            "raw.batch_count": raw.get("batch_count"),
            "raw.gen_count": raw.get("gen_count"),
            "raw.bus_count": raw.get("bus_count"),
            "raw.branch_count": len(raw.get("branch_specs", [])) if isinstance(raw.get("branch_specs"), list) else raw.get("branch_count"),
            "raw.keys": ",".join(sorted(str(key) for key in raw.keys())),
            "raw.tensor_keys": ",".join(sorted(str(key) for key, value in raw.items() if isinstance(value, torch.Tensor))),
            "raw.branch_specs_json": raw.get("branch_specs"),
        }
        row.update(_selected_tensor_summary(raw))
        return row

    def _base_row(self, payload: Dict[str, Any], *, record_type: str, state_kind: Optional[str] = None) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "record_type": record_type,
            "kind": payload.get("kind"),
            "state_kind": state_kind,
            "model_name": self.model_name,
            "case_name": self.case_name,
            "phase": self.phase,
            "task_id": int(self.task_id),
            "stage_name": payload.get("stage_name"),
            "split": payload.get("split"),
            "epoch": payload.get("epoch"),
            "batch_index": payload.get("batch_index"),
            "global_batch": payload.get("global_batch"),
            "batch_size": payload.get("batch_size"),
            "batch_count": payload.get("batch_count"),
            "patience": payload.get("patience"),
            "width": payload.get("width"),
            "depth": payload.get("depth"),
            "best_val": payload.get("best_val"),
            "val_loss": payload.get("val_loss"),
            "best_batch_loss": payload.get("best_batch_loss"),
            "stage_index": payload.get("stage_index"),
            "best_width": payload.get("best_width"),
            "best_depth": payload.get("best_depth"),
            "width_fail": payload.get("width_fail"),
            "completed": payload.get("completed"),
            "checkpoint_last": payload.get("checkpoint_last"),
            "checkpoint_best": payload.get("checkpoint_best"),
            "checkpoint_best_batch": payload.get("checkpoint_best_batch"),
            "best_checkpoint": payload.get("best_checkpoint"),
            "batch_raw_artifact": payload.get("batch_raw_artifact"),
            "epoch_raw_artifact": payload.get("epoch_raw_artifact"),
        }
        for key in self._scalar_metric_keys():
            row[key] = payload.get(key)
        if "batch_metrics" in payload and isinstance(payload["batch_metrics"], dict):
            for key in self._scalar_metric_keys():
                row[f"batch_metrics.{key}"] = payload["batch_metrics"].get(key)
        if "train_epoch_metrics" in payload and isinstance(payload["train_epoch_metrics"], dict):
            for key in self._scalar_metric_keys():
                row[f"train_epoch_metrics.{key}"] = payload["train_epoch_metrics"].get(key)
        if "val_epoch_metrics" in payload and isinstance(payload["val_epoch_metrics"], dict):
            for key in self._scalar_metric_keys():
                row[f"val_epoch_metrics.{key}"] = payload["val_epoch_metrics"].get(key)
        if "train_epoch_metrics" in payload and isinstance(payload["train_epoch_metrics"], dict):
            row["raw.kind"] = payload.get("kind")
            row["raw.split"] = payload["train_epoch_metrics"].get("split")
            row["raw.batch_count"] = payload["train_epoch_metrics"].get("batch_size")
        if "val_epoch_metrics" in payload and isinstance(payload["val_epoch_metrics"], dict):
            row["raw.batch_count"] = row.get("raw.batch_count") or payload["val_epoch_metrics"].get("batch_size")
        return row

    def record_batch(self, payload: Dict[str, Any], raw: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        row = self._base_row(dict(payload), record_type="batch")
        if raw is not None:
            row.update(self._raw_row(raw))
        self._append_row(row)
        self.batches_done += 1
        self._write_progress_state(dict(payload), kind="batch")
        return None

    def record_epoch(self, payload: Dict[str, Any], raw: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        row = self._base_row(dict(payload), record_type="epoch")
        if raw is not None:
            epoch_raw = dict(raw)
            train_raw = epoch_raw.get("train")
            val_raw = epoch_raw.get("val")
            if isinstance(train_raw, dict):
                row["raw.kind"] = epoch_raw.get("kind")
                row["raw.split"] = train_raw.get("split")
                row["raw.batch_count"] = train_raw.get("batch_count")
                row["raw.batch_size"] = train_raw.get("batch_count")
            if isinstance(val_raw, dict):
                row["raw.batch_count"] = row.get("raw.batch_count") or val_raw.get("batch_count")
            row["raw.keys"] = ",".join(sorted(str(key) for key in epoch_raw.keys()))
            row["raw.tensor_keys"] = ""
            row["raw.branch_specs_json"] = epoch_raw.get("stage_name")
        self._append_row(row)
        if str(payload.get("kind") or "epoch").lower() == "epoch":
            self.epochs_done += 1
        self._write_progress_state(dict(payload), kind=str(payload.get("kind") or "epoch"))
        return None

    def update_state(self, payload: Dict[str, Any]) -> None:
        row = self._base_row(dict(payload), record_type="state", state_kind=str(payload.get("kind")) if payload.get("kind") is not None else None)
        self._append_row(row)
        self._write_progress_state(dict(payload), kind=str(payload.get("kind")) if payload.get("kind") is not None else None)

    def save_stage_state(self, payload: Dict[str, Any]) -> None:
        from Dyn_DNN4OPF.utils.resume_state import capture_rng_state

        state = dict(payload)
        rng_name = f"stage_state.rng.{time.time_ns()}.{uuid.uuid4().hex}.pt"
        atomic_torch_save(self.stage_state_path.parent / rng_name, capture_rng_state())
        state.update(
            {
                "model_name": self.model_name,
                "case_name": self.case_name,
                "phase": self.phase,
                "task_id": int(self.task_id),
                "schema": 2,
                "rng_checkpoint": rng_name,
                "rng_checkpoint_path": str((self.stage_state_path.parent / rng_name).relative_to(self.results_dir)),
            }
        )
        atomic_write_json(self.stage_state_path, state)
        row = self._base_row(state, record_type="stage_state", state_kind="stage")
        self._append_row(row)
        self._write_progress_state(state, kind="stage")

    def refresh_active_stage_architecture(self, model: Any, checkpoint_path: Any) -> bool:
        """Align an active controller sidecar after a new-architecture seed checkpoint is durable."""
        if not self.stage_state_path.exists():
            return False
        try:
            state = json.loads(self.stage_state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"exact-resume failure: stage state cannot be read: {self.stage_state_path}") from exc
        if not isinstance(state, dict) or not bool(state.get("active_stage")) or bool(state.get("completed")):
            return False
        stage_checkpoint = state.get("stage_checkpoint")
        if not stage_checkpoint or Path(str(stage_checkpoint)).resolve() != Path(str(checkpoint_path)).resolve():
            return False
        hidden_widths = getattr(model, "hidden_widths", None)
        if not callable(hidden_widths):
            return False
        widths = [int(value) for value in hidden_widths()]
        if not widths:
            raise RuntimeError("exact-resume failure: active model has no hidden-width vector")
        state["current_hidden_widths"] = widths
        state["current_width_growth_cursor"] = int(getattr(model, "_width_growth_cursor", 0))
        state["architecture_reconciled_from_checkpoint"] = True
        self.save_stage_state(state)
        return True

    def exists(self) -> bool:
        return self.load_stage_state() is not None

    def read_text(self, encoding: str = "utf-8") -> str:
        state = self.load_stage_state()
        if state is None:
            return ""
        return json.dumps(state, indent=2, sort_keys=True)

    def write_text(self, payload: str, encoding: str = "utf-8") -> int:
        try:
            data = json.loads(payload)
        except Exception:
            data = {"kind": "stage_state", "raw_text": payload}
        if isinstance(data, dict):
            self.save_stage_state(data)
        return len(payload)

    def load_stage_state(self) -> Optional[Dict[str, Any]]:
        self._stage_rng_deferred_to_checkpoint = False
        if self.stage_state_path.exists():
            try:
                state = json.loads(self.stage_state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"exact-resume failure: stage state cannot be read: {self.stage_state_path}") from exc
            if not isinstance(state, dict):
                raise ValueError(f"stage state must be a JSON object: {self.stage_state_path}")
            if self._active_stage_checkpoint_has_exact_state(state):
                # The active checkpoint is the RNG authority. Rebuilding its
                # architecture consumes RNG, so defer restoration until the
                # training checkpoint has been loaded by the stage trainer.
                self._stage_rng_deferred_to_checkpoint = True
                return state
            if bool(state.get("active_stage")) and not bool(state.get("completed")):
                if self._active_stage_checkpoint_architecture_mismatch(state):
                    raise RuntimeError(
                        f"exact-resume failure: active stage checkpoint architecture disagrees with its sidecar: {self.stage_state_path}"
                    )
                try:
                    stage_index = int(state["stage_index"])
                    stage_name = str(state["stage_name"])
                    expected = (self.results_dir / "stages" / f"{stage_index:03d}_{stage_name}" / "checkpoint_last.pt").resolve()
                    candidate = Path(str(state["stage_checkpoint"])).resolve()
                    if candidate == expected and not candidate.exists():
                        # The controller sidecar was saved after the stage
                        # architecture change but before the trainer wrote its
                        # initial optimizer checkpoint. Rebuild, then restore
                        # the sidecar RNG before creating that checkpoint.
                        self._stage_rng_deferred_to_checkpoint = True
                        return state
                except Exception:
                    pass
                # A legacy active checkpoint may still be discoverable from
                # the stage directory. Let the controller's exact-stage
                # recovery inspect it instead of treating its sidecar as a
                # complete training checkpoint.
                return None
            if state.get("current_hidden_widths") and state.get("stage_checkpoint"):
                # Controller-state RNG is restored after architecture rebuild.
                # Rebuilding dynamic layers consumes random numbers.
                self._stage_rng_deferred_to_checkpoint = True
                return state
            try:
                self._restore_stage_rng(state)
            except Exception as exc:
                if not self._active_stage_checkpoint_has_exact_state(state):
                    raise RuntimeError(f"exact-resume failure: stage RNG state cannot be restored: {exc}") from exc
                # Do not restore here: callers may still rebuild the saved
                # architecture, which can consume RNG. The training checkpoint
                # restores RNG at the point training resumes.
                self._stage_rng_deferred_to_checkpoint = True
            return state
        if not self.telemetry_csv_path.exists():
            return None
        latest: Optional[Dict[str, Any]] = None
        with self.telemetry_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("record_type") != "stage_state":
                    continue
                if row.get("model_name") != self.model_name:
                    continue
                if row.get("case_name") != self.case_name:
                    continue
                if row.get("phase") != self.phase:
                    continue
                if int(row.get("task_id") or 0) != int(self.task_id):
                    continue
                latest = dict(row)
        if latest is not None:
            # CSV has no native booleans. Preserve the type expected by the
            # growth controllers; bool("False") would otherwise be True.
            for key in ("completed",):
                value = latest.get(key)
                if isinstance(value, str):
                    latest[key] = value.strip().lower() in {"1", "true", "yes", "y", "on"}
            if self._active_stage_checkpoint_has_exact_state(latest):
                self._stage_rng_deferred_to_checkpoint = True
                return latest
            if latest.get("current_hidden_widths") and latest.get("stage_checkpoint"):
                self._stage_rng_deferred_to_checkpoint = True
                return latest
            try:
                self._restore_stage_rng(latest)
            except Exception as exc:
                if not self._active_stage_checkpoint_has_exact_state(latest):
                    raise RuntimeError(f"exact-resume failure: stage RNG state cannot be restored: {exc}") from exc
                self._stage_rng_deferred_to_checkpoint = True
        return latest

    def _active_stage_checkpoint_has_exact_state(self, state: Dict[str, Any]) -> bool:
        """Validate an active stage's last checkpoint as an RNG source.

        Older stage-state JSON may lack its separate RNG companion even though
        the active stage's training checkpoint already stores complete resume
        state. Restrict this fallback to the canonical checkpoint for this
        exact stage; completed stages still require their stage RNG snapshot.
        """
        if not bool(state.get("active_stage")) or bool(state.get("completed")):
            return False
        try:
            stage_index = int(state["stage_index"])
            stage_name = str(state["stage_name"])
            if not stage_name or Path(stage_name).name != stage_name:
                return False
            expected = (self.results_dir / "stages" / f"{stage_index:03d}_{stage_name}" / "checkpoint_last.pt").resolve()
            candidate = Path(str(state["stage_checkpoint"])).resolve()
            if candidate != expected or not candidate.is_file():
                return False
            root = self.results_dir.resolve()
            if not candidate.is_relative_to(root):
                return False
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                return False
            if int(checkpoint.get("checkpoint_schema_version", 0) or 0) < 2:
                return False
            required = (
                "model_state", "opt_state", "sched_state", "epoch", "global_batch",
                "rng_state", "train_sampler_state", "best_snap", "best_val",
                "best_batch_loss", "best_batch_snap", "patience", "loader_contract", "backend_state",
            )
            if any(key not in checkpoint or checkpoint[key] is None for key in required):
                return False
            rng = checkpoint["rng_state"]
            if (
                not isinstance(rng, dict)
                or not rng.get("python")
                or rng.get("torch_cpu") is None
                or rng.get("numpy") is None
            ):
                return False
            for generator_key in ("train_loader_generator_state", "val_loader_generator_state"):
                generator_state = checkpoint.get(generator_key)
                if generator_state is not None and not isinstance(generator_state, torch.Tensor):
                    return False
            if not isinstance(checkpoint["loader_contract"], dict) or not isinstance(checkpoint["backend_state"], dict):
                return False
            cursor = int(checkpoint.get("global_batch", 0) or 0)
            sampler = checkpoint["train_sampler_state"]
            if cursor > 0 and (not isinstance(sampler, dict) or sampler.get("order") is None):
                return False
            current_widths = state.get("current_hidden_widths")
            snapshot = checkpoint.get("best_snap")
            saved_widths = snapshot.get("hidden_widths") if isinstance(snapshot, dict) else None
            if not isinstance(current_widths, (list, tuple)) or not isinstance(saved_widths, (list, tuple)):
                return False
            current_width_tuple = tuple(int(value) for value in current_widths)
            saved_width_tuple = tuple(int(value) for value in saved_widths)
            if current_width_tuple != saved_width_tuple:
                if not self._active_stage_architecture_can_be_recovered(state, checkpoint, saved_width_tuple):
                    return False
                # A depth controller can grow widths inside one stage while
                # the last checkpoint path remains unchanged. If interruption
                # lands after that checkpoint is written but before the
                # controller sidecar is refreshed, the complete checkpoint
                # is the authority for the active architecture.
                logger.warning(
                    "[RESUME] reconciling active depth-stage sidecar architecture from checkpoint: %s -> %s (%s)",
                    list(current_width_tuple),
                    list(saved_width_tuple),
                    self.stage_state_path,
                )
                state["current_hidden_widths"] = list(saved_width_tuple)
                snapshot_cursor = snapshot.get("width_growth_cursor")
                if snapshot_cursor is not None:
                    state["current_width_growth_cursor"] = int(snapshot_cursor)
                state["architecture_reconciled_from_checkpoint"] = True
                atomic_write_json(self.stage_state_path, state)
            return True
        except Exception:
            return False

    def _active_stage_architecture_can_be_recovered(
        self,
        state: Dict[str, Any],
        checkpoint: Dict[str, Any],
        saved_widths: tuple[int, ...],
    ) -> bool:
        """Permit only a self-consistent in-stage depth-width growth recovery."""
        try:
            stage_index = int(state["stage_index"])
            stage_name = str(state["stage_name"])
            parts = stage_name.split("_")
            if len(parts) < 3 or parts[0] != "depth" or int(parts[1]) != stage_index + 1:
                return False
            current_widths = tuple(int(value) for value in state["current_hidden_widths"])
            start_width = int(state["stage_start_width"])
            target_value = state.get("stage_target_width")
            if target_value is None:
                # Legacy depth discovery did not preserve the target. Its
                # only safe in-stage width transition is the simultaneous
                # one-step seed used by all-layers growth.
                if (
                    not bool(state.get("legacy_stage_recovery"))
                    or not stage_name.endswith("_seed")
                    or any(value != start_width for value in current_widths)
                    or any(value != start_width + 1 for value in saved_widths)
                ):
                    return False
                target_width = start_width + 1
                state["stage_target_width"] = target_width
            else:
                target_width = int(target_value)
            if (
                not current_widths
                or len(current_widths) != len(saved_widths)
                or len(saved_widths) != int(parts[1])
                or target_width < start_width
                or any(value < start_width or value > target_width for value in current_widths)
                or any(value < start_width or value > target_width for value in saved_widths)
                or any(saved < current for current, saved in zip(current_widths, saved_widths))
            ):
                return False
            snapshot = checkpoint.get("best_snap")
            snapshot_state = snapshot.get("state_dict") if isinstance(snapshot, dict) else None
            model_state = checkpoint.get("model_state")
            if not isinstance(snapshot_state, dict) or not isinstance(model_state, dict):
                return False
            if set(snapshot_state) != set(model_state):
                return False
            for key in snapshot_state:
                left, right = snapshot_state[key], model_state[key]
                if not hasattr(left, "shape") or not hasattr(right, "shape") or tuple(left.shape) != tuple(right.shape):
                    return False
            return True
        except Exception:
            return False

    def _active_stage_checkpoint_architecture_mismatch(self, state: Dict[str, Any]) -> bool:
        """Detect a loadable active checkpoint whose architecture contradicts its sidecar."""
        try:
            stage_index = int(state["stage_index"])
            stage_name = str(state["stage_name"])
            expected = (self.results_dir / "stages" / f"{stage_index:03d}_{stage_name}" / "checkpoint_last.pt").resolve()
            candidate = Path(str(state["stage_checkpoint"])).resolve()
            if candidate != expected or not candidate.is_file():
                return False
            checkpoint = torch.load(str(candidate), map_location="cpu", weights_only=False)
            snapshot = checkpoint.get("best_snap") if isinstance(checkpoint, dict) else None
            saved_widths = snapshot.get("hidden_widths") if isinstance(snapshot, dict) else None
            current_widths = state.get("current_hidden_widths")
            if not isinstance(saved_widths, (list, tuple)) or not isinstance(current_widths, (list, tuple)):
                return False
            return tuple(int(value) for value in saved_widths) != tuple(int(value) for value in current_widths)
        except Exception:
            return False

    def _restore_stage_rng(self, state: Dict[str, Any]) -> None:
        rng_name = state.get("rng_checkpoint")
        if not rng_name:
            raise ValueError("exact-resume failure: stage state lacks its RNG companion")
        if Path(str(rng_name)).name != str(rng_name):
            raise ValueError("exact-resume failure: stage RNG checkpoint name must be a local filename")
        rng_path = self.stage_state_path.parent / str(rng_name)
        if not rng_path.is_file():
            raise FileNotFoundError(f"exact-resume failure: stage RNG checkpoint is missing: {rng_path}")
        from Dyn_DNN4OPF.utils.resume_state import restore_rng_state
        try:
            rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"exact-resume failure: stage RNG checkpoint cannot be loaded: {rng_path}") from exc
        if not restore_rng_state(rng_state):
            raise ValueError(f"exact-resume failure: stage RNG checkpoint is invalid: {rng_path}")

    def save_last_checkpoint(self, payload: Any, *, mirror_path: Optional[Path] = None) -> None:
        atomic_torch_save(self.last_checkpoint_path, payload)
        if mirror_path is not None:
            atomic_torch_save(Path(mirror_path), payload)

    def save_best_checkpoint(self, payload: Any, *, mirror_path: Optional[Path] = None) -> None:
        atomic_torch_save(self.best_checkpoint_path, payload)
        if mirror_path is not None:
            atomic_torch_save(Path(mirror_path), payload)

    def save_best_batch_checkpoint(self, payload: Any, *, mirror_path: Optional[Path] = None) -> None:
        atomic_torch_save(self.best_batch_checkpoint_path, payload)
        if mirror_path is not None:
            atomic_torch_save(Path(mirror_path), payload)
