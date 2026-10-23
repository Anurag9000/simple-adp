import json

import pandas as pd

from simple_adp.benchmark_report import summarize


def test_report_ranks_task_metrics_in_the_declared_direction_and_keeps_variants_separate(tmp_path):
    signature = "task-aware-signature"
    (tmp_path / "active-run.json").write_text(json.dumps({"run_signature": signature}), encoding="utf-8")
    records = []
    for model, ndcg, family in (("good", 0.9, "grouped_ranking"), ("weak", 0.5, "grouped_ranking"),
                                ("good", 0.2, "recommendation_preference")):
        records.append({
            "status": "completed", "run_signature": signature, "provider": "fixture",
            "suite": "toy", "task_id": "one", "dataset_id": "d1", "dataset_name": "toy data",
            "task_family": family, "task_variant": "rank" if family == "grouped_ranking" else "preference",
            "model": model, "model_id": "MLP-Fixed", "generic_profile": None,
            "metric_directions": {"ndcg@10": "max", "accuracy": "max"},
            # roc_auc is a standard metric elsewhere, but is intentionally not
            # declared by these ranking/preference adapters.
            "test_metrics": {"ndcg@10": ndcg, "accuracy": ndcg, "roc_auc": 0.99},
        })
    with (tmp_path / "results.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    paths = summarize(tmp_path)
    frame = pd.read_csv(paths["dataset_summary"])
    ranking = frame.loc[frame["task_family"] == "grouped_ranking"]
    assert ranking.set_index("model")["rank_ndcg@10"].to_dict() == {"good": 1.0, "weak": 2.0}
    assert set(frame["task_family"]) == {"grouped_ranking", "recommendation_preference"}
    assert "roc_auc" not in frame.columns
