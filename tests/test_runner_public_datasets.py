import zipfile
import json
import hashlib
from pathlib import Path

import numpy as np

from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.public_datasets import MovieLensProvider
from simple_adp.runner.temporal_datasets import UCIElectricityProvider
from simple_adp.runner.sequence_datasets import UCIHAPTProvider
from simple_adp.runner.nab_datasets import NABProvider
from simple_adp.runner.monash_datasets import MonashForecastingProvider
from simple_adp.runner.physionet_datasets import PhysioNetECGProvider
from simple_adp.runner.odds_datasets import ODDSProvider, _ARTIFACTS, _EXPECTED_COUNTS
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.models import SimpleADPModelAdapter
from simple_adp.runner.plan import SuitePlan


def _write_movielens(path, dataset="ml-100k"):
    member = "ml-100k/u.data" if dataset == "ml-100k" else "ml-1m/ratings.dat"
    delimiter = "\t" if dataset == "ml-100k" else "::"
    with zipfile.ZipFile(path, "w") as archive:
        lines = []
        for user in range(1, 13):
            for item in range(1, 21):
                rating = 5 if item % 2 else 2
                lines.append(delimiter.join(map(str, (user, item, rating, 1_000_000 + item))))
        archive.writestr(member, "\n".join(lines) + "\n")


def _write_nab(path):
    import json
    import pandas as pd
    from simple_adp.runner.nab_datasets import NABProvider

    labels = {}
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(6):
            relative = f"realKnownCause/series_{index}.csv"
            timestamps = [stamp.strftime("%Y-%m-%d %H:%M:%S") for stamp in
                          pd.date_range("2020-01-01", periods=256, freq="min")]
            lines = ["timestamp,value", *(f"{stamp},{minute + index}" for minute, stamp in enumerate(timestamps))]
            archive.writestr(f"NAB-{NABProvider.commit}/data/{relative}", "\n".join(lines) + "\n")
            labels[relative] = [[timestamps[80], timestamps[110]]] if index < 3 else []
        archive.writestr(f"NAB-{NABProvider.commit}/labels/combined_labels.json", json.dumps(labels))


def _write_monash(path):
    lines = ["@attribute series_name string", "@attribute state string",
             "@attribute start_timestamp date", "@frequency half_hourly",
             "@missing false", "@equallength true", "@data"]
    for series, state in enumerate(("VIC", "NSW", "QLD", "TAS", "SA")):
        values = ",".join(str(series + index / 10) for index in range(11_000))
        lines.append(f"series-{series}:{state}:1997-01-01 00-00-00:{values}")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("m4_monthly_dataset.tsf", "\n".join(lines) + "\n")


def test_movielens_provider_has_three_task_variants_and_grouped_chronological_splits(tmp_path):
    archive = tmp_path / "ml-100k.zip"
    _write_movielens(archive)
    provider = MovieLensProvider(tmp_path, offline=True)
    refs = provider.list_tasks("ml-100k")
    assert [ref.metadata["task_variant"] for ref in refs] == ["rating", "preference", "ranking"]
    for ref in refs:
        data = provider.load_task(ref)
        split = provider.list_splits(ref, data, object())[0]
        assert data.X.columns.tolist() == ["user_id", "item_id"]
        assert data.dataset_metadata["archive_sha256"]
        assert [len(split.train_indices), len(split.validation_indices), len(split.test_indices)] == [192, 24, 24]
        assert not (set(split.train_indices) & set(split.validation_indices))
        assert data.context["group_id"].shape == (240,)
    rank_ref = refs[2]
    rank_data = provider.load_task(rank_ref)
    assert rank_data.task_kind == "ranking"
    assert rank_data.task_adapter.primary_metric == "ndcg@10"


def test_movielens_1m_uses_readme_archive_format(tmp_path):
    archive = tmp_path / "ml-1m.zip"
    _write_movielens(archive, dataset="ml-1m")
    provider = MovieLensProvider(tmp_path, offline=True)
    ref = DatasetTaskRef("movielens", "ml-1m", "ml-1m-rating", dataset_id="ml-1m",
                         metadata={"task_variant": "rating"})
    data = provider.load_task(ref)
    assert len(data.y) == 240
    assert np.isfinite(data.y).all()


def test_uci_electricity_windows_use_target_time_splits_and_keep_missing_features_for_train_imputation(tmp_path):
    archive_path = tmp_path / "electricityloaddiagrams20112014.zip"
    rows = [";meter_a;meter_b"]
    for index in range(180):
        value_b = "?" if index == 50 else f"{index / 10:.1f}"
        rows.append(f"2011-01-{1 + index // 96:02d} {index % 24 // 4:02d}:{index % 4 * 15:02d}:00;{index / 20:.2f};{value_b}")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("LD2011_2014.txt", "\n".join(rows) + "\n")
    provider = UCIElectricityProvider(tmp_path, offline=True)
    ref = provider.list_tasks("electricity")[0]
    data = provider.load_task(ref)
    split = provider.list_splits(ref, data, object())[0]
    assert data.X.shape[1] == 128 * 2
    assert data.dataset_metadata["stride"] == 1
    assert len(data.y) == 52
    assert data.content_sha256 == hashlib.sha256(
        f"{data.dataset_metadata['archive_sha256']}|lookback=128|horizon=1|stride=1".encode()
    ).hexdigest()
    assert np.isnan(data.X).any()
    assert np.all(data.context["target_position"][list(split.train_indices)] <
                  data.context["target_position"][list(split.validation_indices)].min())
    assert [len(split.train_indices), len(split.validation_indices), len(split.test_indices)] == [36, 5, 11]


def test_enabled_public_sources_expand_through_the_same_suite_plan_routes(tmp_path):
    movielens_root = tmp_path / "movielens-cache"
    movielens_root.mkdir()
    _write_movielens(movielens_root / "ml-100k.zip")
    electricity_root = tmp_path / "electricity-cache"
    electricity_root.mkdir()
    rows = [";meter_a"]
    rows.extend(f"2011-01-{1 + index // 96:02d} {index % 24 // 4:02d}:{index % 4 * 15:02d}:00;{index / 20:.2f}" for index in range(180))
    with zipfile.ZipFile(electricity_root / "electricityloaddiagrams20112014.zip", "w") as archive:
        archive.writestr("LD2011_2014.txt", "\n".join(rows) + "\n")
    nab_root = tmp_path / "nab-cache"
    nab_root.mkdir()
    from simple_adp.runner.nab_datasets import NABProvider
    _write_nab(nab_root / NABProvider.archive_name)
    monash_root = tmp_path / "monash-cache"
    monash_root.mkdir()
    _write_monash(monash_root / "monash-aus-electricity-demand.zip")
    providers = (MovieLensProvider(movielens_root, offline=True), UCIElectricityProvider(electricity_root, offline=True), NABProvider(nab_root, offline=True), MonashForecastingProvider(monash_root, offline=True))
    settings = RunnerSettings(
        cpu_only=True, run_hardness_screen=False, model_seeds=(0,), generic_profiles=("faithful", "corrected"),
        tasks=("all",), include_vanilla=True, width_only_depths=(1,),
        width_only_width=1, task_limit=None, fold_limit=1, job_limit=0,
        results_dir=str(tmp_path / "results"), run_root=str(tmp_path / "runs"),
    )
    plan = SuitePlan.build(
        settings=settings, results_dir=tmp_path / "results", run_root=tmp_path / "runs",
        dataset_providers=providers,
        dataset_suites=(("uci_temporal", "electricity"), ("movielens", "ml-100k"), ("nab", "v1.1"), ("monash", "aus-electricity-demand")),
        model_adapters=(SimpleADPModelAdapter(),),
    )
    refs = {run.task_ref.task_id: run.task_ref for run in plan.runs}
    assert {ref.task_id for ref in refs.values()} == {
        "uci-electricity-load-h1", "ml-100k-rating", "ml-100k-preference", "ml-100k-ranking",
        "nab-v1.1-window-anomaly", "monash-aus-electricity-demand-h1",
        "monash-aus-electricity-demand-h7", "monash-aus-electricity-demand-h28",
    }
    routes_by_task = {
        task_id: {run.route.route_label for run in plan.runs if run.task_ref.task_id == task_id}
        for task_id in refs
    }
    assert len(next(iter(routes_by_task.values()))) > 0
    assert all(routes == next(iter(routes_by_task.values())) for routes in routes_by_task.values())
    all_task_routes = [run.route for run in plan.runs if run.task_ref.task_id == next(iter(refs))]
    vanilla_labels = {route.label for route in all_task_routes if route.label and route.label.startswith("vanilla-d")}
    assert len(vanilla_labels) == 50
    from simple_adp.registry import MODEL_REGISTRY
    assert {route.model_id for route in all_task_routes} >= set(MODEL_REGISTRY)
    assert {route.generic_profile for route in all_task_routes if route.generic_profile is not None} == {"faithful", "corrected"}
    for ref in refs.values():
        assert ref.metadata["task_family"]
        assert ref.metadata["task_variant"]
        assert ref.metadata["split_protocol"]


def test_hapt_raw_parser_creates_per_step_windows_and_disjoint_subject_partitions(tmp_path):
    archive_path = tmp_path / "uci-hapt.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("HAPT/Train/subject_id_train.txt", "1\n2\n3\n4\n5\n")
        archive.writestr("HAPT/Test/subject_id_test.txt", "6\n")
        label_lines = []
        for subject in range(1, 7):
            experiment = subject
            stem = f"HAPT/RawData"
            acc = np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (256, 1))
            gyro = np.tile(np.array([[0.4, 0.5, 0.6]], dtype=np.float32), (256, 1))
            from io import StringIO
            acc_buffer, gyro_buffer = StringIO(), StringIO()
            np.savetxt(acc_buffer, acc)
            np.savetxt(gyro_buffer, gyro)
            archive.writestr(f"{stem}/acc_exp{experiment:02d}_user{subject:02d}.txt", acc_buffer.getvalue())
            archive.writestr(f"{stem}/gyro_exp{experiment:02d}_user{subject:02d}.txt", gyro_buffer.getvalue())
            label_lines.extend((f"{experiment} {subject} 1 1 128", f"{experiment} {subject} 2 129 256"))
        archive.writestr("HAPT/RawData/labels.txt", "\n".join(label_lines) + "\n")
    provider = UCIHAPTProvider(tmp_path, offline=True)
    ref = provider.list_tasks("hapt")[0]
    data = provider.load_task(ref)
    split = provider.list_splits(ref, data, object())[0]
    assert data.X.shape == (12, 128 * 6)
    assert data.y.shape == (12, 128)
    assert [len(split.train_indices), len(split.validation_indices), len(split.test_indices)] == [8, 2, 2]
    subjects = data.context["subject_id"]
    assert set(subjects[list(split.train_indices)]) == {1, 2, 3, 4}
    assert set(subjects[list(split.validation_indices)]) == {5}
    assert set(subjects[list(split.test_indices)]) == {6}


def test_nab_provider_parses_published_intervals_and_splits_by_source_file(tmp_path):
    archive_path = tmp_path / f"nab-{NABProvider.commit}.zip"
    _write_nab(archive_path)
    provider = NABProvider(tmp_path, offline=True)
    ref = provider.list_tasks("v1.1")[0]
    data = provider.load_task(ref)
    split = provider.list_splits(ref, data, object())[0]
    assert data.X.shape[1] == 128
    assert set(np.unique(data.y)) == {0, 1}
    assert data.dataset_metadata["repository_commit"] == NABProvider.commit
    assert data.context["event_id"].shape == data.y.shape
    partitions = [set(data.context["series_id"][list(part)]) for part in
                  (split.train_indices, split.validation_indices, split.test_indices)]
    assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])
    assert all(set(np.unique(data.y[list(part)])) == {0, 1} for part in
               (split.train_indices, split.validation_indices, split.test_indices))


def test_nab_file_split_uses_run_seed_deterministically(tmp_path):
    from types import SimpleNamespace

    archive_path = tmp_path / f"nab-{NABProvider.commit}.zip"
    _write_nab(archive_path)
    provider = NABProvider(tmp_path, offline=True)
    ref = provider.list_tasks("v1.1")[0]
    data = provider.load_task(ref)
    first = provider.list_splits(ref, data, SimpleNamespace(seed=19))[0]
    repeated = provider.list_splits(ref, data, SimpleNamespace(seed=19))[0]
    other = provider.list_splits(ref, data, SimpleNamespace(seed=29))[0]
    assert first == repeated
    file_ids = np.asarray(data.context["series_id"])
    first_train = set(file_ids[list(first.train_indices)])
    other_train = set(file_ids[list(other.train_indices)])
    assert first_train != other_train


def test_monash_tsf_loader_uses_declared_frequency_horizons_and_entity_splits(tmp_path):
    archive_path = tmp_path / "monash-aus-electricity-demand.zip"
    _write_monash(archive_path)
    provider = MonashForecastingProvider(tmp_path, offline=True)
    refs = provider.list_tasks("aus-electricity-demand")
    assert [ref.metadata["horizon"] for ref in refs] == [1, 7, 28]
    for ref in refs:
        data = provider.load_task(ref)
        split = provider.list_splits(ref, data, object())[0]
        assert data.X.shape[1] == 128
        assert data.context["entity_id"].shape == data.y.shape
        assert data.dataset_metadata["frequency"] == "half_hourly"
        assert data.dataset_metadata["license"] == "CC BY 4.0"
        assert data.dataset_metadata["stride"].endswith("24")
        assert len(split.train_indices) + len(split.validation_indices) + len(split.test_indices) == len(data.y)
        for entity in np.unique(data.context["entity_id"]):
            partitions = [set(np.flatnonzero(data.context["entity_id"] == entity)) & set(indices)
                          for indices in (split.train_indices, split.validation_indices, split.test_indices)]
            assert all(partitions)


def test_physionet_provider_resamples_and_keeps_records_and_201_202_patient_pair_disjoint(tmp_path):
    from types import SimpleNamespace

    records = ("100", "201", "202", "203", "204")

    class FakeWFDB:
        class processing:
            @staticmethod
            def resample_sig(signal, source_hz, target_hz):
                factor = int(target_hz / source_hz)
                return np.repeat(signal, factor), np.arange(len(signal) * factor) / target_hz

        @staticmethod
        def rdsamp(record_path):
            signal = np.linspace(-1, 1, 2048, dtype=np.float32)
            return signal[:, None], {"fs": 128}

        @staticmethod
        def rdann(record_path, extension):
            return SimpleNamespace(sample=np.asarray([400, 600, 800, 1000, 1200]),
                                   symbol=["N", "A", "V", "F", "Q"])

    cache = tmp_path / "mitdb-1.0.0"
    cache.mkdir()
    for record in records:
        for extension in ("hea", "dat", "atr"):
            (cache / f"{record}.{extension}").write_bytes(f"{record}-{extension}".encode())
    provider = PhysioNetECGProvider(tmp_path, offline=True, wfdb_module=FakeWFDB())
    provider.record_overrides["mitdb"] = records
    ref = provider.list_tasks("mitdb")[0]
    data = provider.load_task(ref)
    split = provider.list_splits(ref, data, RunnerSettings(seed=7))[0]
    assert data.X.shape == (len(records) * 5, 360)
    assert set(np.unique(data.y)) == {"N", "S", "V", "F", "Q"}
    assert set(data.dataset_metadata["source_sampling_frequencies_hz"].values()) == {128.0}
    pair_parts = [index for index, partition in enumerate(
        (split.train_indices, split.validation_indices, split.test_indices))
                  if {201, 202} & set(data.context["record_id"][list(partition)])]
    assert len(pair_parts) == 1
    assert set(np.unique(data.context["patient_group_id"])) == {0, 1, 3, 4}


def test_odds_allowlist_loads_individually_described_artifacts_with_stratified_splits(tmp_path):
    expected_dims = {
        "cardio": 21, "thyroid": 6, "satimage-2": 36, "lympho": 18,
        "glass": 9, "vowels": 12, "musk": 166, "letter": 32,
        "speech": 400, "satellite": 36, "shuttle": 9, "breastw": 9,
        "arrhythmia": 274, "ionosphere": 33, "mnist": 100,
        "optdigits": 64, "http": 3, "smtp": 3, "mammography": 6,
        "annthyroid": 6, "pendigits": 16, "ecoli": 7, "wine": 13,
        "vertebral": 6,
    }
    expected_counts = {
        "cardio": (1831, 176), "thyroid": (3772, 93), "satimage-2": (5803, 71),
        "lympho": (148, 6), "glass": (214, 9), "vowels": (1456, 50),
        "musk": (3062, 97), "letter": (1600, 100), "speech": (3686, 61),
        "satellite": (6435, 2036), "shuttle": (49097, 3511), "breastw": (683, 239),
        "arrhythmia": (452, 66), "ionosphere": (351, 126), "mnist": (7603, 700),
        "optdigits": (5216, 150), "http": (567498, 2211), "smtp": (95156, 30),
        "mammography": (11183, 260), "annthyroid": (7200, 534),
        "pendigits": (6870, 156), "ecoli": (336, 9), "wine": (129, 10),
        "vertebral": (240, 30),
    }
    cache = tmp_path / "odds-cache"
    cache.mkdir()
    for suite in expected_dims:
        (cache / f"odds-{suite}.mat").write_bytes(f"fixture-{suite}".encode())

    def mat_reader(path):
        suite = path.stem.removeprefix("odds-")
        rows, positives = expected_counts[suite]
        dims = expected_dims[suite]
        rng = np.random.default_rng(sum(path.name.encode()))
        x = rng.normal(size=(rows, dims)).astype(np.float32)
        y = np.asarray([0] * (rows - positives) + [1] * positives, dtype=np.int64).reshape(-1, 1)
        return {"X": x, "y": y}

    provider = ODDSProvider(cache, offline=True, mat_reader=mat_reader)
    assert set(provider.suites) == set(expected_dims)
    assert len(provider.suites) == 24
    for suite, feature_count in expected_dims.items():
        ref = provider.list_tasks(suite)[0]
        data = provider.load_task(ref)
        split = provider.list_splits(ref, data, RunnerSettings())[0]
        expected_rows, expected_outliers = expected_counts[suite]
        assert data.X.shape == (expected_rows, feature_count)
        assert data.task_family == "supervised_anomaly"
        assert data.dataset_metadata["artifact_sha256"]
        assert data.dataset_metadata["odds_transformation"]
        assert "no separate license stated" in data.dataset_metadata["license"]
        assert data.dataset_metadata["outlier_count"] == expected_outliers
        assert sum(map(len, (split.train_indices, split.validation_indices, split.test_indices))) == expected_rows
        for part in (split.train_indices, split.validation_indices, split.test_indices):
            assert set(np.unique(data.y[list(part)])) == {0, 1}
    with __import__("pytest").raises(ValueError, match="individually reviewed allowlist"):
        provider.list_tasks("unknown")


def test_odds_catalog_matches_every_enabled_artifact_and_every_explicit_exclusion():
    catalog_path = Path(__file__).parents[1] / "docs" / "SUPERVISED_DATASET_CATALOG.json"
    catalog = json.loads(catalog_path.read_text())
    entries = {entry["suite"]: entry for entry in catalog["enabled"] if entry["provider"] == "odds"}
    assert set(entries) == set(_ARTIFACTS)
    for suite, info in _ARTIFACTS.items():
        entry = entries[suite]
        rows, outliers = _EXPECTED_COUNTS[suite]
        assert entry["source_url"] == info["odds_page"]
        assert entry["artifact_url"] == info["mat_url"]
        assert entry["source_dataset_url"] == info["source_url"]
        assert entry["transformation"] == info["transformation"]
        assert entry["schema"] == {
            "rows": rows, "features": info["feature_count"], "outliers": outliers,
            "target": "binary 0=inlier, 1=outlier",
        }
        assert "does not state a separate license" in entry["license_and_terms"]
    excluded = {entry["suite"] for entry in catalog["planned_not_enabled"]}
    assert excluded == {"wbc", "forestcover", "pima", "mulcross", "yeast", "heart", "seismic"}
    pools = {entry["suite"]: entry for entry in catalog["enabled_suite_pools"]}
    assert {name: (entry["openml_suite_id"], entry["task_count"], entry["task_family"])
            for name, entry in pools.items()} == {
        "cc18": (99, 72, "classification"),
        "tabzilla": (379, 36, "classification"),
        "ctr23": (353, 35, "regression"),
    }


def test_odds_loader_rejects_artifact_with_wrong_published_row_and_outlier_counts(tmp_path):
    cache = tmp_path / "odds-cache"
    cache.mkdir()
    (cache / "odds-lympho.mat").write_bytes(b"not-a-real-mat-file; injected reader supplies fixture")

    def wrong_counts(_path):
        return {"X": np.zeros((148, 18), dtype=np.float32),
                "y": np.asarray([0] * 141 + [1] * 7, dtype=np.int64)}

    provider = ODDSProvider(cache, offline=True, mat_reader=wrong_counts)
    with __import__("pytest").raises(ValueError, match="must have 148 rows and 6 outliers"):
        provider.load_task(provider.list_tasks("lympho")[0])


def test_odds_loader_reads_matlab_v73_hdf5_and_normalizes_feature_major_arrays(tmp_path):
    h5py = __import__("pytest").importorskip("h5py")
    cache = tmp_path / "odds-cache"
    cache.mkdir()
    path = cache / "odds-lympho.mat"
    with h5py.File(path, "w", userblock_size=512) as handle:
        handle.create_dataset("X", data=np.zeros((18, 148), dtype=np.float64))
        handle.create_dataset("y", data=np.asarray([[0] * 142 + [1] * 6], dtype=np.float64))
    header = bytearray(512)
    header[:38] = b"MATLAB 7.3 MAT-file, Platform: GLNXA64"
    header[124:128] = b"\x00\x02IM"
    with path.open("r+b") as stream:
        stream.write(header)
    provider = ODDSProvider(cache, offline=True)
    data = provider.load_task(provider.list_tasks("lympho")[0])
    assert data.X.shape == (148, 18)
    assert data.y.shape == (148,)
    assert int(data.y.sum()) == 6
