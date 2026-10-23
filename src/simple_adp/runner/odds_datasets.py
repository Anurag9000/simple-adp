"""Individually inventoried, public ODDS binary tabular anomaly datasets."""

from __future__ import annotations

import hashlib
import importlib
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..advanced_tasks import SupervisedAnomalyTask
from ..task_preparation import stratified_split
from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices


def _artifact(slug: str, dropbox_id: str, features: int, source: str,
              transformation: str, citation: str,
              *, dropbox_legacy: bool = True) -> dict[str, Any]:
    """Build a data-only entry; provenance and terms stay artifact-specific."""
    if dropbox_legacy:
        url = f"https://www.dropbox.com/s/{dropbox_id}/{slug}.mat?dl=1"
    else:
        # The existing files use Dropbox's newer shared-file URL form.
        url = dropbox_id
    return {
        "mat_url": url,
        "odds_page": f"https://shebuti.com/{slug}-dataset/",
        "source_url": source,
        "source_id": slug,
        "source_title": slug.replace("-", " ").title(),
        "feature_count": features,
        "transformation": transformation,
        "citation": citation,
    }


# This is an explicit allowlist, not a blanket claim that every ODDS entry has
# equivalent source terms. Each artifact records its own transformation and
# original source. ODDS does not declare a separate license for these MAT files.
_ARTIFACTS: dict[str, dict[str, Any]] = {
    "cardio": _artifact(
        "cardiotocogrpahy", "https://www.dropbox.com/scl/fi/a1v7m3p1kbcnte1/cardio.mat?rlkey=84oboq7oqussys4iqi2b3kgv2&dl=1",
        21, "https://archive.ics.uci.edu/dataset/193/cardiotocography",
        "normal fetal-state class is inlier; pathologic class is downsampled to 176 anomalies; suspect class is discarded",
        "Campos and Bernardes (2000), UCI DOI 10.24432/C51S4N; Rayana (2016), ODDS", dropbox_legacy=False),
    "thyroid": _artifact(
        "thyroid-disease", "https://www.dropbox.com/scl/fi/bfwe561x2bvbbjq/thyroid.mat?rlkey=rrwjoqv9eymgyey3rmy5s5id1&dl=1",
        6, "https://archive.ics.uci.edu/dataset/102/thyroid+disease",
        "six real attributes from the ODDS Thyroid training subset; hyperfunction is anomaly; normal and subnormal are inliers",
        "Quinlan (1986), UCI DOI 10.24432/C5D010; Rayana (2016), ODDS", dropbox_legacy=False),
    "satimage-2": _artifact(
        "satimage-2", "https://www.dropbox.com/scl/fi/s4uymhnzj1ec1hu/satimage-2.mat?rlkey=f40h6so1cy2njmvz3nuu57vhy&dl=1",
        36, "https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite",
        "combines source rows, downsamples class 2 to 71 anomalies, and labels all other classes inlier",
        "Srinivasan (1993), UCI DOI 10.24432/C55887; Rayana (2016), ODDS", dropbox_legacy=False),
    "lympho": _artifact("lympho", "ag469ssk0lmctco", 18,
        "https://archive.ics.uci.edu/ml/datasets/Lymphography",
        "two smallest source classes (counts 2 and 4) are outliers; the two larger classes are inliers",
        "UCI Lymphography; Rayana (2016), ODDS"),
    "glass": _artifact("glass", "iq3hjxw77gpbl7u", 9,
        "https://archive.ics.uci.edu/ml/datasets/Glass+Identification",
        "source class 6 is the outlier; all other classes are inliers",
        "UCI Glass Identification; Rayana (2016), ODDS"),
    "vowels": _artifact("vowels", "pa26odoq6atq9vx", 12,
        "https://archive.ics.uci.edu/ml/datasets/Japanese+Vowels",
        "frame rows from 12-dimensional sequences; speaker 1 is downsampled to 50 outliers; speakers 6, 7, and 8 are inliers; remaining speakers are discarded",
        "UCI Japanese Vowels; Rayana (2016), ODDS"),
    "musk": _artifact("musk", "we6aqhb0m38i60t", 166,
        "https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)",
        "non-musk classes j146, j147, and 252 are inliers; musk classes 213 and 211 are outliers; remaining classes are discarded",
        "UCI Musk Version 2; Rayana (2016), ODDS"),
    "letter": _artifact("letter", "rt9i95h9jywrtiy", 32,
        "https://archive.ics.uci.edu/ml/datasets/Letter+Recognition",
        "constructs 1,500 inliers from three letters using paired feature concatenation and 100 outliers from other letters concatenated with normal components",
        "UCI Letter Recognition; Rayana (2016), ODDS"),
    "speech": _artifact("speech", "w6xv51ctea6uauc", 400,
        "http://speech.fit.vutbr.cz/",
        "American-accent i-vectors are inliers; the other seven accents are outliers",
        "VUT Brno speech corpus; Rayana (2016), ODDS"),
    "satellite": _artifact("satellite", "dpzxp8jyr9h93k5", 36,
        "https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)",
        "the three smallest classes (2, 4, and 5) are outliers; all other classes are inliers",
        "UCI Statlog Landsat Satellite; Rayana (2016), ODDS"),
    "shuttle": _artifact("shuttle", "mk8ozgisimfn3dw", 9,
        "https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)",
        "classes 2, 3, 5, 6, and 7 are outliers; class 1 is inlier; class 4 is discarded",
        "UCI Statlog Shuttle; Rayana (2016), ODDS"),
    "breastw": _artifact("breastw", "g3hlnucj71kfvq4", 9,
        "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)",
        "malignant rows are outliers and benign rows are inliers",
        "UCI Breast Cancer Wisconsin Original; Rayana (2016), ODDS"),
    "arrhythmia": _artifact("arrhythmia", "lmlwuspn1sey48r", 274,
        "https://archive.ics.uci.edu/ml/datasets/Arrhythmia",
        "classes 3, 4, 5, 7, 8, 9, 14, and 15 are outliers; all other classes are inliers; five nominal attributes are discarded",
        "UCI Arrhythmia; Rayana (2016), ODDS"),
    "ionosphere": _artifact("ionosphere", "lpn4z73fico4uup", 33,
        "https://archive.ics.uci.edu/ml/datasets/Ionosphere",
        "bad-radar returns are outliers and good returns are inliers; the all-zero feature is discarded",
        "UCI Ionosphere; Rayana (2016), ODDS"),
    "mnist": _artifact("mnist", "n3wurjt8v9qi6nc", 100,
        "http://yann.lecun.com/exdb/mnist/",
        "digit 0 examples are inliers; 700 digit 6 examples are sampled as outliers; 100 of 784 pixel features are selected",
        "LeCun et al., MNIST; Rayana (2016), ODDS"),
    "optdigits": _artifact("optdigits", "w52ndgz5k75s514", 64,
        "https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits",
        "digit 0 is downsampled to 150 outliers; digits 1 through 9 are inliers",
        "UCI Optical Recognition of Handwritten Digits; Rayana (2016), ODDS"),
    "http": _artifact("http", "iy9ucsifal754tp", 3,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/",
        "service-specific KDD Cup 99 subset; continuous source values are log(x+0.1); MAT artifact has 3 features",
        "UCI KDD Cup 99; Rayana (2016), ODDS"),
    "smtp": _artifact("smtp", "dbv2u4830xri7og", 3,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/",
        "service-specific KDD Cup 99 subset; continuous source values are log(x+0.1); MAT artifact has 3 features",
        "UCI KDD Cup 99; Rayana (2016), ODDS"),
    "mammography": _artifact("mammography", "tq2v4hhwyv17hlk", 6,
        "https://www.openml.org/d/310",
        "minority calcification class is outlier; non-calcification class is inlier",
        "OpenML dataset 310; Rayana (2016), ODDS"),
    "annthyroid": _artifact("annthyroid", "aifk51owxbogwav", 6,
        "https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease",
        "six real features; hyperfunction and subnormal classes are outliers; the remaining class is inlier",
        "UCI Thyroid Disease; Rayana (2016), ODDS"),
    "pendigits": _artifact("pendigits", "1x8rzb4a0lia6t1", 16,
        "https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits",
        "equal-frequency source classes; digit 0 is reduced to approximately 10 percent as the outlier class",
        "UCI Pen-Based Recognition of Handwritten Digits; Rayana (2016), ODDS"),
    "ecoli": _artifact("ecoli", "f38mquv0g08xxgp", 7,
        "https://archive.ics.uci.edu/ml/datasets/Ecoli",
        "omL, imL, and imS minority classes are outliers; other classes are inliers; sequence-name field is removed",
        "UCI Ecoli; Rayana (2016), ODDS"),
    "wine": _artifact("wine", "uvjaudt2uto7zal", 13,
        "https://archive.ics.uci.edu/ml/datasets/Wine",
        "classes 2 and 3 are inliers; class 1 is downsampled to 10 outliers",
        "UCI Wine; Rayana (2016), ODDS"),
    "vertebral": _artifact("vertebral", "5kuqb387sgvwmrb", 6,
        "https://archive.ics.uci.edu/ml/datasets/Vertebral+Column",
        "AB class is inlier; NO class is downsampled from 100 to 30 outliers",
        "UCI Vertebral Column; Rayana (2016), ODDS"),
}

# Published ODDS index dimensions and positive-label counts. These pin the
# artifact identity in addition to X's feature width and the file digest.
_EXPECTED_COUNTS = {
    "lympho": (148, 6), "glass": (214, 9), "vowels": (1456, 50),
    "cardio": (1831, 176), "thyroid": (3772, 93), "musk": (3062, 97),
    "satimage-2": (5803, 71), "letter": (1600, 100), "speech": (3686, 61),
    "satellite": (6435, 2036), "shuttle": (49097, 3511), "breastw": (683, 239),
    "arrhythmia": (452, 66), "ionosphere": (351, 126), "mnist": (7603, 700),
    # The downloaded HTTP MAT artifact is 567,498 rows; ODDS index/detail
    # prose currently disagree (567,479 and 567,497 respectively).
    "optdigits": (5216, 150), "http": (567498, 2211), "smtp": (95156, 30),
    "mammography": (11183, 260), "annthyroid": (7200, 534),
    "pendigits": (6870, 156), "ecoli": (336, 9), "wine": (129, 10),
    "vertebral": (240, 30),
}

_ODDS_PAGES = {
    "cardio": "cardiotocogrpahy-dataset", "thyroid": "thyroid-disease-dataset",
    "satimage-2": "satimage-2-dataset", "lympho": "lymphography-dataset",
    "glass": "glass-dataset", "vowels": "japanese-vowels-dataset",
    "musk": "musk-dataset", "letter": "letter-recognition-dataset",
    "speech": "speech-dataset", "satellite": "satellite-dataset",
    "shuttle": "shuttle-dataset", "breastw": "breast-cancer-wisconsin-original-dataset",
    "arrhythmia": "arrhythmia-dataset", "ionosphere": "ionosphere-dataset",
    "mnist": "mnist-dataset", "optdigits": "optdigits-dataset",
    "http": "http-kddcup99-dataset", "smtp": "smtp-kddcup99-dataset",
    "mammography": "mammography-dataset", "annthyroid": "annthyroid-dataset",
    "pendigits": "pendigits-dataset", "ecoli": "ecoli-dataset",
    "wine": "wine-dataset", "vertebral": "vertebral-dataset",
}
for _suite, _page in _ODDS_PAGES.items():
    _ARTIFACTS[_suite]["odds_page"] = f"https://shebuti.com/{_page}/"
_ARTIFACTS["http"]["published_count_note"] = (
    "Directly downloaded MAT file is 567,498 rows; the ODDS detail text says 567,497 "
    "and the index table says 567,479. The acquired file shape is treated as the artifact contract."
)


class ODDSProvider:
    """Cached ODDS MAT files with artifact-specific provenance and splits."""

    name = "odds"
    split_seed = 42

    def __init__(self, cache_dir: str | Path | None = None, *, offline: bool = False,
                 mat_reader: Callable[[Path], dict[str, Any]] | None = None) -> None:
        configured = cache_dir or os.environ.get("SIMPLE_ADP_DATA_CACHE")
        self.cache_dir = Path(configured).expanduser() if configured else Path.home() / ".cache/simple-adp/datasets"
        self.offline = offline
        self._mat_reader = mat_reader

    @property
    def suites(self) -> tuple[str, ...]:
        return tuple(_ARTIFACTS)

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite not in _ARTIFACTS:
            raise ValueError(f"ODDS artifact {suite!r} is not in the individually reviewed allowlist")
        return [DatasetTaskRef(self.name, suite, f"odds-{suite}-supervised-anomaly",
                               dataset_id=f"odds-{suite}")]

    def _path(self, suite: str) -> Path:
        path = self.cache_dir / f"odds-{suite}.mat"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            if self.offline:
                raise FileNotFoundError(f"ODDS {suite} artifact is not cached: {path}")
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                request = urllib.request.Request(_ARTIFACTS[suite]["mat_url"],
                                                 headers={"User-Agent": "simple-adp/0.1"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    @staticmethod
    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _read(self, path: Path) -> dict[str, Any]:
        if self._mat_reader is not None:
            return self._mat_reader(path)
        try:
            scipy_io = importlib.import_module("scipy.io")
        except ImportError as exc:
            raise RuntimeError("ODDS MAT loading requires the optional 'odds' extra (scipy)") from exc
        try:
            return scipy_io.loadmat(path)
        except NotImplementedError as exc:
            # MATLAB v7.3 files use HDF5 and are not supported by scipy.io.
            try:
                h5py = importlib.import_module("h5py")
            except ImportError as missing_hdf5:
                raise RuntimeError(
                    "This ODDS MAT file uses MATLAB v7.3/HDF5; install the optional 'odds' extra (h5py)"
                ) from missing_hdf5
            with h5py.File(path, "r") as handle:
                if "X" not in handle or "y" not in handle:
                    raise ValueError("ODDS v7.3 MAT artifact must contain X and y datasets") from exc
                # MATLAB stores arrays in column-major order; row/feature
                # orientation is normalized against y in load_task().
                return {"X": np.asarray(handle["X"]), "y": np.asarray(handle["y"])}

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if ref.provider != self.name or ref.suite not in _ARTIFACTS:
            raise ValueError("ODDS task reference is not in the individually reviewed allowlist")
        if ref.task_id != f"odds-{ref.suite}-supervised-anomaly":
            raise ValueError("unsupported ODDS task ID")
        info = _ARTIFACTS[ref.suite]
        path = self._path(ref.suite)
        parsed = self._read(path)
        if "X" not in parsed or "y" not in parsed:
            raise ValueError(f"ODDS MAT artifact {ref.suite} must contain X and y arrays")
        x = np.asarray(parsed["X"], dtype=np.float32)
        y = np.asarray(parsed["y"]).reshape(-1)
        if x.ndim == 2 and x.shape[0] != len(y) and x.shape[1] == len(y):
            x = x.T
        if x.ndim != 2 or x.shape[0] != len(y):
            raise ValueError("ODDS X/y arrays must be aligned row-major examples")
        if x.shape[1] != int(info["feature_count"]):
            raise ValueError(f"ODDS {ref.suite} must have {info['feature_count']} features, found {x.shape[1]}")
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("ODDS anomaly labels must contain exactly 0=inlier and 1=outlier")
        expected_rows, expected_outliers = _EXPECTED_COUNTS[ref.suite]
        if len(y) != expected_rows or int(np.count_nonzero(y == 1)) != expected_outliers:
            raise ValueError(
                f"ODDS {ref.suite} must have {expected_rows} rows and {expected_outliers} outliers; "
                f"found {len(y)} rows and {int(np.count_nonzero(y == 1))} outliers"
            )
        if not np.isfinite(x).all():
            raise ValueError("ODDS artifact contains missing or non-finite feature values")
        digest = self._sha(path)
        return PreparedTaskData(
            dataset_id=f"odds-{ref.suite}", name=f"ODDS {ref.suite}",
            target_name="outlier_label", X=x, y=y.astype(np.int64),
            feature_schema=["numeric"] * x.shape[1],
            content_sha256=hashlib.sha256(f"{digest}|source={info['source_id']}".encode()).hexdigest(),
            dataset_metadata={
                "source_url": info["odds_page"], "artifact_url": info["mat_url"],
                "artifact_sha256": digest, "artifact_version": "ODDS public MAT file, content-hash pinned per run",
                "source_dataset_url": info["source_url"], "source_dataset_id": info["source_id"],
                "source_dataset_title": info["source_title"],
                "license": "ODDS-derived MAT artifact has no separate license stated on its ODDS page; original-source terms must be followed",
                "attribution": info["citation"], "odds_transformation": info["transformation"],
                "source_count_note": info.get("published_count_note"),
                "access_policy_result": "public direct file download; no account or click-through agreement",
                "rows": int(len(x)), "features": int(x.shape[1]),
                "outlier_count": int(np.count_nonzero(y == 1)),
            },
            task_kind="classification", task_adapter=SupervisedAnomalyTask(),
            task_family="supervised_anomaly", task_variant=ref.suite,
            split_protocol="seeded_stratified_60_20_20",
        )

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings: Any) -> list[SplitSpec]:
        train, validation, test = stratified_split(
            np.asarray(data.y), seed=int(getattr(settings, "seed", self.split_seed)),
        )
        train_tuple, valid_tuple, test_tuple = tuple(train.tolist()), tuple(validation.tolist()), tuple(test.tolist())
        return [SplitSpec(0, 0, 0, train_tuple, valid_tuple, test_tuple,
                          _sha_indices(train_tuple, valid_tuple, test_tuple))]
