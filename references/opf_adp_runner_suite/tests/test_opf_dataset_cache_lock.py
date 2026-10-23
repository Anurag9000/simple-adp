import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path

from utils.opf_dataset_cache import _case_cache_lock, _global_sample_index, prepare_case_cache


def _lock_worker(root: str, events: str, ready) -> None:
    event_path = Path(events)
    ready.wait(10)
    with _case_cache_lock(Path(root), "dataset_release_1", "pglib_opf_case14_ieee"):
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write("enter\n")
        time.sleep(0.15)
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write("exit\n")


def _prepare_worker(root: str, ready) -> None:
    ready.wait(10)
    prepare_case_cache(
        root=Path(root),
        case_name="pglib_opf_case14_ieee",
        num_groups=0,
        chunk_size=1,
    )


class DatasetCacheLockTests(unittest.TestCase):
    def test_group_sample_names_are_mapped_to_global_indices(self) -> None:
        self.assertEqual(_global_sample_index(0, 0), 0)
        self.assertEqual(_global_sample_index(1, 0), 15000)
        self.assertEqual(_global_sample_index(2, 14999), 44999)
        self.assertEqual(_global_sample_index(1, 15000), 15000)
        self.assertEqual(_global_sample_index(2, 29999), 29999)

    def test_serializes_concurrent_case_cache_workers(self) -> None:
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("cross-process lock test requires a POSIX fork context")
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory(prefix="opf_cache_lock_test_") as temp_dir:
            root = Path(temp_dir)
            events = root / "events.log"
            ready = context.Event()
            workers = [
                context.Process(target=_lock_worker, args=(str(root), str(events), ready))
                for _ in range(3)
            ]
            for worker in workers:
                worker.start()
            ready.set()
            for worker in workers:
                worker.join(10)
                self.assertEqual(worker.exitcode, 0)

            self.assertEqual(events.read_text(encoding="utf-8").splitlines(), [
                "enter", "exit", "enter", "exit", "enter", "exit",
            ])

    def test_public_prepare_is_safe_for_concurrent_first_use(self) -> None:
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("cross-process lock test requires a POSIX fork context")
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory(prefix="opf_cache_prepare_test_") as temp_dir:
            root = Path(temp_dir)
            ready = context.Event()
            workers = [
                context.Process(target=_prepare_worker, args=(str(root), ready))
                for _ in range(3)
            ]
            for worker in workers:
                worker.start()
            ready.set()
            for worker in workers:
                worker.join(10)
                self.assertEqual(worker.exitcode, 0)

            manifest = root / "dataset_release_1" / "pglib_opf_case14_ieee" / "processed_0" / "manifest.json"
            self.assertTrue(manifest.exists())


if __name__ == "__main__":
    unittest.main()
