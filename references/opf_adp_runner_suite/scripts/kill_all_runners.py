import os
from pathlib import Path

import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
OWNED_MARKERS = (
    "opf_suite_launcher.py",
    "opf_massive_suite_runner.py",
    "universal_main.py",
    "pressure_aware_wrapper.py",
)


def _is_owned_command(cmdline: list[str]) -> bool:
    """Match runner entrypoints, not source filenames passed to another tool."""
    for index, token in enumerate(cmdline):
        if Path(token).name not in OWNED_MARKERS:
            continue
        arguments = cmdline[index + 1 :]
        marker = Path(token).name
        if marker in {"opf_suite_launcher.py", "opf_massive_suite_runner.py"}:
            return "--mode" in arguments
        if marker == "universal_main.py":
            return "--model" in arguments
        if marker == "pressure_aware_wrapper.py":
            return "--" in arguments
    return False


def _is_owned_runner(proc: psutil.Process) -> bool:
    try:
        info = proc.as_dict(attrs=["pid", "cmdline", "cwd"])
        cmdline_parts = [str(part) for part in (info.get("cmdline") or [])]
        if not _is_owned_command(cmdline_parts):
            return False
        cwd = info.get("cwd")
        if cwd:
            try:
                Path(cwd).resolve().relative_to(REPO_ROOT)
                return True
            except ValueError:
                pass
        return str(REPO_ROOT) in " ".join(cmdline_parts)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return False


def _descendants(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return []

def main():
    print(f"Emergency Kill Switch: terminating OPF_ADP runner processes under {REPO_ROOT}...")
    my_pid = os.getpid()

    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            pid = proc.info['pid']
            if pid == my_pid or not _is_owned_runner(proc):
                continue
            print(f"Terminating owned runner PID {pid}...")
            targets = _descendants(proc) + [proc]
            seen = set()
            for target in reversed(targets):
                if target.pid in seen or target.pid == my_pid:
                    continue
                seen.add(target.pid)
                try:
                    target.kill()
                    killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    print(f"Finished. Successfully terminated {killed} owned runner process(es).")

if __name__ == "__main__":
    main()
