import os
import sys
import time
import argparse
import subprocess
import signal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psutil

from DNN.VANILLA.Dyn_DNN4OPF.utils.run_defaults import PRESSURE_WRAPPER_DEFAULTS

_ACTIVE_CHILD_PROCESS: subprocess.Popen | None = None


def _external_drop_reopens_gates(
    *,
    host_ready: bool,
    gpu_ready: bool,
    host_drop_mib: float,
    gpu_drop_mib: float,
    threshold_mib: float,
) -> bool:
    """Require each pressure domain to be safe before resuming the tree."""
    drop_enabled = threshold_mib > 0.0
    host_gate_ready = host_ready or (drop_enabled and host_drop_mib >= threshold_mib)
    gpu_gate_ready = gpu_ready or (drop_enabled and gpu_drop_mib >= threshold_mib)
    return host_gate_ready and gpu_gate_ready


def _terminate_active_child_tree(timeout_sec: float = 10.0) -> None:
    global _ACTIVE_CHILD_PROCESS
    proc = _ACTIVE_CHILD_PROCESS
    if proc is None:
        return
    try:
        parent = psutil.Process(proc.pid)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
        _ACTIVE_CHILD_PROCESS = None
        return

    children = parent.children(recursive=True)
    processes = [parent, *children]
    for child in reversed(children):
        try:
            child.terminate()
        except Exception:
            pass
    try:
        parent.terminate()
    except Exception:
        pass

    _, alive = psutil.wait_procs(processes, timeout=max(0.1, float(timeout_sec)))
    for child in alive:
        try:
            child.kill()
        except Exception:
            pass
    _, still_alive = psutil.wait_procs(alive, timeout=max(0.1, float(timeout_sec)))
    for child in still_alive:
        try:
            child.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=max(0.1, float(timeout_sec)))
    except Exception:
        pass
    _ACTIVE_CHILD_PROCESS = None


def _pause_active_child_tree() -> None:
    """Suspend the whole command group without losing its in-memory state."""
    proc = _ACTIVE_CHILD_PROCESS
    if proc is None or proc.poll() is not None:
        return
    if os.name != "nt":
        os.killpg(os.getpgid(proc.pid), signal.SIGSTOP)
        return
    parent = psutil.Process(proc.pid)
    for child in parent.children(recursive=True):
        child.suspend()
    parent.suspend()


def _resume_active_child_tree() -> None:
    """Resume a command previously paused by _pause_active_child_tree."""
    proc = _ACTIVE_CHILD_PROCESS
    if proc is None or proc.poll() is not None:
        return
    if os.name != "nt":
        os.killpg(os.getpgid(proc.pid), signal.SIGCONT)
        return
    parent = psutil.Process(proc.pid)
    parent.resume()
    for child in parent.children(recursive=True):
        child.resume()


def _install_signal_handlers() -> None:
    def _handle_shutdown(signum, frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"\n[INTERRUPT] Caught {signame}. Terminating active child tree...")
        _terminate_active_child_tree()
        sys.exit(130)

    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(_sig, _handle_shutdown)
        except Exception:
            pass

def sample_gpu_memory_pressure(device_index=0):
    total_mib = 0
    used_mib = 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        row = out.decode("utf-8").strip().splitlines()[0]
        total_text, used_text = [part.strip() for part in row.split(",", 1)]
        total_mib = int(float(total_text))
        used_mib = int(float(used_text))
    except Exception:
        pass
    if total_mib <= 0:
        return 0.0, 0
    return max(0.0, min(100.0, (float(used_mib) / float(total_mib)) * 100.0)), used_mib

def sample_python_host_memory_mib():
    total_py_mib = 0.0
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            name = p.info['name']
            if name and ('python' in name.lower() or 'python3' in name.lower()):
                total_py_mib += p.info['memory_info'].rss / 1048576.0
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, TypeError, KeyError):
            pass
    return total_py_mib

def sample_python_gpu_memory_mib(device_index=0):
    total_py_gpu_mib = 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        for line in out.decode("utf-8").strip().splitlines():
            parts = line.split(',')
            if len(parts) >= 2:
                pname = parts[0].strip()
                mem_str = parts[1].strip()
                if mem_str.isdigit() and ('python' in pname.lower() or 'python3' in pname.lower()):
                    total_py_gpu_mib += float(mem_str)
    except Exception:
        pass
    return total_py_gpu_mib

def main():
    _install_signal_handlers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-ram-pressure-limit-pct", type=float, default=float(PRESSURE_WRAPPER_DEFAULTS["host_ram_pressure_limit_pct"]))
    parser.add_argument("--host-ram-resume-pct", type=float, default=float(PRESSURE_WRAPPER_DEFAULTS["host_ram_resume_pct"]))
    parser.add_argument("--swap-pressure-limit-pct", type=float, default=float(PRESSURE_WRAPPER_DEFAULTS["swap_pressure_limit_pct"]), help="Legacy compatibility option; swap occupancy does not gate this wrapper.")
    parser.add_argument("--swap-resume-pct", type=float, default=float(PRESSURE_WRAPPER_DEFAULTS["swap_resume_pct"]), help="Legacy compatibility option; swap occupancy does not gate this wrapper.")
    parser.add_argument("--gpu-memory-pressure-limit-pct", type=float, default=float(PRESSURE_WRAPPER_DEFAULTS["gpu_memory_pressure_limit_pct"]))
    parser.add_argument("--gpu-memory-resume-pct", type=float, default=float(PRESSURE_WRAPPER_DEFAULTS["gpu_memory_resume_pct"]))
    parser.add_argument("--external-memory-drop-reopen-mib", type=float, default=float(PRESSURE_WRAPPER_DEFAULTS["external_memory_drop_reopen_mib"]))
    parser.add_argument("--gpu-device-index", type=int, default=0)
    parser.add_argument("--pressure-poll-interval-sec", type=float, default=float(PRESSURE_WRAPPER_DEFAULTS["pressure_poll_interval_sec"]))
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.exit("No command provided")
    if not 0.0 <= float(args.host_ram_resume_pct) <= float(args.host_ram_pressure_limit_pct) <= 100.0:
        raise SystemExit("host RAM resume/pressure percentages must satisfy 0 <= resume <= pressure <= 100")
    if not 0.0 <= float(args.gpu_memory_resume_pct) <= float(args.gpu_memory_pressure_limit_pct) <= 100.0:
        raise SystemExit("GPU resume/pressure percentages must satisfy 0 <= resume <= pressure <= 100")
    if float(args.external_memory_drop_reopen_mib) <= 0.0:
        raise SystemExit("--external-memory-drop-reopen-mib must be > 0")
    if float(args.pressure_poll_interval_sec) <= 0.0:
        raise SystemExit("--pressure-poll-interval-sec must be > 0")

    print(f"[PRESSURE AWARE] Launching: {' '.join(cmd)}")
    popen_kwargs = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    global _ACTIVE_CHILD_PROCESS
    paused = False
    waiting_for_resume = False
    peak_paused_host_mib = 0.0
    peak_paused_gpu_mib = 0.0
    try:
        while True:
            if waiting_for_resume:
                mem = psutil.virtual_memory()
                host_used_pct = mem.percent
                current_host_mib = mem.used / 1048576.0

                swap = psutil.swap_memory()
                swap_used_pct = swap.percent

                gpu_used_pct, current_gpu_mib = sample_gpu_memory_pressure(args.gpu_device_index)

                host_resume = host_used_pct <= args.host_ram_resume_pct
                gpu_resume = gpu_used_pct <= args.gpu_memory_resume_pct

                effective_host_mib = current_host_mib - sample_python_host_memory_mib()
                effective_gpu_mib = current_gpu_mib - sample_python_gpu_memory_mib(args.gpu_device_index)

                peak_paused_host_mib = max(peak_paused_host_mib, effective_host_mib)
                peak_paused_gpu_mib = max(peak_paused_gpu_mib, effective_gpu_mib)

                host_drop = peak_paused_host_mib - effective_host_mib
                gpu_drop = peak_paused_gpu_mib - effective_gpu_mib

                if _external_drop_reopens_gates(
                    host_ready=host_resume,
                    gpu_ready=gpu_resume,
                    host_drop_mib=host_drop,
                    gpu_drop_mib=gpu_drop,
                    threshold_mib=float(args.external_memory_drop_reopen_mib),
                ):
                    print(
                        f"\n[STATE] admission gate reopened by RAM/GPU drop or thresholds met "
                        f"host_drop={host_drop:.1f} gpu_drop={gpu_drop:.1f} "
                        f"host={host_used_pct:.2f} swap={swap_used_pct:.2f} gpu={gpu_used_pct:.2f}"
                    )
                    _resume_active_child_tree()
                    waiting_for_resume = False
                    paused = False
                    peak_paused_host_mib = 0.0
                    peak_paused_gpu_mib = 0.0
                    continue

                time.sleep(max(0.1, args.pressure_poll_interval_sec))
                continue

            proc = _ACTIVE_CHILD_PROCESS
            if proc is None or proc.poll() is not None:
                proc = subprocess.Popen(cmd, **popen_kwargs)
                _ACTIVE_CHILD_PROCESS = proc
            try:
                psutil.Process(proc.pid)
            except psutil.NoSuchProcess:
                sys.exit(proc.wait())

            while proc.poll() is None:
                mem = psutil.virtual_memory()
                host_used_pct = mem.percent
                current_host_mib = mem.used / 1048576.0

                swap = psutil.swap_memory()
                swap_used_pct = swap.percent

                gpu_used_pct, current_gpu_mib = sample_gpu_memory_pressure(args.gpu_device_index)

                host_pressure = host_used_pct > args.host_ram_pressure_limit_pct
                gpu_pressure = gpu_used_pct > args.gpu_memory_pressure_limit_pct

                if not paused and (host_pressure or gpu_pressure):
                    reason = []
                    if host_pressure:
                        reason.append(f"host={host_used_pct:.2f}%>{args.host_ram_pressure_limit_pct}")
                    if gpu_pressure:
                        reason.append(f"gpu={gpu_used_pct:.2f}%>{args.gpu_memory_pressure_limit_pct}")
                    print(f"\n[PRESSURE] Pause requested. {' '.join(reason)}")
                    try:
                        _pause_active_child_tree()
                        paused = True
                        waiting_for_resume = True
                        peak_paused_host_mib = current_host_mib - sample_python_host_memory_mib()
                        peak_paused_gpu_mib = current_gpu_mib - sample_python_gpu_memory_mib(args.gpu_device_index)
                    except Exception as e:
                        print(f"Failed to terminate for pause: {e}")
                    break

                time.sleep(max(0.1, args.pressure_poll_interval_sec))

            if proc.poll() is None:
                continue
            if waiting_for_resume:
                continue
            _ACTIVE_CHILD_PROCESS = None
            sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Caught KeyboardInterrupt. Terminating active child tree...")
        if waiting_for_resume:
            try:
                _resume_active_child_tree()
            except Exception:
                pass
        _terminate_active_child_tree()
        sys.exit(130)
    finally:
        _ACTIVE_CHILD_PROCESS = None

    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
