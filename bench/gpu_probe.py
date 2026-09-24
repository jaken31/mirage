"""Does the GPU clock up under sustained load, and how fast is it then?

Written to settle a planning question: the speed estimates assumed 448 GB/s of
memory bandwidth, and the first attempt measured 66-77 GB/s at 6.16 W, which
was a low-power state and so proved nothing either way.

Two phases, because the two loads raise different clocks:

  compute  fp16 matmul. The compute cores (SMs) speed up and the driver
           *lowers* the memory clock, which compute-bound work does not need.
           The reported power state (pstate) follows the memory clock, so it
           reads P4 here. That is correct, not a failure; judge this phase on
           SM clock and power draw instead.
  memory   large copies. The memory clock goes to max and pstate reads P0.
           Only this phase gives a valid bandwidth number.

Each phase passes or fails on its own. "P0 the whole time" cannot work as a
test, because no single load raises both clocks at once.

The compute verdict reads only samples taken after the first matmul has
finished. The sampler starts before the load, so its first sample is the idle
clock (180 MHz on a quiet GPU), and counting it made clocks look like they
rose, which failed the decay clause on a GPU that was holding steady.

    python bench/gpu_probe.py
    python bench/gpu_probe.py --self-check      # no GPU needed
"""
import argparse
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

# nvidia-smi reports half the memory's data rate, so bandwidth = clock x 2 x bus bytes.
# 128-bit bus = 16 B, from the RTX 5060 Laptop spec (cannot be measured here).
BUS_BYTES = 16
SUSTAIN_S, MEASURE_AT_S, MEM_WARM_S = 35.0, 20.0, 10.0
K = 8192
NBYTES = 1024 << 20

FIELDS = ["pstate", "clocks.current.sm", "clocks.max.sm", "clocks.current.memory",
          "clocks.max.memory", "power.draw", "enforced.power.limit", "temperature.gpu",
          "utilization.gpu", "display_active"]


def sample():
    """One nvidia-smi reading, stamped with `t` (perf_counter) just before the query."""
    t = time.perf_counter()
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    p, sm, smx, mem, memx, pw, lim, tmp, util, disp = [x.strip() for x in out.split(",")]
    return dict(t=t, pstate=p, sm=int(sm), sm_max=int(smx), mem=int(mem), mem_max=int(memx),
                power=float(pw), limit=float(lim), temp=float(tmp), util=util, display=disp)


def nvidia_monitors():
    """Linux: each connected monitor wired to an NVIDIA GPU, with its power state.

    An awake one means the compositor shares the GPU with the load: the
    2026-09-24 recheck measured about 16% less fp16 throughput with the external
    monitor on. nvidia-smi's `display_active` is no substitute: its own help says
    it can read Enabled with no monitor attached. None where there is no
    /sys/class/drm.
    """
    drm = Path("/sys/class/drm")
    if not drm.exists():
        return None
    out = []
    for c in sorted(drm.glob("card*-*")):
        card, name = c.name.split("-", 1)
        try:
            if ((drm / card / "device" / "vendor").read_text().strip() == "0x10de"
                    and (c / "status").read_text().strip() == "connected"):
                out.append(f"{name} {(c / 'dpms').read_text().strip()}")
        except OSError:
            continue
    return out


def timed(fn, iters=30):
    """Median seconds per call, timed on the CUDA stream, not the host clock."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        t.append(s.elapsed_time(e) / 1e3)
    return float(np.median(t))


class Sampler:
    def __init__(self):
        self.rows, self._stop = [], threading.Event()

    def __enter__(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self.rows.append(sample())
            except Exception as e:
                print(f"  sample failed: {e}")
            self._stop.wait(1.0)

    def __exit__(self, *_):
        self._stop.set()
        time.sleep(1.2)


def clock_decay(sm):
    """Percent drop from the first third's mean SM clock to the last third's."""
    third = len(sm) // 3
    return (sm[:third].mean() - sm[-third:].mean()) / sm[:third].mean() * 100


def compute_verdict(rows, t_load):
    """Judge the compute phase on the samples stamped at or after `t_load` only.

    Passes when the peak SM clock reaches 80% of max, the clock holds within 5%
    from the first third of the load to the last, and peak power reaches 80% of
    the enforced limit. The limit is read under load too: on Linux it sits at
    85 W idle and rises to 100 W once the load starts.
    """
    load = [r for r in rows if r["t"] >= t_load]
    assert len(load) >= 3, f"only {len(load)} samples during the load, need 3 for the decay"
    sm = np.array([r["sm"] for r in load])
    pw = np.array([r["power"] for r in load])
    tmp = np.array([r["temp"] for r in load])
    sm_max, limit = load[0]["sm_max"], max(r["limit"] for r in load)
    decay = clock_decay(sm)
    ok = bool(sm.max() >= 0.80 * sm_max and abs(decay) < 5 and pw.max() >= 0.80 * limit)
    return dict(n=len(load), dropped=len(rows) - len(load), sm=sm, pw=pw, tmp=tmp,
                sm_max=sm_max, limit=limit, decay=decay, ok=ok)


def main():
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda")
    print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}"
          f"  |  capability {torch.cuda.get_device_capability(0)}"
          f"  |  {torch.cuda.get_device_properties(0).multi_processor_count} SMs")
    s0 = sample()
    print(f"idle: {s0['pstate']} {s0['sm']} MHz sm, {s0['mem']} MHz mem, "
          f"{s0['power']:.1f} W of {s0['limit']:.0f} W, {s0['temp']:.0f} C, {s0['util']}% util")
    mons = nvidia_monitors()
    print(f"display: NVIDIA-driven monitors "
          f"{'unknown (no /sys/class/drm)' if mons is None else ', '.join(mons) or 'none connected'}"
          f"; nvidia-smi display_active {s0['display']}\n")

    # --- phase 1: compute ---------------------------------------------------
    a = torch.randn(K, K, device=dev, dtype=torch.float16)
    b = torch.randn(K, K, device=dev, dtype=torch.float16)
    c = torch.empty_like(a)
    flop = 2.0 * K ** 3

    print(f"compute phase: fp16 matmul {K}^3 for {SUSTAIN_S:.0f} s ...")
    with Sampler() as smp:
        t0 = time.perf_counter()
        t_load, matmul_s = None, None
        while time.perf_counter() - t0 < SUSTAIN_S:
            # Sync every iteration: otherwise the CPU queues work faster than the GPU
            # finishes it and builds up minutes of backlog.
            torch.matmul(a, b, out=c)
            torch.cuda.synchronize()
            if t_load is None:
                t_load = time.perf_counter()
            if matmul_s is None and time.perf_counter() - t0 >= MEASURE_AT_S:
                at = sample()
                matmul_s = timed(lambda: torch.matmul(a, b, out=c))

    v = compute_verdict(smp.rows, t_load)
    sm, pw, tmp = v["sm"], v["pw"], v["tmp"]
    tflops = flop / matmul_s / 1e12
    print(f"  {v['n']} samples under load ({v['dropped']} before it dropped)")
    print(f"  sm {sm.min()}-{sm.max()} MHz of {v['sm_max']} ({sm.max()/v['sm_max']*100:.0f}% peak), "
          f"decay {v['decay']:+.1f}%")
    print(f"  power {pw.min():.0f}-{pw.max():.0f} W of {v['limit']:.0f}   "
          f"temp {tmp.min():.0f}-{tmp.max():.0f} C")
    print(f"  fp16 matmul {matmul_s*1e3:.1f} ms -> {tflops:.1f} TFLOP/s   (measured at {at['pstate']}, "
          f"{at['sm']} MHz sm / {at['mem']} MHz mem)")
    compute_ok = v["ok"]
    print(f"  clocked up and holding: {compute_ok}\n")

    del a, b, c
    torch.cuda.empty_cache()

    # --- phase 2: memory ----------------------------------------------------
    n = NBYTES // 2
    src = torch.empty(n, device=dev, dtype=torch.float16).fill_(1.0)
    dst = torch.empty_like(src)

    print(f"memory phase: {NBYTES >> 20} MB transfers, {MEM_WARM_S:.0f} s warm ...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < MEM_WARM_S:
        dst.copy_(src)
    torch.cuda.synchronize()

    at = sample()
    res = {"copy": (timed(lambda: dst.copy_(src)), 2), "read": (timed(lambda: src.sum()), 1),
           "write": (timed(lambda: dst.fill_(2.0)), 1)}
    peak = at["mem_max"] * 2 * BUS_BYTES / 1e3       # MHz -> GB/s
    print(f"  at {at['pstate']}, {at['mem']} MHz mem of {at['mem_max']} max, "
          f"{at['power']:.0f} W, {at['temp']:.0f} C")
    for k, (s, traffic) in res.items():
        gbps = traffic * NBYTES / s / 1e9
        print(f"  {k:6s} {gbps:6.1f} GB/s   {gbps/peak*100:3.0f}% of {peak:.0f} theoretical"
              f"   {gbps/448*100:3.0f}% of the assumed 448")
    mem_ok = at["mem"] == at["mem_max"]
    print(f"  memory at max clock during measurement: {mem_ok}")

    print(f"\ntheoretical peak from clocks.max.memory: {at['mem_max']} MHz x 2 x {BUS_BYTES} B"
          f" = {peak:.0f} GB/s, NOT the 448 the fork table assumes")
    print(f"VERDICT  compute {'PASS' if compute_ok else 'FAIL'}   "
          f"memory {'PASS' if mem_ok else 'FAIL'}")


def _self_check():
    """No GPU: an idle sample before the load cannot fail the verdict, a real drop still does."""
    def rows(sm, t_first=1.0, idle=None):
        # 1 Hz samples at 99 W of a 100 W limit; an optional idle sample at t=0.
        out = [dict(t=t_first + i, sm=s, sm_max=3090, power=99.0, limit=100.0, temp=70.0)
               for i, s in enumerate(sm)]
        if idle is not None:
            out.insert(0, dict(t=0.0, sm=idle, sm_max=3090, power=3.3, limit=85.0, temp=56.0))
        return out

    rng = np.random.default_rng(0)
    steady = (2480 + rng.integers(-60, 60, 34)).tolist()   # peak ~2540 = 82% of 3090

    # Run 1 of the 2026-09-24 recheck: an idle 180 MHz sample, then a steady load.
    r = rows(steady, idle=180)
    assert abs(clock_decay(np.array([x["sm"] for x in r]))) >= 5, "case does not reproduce the bug"
    v = compute_verdict(r, t_load=0.5)
    assert v["ok"] and v["dropped"] == 1 and abs(v["decay"]) < 1, v
    assert v["limit"] == 100.0, "limit read from the idle sample, not the load"

    # A sample stamped exactly at t_load was taken with the load running, so it counts.
    assert compute_verdict(rows(steady), t_load=1.0)["dropped"] == 0

    # A genuine mid-load drop, 2500 -> 2200 MHz, still fails, idle sample or not.
    drop = [2500] * 17 + [2200] * 17
    for idle in (None, 180):
        v = compute_verdict(rows(drop, idle=idle), t_load=0.5)
        assert not v["ok"] and v["decay"] >= 5, v

    # Too few load samples is an error, not a verdict.
    try:
        compute_verdict(rows([2500, 2500], idle=180), t_load=0.5)
    except AssertionError:
        pass
    else:
        raise AssertionError("two load samples gave a verdict")
    print("gpu_probe self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-check", action="store_true", help="check the verdict logic, no GPU")
    if ap.parse_args().self_check:
        _self_check()
    else:
        main()
