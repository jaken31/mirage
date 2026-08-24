"""Is the GPU clocked up under sustained load, and what does it deliver there?

The AGENDA blocker. Every compute floor in the fork table derives from an
assumed 448 GB/s, and the first attempt measured 66-77 GB/s at 6.16 W - a
low-power state, so it neither confirmed nor refuted the figure.

Two phases, because the two loads clock different domains:

  compute  fp16 matmul. SMs boost, and the driver *drops* the memory clock
           because compute-bound work does not need it. Reported pstate follows
           the memory domain, so it reads P4 here - that is correct behaviour,
           not a failure. Judge this phase on SM clock and power instead.
  memory   large copies. Memory clock goes to max and the pstate reads P0.
           This is the only phase in which a bandwidth number is valid.

Pass is per phase. "P0 for the whole window" is not a usable criterion: no
single load clocks both domains at once.
"""
import subprocess
import threading
import time

import numpy as np
import torch

assert torch.cuda.is_available(), "no CUDA device"
DEV = torch.device("cuda")

# nvidia-smi reports half the GDDR7 data rate, so bandwidth is clock x 2 x bytes.
# 128-bit bus = 16 B, per the RTX 5060 Laptop product spec (not measurable here).
BUS_BYTES = 16
SUSTAIN_S, MEASURE_AT_S, MEM_WARM_S = 35.0, 20.0, 10.0
K = 8192
NBYTES = 1024 << 20

FIELDS = ["pstate", "clocks.current.sm", "clocks.max.sm", "clocks.current.memory",
          "clocks.max.memory", "power.draw", "enforced.power.limit", "temperature.gpu"]


def sample():
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    p, sm, smx, mem, memx, pw, lim, tmp = [x.strip() for x in out.split(",")]
    return dict(pstate=p, sm=int(sm), sm_max=int(smx), mem=int(mem), mem_max=int(memx),
                power=float(pw), limit=float(lim), temp=float(tmp))


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

    def col(self, k):
        return np.array([r[k] for r in self.rows])


print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}"
      f"  |  capability {torch.cuda.get_device_capability(0)}"
      f"  |  {torch.cuda.get_device_properties(0).multi_processor_count} SMs")
s0 = sample()
print(f"idle: {s0['pstate']} {s0['sm']} MHz sm, {s0['mem']} MHz mem, "
      f"{s0['power']:.1f} W of {s0['limit']:.0f} W, {s0['temp']:.0f} C\n")

# --- phase 1: compute ---------------------------------------------------
a = torch.randn(K, K, device=DEV, dtype=torch.float16)
b = torch.randn(K, K, device=DEV, dtype=torch.float16)
c = torch.empty_like(a)
FLOP = 2.0 * K ** 3

print(f"compute phase: fp16 matmul {K}^3 for {SUSTAIN_S:.0f} s ...")
with Sampler() as smp:
    t0 = time.perf_counter()
    matmul_s = None
    while time.perf_counter() - t0 < SUSTAIN_S:
        # Sync each iteration: without it the host queues kernels faster than the
        # GPU drains them and banks minutes of backlog.
        torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        if matmul_s is None and time.perf_counter() - t0 >= MEASURE_AT_S:
            at = sample()
            matmul_s = timed(lambda: torch.matmul(a, b, out=c))

sm, pw, tmp = smp.col("sm"), smp.col("power"), smp.col("temp")
sm_max, limit = smp.rows[0]["sm_max"], smp.rows[0]["limit"]
third = len(sm) // 3
decay = (sm[:third].mean() - sm[-third:].mean()) / sm[:third].mean() * 100
tflops = FLOP / matmul_s / 1e12
print(f"  sm {sm.min()}-{sm.max()} MHz of {sm_max} ({sm.max()/sm_max*100:.0f}% peak), "
      f"decay {decay:+.1f}%")
print(f"  power {pw.min():.0f}-{pw.max():.0f} W of {limit:.0f}   temp {tmp.min():.0f}-{tmp.max():.0f} C")
print(f"  fp16 matmul {matmul_s*1e3:.1f} ms -> {tflops:.1f} TFLOP/s   (measured at {at['pstate']}, "
      f"{at['sm']} MHz sm / {at['mem']} MHz mem)")
compute_ok = sm.max() >= 0.80 * sm_max and abs(decay) < 5 and pw.max() >= 0.80 * limit
print(f"  clocked up and holding: {compute_ok}\n")

del a, b, c
torch.cuda.empty_cache()

# --- phase 2: memory ----------------------------------------------------
n = NBYTES // 2
src = torch.empty(n, device=DEV, dtype=torch.float16).fill_(1.0)
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
