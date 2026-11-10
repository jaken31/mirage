"""Play the dynamics model from the keyboard, with no simulator.

The context is seeded from a real clip of the val split (cached tokens and
their recorded actions); every later frame is the model's own, generated one
step at a time by `dynamics_eval.rollout` from the last `ctx` frames and the
action of the keys held now. Frames decode through `Tokenizer.decode`
(`dynamics_eval.decode_u8`) and upscale nearest neighbour.

Nothing here names a checkpoint, a tokenizer or a dataset: all three come from
the command line, and the checkpoint and token cache are refused by their own
loaders unless their hashes match the config.

    python -m mirage.play --dynamics RUN_OR_PATH --tokenizer RUN_ID --config CONFIG
    python -m mirage.play ... --side-by-side --record demo.mp4   # real | model, the episode's actions
    python -m mirage.play ... --bench 300                        # headless timing table

Keys (play mode): A/D or Left/Right drive joint 0, W/S or Up/Down joint 1
(both of a pair held cancel to 0). R reseeds from a new val clip, Space pauses,
N steps once while paused, Q or Esc quits. In side-by-side mode the actions are
the episode's own and R moves to another episode.
"""

import argparse
import math
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mirage import config, dynamics, fsq_eval
from mirage.dynamics_eval import COMMAND, JOINTS, decode_u8, load_checkpoint, rollout

ROOT = Path(__file__).resolve().parent.parent
WARMUP = 10   # steps left out of every timing figure: CUDA kernel selection, allocator growth


def action_index(command: tuple[int, ...]) -> int:
    """A per-joint command in {-1, 0, +1} -> its action id, read back from `COMMAND`."""
    hit = np.flatnonzero((COMMAND == np.asarray(command)).all(1))
    assert len(hit) == 1, command
    return int(hit[0])


def resolve_checkpoint(arg: str, which: str) -> Path:
    """A checkpoint file, a run directory, or a run id under `runs/`."""
    p = Path(arg)
    if not p.exists():
        p = ROOT / "runs" / arg
    if p.is_dir():
        p = p / {"best": "best.pt", "final": "model.pt"}[which]
    if not p.is_file():
        raise SystemExit(f"no dynamics checkpoint at {p}")
    return p


# ---------------------------------------------------------------- the model

class Player:
    """The model, the tokenizer and the val episodes, stepped one frame at a time."""

    def __init__(self, args: argparse.Namespace):
        self.dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = config.load(args.config)
        self.ckpt_path = resolve_checkpoint(args.dynamics, args.checkpoint)
        self.model, ck = load_checkpoint(self.ckpt_path, self.cfg, self.dev)
        self.step_no = ck.get("step")
        self.tok, _ = fsq_eval.load_run(args.tokenizer, self.cfg, self.dev)
        self.k_round = getattr(self.model, "mask_id", None) is not None
        if args.rounds is not None and not self.k_round:
            raise SystemExit("--rounds is for a K-round checkpoint; this one decodes in one pass")
        self.rounds = args.rounds if self.k_round else None
        self.guidance = args.guidance
        self.ctx = args.ctx or self.cfg.data["ctx"]
        self.grid = tuple(self.cfg.shapes.token_grid)
        self.cells = math.prod(self.grid)

        val = dynamics.load_splits(self.cfg, args.tokenizer).val
        self.episodes = []   # (episode id, tokens (T, cells) int64, actions (T,) int64)
        for ep in val.windows.episodes:
            s = slice(ep.start, ep.start + ep.length)
            self.episodes.append((ep.episode_id, np.asarray(val.tokens[ep.shard][s], np.int64),
                                  np.asarray(val.actions[ep.shard][s], np.int64)))
        if not self.episodes:
            raise SystemExit("the val split has no episodes")
        self.rng = np.random.default_rng(args.seed)
        self.frames: deque = deque(maxlen=self.ctx)
        self.acts: deque = deque(maxlen=self.ctx + 1)
        self.episode = self.offset = None

    def describe(self) -> str:
        kind = f"K-round, {self.rounds or 1} round(s)" if self.k_round else "one-pass"
        return (f"{self.ckpt_path} (step {self.step_no}, {kind}) on {self.dev}; "
                f"ctx {self.ctx}, {len(self.episodes)} val episodes")

    def seed(self, episode: int | None = None, offset: int | None = None) -> np.ndarray:
        """Fill the context from a real val clip; returns its last frame's tokens."""
        if episode is None:
            episode = int(self.rng.integers(len(self.episodes)))
        _, tokens, actions = self.episodes[episode]
        if len(tokens) < self.ctx:
            raise SystemExit(f"episode {episode} has {len(tokens)} frames, under ctx {self.ctx}")
        if offset is None:
            offset = int(self.rng.integers(len(tokens) - self.ctx + 1))
        self.episode, self.offset = episode, offset
        self.frames.clear()
        self.acts.clear()
        self.frames.extend(tokens[offset:offset + self.ctx])
        self.acts.extend(actions[offset:offset + self.ctx])
        return self.frames[-1]

    def step(self, action: int) -> tuple[np.ndarray, float]:
        """One generated frame under `action`, and the seconds the model took.

        `rollout`'s own contract: the window's actions have `ctx + 1` columns,
        the last being the action that produces the predicted frame.
        """
        self.acts.append(action)
        seed = np.stack(self.frames)[None]
        acts = np.fromiter(self.acts, np.int64)[None]
        t0 = time.perf_counter()
        out = rollout(self.model, seed, acts, 1, self.dev, rounds=self.rounds,
                      guidance=self.guidance).frames[0, -1]
        dt = time.perf_counter() - t0   # rollout ends in .cpu(), so this waits for the GPU
        self.frames.append(out)
        return out, dt

    def decode(self, rows: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Token rows -> `(n, h, w, 3)` uint8 frames, and the seconds it took."""
        t0 = time.perf_counter()
        px = decode_u8(self.tok, np.stack(rows), self.grid, self.dev)
        return px, time.perf_counter() - t0


# ------------------------------------------------------------- the pictures

def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except TypeError:   # Pillow < 10.1 has one bitmap size
        return ImageFont.load_default()


def upscale(img: np.ndarray, scale: int) -> np.ndarray:
    """Nearest neighbour: every pixel becomes a `scale` x `scale` block."""
    return img.repeat(scale, 0).repeat(scale, 1)


def labelled(panels: np.ndarray, scale: int, labels: list[str] | None) -> np.ndarray:
    """`(n, h, w, 3)` panels -> one upscaled row, each panel labelled if labels are given."""
    body = upscale(np.concatenate(list(panels), 1), scale)
    if not labels:
        return body
    im = Image.fromarray(body)
    d = ImageDraw.Draw(im)
    f = _font(max(10, 3 * scale))
    w = panels.shape[2] * scale
    for i, text in enumerate(labels):
        d.text((i * w + 2 * scale, scale), text, fill=(255, 255, 255), font=f,
               stroke_width=max(1, scale // 4), stroke_fill=(0, 0, 0))
    return np.asarray(im)


def with_hud(body: np.ndarray, lines: list[str], size: int) -> np.ndarray:
    """The body with a text strip under it, so the HUD never covers a frame."""
    f = _font(size)
    strip = Image.new("RGB", (body.shape[1], (size + 4) * len(lines) + 6), (16, 16, 16))
    d = ImageDraw.Draw(strip)
    for i, text in enumerate(lines):
        d.text((6, 3 + i * (size + 4)), text, fill=(220, 220, 220), font=f)
    return np.concatenate((body, np.asarray(strip)), 0)


class Timings:
    """Rolling per-stage seconds; the HUD reads these."""

    def __init__(self, window: int = 300):
        self.s = {k: deque(maxlen=window) for k in ("model", "decode", "display", "frame")}
        self.ticks: deque = deque(maxlen=window)

    def add(self, **dt: float) -> None:
        for k, v in dt.items():
            self.s[k].append(v)
        self.ticks.append(time.perf_counter())

    def ms(self, key: str, q: float) -> float:
        v = self.s[key]
        return 1e3 * float(np.percentile(v, q)) if v else float("nan")

    def fps(self) -> float:
        t = self.ticks
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else float("nan")

    def hud(self) -> list[str]:
        return [f"{self.fps():5.1f} fps   frame p50 {self.ms('frame', 50):6.1f} ms  "
                f"p99 {self.ms('frame', 99):6.1f} ms",
                f"model {self.ms('model', 50):6.1f} ms   decode {self.ms('decode', 50):5.1f} ms   "
                f"display {self.ms('display', 50):5.1f} ms   (p50)"]


class Recorder:
    """`--record PATH`: .gif through Pillow, .mp4 through the system ffmpeg.

    GIF frames are kept at native size and upscaled when written, so a long
    recording holds 12 KB a panel in memory rather than its upscaled size.
    """

    def __init__(self, path: str, fps: float, scale: int, labels: list[str] | None):
        self.path, self.fps, self.scale, self.labels = Path(path), fps, scale, labels
        self.kind = self.path.suffix.lower()
        if self.kind not in (".gif", ".mp4"):
            raise SystemExit(f"--record {path}: write .gif or .mp4")
        if self.kind == ".mp4" and not shutil.which("ffmpeg"):
            raise SystemExit("--record .mp4 needs ffmpeg on PATH; record a .gif instead")
        self.frames: list[np.ndarray] = []
        self.proc = None
        self.count = 0

    def add(self, panels: np.ndarray) -> None:
        self.count += 1
        if self.kind == ".gif":
            self.frames.append(panels.copy())
            return
        img = labelled(panels, self.scale, self.labels)
        if self.proc is None:
            h, w = img.shape[:2]
            self.proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "16", str(self.path)], stdin=subprocess.PIPE)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self) -> None:
        if self.count == 0:
            print(f"--record {self.path}: no frames generated, nothing written")
            return
        if self.kind == ".gif":
            ims = [Image.fromarray(labelled(p, self.scale, self.labels)) for p in self.frames]
            # GIF counts in 10 ms units, so 33 ms would play 30 fps 10% fast. Round
            # each frame's end time instead: durations alternate 30/40 ms and the
            # clip's length stays the true one. 20 ms is the shortest viewers honour.
            ends = np.round(np.arange(1, len(ims) + 1) * max(20.0, 1000 / self.fps), -1)
            ims[0].save(self.path, save_all=True, append_images=ims[1:], loop=0,
                        duration=np.diff(ends, prepend=0).astype(int).tolist())
        else:
            self.proc.stdin.close()
            if self.proc.wait():
                raise SystemExit(f"ffmpeg exited {self.proc.returncode} writing {self.path}")
        print(f"wrote {self.path}: {self.count} frames")


# --------------------------------------------------------------- the window

class Window:
    """A GLFW window showing one RGB image, nearest-filtered, with key state.

    A key pressed between two polls counts as held for the next frame even if
    it was released already, so a tap always drives at least one step.
    """

    def __init__(self, title: str, w: int, h: int):
        import glfw
        from OpenGL import GL
        self.glfw, self.GL = glfw, GL
        if not glfw.init():
            raise SystemExit("GLFW could not initialise: no display?")
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        self.win = glfw.create_window(w, h, title, None, None)
        if not self.win:
            glfw.terminate()
            raise SystemExit("GLFW could not open a window")
        glfw.make_context_current(self.win)
        glfw.swap_interval(0)   # the --fps cap paces frames, not vsync
        self.held: set[int] = set()
        self.tapped: set[int] = set()
        self.pressed: list[int] = []
        glfw.set_key_callback(self.win, self._key)
        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        for p in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, GL.GL_NEAREST)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glEnable(GL.GL_TEXTURE_2D)
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode()

    def _key(self, win, key, scancode, action, mods):
        g = self.glfw
        if action == g.PRESS:
            self.held.add(key)
            self.tapped.add(key)
            self.pressed.append(key)
        elif action == g.RELEASE:
            self.held.discard(key)

    def poll(self) -> tuple[set[int], list[int]]:
        """Keys down this frame (held or tapped since the last poll), and new presses."""
        self.glfw.poll_events()
        down = self.held | self.tapped
        pressed, self.tapped, self.pressed = self.pressed, set(), []
        return down, pressed

    def closed(self) -> bool:
        return bool(self.glfw.window_should_close(self.win))

    def show(self, img: np.ndarray) -> None:
        GL, glfw = self.GL, self.glfw
        h, w = img.shape[:2]
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB,
                        GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(img))
        fw, fh = glfw.get_framebuffer_size(self.win)
        GL.glViewport(0, 0, fw, fh)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        # Letterbox to the image's aspect so pixels stay square when resized.
        sx = min(1.0, (fw / fh) / (w / h)) if fh else 1.0
        sy = min(1.0, (w / h) / (fw / fh)) if fh else 1.0
        GL.glBegin(GL.GL_QUADS)
        for (u, v), (x, y) in zip(((0, 1), (1, 1), (1, 0), (0, 0)),
                                  ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))):
            GL.glTexCoord2f(u, v)
            GL.glVertex2f(x, y)
        GL.glEnd()
        glfw.swap_buffers(self.win)

    def close(self) -> None:
        self.glfw.terminate()


def _keys():
    import glfw
    return {"j0-": {glfw.KEY_A, glfw.KEY_LEFT}, "j0+": {glfw.KEY_D, glfw.KEY_RIGHT},
            "j1-": {glfw.KEY_S, glfw.KEY_DOWN}, "j1+": {glfw.KEY_W, glfw.KEY_UP},
            "reset": {glfw.KEY_R}, "pause": {glfw.KEY_SPACE}, "step": {glfw.KEY_N},
            "quit": {glfw.KEY_Q, glfw.KEY_ESCAPE}}


def command_from(down: set[int], keys: dict) -> tuple[int, ...]:
    """Held keys -> a per-joint command; opposite keys of one joint cancel."""
    return tuple(int(bool(down & keys[f"j{j}+"])) - int(bool(down & keys[f"j{j}-"]))
                 for j in range(JOINTS))


def _pace(t_start: float, fps: float) -> None:
    left = t_start + 1.0 / fps - time.perf_counter()
    if left > 0:
        time.sleep(left)


# ------------------------------------------------------------------- modes

def play(p: Player, args: argparse.Namespace) -> None:
    keys = _keys()
    last = p.seed()
    px, _ = p.decode([last])
    hud_size = max(12, 2 * args.scale)
    first = with_hud(labelled(px, args.scale, None), ["", "", ""], hud_size)
    win = Window("mirage play", first.shape[1], first.shape[0])
    print(f"window on {win.renderer}")
    rec = Recorder(args.record, args.fps, args.scale, None) if args.record else None
    tm, paused, frames, prev_action = Timings(), False, 0, None
    try:
        while not win.closed():
            t0 = time.perf_counter()
            down, pressed = win.poll()
            if any(k in keys["quit"] for k in pressed):
                break
            if any(k in keys["pause"] for k in pressed):
                paused = not paused
            if any(k in keys["reset"] for k in pressed):
                last = p.seed()
                px, _ = p.decode([last])
                print(f"reseeded: episode {p.episodes[p.episode][0]} offset {p.offset}")
            cmd = command_from(down, keys)
            action = action_index(cmd)
            if action != prev_action and args.verbose:
                print(f"frame {frames}: action {action} command {cmd}", flush=True)
            prev_action = action
            stepping = not paused or any(k in keys["step"] for k in pressed)
            dm = dd = 0.0
            if stepping:
                last, dm = p.step(action)
                px, dd = p.decode([last])
                frames += 1
                if rec:
                    rec.add(px)
            t1 = time.perf_counter()
            state = "PAUSED" if paused else f"frame {frames}"
            lines = tm.hud() + [f"action {action} cmd j0 {cmd[0]:+d} j1 {cmd[1]:+d}   "
                                f"ep {p.episodes[p.episode][0]}   {state}"
                                + ("   REC" if rec else "")]
            win.show(with_hud(labelled(px, args.scale, None), lines, hud_size))
            dt_disp = time.perf_counter() - t1
            if stepping and frames > WARMUP:
                tm.add(model=dm, decode=dd, display=dt_disp, frame=dm + dd + dt_disp)
            if args.max_frames and frames >= args.max_frames:
                break
            _pace(t0, args.fps)
    finally:
        win.close()
        if rec:
            rec.close()
    print(*tm.hud(), sep="\n")


def side_by_side(p: Player, args: argparse.Namespace) -> None:
    """Left the real frames, right the model's rollout under the episode's own actions."""
    keys = _keys()
    episode = args.episode if args.episode is not None else int(p.rng.integers(len(p.episodes)))
    labels = ["real", "model"]

    def start(e: int):
        p.seed(e, 0)
        _, tokens, actions = p.episodes[e]
        print(f"episode {p.episodes[e][0]}: {len(tokens)} frames, seeded on 0..{p.ctx - 1}")
        return tokens, actions, p.ctx

    tokens, actions, t = start(episode)
    px, _ = p.decode([tokens[t - 1], tokens[t - 1]])
    hud_size = max(12, 2 * args.scale)
    first = with_hud(labelled(px, args.scale, labels), ["", "", ""], hud_size)
    win = None if args.headless else Window("mirage side-by-side", first.shape[1], first.shape[0])
    if win:
        print(f"window on {win.renderer}")
    rec = Recorder(args.record, args.fps, args.scale, labels) if args.record else None
    tm, paused, frames = Timings(), False, 0
    try:
        while win is None or not win.closed():
            t0 = time.perf_counter()
            pressed = win.poll()[1] if win else []
            if any(k in keys["quit"] for k in pressed):
                break
            if any(k in keys["pause"] for k in pressed):
                paused = not paused
            if any(k in keys["reset"] for k in pressed) or t >= len(tokens):
                if t >= len(tokens) and (rec or win is None):
                    break   # one episode per recording
                episode = (episode + 1) % len(p.episodes) if t >= len(tokens) \
                    else int(p.rng.integers(len(p.episodes)))
                tokens, actions, t = start(episode)
            stepping = not paused or any(k in keys["step"] for k in pressed)
            dm = dd = 0.0
            if stepping:
                gen, dm = p.step(int(actions[t]))
                px, dd = p.decode([tokens[t], gen])
                if rec:
                    rec.add(px)
                t += 1
                frames += 1
            t1 = time.perf_counter()
            if win:
                cmd = COMMAND[actions[t - 1]]
                lines = tm.hud() + [f"action {actions[t - 1]} cmd j0 {cmd[0]:+d} j1 {cmd[1]:+d}   "
                                    f"ep {p.episodes[episode][0]} frame {t - 1}/{len(tokens) - 1}"
                                    + ("   PAUSED" if paused else "") + ("   REC" if rec else "")]
                win.show(with_hud(labelled(px, args.scale, labels), lines, hud_size))
            dt_disp = time.perf_counter() - t1
            if stepping and frames > WARMUP:
                tm.add(model=dm, decode=dd, display=dt_disp, frame=dm + dd + dt_disp)
            if args.max_frames and frames >= args.max_frames:
                break
            if win:
                _pace(t0, args.fps)
    finally:
        if win:
            win.close()
        if rec:
            rec.close()
    print(*tm.hud(), sep="\n")


def bench(p: Player, n: int) -> None:
    """Headless: `n` timed steps after `WARMUP`, actions cycling through all nine."""
    p.seed()
    t_model, t_decode = [], []
    for i in range(WARMUP + n):
        row, dm = p.step(i // 5 % dynamics.N_ACTIONS)   # each action held 5 frames
        _, dd = p.decode([row])
        if i >= WARMUP:
            t_model.append(dm)
            t_decode.append(dd)
    m, d = np.array(t_model) * 1e3, np.array(t_decode) * 1e3
    tot = m + d
    print(f"\n{n} steps after {WARMUP} warm-up, batch 1, one frame per step")
    print(f"{'stage':<8}{'mean':>9}{'p50':>9}{'p99':>9}{'max':>9}   ms")
    for name, v in (("model", m), ("decode", d), ("total", tot)):
        print(f"{name:<8}{v.mean():9.2f}{np.percentile(v, 50):9.2f}"
              f"{np.percentile(v, 99):9.2f}{v.max():9.2f}")
    print(f"model share of total {m.sum() / tot.sum():.1%}; "
          f"ceiling {1e3 / np.percentile(tot, 50):.1f} fps at p50, "
          f"{1e3 / np.percentile(tot, 99):.1f} fps at p99 (display excluded)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 3)[3])
    ap.add_argument("--dynamics", required=True, help="checkpoint file, run directory, or run id")
    ap.add_argument("--checkpoint", choices=("best", "final"), default="best",
                    help="which file of a run: best.pt or model.pt (default best)")
    ap.add_argument("--tokenizer", required=True, help="tokenizer run id (its token cache seeds)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ctx", type=int, default=None, help="context frames (default the config's)")
    ap.add_argument("--rounds", type=int, default=None, help="decoding rounds, K-round models only")
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance w")
    ap.add_argument("--fps", type=float, default=30.0, help="frame cap (default 30)")
    ap.add_argument("--scale", type=int, default=8, help="nearest-neighbour upscale (default 8)")
    ap.add_argument("--seed", type=int, default=None, help="RNG for picking seed clips")
    ap.add_argument("--record", default=None, metavar="PATH", help="write a .gif or .mp4")
    ap.add_argument("--side-by-side", action="store_true", help="real | model on a val episode")
    ap.add_argument("--episode", type=int, default=None, help="side-by-side: val episode index")
    ap.add_argument("--headless", action="store_true",
                    help="side-by-side without a window, for recording only")
    ap.add_argument("--bench", type=int, default=None, metavar="N",
                    help="time N steps headless and print the table")
    ap.add_argument("--max-frames", type=int, default=None, help="quit after this many frames")
    ap.add_argument("--verbose", action="store_true", help="print each action change")
    args = ap.parse_args()
    if args.fps <= 0 or args.scale < 1:
        raise SystemExit("--fps must be > 0 and --scale >= 1")

    p = Player(args)
    print(p.describe())
    if args.bench is not None:
        bench(p, args.bench)
    elif args.side_by_side:
        side_by_side(p, args)
    elif args.headless:
        raise SystemExit("--headless is for --side-by-side; use --bench for timing")
    else:
        play(p, args)


if __name__ == "__main__":
    sys.exit(main())
