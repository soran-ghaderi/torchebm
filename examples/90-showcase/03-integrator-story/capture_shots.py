r"""Render the explainer to MP4 via parallel headless-Chrome screenshots.

CDP screencast is blocked in some sandboxes (Chrome is killed when it binds a
debug port), so we instead capture one clean frame per `--screenshot` launch
(?capture=1&t=SECONDS) across a thread pool, then encode with ffmpeg.

Run:  python capture_shots.py [--fps 20] [--workers 6]
Output: integrator_story.mp4  (1280x720, H.264, yuv420p)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
URL = (HERE / "index.html").as_uri()
DECK_DUR = {"story": 66, "sde": 16, "ot": 26, "fluid": 34}

CHROME_FLAGS = [
    "google-chrome", "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
    "--hide-scrollbars", "--window-size=1280,720", "--force-device-scale-factor=1",
    "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
]


def shot(i: int, fps: int, budget: int, frames: Path, deck: str):
    out = frames / f"f{i:05d}.png"
    if out.exists() and out.stat().st_size > 2000:
        return True
    try:
        subprocess.run(
            CHROME_FLAGS + [f"--virtual-time-budget={budget}", f"--screenshot={out}",
                            f"{URL}?deck={deck}&capture=1&t={i / fps}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except Exception:
        pass
    return out.exists() and out.stat().st_size > 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="story", choices=list(DECK_DUR))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--duration", type=float, default=0.0)   # 0 -> per-deck default
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=int, default=1600)
    ap.add_argument("--out", default="")
    ap.add_argument("--encode-only", action="store_true")
    a = ap.parse_args()

    deck = a.deck
    a.out = a.out or str(HERE / f"integrator_{deck}.mp4")
    FRAMES = Path(f"/tmp/frames_{deck}")
    dur = a.duration or DECK_DUR[deck]
    n = int(round(dur * a.fps))
    FRAMES.mkdir(exist_ok=True)
    print(f"duration={dur:.1f}s  fps={a.fps}  frames={n}  workers={a.workers}", flush=True)

    if not a.encode_only:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(shot, i, a.fps, a.budget, FRAMES, deck) for i in range(n)]
            done = ok = 0
            for f in concurrent.futures.as_completed(futs):
                done += 1
                ok += 1 if f.result() else 0
                if done % 40 == 0 or done == n:
                    print(f"  {done}/{n} captured ({ok} ok)  {time.time() - t0:.0f}s", flush=True)
        missing = [i for i in range(n) if not (FRAMES / f"f{i:05d}.png").exists()]
        if missing:
            print(f"  retrying {len(missing)} missing frames serially...", flush=True)
            for i in missing:
                shot(i, a.fps, a.budget + 400, FRAMES, deck)
        missing = [i for i in range(n) if not (FRAMES / f"f{i:05d}.png").exists()]
        print(f"missing after retry: {len(missing)}", flush=True)

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", str(a.fps), "-i", str(FRAMES / "f%05d.png"),
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18",
                    "-preset", "medium", "-movflags", "+faststart", a.out], check=True)
    kb = Path(a.out).stat().st_size // 1000
    print(f"wrote {a.out}  ({kb} KB)", flush=True)


if __name__ == "__main__":
    main()
