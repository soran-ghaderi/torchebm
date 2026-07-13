r"""Render the explainer to an MP4 by driving headless Chrome via the DevTools
protocol. For each frame we call window.__seek(t) (deterministic) and grab the
canvas with window.__snap(), then pipe the JPEGs to ffmpeg (from imageio-ffmpeg).

Run:  python examples/visualization/integrator_story/capture.py [--fps 30]
Output: integrator_story.mp4  (1280x720, H.264, yuv420p — LinkedIn/Twitter ready)
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
URL = (HERE / "index.html").as_uri()
PORT = 9333


def get_page_ws():
    for _ in range(120):
        try:
            data = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
            for t in data:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("DevTools page target not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(HERE / "integrator_story.mp4"))
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    chrome = subprocess.Popen([
        "google-chrome", "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
        "--hide-scrollbars", "--window-size=1320,840",
        "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        f"--remote-debugging-port={PORT}", "--remote-allow-origins=*", URL,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ws = websocket.create_connection(get_page_ws(), suppress_origin=True,
                                     max_int=2**31, timeout=60)
    _id = [0]

    def cmd(method, params=None):
        _id[0] += 1
        mid = _id[0]
        ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result")

    def ev(expr):
        r = cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return r["result"].get("value")

    cmd("Page.enable")
    cmd("Runtime.enable")
    for _ in range(200):
        try:
            if ev("typeof window.__ready==='function' && window.__ready()"):
                break
        except Exception:
            pass
        time.sleep(0.1)

    total = ev("window.__total")
    if not total:
        raise RuntimeError("page did not initialise (window.__total missing)")
    nframes = int(round(total * args.fps))
    print(f"total={total:.1f}s  ->  {nframes} frames @ {args.fps}fps  ->  {args.out}")

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    enc = subprocess.Popen([
        ff, "-y", "-f", "image2pipe", "-r", str(args.fps), "-vcodec", "mjpeg", "-i", "-",
        "-vf", "scale=1280:720:flags=lanczos", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium",
        "-movflags", "+faststart", args.out,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t0 = time.time()
    for i in range(nframes):
        ev(f"window.__seek({i / args.fps})")
        durl = ev("window.__snap(0.95)")
        enc.stdin.write(base64.b64decode(durl.split(",", 1)[1]))
        if i % 120 == 0:
            print(f"  frame {i}/{nframes}  ({time.time() - t0:.0f}s)")
    enc.stdin.close()
    enc.wait()
    ws.close()
    chrome.terminate()
    kb = Path(args.out).stat().st_size // 1000
    print(f"wrote {args.out}  ({kb} KB)")


if __name__ == "__main__":
    main()
