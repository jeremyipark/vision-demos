"""Probing, cached conversion, and cutting of the source clip."""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

HDR_TRANSFERS = frozenset({"arib-std-b67", "smpte2084"})   # HLG (iPhone) and PQ


@dataclass(frozen=True)
class VideoInfo:
    """A decoded clip, as OpenCV sees it."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def __str__(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps:.2f} fps, {self.n_frames} frames, {self.duration:.1f}s"


@functools.lru_cache(maxsize=None)
def tool(name: str) -> str:
    """The ffmpeg/ffprobe next to this Python (conda's has libplacebo), else PATH's."""
    local = Path(sys.executable).parent / name
    return str(local) if local.is_file() else name


@functools.lru_cache(maxsize=None)
def has_filter(name: str) -> bool:
    try:
        proc = subprocess.run([tool("ffmpeg"), "-hide_banner", "-filters"],
                              capture_output=True, text=True)
    except OSError:
        return False
    return any(line.split()[1:2] == [name] for line in proc.stdout.splitlines() if line.strip())


def tonemap_filter(source: dict, *, mode: str, algorithm: str) -> tuple[str | None, str]:
    """``(ffmpeg filter or None, reason)``: HDR to Rec. 709, retagged so players don't re-apply HDR."""
    transfer = (source.get("color_transfer") or "").lower()
    if mode == "off":
        return None, "off"
    if mode == "auto" and transfer not in HDR_TRANSFERS:
        return None, f"not needed, source is SDR ({transfer or 'untagged'})"
    if not has_filter("libplacebo"):
        return None, (f"[yellow]source is HDR ({transfer}) but this ffmpeg has no libplacebo[/]; "
                      "use the conda env's ffmpeg, not Homebrew's")
    return (f"libplacebo=tonemapping={algorithm}:colorspace=bt709:"
            f"color_primaries=bt709:color_trc=bt709:range=tv:format=yuv420p"
            ), f"{algorithm}, {transfer} -> bt709"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr[-3000:]}")
    return proc


def probe_source(path: Path) -> dict:
    proc = _run([
        tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,duration,pix_fmt,"
                         "color_transfer,color_primaries,color_space",
        "-of", "json", str(path),
    ])
    stream = json.loads(proc.stdout)["streams"][0]
    transfer = stream.get("color_transfer")
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "pix_fmt": stream.get("pix_fmt"),
        "n_frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
        "color_transfer": transfer,
        "color_primaries": stream.get("color_primaries"),
        "color_space": stream.get("color_space"),
        "is_hdr": (transfer or "").lower() in HDR_TRANSFERS,
    }


def cache_path(src: Path, cache_dir: Path, *, target_height, trim_seconds, crf,
               tonemap: str | None = None) -> Path:
    """The converted MP4's name, keyed on the source and every conversion setting."""
    stat = src.stat()
    key = json.dumps({
        "name": src.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "target_height": target_height, "trim_seconds": trim_seconds, "crf": crf,
        **({"tonemap": tonemap} if tonemap else {}),
    }, sort_keys=True)
    return cache_dir / f"{src.stem}.{hashlib.sha256(key.encode()).hexdigest()[:12]}.mp4"


def convert(src: Path, dst: Path, *, target_height: int | None, trim_seconds: float | None,
            crf: int, total_frames: int, console, tonemap: str | None = None) -> None:
    """Transcode to 8-bit H.264 (rotation applied, no audio), with a progress bar."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    cmd = [tool("ffmpeg"), "-y", "-nostats", "-loglevel", "error", "-progress", "pipe:1",
           "-i", str(src)]
    if trim_seconds:
        cmd += ["-t", str(trim_seconds)]
    chain = [tonemap] if tonemap else []   # tone-map before scaling
    if target_height:
        chain.append(f"scale=-2:'min({target_height},ih)'")   # only ever downscale
    if chain:
        cmd += ["-vf", ",".join(chain)]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-an", str(tmp)]

    columns = [TextColumn("[cyan]converting[/]"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn()]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with Progress(*columns, console=console, transient=True) as progress:
            task = progress.add_task("convert", total=total_frames or None)
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                if key == "frame" and value.isdigit():
                    progress.update(task, completed=min(int(value), total_frames or int(value)))
            proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg conversion failed:\n{proc.stderr.read()[-3000:]}")
    tmp.replace(dst)   # atomic, so a killed run never leaves half a file in the cache


def inspect(path: Path) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    info = VideoInfo(path=path, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                     n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def encode_h264(src: Path, dst: Path, *, crf: int) -> None:
    """Re-encode OpenCV's MPEG-4 output to H.264 tagged Rec. 709, which browsers play."""
    _run([tool("ffmpeg"), "-y", "-loglevel", "error", "-i", str(src),
          "-vf", "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
          "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)])


def segment(src: Path, dst: Path, *, start_frame: int, n_frames: int, fps: float,
            crf: int = 20) -> None:
    """Cut frames ``[start_frame, start_frame + n_frames)`` into their own MP4.

    Re-encoded, not stream-copied, so the cut is frame-exact rather than keyframe-aligned.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    _run([tool("ffmpeg"), "-y", "-loglevel", "error", "-ss", f"{start_frame / fps:.6f}",
          "-i", str(src), "-frames:v", str(int(n_frames)), "-c:v", "libx264",
          "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(tmp)])
    tmp.replace(dst)
