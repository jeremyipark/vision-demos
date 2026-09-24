"""Probing and (cached) conversion of the source clip."""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn


@dataclass(frozen=True)
class VideoInfo:
    """Geometry of a decoded clip, as OpenCV will see it."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def __str__(self) -> str:
        return (
            f"{self.width}x{self.height} @ {self.fps:.2f} fps, "
            f"{self.n_frames} frames, {self.duration:.1f}s"
        )


# Transfer curves that mean the clip is HDR. An iPhone shooting in HDR writes
# HLG; PQ is here because other cameras and edited exports use it and the
# handling is identical.
HDR_TRANSFERS = frozenset({"arib-std-b67", "smpte2084"})


@functools.lru_cache(maxsize=None)
def tool(name: str) -> str:
    """Resolve *name* next to the running interpreter, falling back to PATH.

    `conda activate` puts the environment's bin first, but running its python by
    absolute path does not, and what PATH finds instead is usually Homebrew's
    ffmpeg — which has no `libplacebo`, so HDR silently goes untone-mapped.
    """
    local = Path(sys.executable).parent / name
    return str(local) if local.is_file() else name


@functools.lru_cache(maxsize=None)
def has_filter(name: str) -> bool:
    """Whether this ffmpeg build ships *name*.

    Worth asking rather than assuming: tone-mapping needs `libplacebo`, which
    conda-forge's ffmpeg has and Homebrew's does not, so the answer depends on
    which one is first on PATH — not on the machine.
    """
    try:
        proc = subprocess.run([tool("ffmpeg"), "-hide_banner", "-filters"],
                              capture_output=True, text=True)
    except OSError:
        return False
    return any(line.split()[1:2] == [name] for line in proc.stdout.splitlines() if line.strip())


def tonemap_filter(source: dict, *, mode: str, algorithm: str) -> tuple[str | None, str]:
    """The tone-map filter for this source, and a line explaining the choice.

    Returns ``(filter_string_or_None, reason)``. The filter converts HDR to
    Rec. 709 and *retags* the output, which matters as much as the pixels: the
    conversion this replaces left 8-bit frames still labelled `bt2020`/`HLG`,
    so players applied an HDR curve to video that no longer had one.
    """
    transfer = (source.get("color_transfer") or "").lower()
    is_hdr = transfer in HDR_TRANSFERS

    if mode == "off":
        return None, "disabled (TONEMAP = 'off')"
    if mode == "auto" and not is_hdr:
        return None, f"not needed — source is SDR ({transfer or 'untagged'})"
    if not has_filter("libplacebo"):
        return None, (
            f"[yellow]source is HDR ({transfer}) but this ffmpeg has no libplacebo[/] — "
            "colours will stay washed out and detections will suffer. The conda-forge "
            "ffmpeg in this project's environment has it; Homebrew's does not, so check "
            "which one `which ffmpeg` finds."
        )

    return (
        f"libplacebo=tonemapping={algorithm}:colorspace=bt709:"
        f"color_primaries=bt709:color_trc=bt709:range=tv:format=yuv420p"
    ), f"{algorithm} — {transfer or 'untagged'} -> bt709"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr[-3000:]}")
    return proc


def probe_source(path: Path) -> dict:
    """ffprobe the source, including the rotation OpenCV would miss."""
    proc = _run([
        tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,duration,pix_fmt,"
                         "color_transfer,color_primaries,color_space",
        "-show_entries", "stream_side_data=rotation",
        "-of", "json", str(path),
    ])
    stream = json.loads(proc.stdout)["streams"][0]
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "pix_fmt": stream.get("pix_fmt"),
        "n_frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
        "rotation": rotation,
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "color_space": stream.get("color_space"),
        "is_hdr": (stream.get("color_transfer") or "").lower() in HDR_TRANSFERS,
    }


def cache_path(src: Path, cache_dir: Path, *, target_height, trim_seconds, crf,
               tonemap: str | None = None, passthrough: bool = False) -> Path:
    """Deterministic name for the converted MP4.

    Keyed on the source's identity *and* the settings that shaped the output, so
    a changed height or CRF produces a different file rather than silently
    reusing the previous one.
    """
    stat = src.stat()
    key = json.dumps(
        {
            "name": src.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "target_height": target_height,
            "trim_seconds": trim_seconds,
            "crf": crf,
            "tonemap": tonemap,
            **({"passthrough": True} if passthrough else {}),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"{src.stem}.{digest}.mp4"


def convert(
    src: Path,
    dst: Path,
    *,
    target_height: int | None,
    trim_seconds: float | None,
    crf: int,
    total_frames: int,
    console,
    tonemap: str | None = None,
    passthrough: bool = False,
) -> None:
    """Transcode to H.264 MP4, printing ffmpeg's own frame counter as progress.

    *passthrough* keeps every source frame exactly once, whatever its timestamp.
    A LiDAR capture pairs depth slab *i* with video frame *i*, and its clip runs
    at 29.99 fps against a nominal 30 — a constant-rate re-mux would duplicate
    or drop the odd frame to make up the difference, and every depth map after
    it would be one frame out.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")

    cmd = [tool("ffmpeg"), "-y", "-nostats", "-loglevel", "error",
           "-progress", "pipe:1", "-i", str(src)]
    if trim_seconds:
        cmd += ["-t", str(trim_seconds)]
    # Tone-map before scaling, not after. Resampling is an average of
    # neighbouring pixels, and averaging is only meaningful in linear-ish light
    # — doing it across an HLG curve mixes values that do not mean what their
    # spacing implies. It also costs nothing here: the filter runs on the GPU.
    chain = []
    if tonemap:
        chain.append(tonemap)
    if target_height:
        # Width follows the aspect ratio; `-2` keeps it even (H.264 needs that)
        # and `min(...,ih)` means this only ever downscales. `ih` is the
        # *rotated* height. ffmpeg applies the container's rotation before
        # filters run, so a clip stored 3840x2160 scales as the portrait it is.
        chain.append(f"scale=-2:'min({target_height},ih)'")
    if chain:
        cmd += ["-vf", ",".join(chain)]
    if passthrough:
        cmd += ["-fps_mode", "passthrough"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",   # 10-bit HEVC -> 8-bit H.264, which everything decodes
        "-an",                   # a depth capture records no sound
        str(tmp),
    ]

    columns = [
        TextColumn("[cyan]converting[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} frames"),
        TimeElapsedColumn(),
    ]
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

    tmp.replace(dst)   # atomic: a killed run never leaves a half-file in the cache


def segment(src: Path, dst: Path, *, start_frame: int, n_frames: int, fps: float,
            crf: int = 20) -> None:
    """Cut frames ``[start_frame, start_frame + n_frames)`` into their own MP4.

    Re-encoded rather than stream-copied. A copy can only cut on a keyframe, and
    every frame index downstream — SAM's `frame_id`, the camera track, the pose
    — is counted from the start of the file, so a segment that silently began
    three frames early would put the whole route three frames out of step.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    _run([tool("ffmpeg"), "-y", "-loglevel", "error",
          "-ss", f"{start_frame / fps:.6f}", "-i", str(src),
          "-frames:v", str(int(n_frames)),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
          "-pix_fmt", "yuv420p", "-an", str(tmp)])
    tmp.replace(dst)


def inspect(path: Path) -> VideoInfo:
    """Read geometry the way the renderer will, i.e. through OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def encode_h264(src: Path, dst: Path, *, crf: int, audio_from: Path | None = None,
                duration: float | None = None) -> None:
    """Re-encode the rendered video, optionally muxing audio back in.

    OpenCV's VideoWriter emits MPEG-4 Part 2 and no audio at all, which most
    players and every browser refuse. This is the step that makes the output
    shareable. *audio_from* is the original clip: the grunt at the crux is part
    of the climb, and the render has dropped it by this point.

    The Rec. 709 tags are written explicitly rather than left blank. Players
    assume 709 for untagged video and would land in the right place anyway, but
    the whole HDR problem upstream of here was a file whose tags disagreed with
    its pixels — so this one says what it is.

    They go on with `setparams` rather than the `-color_*` output options. The
    frames arrive from OpenCV carrying no colour description at all, and
    against an unspecified input those options set the matrix and then silently
    drop the primaries and the transfer — two thirds of a tag is worse than
    none. `setparams` stamps all three onto the frames themselves.

    *duration* caps the output in seconds, and is what a render longer than its
    own source needs. The completion panel holds the last frame past the end of
    the clip, and `-shortest` — otherwise what keeps a long audio track from
    running on over nothing — would cut exactly that tail back off. Given a
    duration the length is stated rather than inferred, so the held tail
    survives and the audio still stops where it should.
    """
    cmd = [tool("ffmpeg"), "-y", "-loglevel", "error", "-i", str(src)]
    if audio_from is not None:
        cmd += ["-i", str(audio_from)]
    cmd += [
        "-vf", "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
    ]
    if audio_from is not None:
        # `0:a?` is not enough: the render has no audio stream, so the mapping
        # has to come from the second input, and `?` keeps a silent source from
        # failing the encode.
        cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
        if duration is None:
            cmd += ["-shortest"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [str(dst)]
    _run(cmd)
