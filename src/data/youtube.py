"""YouTube video downloader module using yt-dlp.

Enables downloading public YouTube traffic videos for intersection analysis.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import imageio_ffmpeg
import yt_dlp

from app.utils import temp_dir

logger = logging.getLogger(__name__)

YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com|youtu\.be)/(watch\?v=|embed/|v/|shorts/)?([a-zA-Z0-9_-]{11})"
)


def is_youtube_url(url: str) -> bool:
    """Check if the provided string matches a YouTube video URL pattern."""
    if not url:
        return False
    return bool(YOUTUBE_REGEX.search(url.strip()))


def _ensure_ffmpeg() -> str:
    """Ensure ffmpeg.exe exists in imageio_ffmpeg binary path and is registered in os.environ['PATH']."""
    try:
        exe_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
        bin_dir = exe_path.parent
        target_exe = bin_dir / "ffmpeg.exe"
        if not target_exe.exists():
            shutil.copy(exe_path, target_exe)

        bin_dir_str = str(bin_dir)
        path_env = os.environ.get("PATH", "")
        if bin_dir_str not in path_env.split(os.pathsep):
            os.environ["PATH"] = bin_dir_str + os.pathsep + path_env

        return str(target_exe if target_exe.exists() else exe_path)
    except Exception as exc:
        logger.warning(f"Could not register ffmpeg from imageio_ffmpeg: {exc}")
        return ""


def _trim_video(input_path: Path, max_duration_s: float, ffmpeg_exe: str) -> Path:
    """Trim a video file to `max_duration_s` using local ffmpeg binary."""
    if not ffmpeg_exe or not Path(ffmpeg_exe).exists():
        return input_path

    trimmed_path = input_path.parent / f"{input_path.stem}_clip{input_path.suffix}"
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss",
        "0",
        "-i",
        str(input_path),
        "-t",
        str(max_duration_s),
        "-c",
        "copy",
        str(trimmed_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=30)
        if res.returncode == 0 and trimmed_path.exists() and trimmed_path.stat().st_size > 0:
            return trimmed_path
    except Exception as exc:
        logger.warning(f"Local ffmpeg trim failed: {exc}")

    return input_path


def download_youtube_video(
    url: str,
    output_dir: Path | None = None,
    max_duration_s: float = 60.0,
) -> Tuple[Path, str, float]:
    """Download a YouTube video clipped to `max_duration_s` seconds.

    Args:
        url: The YouTube video URL.
        output_dir: Target directory for saving the file (defaults to temp_dir()).
        max_duration_s: Maximum duration in seconds to download (default: 60s).

    Returns:
        Tuple of (saved_file_path, video_title, duration_seconds).
    """
    clean_url = url.strip()
    if not is_youtube_url(clean_url):
        raise ValueError(f"Invalid YouTube URL: '{url}'")

    out_dir = output_dir or temp_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_template = str(out_dir / "%(id)s.%(ext)s")
    ffmpeg_exe = _ensure_ffmpeg()

    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
    }
    if ffmpeg_exe:
        ydl_opts["ffmpeg_location"] = ffmpeg_exe

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)

    if not info:
        raise RuntimeError(f"Failed to fetch video information from {clean_url}")

    video_id = info.get("id", "yt_video")
    filename = ydl.prepare_filename(info)
    actual_path = Path(filename)

    if not actual_path.exists():
        candidates = list(out_dir.glob(f"{video_id}.*"))
        if candidates:
            actual_path = candidates[0]
        else:
            raise FileNotFoundError(f"Downloaded YouTube video file not found at {filename}")

    title = info.get("title", "YouTube Traffic Video")
    total_dur = float(info.get("duration") or max_duration_s)

    # Trim to max_duration_s locally using ffmpeg if video exceeds max_duration_s
    if total_dur > max_duration_s:
        actual_path = _trim_video(actual_path, max_duration_s, ffmpeg_exe)
        duration = max_duration_s
    else:
        duration = total_dur

    logger.info(f"Downloaded YouTube video '{title}' ({duration:.1f}s) to {actual_path}")
    return actual_path, title, duration
