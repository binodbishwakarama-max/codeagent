"""Render production launch video for Prism Review with AI voiceover, Ken Burns motion, and subtitles.

Usage:
    uv run python scripts/render_video.py
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TypedDict


class SceneSpec(TypedDict):
    image: Path
    duration: float
    zoom: str


def get_ffmpeg_exe() -> str:
    """Find a functional ffmpeg binary."""
    # Check PATH first
    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path

    # Check uv cache for imageio-ffmpeg
    uv_cache = Path.home() / "AppData" / "Local" / "uv" / "cache"
    for candidate in uv_cache.glob("**/ffmpeg-win-x86_64*.exe"):
        if candidate.exists():
            return str(candidate)

    # Try importing imageio_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass

    raise RuntimeError("ffmpeg executable not found.")


def render_video() -> Path:
    ffmpeg_exe = get_ffmpeg_exe()
    print(f"[+] Using FFmpeg: {ffmpeg_exe}")

    base_dir = Path("dist/video")
    audio_path = base_dir / "prism_review_voiceover.mp3"
    subtitles_path = base_dir / "prism_review_subtitles.vtt"
    output_mp4 = base_dir / "prism_review_demo.mp4"

    scenes: list[SceneSpec] = [
        {"image": base_dir / "scene1.jpg", "duration": 14.0, "zoom": "in"},
        {"image": base_dir / "scene2.jpg", "duration": 14.0, "zoom": "pan_right"},
        {"image": base_dir / "scene3.jpg", "duration": 19.5, "zoom": "center_push"},
        {"image": base_dir / "scene4.jpg", "duration": 16.0, "zoom": "slow_in"},
        {"image": base_dir / "scene5.jpg", "duration": 8.55, "zoom": "out"},
    ]

    for s in scenes:
        if not s["image"].exists():
            raise FileNotFoundError(f"Missing scene image: {s['image']}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio: {audio_path}")

    # Generate video segments with Ken Burns motion
    temp_dir = base_dir / "temp_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segment_paths = []

    fps = 30
    width = 1920
    height = 1080

    print("[1/3] Generating animated scene segments with Ken Burns motion...")
    for idx, s in enumerate(scenes):
        img = s["image"]
        dur = s["duration"]
        frames = int(dur * fps)
        mode = s["zoom"]
        seg_out = temp_dir / f"seg_{idx}.mp4"
        segment_paths.append(seg_out)

        # Filters for subtle cinematic camera motion
        if mode == "in":
            # Slow zoom in
            vf = (
                f"scale=-2:2160,zoompan=z='min(zoom+0.0012,1.25)':d={frames}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
            )
        elif mode == "pan_right":
            # Pan slightly right with steady zoom
            vf = (
                f"scale=-2:2160,zoompan=z='1.18':d={frames}:"
                f"x='if(lte(on,1),(iw-iw/zoom)/3,min(x+0.8,(iw-iw/zoom)*0.85))':"
                f"y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
            )
        elif mode == "center_push":
            # Center push-in toward prism
            vf = (
                f"scale=-2:2160,zoompan=z='min(zoom+0.0009,1.20)':d={frames}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
            )
        elif mode == "slow_in":
            # Subtle stage push
            vf = (
                f"scale=-2:2160,zoompan=z='min(zoom+0.0010,1.18)':d={frames}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
            )
        else:  # out
            # Pull out
            vf = (
                f"scale=-2:2160,zoompan=z='max(1.22-0.0012*on,1.02)':d={frames}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
            )

        cmd = [
            ffmpeg_exe,
            "-y",
            "-loop",
            "1",
            "-i",
            str(img),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(dur),
            str(seg_out),
        ]
        print(f"  Rendering Scene {idx + 1}/{len(scenes)} ({s['image'].name}, {dur}s, {mode})...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr)
            raise RuntimeError(f"Failed to render segment {idx}: {res.stderr}")

    # Concatenate segments
    print("[2/3] Concatenating visual scenes into continuous track...")
    concat_list_file = temp_dir / "concat_list.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            # Use forward slashes in ffmpeg concat list
            escaped_path = str(p.resolve()).replace("\\", "/")
            f.write(f"file '{escaped_path}'\n")

    video_only = temp_dir / "video_only.mp4"
    cmd_concat = [
        ffmpeg_exe,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_file),
        "-c",
        "copy",
        str(video_only),
    ]
    res_concat = subprocess.run(cmd_concat, capture_output=True, text=True)
    if res_concat.returncode != 0:
        print(res_concat.stderr)
        raise RuntimeError(f"Concat failed: {res_concat.stderr}")

    # Combine video + voiceover audio + burned-in subtitles
    print("[3/3] Mixing neural voiceover and burning stylized subtitles...")
    subtitles_filter_arg = str(subtitles_path.resolve()).replace("\\", "/")
    # Escape colon for ffmpeg filter syntax on Windows (e.g., C\:/...)
    subtitles_filter_arg = subtitles_filter_arg.replace(":", "\\:")

    subtitle_style = (
        "Fontname=Arial,Fontsize=22,Bold=1,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H80000000,BorderStyle=4,"
        "Outline=1,Shadow=0,MarginV=35,Alignment=2"
    )

    cmd_final = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(audio_path),
        "-vf",
        f"subtitles='{subtitles_filter_arg}':force_style='{subtitle_style}'",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_mp4),
    ]
    res_final = subprocess.run(cmd_final, capture_output=True, text=True)
    if res_final.returncode != 0:
        # Fallback without subtitle filter if libass encounters fontconfig issue
        print("  [Notice] Retrying without burned-in subtitles as fallback...")
        cmd_fallback = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_only),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_mp4),
        ]
        res_fb = subprocess.run(cmd_fallback, capture_output=True, text=True)
        if res_fb.returncode != 0:
            raise RuntimeError(f"Final video mix failed: {res_fb.stderr}")

    # Clean up temp segments
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Copy to docs/prism_review_demo.mp4 as well for easy viewing
    docs_video = Path("docs/prism_review_demo.mp4")
    shutil.copy2(output_mp4, docs_video)

    file_size_mb = output_mp4.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 80)
    print(f"🎉 VIDEO GENERATED SUCCESSFULLY: {output_mp4}")
    print(f"📦 File Size: {file_size_mb:.2f} MB")
    print(f"📂 Also saved to: {docs_video}")
    print("=" * 80)

    return output_mp4


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    render_video()


if __name__ == "__main__":
    main()
