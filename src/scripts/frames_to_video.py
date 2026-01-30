# scripts/frames_to_video.py
import cv2
import glob
import os
from src.utils.paths import load_paths

def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: int = 15,
    pattern: str = "frame_*.png",
):
    """
    Convert saved Bloch-sphere frames into a video.

    Parameters
    ----------
    frames_dir : str
        Directory containing frame images.
    output_path : str
        Path to output video (e.g. .mp4).
    fps : int
        Frames per second.
    pattern : str
        Glob pattern for frame files.
    """
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, pattern)))

    if not frame_paths:
        raise RuntimeError("No frames found to convert into video.")

    # Read first frame to get dimensions
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError("Failed to read first frame.")

    height, width, _ = first.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        output_path,
        cv2.CAP_FFMPEG,
        fourcc,
        fps,
        (width, height),
    )

    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        video.write(img)

    video.release()
    print(f"✅ Video written to: {output_path}")


if __name__ == "__main__":
    _ , path = load_paths()
    frames_dir = path["frames_adaptive"]
    output_path = path["video_adaptive"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames_to_video(
        frames_dir=frames_dir,
        output_path=output_path,
        fps=15,
    )
