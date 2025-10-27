import os
import re
import sys
import glob
import cv2
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog, simpledialog

# ---------- helpers ----------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def pick_folder():
    root = tk.Tk(); root.withdraw(); root.update()
    folder = filedialog.askdirectory(title="select image folder")
    root.destroy()
    return folder

def ask_fps(default=30.0):
    root = tk.Tk(); root.withdraw()
    try:
        val = simpledialog.askfloat("frame rate", "fps for output video:", initialvalue=default, minvalue=0.1)
    finally:
        root.destroy()
    return val if val else default

def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort(key=natural_key)
    return files

def ensure_bgr8(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.dtype != 'uint8':
        return cv2.convertScaleAbs(frame)
    return frame

def check_same_size(paths, w_ref, h_ref):
    bad = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            bad.append((p, "unreadable"))
            continue
        h, w = img.shape[:2]
        if (w, h) != (w_ref, h_ref):
            bad.append((p, f"{w}x{h}"))
    return bad

# ---------- exporters ----------
def export_with_ffmpeg_pipe(images, out_path_mp4, fps, w, h):
    """
    stream raw bgr frames to ffmpeg (h.264 ultrafast, crf 15).
    keeps exact resolution; no scaling.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s:v", f"{w}x{h}",
        "-r", f"{fps}",
        "-i", "-",                # stdin
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "15",             # smaller number => higher quality
        "-pix_fmt", "yuv420p",    # broad compatibility; no size change
        out_path_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for idx, p in enumerate(images, 1):
            frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            frame = ensure_bgr8(frame)
            if frame is None or frame.shape[1] != w or frame.shape[0] != h:
                print(f"[skip] {os.path.basename(p)} unreadable or wrong size")
                continue
            proc.stdin.write(frame.tobytes())
            if idx % 50 == 0:
                print(f"wrote {idx}/{len(images)} frames...")
    finally:
        proc.stdin.close()
        proc.wait()
    return out_path_mp4

def export_with_opencv_mjpg(images, out_path_avi, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(out_path_avi, fourcc, fps, (w, h), True)
    if not vw.isOpened():
        return None
    for idx, p in enumerate(images, 1):
        frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        frame = ensure_bgr8(frame)
        if frame is None or frame.shape[1] != w or frame.shape[0] != h:
            print(f"[skip] {os.path.basename(p)} unreadable or wrong size")
            continue
        vw.write(frame)
        if idx % 50 == 0:
            print(f"wrote {idx}/{len(images)} frames...")
    vw.release()
    return out_path_avi

# ---------- main ----------
def main():
    # cli: python script.py <folder> [fps]
    if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        fps = float(sys.argv[2]) if len(sys.argv) >= 3 else 30.0
    else:
        folder = pick_folder()
        if not folder:
            print("no folder selected. exiting.")
            return
        fps = ask_fps(30.0)

    images = list_images(folder)
    if not images:
        print("no images found in the selected folder.")
        return

    # probe first frame for size (this is the enforced size)
    first = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        print(f"failed to read first image: {images[0]}")
        return
    first = ensure_bgr8(first)
    h_ref, w_ref = first.shape[:2]
    print(f"frame size: {w_ref}x{h_ref} | fps: {fps}")

    # verify all are identical size (no resizing will be done)
    mismatches = check_same_size(images, w_ref, h_ref)
    if mismatches:
        print("error: found images with different sizes. to keep exact size/aspect, no resizing is allowed.\n"
              "mismatches (first 20 shown):")
        for p, sz in mismatches[:20]:
            print(f" - {os.path.basename(p)} : {sz}")
        if len(mismatches) > 20:
            print(f"...and {len(mismatches)-20} more.")
        return

    folder_name = os.path.basename(os.path.normpath(folder))
    out_base = os.path.join(folder, f"{folder_name}_from_images_fast")

    # prefer ffmpeg (fast + minimal compression, high quality)
    if shutil.which("ffmpeg") is not None:
        out_mp4 = out_base + ".mp4"
        print("[info] using ffmpeg: h.264 ultrafast, crf=15 (fast, minimal compression)")
        path = export_with_ffmpeg_pipe(images, out_mp4, fps, w_ref, h_ref)
        print(f"done. saved: {path}")
        return

    # fallback: opencv mjpg (very fast; mild compression, avi)
    print("[warn] ffmpeg not found — falling back to opencv MJPG (fast).")
    out_avi = out_base + ".avi"
    path = export_with_opencv_mjpg(images, out_avi, fps, w_ref, h_ref)
    if path:
        print(f"done. saved: {path}")
    else:
        print("could not create video writer. install ffmpeg or check your opencv build.")

if __name__ == "__main__":
    main()
