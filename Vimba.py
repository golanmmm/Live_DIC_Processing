import cv2
import numpy as np
import time
from vmbpy import VmbSystem, FrameStatus, PixelFormat
from threading import Lock

# === Globals ===
latest_frame = None
frame_lock = Lock()
running = True


# === Convert frame to OpenCV grayscale image ===
def to_gray(frame):
    img = frame.as_numpy_ndarray()
    return img  # Mono8 = already 2D array


# === Vimba frame handler (runs in camera thread) ===
def handle_frame(cam, stream, frame):
    global latest_frame
    if frame.get_status() == FrameStatus.Complete:
        img = to_gray(frame)
        with frame_lock:
            latest_frame = img
    stream.queue_frame(frame)


# === Main function ===
def main():
    global running
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            raise RuntimeError("No Allied Vision cameras found.")

        with cams[0] as cam:
            if PixelFormat.Mono8 in cam.get_pixel_formats():
                cam.set_pixel_format(PixelFormat.Mono8)
            else:
                raise RuntimeError("Camera does not support Mono8 format.")

            # Set GVSPPacketSize
            try:
                feature = cam.get_feature_by_name("GVSPPacketSize")
                feature.set(8228)
                print("✅ GVSPPacketSize set to 8228")
            except Exception as e:
                print(f"⚠️ Failed to set GVSPPacketSize: {e}")

            # --- Optional tuning ---
            # cam.ExposureTime.set(3000)  # in µs
            # cam.AcquisitionFrameRate.set(60)

            # --- Start streaming ---
            cam.start_streaming(handle_frame, buffer_count=10)
            print("🎥 Streaming started. Press ESC to quit.")

            # --- Display loop in main thread ---
            prev_time = time.time()
            frame_count = 0

            while cam.is_streaming() and running:
                with frame_lock:
                    frame = latest_frame.copy() if latest_frame is not None else None

                if frame is not None:
                    cv2.imshow("Low-Latency Live Stream", frame)
                    frame_count += 1

                # Exit on ESC
                if cv2.waitKey(1) & 0xFF == 27:
                    running = False
                    cam.stop_streaming()
                    break

                # Print FPS every second
                if time.time() - prev_time >= 1.0:
                    print(f"FPS: {frame_count}")
                    frame_count = 0
                    prev_time = time.time()

            cam.stop_streaming()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
