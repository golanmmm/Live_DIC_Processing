import cv2
import numpy as np
from vmbpy import VmbSystem, FrameStatus, PixelFormat
from threading import Thread
from queue import Queue

frame_queue = Queue(maxsize=2)
running = True

def to_bgr(frame):
    try:
        img = frame.as_numpy_ndarray()
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    except AttributeError:
        fmt = frame.get_pixel_format()
        w, h = frame.get_width(), frame.get_height()
        buf = frame.get_buffer()
        if fmt == PixelFormat.Mono8:
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif fmt == PixelFormat.Bgr8:
            return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        raise RuntimeError(f"Unsupported format: {fmt}")

def handle_frame(cam, stream, frame):
    if frame.get_status() == FrameStatus.Complete:
        img = to_bgr(frame)
        if not frame_queue.full():
            frame_queue.put(img)
    stream.queue_frame(frame)

def display_thread():
    global running
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow("Live Stream", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                running = False
                break
    cv2.destroyAllWindows()

def main():
    global running
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            raise RuntimeError("No cameras found")

        with cams[0] as cam:
            if PixelFormat.Bgr8 in cam.get_pixel_formats():
                cam.set_pixel_format(PixelFormat.Bgr8)
            elif PixelFormat.Mono8 in cam.get_pixel_formats():
                cam.set_pixel_format(PixelFormat.Mono8)

            # Optional performance tuning
            # cam.GVSPPacketSize.set(8228)
            # cam.ExposureTime.set(5000)

            cam.start_streaming(handle_frame, buffer_count=8)
            print("🚀 Streaming... Press ESC in window to stop")

            thread = Thread(target=display_thread)
            thread.start()

            while cam.is_streaming() and running:
                pass

            cam.stop_streaming()
            thread.join()

if __name__ == "__main__":
    main()
