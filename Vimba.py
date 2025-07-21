import cv2, time, threading, queue
from vmbpy import (VmbSystem, PixelFormat, Camera, Stream, Frame,
                   FeaturePersistOptions)

# --------------------------- display thread ---------------------------------
def display_worker(img_q: "queue.Queue[None | tuple]"):
    fps_frames, fps_t0 = 0, time.time()
    while True:
        item = img_q.get()
        if item is None:                      # sentinel -> quit
            break
        img = item
        cv2.imshow("Manta G‑125B  live", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # fps counter
        fps_frames += 1
        now = time.time()
        if now - fps_t0 >= 1:
            print(f"FPS: {fps_frames}")
            fps_frames, fps_t0 = 0, now
    cv2.destroyAllWindows()

# --------------------------- frame handler ----------------------------------
def frame_handler(cam: Camera, stream: Stream, frame: Frame, img_q):
    # *** ONLY light‑weight work here ***
    if frame.get_pixel_format() != PixelFormat.Mono8:
        frame.convert_pixel_format(PixelFormat.Bgr8)
        img = frame.as_opencv_image()         # 3‑channel
    else:
        gray = frame.as_numpy_ndarray()
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_q.put(img)                            # hand frame to GUI thread
    stream.queue_frame(frame)                 # re‑queue for reuse

# --------------------------- main -------------------------------------------
def run_fast_live_view(buffer_count: int = 20):
    img_q: "queue.Queue[None | tuple]" = queue.Queue(maxsize=buffer_count * 2)
    disp_th = threading.Thread(target=display_worker, args=(img_q,), daemon=True)

    with VmbSystem.get_instance() as vmb:
        cam: Camera = vmb.get_all_cameras()[0]
        print("✅ Opening:", cam.get_name())

        with cam:
            # ---------- one‑time performance tweaks ----------
            # 1) Packet‑size auto tune (retry safe)
            try:
                cam.run_command("GVSPAdjustPacketSize")
                cam.get_feature_by_name("GVSPAdjustPacketSize").wait_to_complete(1000)
            except Exception:
                pass                              # ignore older firmware

            # 2) Max link throughput (≈ GigE wire speed)
            try:
                cam.get_feature_by_name("DeviceLinkThroughputLimit").set(115_000_000)
            except Exception:
                pass

            # 3) Pixel format & ROI
            cam.set_pixel_format(PixelFormat.Mono8)      # fastest
            # Example ROI: uncomment next two lines for 640×480 @ ~60 fps
            # cam.get_feature_by_name("Width").set(640)
            # cam.get_feature_by_name("Height").set(480)

            # 4) Exposure
            cam.get_feature_by_name("ExposureTime").set(10_000)   # 10 ms

            # ---------- start threads ----------
            disp_th.start()
            cam.start_streaming(lambda c, s, f: frame_handler(c, s, f, img_q),
                                buffer_count=buffer_count)
            print("🔴 press  q  in the window to stop")

            # keep main thread alive until display closes
            disp_th.join()

            cam.stop_streaming()

    img_q.put(None)                           # tell display thread to quit
    print("✅ Done")

if __name__ == "__main__":
    run_fast_live_view()
