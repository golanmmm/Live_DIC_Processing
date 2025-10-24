import numpy as np
import cv2
import pyvirtualcam
import threading
import time
from pyueye import ueye

# ----------- Step 1: Initialize and Detect Camera ------------
hCam = ueye.HIDS(0)
ret = ueye.is_InitCamera(hCam, None)
if ret != ueye.IS_SUCCESS:
    raise RuntimeError("❌ Could not initialize uEye camera.")

# ----------- Step 2: Get Sensor Info ------------------------
sensor_info = ueye.SENSORINFO()
ueye.is_GetSensorInfo(hCam, sensor_info)

sensor_name = sensor_info.strSensorName.decode('utf-8').lower()
is_mono = True  # Force mono mode for your monochrome sensor
is_color = not is_mono

# ----------- Step 3: Configure Camera ------------------------
ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

width = int(sensor_info.nMaxWidth)
height = int(sensor_info.nMaxHeight)

rect = ueye.IS_RECT()
rect.s32X = ueye.int(0)
rect.s32Y = ueye.int(0)
rect.s32Width = ueye.int(width)
rect.s32Height = ueye.int(height)
ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect))

color_mode = ueye.IS_CM_MONO8
bits_per_pixel = 8
ueye.is_SetColorMode(hCam, color_mode)

# ----------- No FPS forcing (removed is_SetFrameRate) ----------

MemPtr = ueye.c_mem_p()
MemID = ueye.int()
pitch = width * (bits_per_pixel // 8)
ueye.is_AllocImageMem(hCam, width, height, bits_per_pixel, MemPtr, MemID)
ueye.is_SetImageMem(hCam, MemPtr, MemID)

ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)

# read current fps AFTER streaming starts; fallback if zero
actual_fps = ueye.double()
ueye.is_GetFramesPerSecond(hCam, actual_fps)
vc_fps = int(actual_fps.value) if actual_fps.value and actual_fps.value > 0 else 30

# ----------- Step 4: Double-Buffered Threaded Capture -----------
latest_frame = None
frame_ready = threading.Event()
stop_flag = False

def capture_thread():
    while not stop_flag:
        raw = ueye.get_data(MemPtr, width, height, bits_per_pixel, pitch, copy=False)
        expected_size = height * pitch
        if raw.nbytes != expected_size:
            continue
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        # Update latest frame
        globals()['latest_frame'] = frame
        frame_ready.set()

cap_thread = threading.Thread(target=capture_thread)
cap_thread.start()

# ----------- Step 5: Virtual Camera Output + FPS Log -----------
with pyvirtualcam.Camera(width=width, height=height, fps=vc_fps, print_fps=False) as cam:
    print(f"🎥 Virtual camera started at {width}x{height} (mono), FPS: {vc_fps}")
    try:
        count = 0
        fps_timer = time.time()
        while True:
            frame_ready.wait()
            frame_ready.clear()
            if latest_frame is None:
                continue
            cam.send(cv2.cvtColor(latest_frame, cv2.COLOR_GRAY2RGB))
            cam.sleep_until_next_frame()

            count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                print(f"FPS: {count}")
                count = 0
                fps_timer = now

    except KeyboardInterrupt:
        print("🛑 Stopped by user")
    finally:
        stop_flag = True
        cap_thread.join()
        ueye.is_StopLiveVideo(hCam, ueye.IS_FORCE_VIDEO_STOP)
        ueye.is_ExitCamera(hCam)
