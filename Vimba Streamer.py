import copy
import cv2
import threading
import queue
import numpy
import pyvirtualcam

from typing import Optional
from vimba import *

FRAME_QUEUE_SIZE = 10

def print_preamble():
    print('////////////////////////////////////////////')
    print('/// Vimba API Multithreading Example ///////')
    print('////////////////////////////////////////////\n')
    print(flush=True)

def add_camera_id(frame: Frame, cam_id: str) -> Frame:
    cv2.putText(frame.as_opencv_image(), 'Cam: {}'.format(cam_id), org=(0, 30), fontScale=1,
                color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return frame

def get_max_resolution(cam: Camera):
    """Return max width, height supported by the camera."""
    width_feat = cam.get_feature_by_name('Width')
    height_feat = cam.get_feature_by_name('Height')
    max_width = width_feat.get_range()[1]
    max_height = height_feat.get_range()[1]
    return max_width, max_height

def resize_if_required(frame: Frame, width, height) -> numpy.ndarray:
    cv_frame = frame.as_opencv_image()
    if (frame.get_height() != height) or (frame.get_width() != width):
        cv_frame = cv2.resize(cv_frame, (width, height), interpolation=cv2.INTER_AREA)
        cv_frame = cv_frame[..., numpy.newaxis]
    return cv_frame

def create_dummy_frame(width, height) -> numpy.ndarray:
    cv_frame = numpy.zeros((height, width, 1), numpy.uint8)
    cv2.putText(cv_frame, 'No Stream available.', org=(30, 60),
                fontScale=1, color=255, thickness=2, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    cv_frame_bgr = cv2.cvtColor(cv_frame, cv2.COLOR_GRAY2BGR)
    return cv_frame_bgr

def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass

def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    feat = cam.get_feature_by_name(feat_name)
    try:
        feat.set(feat_value)
    except VimbaFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()
        if feat_value <= min_:
            val = min_
        elif feat_value >= max_:
            val = max_
        else:
            val = (((feat_value - min_) // inc) * inc) + min_
        feat.set(val)
        msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
               'Using nearest valid value \'{}\'. Note that, this causes resizing '
               'during processing, reducing the frame rate.')
        Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))

class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()
        self.width = None
        self.height = None

    def __call__(self, cam: Camera, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                try_put_frame(self.frame_queue, cam, frame_cpy)
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        max_width, max_height = get_max_resolution(self.cam)
        set_nearest_value(self.cam, 'Width', max_width)
        set_nearest_value(self.cam, 'Height', max_height)
        self.width = max_width
        self.height = max_height
        try:
            self.cam.ExposureAuto.set('Once')
        except (AttributeError, VimbaFeatureError):
            self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(
                          self.cam.get_id()))
        self.cam.set_pixel_format(PixelFormat.Mono8)

    def run(self):
        self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))
        try:
            with self.cam:
                self.setup_camera()
                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                finally:
                    self.cam.stop_streaming()
        except VimbaCameraError:
            pass
        finally:
            try_put_frame(self.frame_queue, self.cam, None)
        self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))

class FrameConsumer(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, width: int, height: int):
        threading.Thread.__init__(self)
        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.width = width
        self.height = height

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit'
        KEY_CODE_ENTER = 13
        frames = {}
        alive = True
        self.log.info('Thread \'FrameConsumer\' started.')
        fps = 31  # Set to your camera FPS
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=fps, print_fps=False) as vcam:
            print(f"Virtual camera started ({vcam.device})")
            while alive:
                frames_left = self.frame_queue.qsize()
                while frames_left:
                    try:
                        cam_id, frame = self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                    if frame:
                        frames[cam_id] = frame
                    else:
                        frames.pop(cam_id, None)
                    frames_left -= 1
                if frames:
                    cv_images = [resize_if_required(frames[cam_id], self.width, self.height) for cam_id in sorted(frames.keys())]
                    image = numpy.concatenate(cv_images, axis=1)
                    if image.ndim == 2 or image.shape[2] == 1:
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    else:
                        image_bgr = image
                    cv2.imshow(IMAGE_CAPTION, image)
                    vcam.send(image_bgr)
                    vcam.sleep_until_next_frame()
                else:
                    dummy_bgr = create_dummy_frame(self.width, self.height)
                    cv2.imshow(IMAGE_CAPTION, dummy_bgr)
                    vcam.send(dummy_bgr)
                    vcam.sleep_until_next_frame()
                if KEY_CODE_ENTER == cv2.waitKey(10):
                    cv2.destroyAllWindows()
                    alive = False
        self.log.info('Thread \'FrameConsumer\' terminated.')

class MainThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def __call__(self, cam: Camera, event: CameraEvent):
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()

    def run(self):
        log = Log.get_instance()
        vimba = Vimba.get_instance()
        vimba.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)
        log.info('Thread \'MainThread\' started.')
        with vimba:
            # Setup all producers first and get max width/height (using first camera)
            cams = vimba.get_all_cameras()
            if not cams:
                print('No cameras found.')
                return
            for cam in cams:
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer
                # Setup camera to get correct resolution (before consumer is created)
                with cam:
                    producer.setup_camera()
            # Assume all cameras have the same resolution for consumer
            width = self.producers[cams[0].get_id()].width
            height = self.producers[cams[0].get_id()].height
            consumer = FrameConsumer(self.frame_queue, width, height)
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()
            vimba.register_camera_change_handler(self)
            consumer.start()
            consumer.join()
            vimba.unregister_camera_change_handler(self)
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.stop()
                for producer in self.producers.values():
                    producer.join()
        log.info('Thread \'MainThread\' terminated.')

if __name__ == '__main__':
    print_preamble()
    main = MainThread()
    main.start()
    main.join()
