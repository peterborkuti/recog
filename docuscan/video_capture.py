# https://www.youtube.com/live/FmtwcqKdXKQ
# all the code here is from Dr. Satya Mallick's colab

import cv2, threading, time
from queue import LifoQueue, Empty

# bufferless VideoCapture
class VideoCapture:

    def __init__(self):
        self.q = LifoQueue()
        self.cap = self._camera_on()
        self.lock = threading.Lock()
        self.exit = False
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _camera_on(self):
        print("OpenCV version:", cv2.__version__)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # May work: capture always the latest image
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, 12)
        # cap.set(cv2.CAP_PROP_CONTRAST , 34)
        # cap.set(cv2.CAP_PROP_SATURATION , 33)
        # cap.set(cv2.CAP_PROP_HUE , -72)
        # cap.set(cv2.CAP_PROP_AUTO_WB , 1) # Enable ??
        # cap.set(cv2.CAP_PROP_GAMMA  , 256)
        # cap.set(cv2.CAP_PROP_SHARPNESS  , 100)
        # cap.set(cv2.CAP_PROP_BACKLIGHT , 0)
        # cap.set(cv2.CAP_PROP_AUTOFOCUS  , 0)
        # cap.set(cv2.CAP_PROP_FOCUS  , 232)
        # cap.set(cv2.CAP_PROP_TILT  , 0)
        return cap

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        print('webcam reader started')
        while True:
                ret, img = self.cap.read()
                if not ret or self.exit:
                    break
                self.q.put(img)
                time.sleep(1.0/30.0)
        self.cap.release()

    def read(self):
        try:
            img = self.q.get(1)
            with self.q.mutex:
                self.q.queue.clear()
            return [True, img]
        except Empty:
            return [False, None]

    def release(self):
        with self.lock:
            self.exit = True

