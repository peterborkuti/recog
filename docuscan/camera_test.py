import cv2, queue, threading, time

class VideoCapture:

    def __init__(self):
        self.cap = self._camera_on()
        self.lock = threading.Lock()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _camera_on(self):
        print("OpenCV version:", cv2.__version__)

        cap = cv2.VideoCapture(2)
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
        while True:
            with self.lock:
                ret = self.cap.grab()
            time.sleep(1.0/30.0)
            if not ret:
                break

    def read(self):
        with self.lock:
            return self.cap.retrieve()

def main():
    cap = VideoCapture()

    while True:
        ret, img = cap.read()
        cv2.imshow('original image', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()