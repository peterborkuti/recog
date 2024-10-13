import queue, threading, cv2
import time

from image_utils import process_image



class ImageProcessor:
    def __init__(self):
        self.in_q = queue.Queue(1)
        self.out_q = queue.Queue()
        self.lock = threading.Lock()
        t = threading.Thread(target=self._processor)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _processor(self):
        while True:
            img = self.in_q.get(True)
            print('Start processing')
            again, final_img, images = process_image(img)
            if again:
                print('processing failed')
            else:
                print('processing done.')
            self.out_q.put([again, True, final_img, images])
            time.sleep(0.01)

    def put(self, img):
        with self.lock:
            try:
                self.in_q.get(False)
            except queue.Empty:
                pass
            self.in_q.put(img)
    
    def pop(self):
        try:
            return self.out_q.get(False)
        except queue.Empty:
            return False, False, None, []
