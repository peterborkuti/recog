import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# https://www.youtube.com/live/FmtwcqKdXKQ
# all the code here is from Dr. Satya Mallick's colab

import cv2, queue, threading, time
from collections import deque

from image_processor import ImageProcessor
from image_utils import process_image
from video_capture import VideoCapture

def resizeImg(img):
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    return img

'''return [end_prog, stable]'''
def check_changes(img, img_old):
    absdiff = np.average(cv2.absdiff(img, img_old))
    print(absdiff)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        return [True, None]
    if absdiff >= 5:
        print('image is changing')
        return [False, False]

    return [False, True]

def main():
    USE_CAM = not len(sys.argv)>1
    prev_stable = False
    imageProcessor = ImageProcessor()

    if USE_CAM:
        cap = VideoCapture()
        ret, img_old = cap.read()
        if not ret:
            print('Image is unavailable')
            return

    while True:
        time.sleep(0.1)
        if USE_CAM:
            again, valid_final_image, final_image, images = imageProcessor.pop()
            for idx, im in enumerate(images):
                cv2.imshow('step-%d' % idx, im)
            if valid_final_image:
                cv2.imshow('output', final_image)
            ret, img = cap.read()
            if not ret:
                print('Image is unavailable')
                break
            else:
                cv2.imshow('input', img)

            end_prog, stable = check_changes(img, img_old)
            img_old = img.copy()
            if end_prog:
                break
            if stable and (not prev_stable or again):
                imageProcessor.put(img.copy())
        else:
            img = cv2.imread('img22.jpg', cv2.IMREAD_COLOR)
            img = resizeImg(img)
            final_img = process_image(img)
            cv2.imshow('output', final_img)

        if not USE_CAM:
            cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()