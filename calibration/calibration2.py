import cv2
import numpy as np

# Load the image
img = cv2.resize(cv2.imread('my_photo-4.jpg'), (0, 0), fx = 1.0, fy = 1.0)

# Color-segmentation to get binary mask
lwr = np.array([0, 0, 124])
upr = np.array([140, 92, 255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
msk = cv2.inRange(hsv, lwr, upr)

# Extract chess-board
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
dlt = cv2.dilate(msk, krn, iterations=5)
res = cv2.bitwise_and(dlt, msk)

# Displaying chess-board features
res = np.uint8(res)
ret, corners = cv2.findChessboardCorners(res, (6, 9),
                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
print(corners)
if ret:
    fnl = cv2.drawChessboardCorners(img, (6, 9), corners, ret)
    cv2.imshow("fnl", fnl)
else:
    print("No Checkerboard Found")
cv2.waitKey(0)
cv2.destroyAllWindows()