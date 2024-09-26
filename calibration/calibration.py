import numpy as np
import cv2 as cv2
import glob
import json
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# substract 1 from the number of rows, cols
rows = 6
cols = 9
objp = np.zeros((rows*cols,3), np.float32)
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for fname in images:
    print(fname)
    lwr = np.array([0, 0, 124])
    upr = np.array([140, 92, 255])
    img = cv2.imread(fname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, lwr, upr)

    # Extract chess-board
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = cv2.bitwise_and(dlt, msk)

# Displaying chess-board features
    #res = np.uint8(res)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(res, (rows, cols), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(res,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)
        cv2.imshow('img', img)
        # cv2.waitKey(5000)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, res.shape[::-1], None, None)

with open('ret.json', 'w') as fjson:
    json.dump(ret, fjson)
with open('mtx.json', 'w') as fjson:
    json.dump(mtx.tolist(), fjson)
with open('dist.json', 'w') as fjson:
    json.dump(dist.tolist(), fjson)
with open('rvecs.json', 'w') as fjson:
    json.dump(list(map(lambda x: x.tolist(), rvecs)), fjson)
with open('tvecs.json', 'w') as fjson:
    json.dump(list(map(lambda x: x.tolist(),tvecs)), fjson)


with open('ret.json', 'r') as fjson:
    ret = json.load(fjson)
with open('mtx.json', 'r') as fjson:
    mtx = np.array(json.load(fjson))
with open('dist.json', 'r') as fjson:
    dist = np.array(json.load(fjson))
with open('rvecs.json', 'r') as fjson:
    rvecs = list(map(lambda x: np.array(x), json.load(fjson)))
with open('tvecs.json', 'r') as fjson:
    tvecs = list(map(lambda x: np.array(x), json.load(fjson)))

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv2.imshow('img', dst)
    cv2.waitKey(5000)

cv2.destroyAllWindows()

