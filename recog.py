import cv2
import numpy as np
import json
from enum import Enum
from typing import TypedDict
import functools

# Function to detect shapes


def transform(frame):
    frame = cv2.flip(frame, -1)
    return frame


# BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class Shape:
    coord: tuple[int, int]
    params: list[int]

    def correct(self, l: list[float]) -> list[int]:
        return list(map(lambda x: (int(x)//5)*5, l))

    def lists_equals(l1: list[int], l2: list[int]) -> bool:
        if len(l1) != len(l2):
            return False

    def __init__(self, x: float, y: float, args: list[float]) -> None:
        self.coord = self.correct([x, y])
        self.params = self.correct(args)

    def __eq__(self, other):
        if isinstance(other, Shape):
            l1 = list(self.coord) + self.params
            l2 = list(other.coord) + other.params
            return functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, l1, l2), True)

        return False


class Circle(Shape):
    def __init__(self, x: float, y: float, r: float) -> None:
        super().__init__(x, y, [r])

    def __eq__(self, other):
        if isinstance(other, Circle):
            return super().__eq__(other)

        return False


class StoredShape(TypedDict):
    shape: Shape
    probability: float
    updated: bool


circles: list[StoredShape] = []


def found_circle(data: tuple[any, float]) -> None:
    (x, y), r = data
    circle = Circle(x, y, r)
    found_index = next(iter(i for (i, c) in enumerate(
        circles) if c['shape'] == circle), None)
    if (found_index == None):
        circles.append({'shape': circle, 'probability': 0.1, 'updated': True})
    else:
        sCircle = circles[found_index]
        sCircle['probability'] = (1.0 + sCircle['probability']) / 2
        sCircle['updated'] = True


def update_not_found_circles() -> None:
    for i, sShape in enumerate(circles):
        if not sShape['updated']:
            sShape['probability'] /= 2.0
        sShape['updated'] = False

    filteredCircles = list(filter(lambda ss: ss['probability'] >= 0.1, circles))
    circles.clear()
    circles.extend(filteredCircles)


def draw_circles(frame) -> None:
    print(circles)
    for sCircle in circles:
        circle = sCircle['shape']
        cv2.circle(frame, circle.coord, circle.params[0], RED, 2)


def detect_shapes(frame):
    # Convert to grayscale
    # Apply GaussianBlur to reduce noise and improve contour detection
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform Canny edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # (treshold, bw) = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 20, 100)

    # Find contours in the edges image
    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to reduce the number of points
        approx = cv2.approxPolyDP(
            contour, 0.04 * cv2.arcLength(contour, True), True)
        # Draw the detected contours (for visualization)
        cv2.drawContours(frame, [approx], 0, GREEN, 2)

        # Check if the shape is a rectangle
        if len(approx) == 4:
            # Compute the bounding box of the contour and draw it
            (x, y, w, h) = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, 2)
        # Check for circles
        elif len(approx) > 4:
            # Calculate the center and radius of the contour
            found_circle(cv2.minEnclosingCircle(contour))

    update_not_found_circles()
    draw_circles(frame)

    return frame


def flip(frame):
    return cv2.flip(frame, -1)


def undistort(img, mapx, mapy, roi):
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]


def camera_on():
    print("OpenCV version:", cv2.__version__)
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
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


def load_cam_params(cap):
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

    ret, img = cap.read()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), 5)

    return (mapx, mapy, roi)


def main():
    # Initialize webcam

    cap = camera_on()
    mapx, mapy, roi = load_cam_params(cap)
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        frame = undistort(frame, mapx, mapy, roi)

        # Detect shapes in the frame
        frame = detect_shapes(frame)

        # frame = transform(frame)
        # frame = detect_shapes(frame)

        # Display the output frame
        cv2.imshow('Shape Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
