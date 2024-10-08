import cv2
import numpy as np
import json
from typing import TypedDict
import functools
from abc import ABC, abstractmethod #Abstract Base Classes


def transform(frame):
    frame = cv2.flip(frame, -1)
    return frame

def quantize(threshold: int, x):
    return (int(x)//threshold) * threshold

# BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

class Shape(ABC):
    params: any
    quantizer: any

    def correct(self, l: any) -> any:
        return self.quantizer(l)

    def __init__(self, args: any) -> None:
        self.params = self.correct(args)

    def __eq__(self, other):
        if isinstance(other, Shape):
            if self.params.shape != other.params.shape:
                return False

            return (self.params == other.params).all()

        return False

    @abstractmethod
    def draw(self, image, intensity: int):
        pass


class Circle(Shape):
    def __init__(self, x: float, y: float, r: float) -> None:
        self.quantizer = np.vectorize(functools.partial(quantize, 5))
        super().__init__(np.array([x, y, r]))

    def __eq__(self, other):
        if isinstance(other, Circle):
            return super().__eq__(other)

        return False

    def draw(self, image, intensity):
        cv2.circle(image, (self.params[0], self.params[1]),
                   self.params[2], (0, 0, intensity), 2)


class Rectangle(Shape):
    def __init__(self, params) -> None:
        self.quantizer = np.vectorize(functools.partial(quantize, 70))
        super().__init__(params)

    def __eq__(self, other):
        if isinstance(other, Rectangle):
            return super().__eq__(other)

        return False

    def draw(self, image, intensity):
        cv2.drawContours(image, [self.params], 0, (0, 0, intensity), 2)


class StoredShape(TypedDict):
    shape: Shape
    probability: float
    updated: bool


shapes: list[StoredShape] = []


def update_detected_shape(shape: Shape):
    found_index = next(iter(i for (i, s) in enumerate(
        shapes) if s['shape'] == shape), None)
    if (found_index == None):
        print('add shape %s' % type(shape))
        print(shape.params)
        shapes.append({'shape': shape, 'probability': 0.1, 'updated': True})
    else:
        print('increase probablitity of %s' % type(shape))
        sShape = shapes[found_index]
        sShape['probability'] = (1.0 + sShape['probability']) / 2
        sShape['updated'] = True


def found_circle(data: tuple[any, float]) -> None:
    (x, y), r = data
    if r < 5:
        return

    update_detected_shape(Circle(x, y, r))


def found_rectangle(box) -> None:
    update_detected_shape(Rectangle(box))


def update_not_found_shapes() -> None:
    for i, sShape in enumerate(shapes):
        if not sShape['updated']:
            sShape['probability'] /= 2.0
        sShape['updated'] = False

    filteredShapes = list(filter(lambda ss: ss['probability'] >= 0.1, shapes))
    shapes.clear()
    shapes.extend(filteredShapes)


def draw_shapes(frame):
    height, width, channels = frame.shape
    blank_image = np.zeros((height, width, 3), np.uint8)

    for sShape in shapes:
        shape = sShape['shape']
        prob = sShape['probability']
        if prob < 0.7:
            continue
        intensity = int(prob * 255)
        shape.draw(blank_image, intensity)

    return blank_image


def detect_shapes(frame):
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
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)  # Get 4 corners of the rectangle
            found_rectangle(box)
        # Check for circles
        elif len(approx) > 4:
            # Calculate the center and radius of the contour
            found_circle(cv2.minEnclosingCircle(contour))

    update_not_found_shapes()

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
    cap = camera_on()
    mapx, mapy, roi = load_cam_params(cap)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = undistort(frame, mapx, mapy, roi)

        frame = detect_shapes(frame)
        cv2.imshow('Input', frame)

        cv2.imshow('Output', draw_shapes(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
