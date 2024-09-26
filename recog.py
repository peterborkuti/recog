import cv2
import numpy as np
import json

# Function to detect shapes

def transform(frame):
    frame = cv2.flip(frame, -1)
    return frame

#BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)



def detect_shapes(frame):
    # Convert to grayscale
    # Apply GaussianBlur to reduce noise and improve contour detection
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform Canny edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #(treshold, bw) = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
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
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, RED, 2)

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
    #cap.set(cv2.CAP_PROP_BRIGHTNESS, 12)
    #cap.set(cv2.CAP_PROP_CONTRAST , 34)
    #cap.set(cv2.CAP_PROP_SATURATION , 33)
    #cap.set(cv2.CAP_PROP_HUE , -72)
    #cap.set(cv2.CAP_PROP_AUTO_WB , 1) # Enable ??
    #cap.set(cv2.CAP_PROP_GAMMA  , 256)
    #cap.set(cv2.CAP_PROP_SHARPNESS  , 100)
    #cap.set(cv2.CAP_PROP_BACKLIGHT , 0)
    #cap.set(cv2.CAP_PROP_AUTOFOCUS  , 0)
    #cap.set(cv2.CAP_PROP_FOCUS  , 232)
    #cap.set(cv2.CAP_PROP_TILT  , 0)
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
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)

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
        #frame = detect_shapes(frame)

        #frame = transform(frame)
        #frame = detect_shapes(frame)

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
