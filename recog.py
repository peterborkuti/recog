import cv2
import numpy as np

# Function to detect shapes


def detect_shapes(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edges image
    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to reduce the number of points
        approx = cv2.approxPolyDP(
            contour, 0.04 * cv2.arcLength(contour, True), True)
        # Draw the detected contours (for visualization)
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

        # Check if the shape is a rectangle
        if len(approx) == 4:
            # Compute the bounding box of the contour and draw it
            (x, y, w, h) = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Check for circles
        elif len(approx) > 4:
            # Calculate the center and radius of the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (255, 0, 0), 2)

    return frame


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Detect shapes in the frame
        frame = detect_shapes(frame)

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
