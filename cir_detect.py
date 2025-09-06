import cv2
import numpy as np

def list_cameras(max_tested=10):
    """Return a list of available camera indices."""
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def detect_circles(camera_index=0):
    # Open selected camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=90,
            minRadius=10,
            maxRadius=200
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        cv2.imshow("Circle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cams = list_cameras()
    if not cams:
        print("No cameras found!")
    else:
        print("Available cameras:")
        for idx in cams:
            print(f"  {idx}")
        choice = int(input("Select camera index: "))
        if choice in cams:
            detect_circles(choice)
        else:
            print("Invalid selection.")
