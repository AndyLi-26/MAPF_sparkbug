import cv2, time, sys
import numpy as np

LOG=False
SHOW=True
# LIGHT=int(sys.argv[1])
LIGHT=False
RADIUS=60 #60 when in 1080, 100 when in 4k
MIN_DIST=RADIUS*1.8
SATURATION_GAIN=2.5
SATURATION_CUTOFF=130
setup=False
clicked_coords = None
frame=None
circles=[]
new_circles=[]

def update_circles(frame):
    global circles
    for i, c in enumerate(circles):
        if LOG:
            print("original pos: ", c)

        # Safe crop bounds (rows=y, cols=x)
        y1, y2, x1, x2 = crop(c, frame)
        if LOG:
            print("cropped corner (y1,y2,x1,x2):", y1, y2, x1, x2)

        # Guard against empty/degenerate crops
        if (y2 - y1) <= 1 or (x2 - x1) <= 1:
            if LOG:
                print("Skipping empty/too-small crop")
            continue

        cropped_frame = frame[y1:y2, x1:x2]

        # Another guard: ensure crop is non-empty
        if cropped_frame is None or cropped_frame.size == 0:
            if LOG:
                print("Skipping: cropped frame is empty")
            continue

        retval = fit_circle(cropped_frame)

        if retval is None:
            if LOG:
                print("no circle detected")
        else:
            retval = np.uint16(np.around(retval))
            if LOG:
                print(f"got {len(retval[0])} circle(s) detected")
                print("from detection", retval)
            x, y, _ = retval[0][0]

            # Convert to global coordinates (x is col, y is row)
            gx = int(x) + int(x1)
            gy = int(y) + int(y1)
            circles[i] = (gx, gy)

        if SHOW:
            vis = cropped_frame.copy()
            hsv = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]

            hue_color = cv2.applyColorMap(cv2.convertScaleAbs(hue, alpha=255/179.0), cv2.COLORMAP_HSV)
            if retval is not None:
                for (x, y, r) in retval[0, :]:
                    cv2.circle(vis, (int(x), int(y)), int(RADIUS), (255, 255, 0), 2)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 3)
            cv2.imshow(f"circle{i}", hue_color)

def fit_circle(frame):
    if LIGHT:
        return fit_bright(frame)
    else:
        # Guard here too, just in case
        if frame is None or frame.size == 0:
            return None
        return fit_dark(frame)

def crop(coord, frame):
    """
    Compute a safe crop window around (x,y) with half-size ~ 1.5*RADIUS.
    Returns (y1, y2, x1, x2) for direct slicing: frame[y1:y2, x1:x2]
    """
    # Ensure plain ints to avoid uint16 wrap-around
    cx = int(coord[0])
    cy = int(coord[1])

    h = int(frame.shape[0])
    w = int(frame.shape[1])

    half = int(RADIUS * 1.5)

    # x are columns (0..w), y are rows (0..h)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)

    # Ensure monotonicity
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    return (y1, y2, x1, x2)

def color_detection(img,coord):
    global frame
    colors="RGB"
    print("detected colors: ",colors)

def click_event(event, x, y, flags, param):
    global circles, setup
    if setup: return
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = (int(x), int(y))
        circles.append(clicked)
        imshow()
        if LOG:
            print("Clicked:", clicked)

def imshow():
    global circles,new_circles,frame
    if frame is None or frame.size == 0:
        return
    vis = frame.copy()
    for x, y in new_circles:
        if LOG:
            print(x, y)
        cv2.circle(vis, (int(x), int(y)), int(RADIUS), (255, 255, 0), 2)
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 3)
    for x, y in circles:
        if LOG:
            print(x, y)
        cv2.circle(vis, (int(x), int(y)), int(RADIUS), (0, 255, 0), 2)
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 3)
    cv2.imshow("Circle Detection", vis)

def imshow_Brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.where(gray < SATURATION_CUTOFF, 0, gray)
    cv2.imshow("bright",gray)

def imshow_Hue(frame):
    """
    Show the hue channel (0-179 in OpenCV).
    Useful for debugging LED color detection.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]

    # Optional: apply a colormap so you can see hue variations better
    hue_color = cv2.applyColorMap(cv2.convertScaleAbs(hue, alpha=255/179.0), cv2.COLORMAP_HSV)

    # cv2.imshow("hue_raw", hue)        # grayscale hue channel
    cv2.imshow("hue_colored", hue_color)  # colored version for easier visualization


def imshow_Saturation(frame):
    hsv_float = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#.astype(np.float32)
    hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1] * SATURATION_GAIN, 0, 255)
    hsv_float[:, :, 1] = np.where(hsv_float[:,:,1] < SATURATION_CUTOFF, 0, hsv_float[:,:,1])
    gray = hsv_float[:,:,1]
    cv2.imshow("bright",gray)

def imshow_RGB(frame):
    blue=frame.copy()
    blue[:,:,1]=0
    blue[:,:,2]=0
    cv2.imshow("blue", blue)
    red=frame.copy()
    red[:,:,1]=0
    red[:,:,0]=0
    cv2.imshow("red", red)
    green=frame.copy()
    green[:,:,2]=0
    green[:,:,0]=0
    cv2.imshow("green", green)

def fit_bright(frame):
    if frame is None or frame.size == 0:
        return None
    hsv_float = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#.astype(np.float32)
    hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1] * SATURATION_GAIN, 0, 255)
    hsv_float[:, :, 1] = np.where(hsv_float[:,:,1] < SATURATION_CUTOFF, 0, hsv_float[:,:,1])
    gray = hsv_float[:,:,1]
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    cv2.imshow("saturation",blurred)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=MIN_DIST,
        param1=30,
        param2=60,
        minRadius=10,
        maxRadius=150
    )
    return circles

def fit_dark(frame):
    if frame is None or frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.where(gray < SATURATION_CUTOFF, 0, gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    
    if SHOW:
        cv2.imshow("bright", blurred)
        

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=MIN_DIST,
        param1=10,
        param2=25,
        minRadius=8,
        maxRadius=150
    )
    return circles

def re_sample(img,block_size):
    ori_size=img.shape
    new_img=down_sample(img,block_size).astype(np.uint8)
    new_img= np.where(new_img < SATURATION_CUTOFF, 0, new_img)
    new_img=up_sample(new_img,ori_size[::-1])
    return new_img

def up_sample(img, new_size):
    #print(img.shape,new_size)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

def down_sample(img,block_size):
    ori_shape=img.shape
    assert len(ori_shape)==2,"this only down_sample monocolor image"
    new_shape=int(ori_shape[0]/block_size[0]),int(ori_shape[1]/block_size[1])
    retval = img.reshape(new_shape[0],block_size[0],new_shape[1],block_size[1])
    temp1=retval.mean(axis=(1,3))
    temp1= np.where(temp1 < SATURATION_CUTOFF, 0, temp1)
    return temp1 * retval.sum(axis=(1,3))

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
    global frame, circles, setup, new_circles
    # Open selected camera
    if camera_index==-1:
        # cap = cv2.VideoCapture("bright.mov" if LIGHT else "dark_mult_2.mov")
        cap = cv2.VideoCapture("bright.mov" if LIGHT else "dark_multiple.mov")
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    total_frame=0
    start=0
    #with open("circles_bright.csv" if LIGHT else "circles_dark.csv",'w') as f:
    while True:
        ret, frame = cap.read()
        if not ret:
            if start != 0:
                runtime=time.time()-start
                if runtime > 0:
                    print(f"fps: {total_frame/runtime}")
            print("Failed to grab frame")
            break

        if LOG:
            print("----------------------------------------------------")
        total_frame+=1

        if not setup:
            if LOG:
                print("frame size: ", frame.shape)
            temp = fit_circle(frame)
            new_circles = []
            if temp is None:
                if LOG:
                    print("no init circle")
            else:
                temp = np.uint16(np.around(temp))
                for (x, y, r) in temp[0, :]:
                    # store as ints to avoid uint16 issues later
                    nx, ny = int(x), int(y)
                    new_circles.append((nx, ny))
                    if LOG:
                        print("detected circle pos:", nx, ny)
                    cv2.circle(frame, (nx, ny), int(RADIUS), (255, 255, 0), 2)
                    cv2.circle(frame, (nx, ny), 2, (0, 0, 255), 3)

            cv2.imshow("Circle Detection", frame)
            cv2.setMouseCallback("Circle Detection", click_event)

            key = cv2.waitKey(0) & 0xFF
            if key == 13:   # Enter
                setup=True
                start = time.time()
                new_circles=[]
                if LOG:
                    print("Enter")

        else:
            if LOG:
                print(f"{total_frame}:{circles}")
            update_circles(frame)

        if SHOW:
            imshow()
            imshow_Hue(frame) 

        key = cv2.waitKey(1) & 0xFF
        if key == 8:  # Backspace
            if circles:
                circles.pop(-1)
                print("Backspace")
        elif key == 27: # Escape key to quit
            print("Escape")
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
        try:
            choice = int(input("Select camera index: "))
        except Exception:
            choice = cams[0]
        detect_circles(choice)
    # detect_circles(-1)
