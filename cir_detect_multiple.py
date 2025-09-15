import cv2, time, sys
import numpy as np
LOG=False
SHOW=True
LIGHT=int(sys.argv[1])
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
    for i,c in enumerate(circles):
        if LOG:
            print("original pos: ",c)
        (x1,x2,y1,y2)=crop(c,frame)
        if LOG:
            print("cropped cornner",x1,x2,y1,y2)
        cropped_frame=frame[x1:x2,y1:y2]
        retval=fit_circle(cropped_frame)

        if retval is None:
            if LOG:
                print("no circle detected")
        else:
            retval = np.uint16(np.around(retval))
            if LOG:
                print(f"got {len(retval)} circle detected")
                print("from detection",retval)
            x,y,_=retval[0][0]
            circles[i]=(x+y1,y+x1)

        if SHOW:
            if retval is not None:
                for (x, y, r) in retval[0, :]:
                    cv2.circle(cropped_frame, (x, y), RADIUS, (255, 255, 0), 2)
                    cv2.circle(cropped_frame, (x, y), 2, (0, 0, 255), 3)
            cv2.imshow(f"circle{i}",cropped_frame)

def fit_circle(frame):
    if LIGHT:
        return fit_bright(frame)
    else:
        return fit_dark(frame)

def crop(coord,frame):
    x1, x2 = max(0, coord[0] - int(RADIUS*1.5)), min(frame.shape[0], int(coord[0] + RADIUS*1.5))
    y1, y2 = max(0, coord[1] - int(RADIUS*1.5)), min(frame.shape[1], int(coord[1] + RADIUS*1.5))
    #print(x1,x2,y1,y2)
    return (y1,y2,x1,x2)
    return frame[y1:y2, x1:x2]

def color_detection(img,coord):
    global frame
    colors="RGB"
    print("detected colors: ",colors)

def click_event(event, x, y, flags, param):
    global circles,setup
    if setup: return
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords = (x, y)
        circles.append((x,y))
        imshow()
        if LOG:
            print("Clicked:", clicked_coords)

def imshow():
    global circles,new_circles,frame
    for x,y in new_circles:
        if LOG:
            print(x,y)
        cv2.circle(frame, (x, y), RADIUS, (255, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    for x,y in circles:
        if LOG:
            print(x,y)
        cv2.circle(frame, (x, y), RADIUS, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    cv2.imshow("Circle Detection", frame)

def imshow_Brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.where(gray < SATURATION_CUTOFF, 0, gray)
    cv2.imshow("bright",gray)

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.where(gray < SATURATION_CUTOFF, 0, gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    if SHOW:
        cv2.imshow("bright",blurred)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=MIN_DIST,
        param1=10,
        param2=35,
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
    #new_img= np.where(new_img < SATURATION_CUTOFF, 0, new_img)
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
        cap = cv2.VideoCapture("bright_1080.mov" if LIGHT else "dark_1080.mov")
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return
    total_frame=0
    start=0
    minx=miny=5000
    maxx=maxy=1000
    #with open("circles_bright.csv" if LIGHT else "circles_dark.csv",'w') as f:
    while True:
        ret, frame = cap.read()
        if not ret:
            runtime=time.time()-start
            print(f"fps: {total_frame/runtime}")
            print("Failed to grab frame")
            break

        ori_frame=frame.copy()
        if LOG:
            print("----------------------------------------------------")
        total_frame+=1
        if not setup:
            if LOG:
                print("frame size: ",frame.shape)
            temp=fit_circle(frame)
            if temp is None:
                if LOG:
                    print("no init circle")
            if temp is not None:
                temp = np.uint16(np.around(temp))
                for (x, y, r) in temp[0, :]:
                    new_circles.append((x,y))
                    if LOG:
                        print("detected circle pos:",x,y)
                    cv2.circle(frame, (x, y), RADIUS, (255, 255, 0), 2)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            cv2.imshow("Circle Detection", frame)
            cv2.setMouseCallback("Circle Detection", click_event)

            key= cv2.waitKey(0) & 0xFF
            if key == 13:   # Enter key ASCII code
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
        imshow()

        key= cv2.waitKey(1) & 0xFF
        if key == 8:  # Backspace
            circles.pop(-1)
            print("Backspace")
        elif key == 27: # Escape key to quit
            print("Escape")
            exit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cams = list_cameras()
    if not cams:
        print("No cameras found!")
        exit()
    print("Available cameras:")
    for idx in cams:
        print(f"  {idx}")
    choice = int(input("Select camera index: "))
    detect_circles(choice)
    #detect_circles(-1)
