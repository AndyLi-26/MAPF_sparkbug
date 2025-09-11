import cv2, time
import numpy as np
LIGHT=True
MIN_DIST=90
SATURATION_GAIN=2.5
SATURATION_CUTOFF=100

RADIUS=110
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

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=MIN_DIST,
        param1=30,
        param2=70,
        minRadius=40,
        maxRadius=150
    )
    return circles

def fit_dark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.where(gray < SATURATION_CUTOFF, 0, gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=MIN_DIST,
        param1=30,
        param2=70,
        minRadius=40,
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
    # Open selected camera
    if camera_index==-1:
        cap = cv2.VideoCapture("bright.mov" if LIGHT else "dark.mov")
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return
    total_frame=0
    start = time.time()
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

        total_frame+=1


        if LIGHT:
            circles=fit_bright(frame)
        else:
            circles=fit_dark(frame)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(frame, (x, y), RADIUS, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                #print(x,y,r)
                minx=min(minx,x)
                maxx=max(maxx,x)
                miny=min(miny,y)
                maxy=max(maxy,y)
                print(f"{total_frame}, ({x},{y})")
                if total_frame>100:
                    print(f"({minx},{miny}), ({maxx},{maxy})")
                    print(f"dif: ({maxx-minx},{maxy-miny})")
                    exit()

        cv2.imshow("Circle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    '''
    cams = list_cameras()
    if not cams:
        print("No cameras found!")
        exit()
    print("Available cameras:")
    for idx in cams:
        print(f"  {idx}")
    choice = int(input("Select camera index: "))
    detect_circles(choice)
    '''
    detect_circles(-1)
