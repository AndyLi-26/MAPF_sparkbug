import cv2
import numpy as np
SATURATION_GAIN=2.5
SATURATION_CUTOFF=100
RADIUS=110
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
    #print("ori shape",ori_shape)
    new_shape=int(ori_shape[0]/block_size[0]),int(ori_shape[1]/block_size[1])
    #print("new shape",new_shape)
    retval = img.reshape(new_shape[0],block_size[0],new_shape[1],block_size[1])
    temp1=retval.mean(axis=(1,3))
    temp1= np.where(temp1 < SATURATION_CUTOFF, 0, temp1)
    #print(temp1.max())

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
        cap = cv2.VideoCapture("bright.mov")
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break


        hsv_float = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#.astype(np.float32)
        hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1] * SATURATION_GAIN, 0, 255)
        hsv_float[:, :, 1] = np.where(hsv_float[:,:,1] < SATURATION_CUTOFF, 0, hsv_float[:,:,1])
        #hsv_float[:, :, 2] = np.clip(hsv_float[:, :, 2] * VALUE_GAIN, 0, 255)
        hsv_img = hsv_float.astype(np.uint8)
        img_proc = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
        #gray = re_sample(hsv_float[:,:,1],(8,8)).astype(np.uint8)
        gray = hsv_float[:,:,1]
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        cv2.imshow("detecting on", blurred)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=40,
            param2=70,
            minRadius=40,
            maxRadius=150
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(frame, (x, y), RADIUS, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                print(x,y,r)

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
