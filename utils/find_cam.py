import cv2

# Test each video device
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            print(f"Device /dev/video{i} is working")
        else:
            print(f"Device /dev/video{i} is not returning frames")
        cap.release()
    else:
        print(f"Device /dev/video{i} could not be opened")

cv2.waitKey(0)
cv2.destroyAllWindows()
