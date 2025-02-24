import cv2
import os

# Test each video device
for i in range(6):
    cap = cv2.VideoCapture(i)
    if os.name == "nt":
        device_path = f"Video Device {i}"
    else:
        device_path = os.path.join("/dev", f"video{i}")

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            print(f"Device {device_path} is working")
        else:
            print(f"Device {device_path} is not returning frames")
        cap.release()
    else:
        print(f"Device {device_path} could not be opened")

cv2.waitKey(0)
cv2.destroyAllWindows()
