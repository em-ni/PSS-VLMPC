import cv2


def find_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
            cap.release()
        index += 1
    return arr


# Open a connection to the camera
cap = cv2.VideoCapture(4)  # Change the index if you have multiple cameras

# Capture a single frame to determine the resolution
ret, frame = cap.read()
if ret:
    frame_size = (frame.shape[1], frame.shape[0])
    width = frame.shape[1]
    height = frame.shape[0]
    # Get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame width: {width}")
    print(f"Frame height: {height}")
    print(f"Frame rate: {fps}")
else:
    print("Failed to capture an image")
    # Get a list of available camera indices
    available_cameras = find_available_cameras()
    print(f"Available camera indices: {available_cameras}")

# Release the camera
cap.release()
