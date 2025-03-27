import cv2
import os
import time

# Camera name
# camera_name = 'Misumi_200x200p'
# camera_name = "Misumi_400x380p"
# camera_name = "videoscope_1280x720p"
# camera_name = "camright_640x480p"
camera_name = "camleft_640x480p"
camera_index = 0  # IMPORTANT: check if it's correct every time you run the code

print("IMPORTANT: Check if the camera index is correct every time you run the code.")

# Settings
save_dir = "calibration_images_" + camera_name

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Camera settings

# camera_index = "rtsp://:@192.168.1.1:8554/session0.mpg"
capture_interval = 0.2  # Time in seconds between captures
setting_time = 10  # Time in seconds to adjust camera settings
total_images = 20  # Total number of images to capture

# Ask user for input source
source = input(
    "Press 'v' to use a video file, anything else will use the live stream: "
)
if source.lower() == "v":
    video_file = input("Enter path to video file: ")
    camera_index = video_file
else:
    print("Using live stream.")

# Open the source
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open source.")
    exit()

# Show the source feed to adjust settings for setting_time seconds
print(f"Adjust settings for {setting_time} seconds.")
for i in range(setting_time):
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    cv2.imshow("Feed", frame)
    cv2.waitKey(1000)  # Display the image for 1000 ms
    print(f"Setting time remaining: {setting_time - i - 1} seconds.")

print(
    f"Starting capture. Capturing {total_images} images every {capture_interval} second(s)."
)

# Capture images
for i in range(total_images):
    # Flush the buffer to ensure the most recent frame is captured
    for _ in range(
        20
    ):  # Adjust number as necessary based on how often your stream updates
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Save the captured frame
    img_filename = os.path.join(save_dir, f"image_{i+1}.jpg")
    cv2.imwrite(img_filename, frame)
    print(f"Captured {img_filename}")

    # Show the frame (optional)
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(500)  # Display the image for 500 ms

    # Wait for the specified interval
    time.sleep(capture_interval)

# Release the source and close windows
cap.release()
cv2.destroyAllWindows()

print("Image capture complete.")
