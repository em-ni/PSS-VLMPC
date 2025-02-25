import cv2
import numpy as np
import sys
import os


def main(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a color range for yellow
    lower_yellow = np.array([23, 88, 0])
    upper_yellow = np.array([36, 254, 255])

    # Define the color ranges for red
    lower_red1 = np.array([0, 55, 0])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([171, 55, 0])
    upper_red2 = np.array([180, 255, 255])

    # Define the color range for blue
    lower_blue = np.array([100, 100, 50])  # Increase saturation
    upper_blue = np.array([130, 255, 255])  # Narrow the hue range

    # Create masks for yellow, red, and blue colors
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Combine the red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Bitwise-AND mask and original image to extract yellow, red, and blue colors
    yellow_detected = cv2.bitwise_and(image, image, mask=mask_yellow)
    red_detected = cv2.bitwise_and(image, image, mask=mask_red)
    blue_detected = cv2.bitwise_and(image, image, mask=mask_blue)

    # Show the original image, yellow detection, red detection, and blue detection side by side
    result_yellow = np.hstack((image, yellow_detected))
    result_red = np.hstack((image, red_detected))
    result_blue = np.hstack((image, blue_detected))

    # Display the resulting frames
    cv2.imshow("Yellow color detection", result_yellow)
    cv2.imshow("Red color detection", result_red)
    cv2.imshow("Blue color detection", result_blue)

    # Save the result images
    output_dir = os.path.join("data", "test")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, "yellow_detection_result.png"), result_yellow)
    cv2.imwrite(os.path.join(output_dir, "red_detection_result.png"), result_red)
    cv2.imwrite(os.path.join(output_dir, "blue_detection_result.png"), result_blue)

    # Wait for the ESC key (27) to close the windows
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide an image path.")
