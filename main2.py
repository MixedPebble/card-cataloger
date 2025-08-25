import cv2
import numpy as np
import time

background_seperator = None

def run():
    # Initialize video capture (0 = default webcam)
    video_capture = cv2.VideoCapture(0) # or 0 for webcam

    # Create background subtractor for motion detection.
    # This seperates the moving objects (foreground) from the stationary objects (background).
    # In other words it takes a frame, compares it to previous frames,
    # then marks the parts that have changed in white.
    # https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    global background_seperator
    background_seperator = cv2.createBackgroundSubtractorMOG2()
    
    while True:
        # Read a frame from the video capture
        is_frame_read_success, frame = video_capture.read()
        if not is_frame_read_success:
            break  # Exit loop if frame not read properly

        process_Frame(frame)
        print("Waiting 1 second...")
        
        if not handle_user_input():
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()


def create_threshold(frame, background_seperator):
    # Apply background subtraction to get the foreground mask
    fgmask = background_seperator.apply(frame)
    # Apply thresholding to binarize the mask
    thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
    # Remove noise with erosion
    thresh = cv2.erode(thresh, None, iterations=2)
    # Restore object size with dilation
    thresh = cv2.dilate(thresh, None, iterations=2)
    return thresh

def process_Frame(frame):
    thresh = create_threshold(frame, background_seperator)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    handle_motion(frame, contours)


    # Display the frame with motion rectangles
    cv2.imshow('Motion Detection', frame)
    
def handle_user_input() -> bool:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return False  # Stop running
    if key == ord('p'):
        print("Pausing for 5 seconds...")
        cv2.waitKey(5000)
    if key == ord('s'):
        print("Skipping to next frame...")
        cv2.waitKey(1)
    # Add more key handling here if needed
    return True  # Continue running

def handle_motion(frame, contours):
    for contour in contours:
        # Ignore small contours to reduce false positives
        if cv2.contourArea(contour) < 1000:  # Adjust sensitivity
            print("Ignoring small contour")
            continue
        # Get bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw rectangle around detected motion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

run()