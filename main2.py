import cv2
import numpy as np
import time
from dotenv import load_dotenv
load_dotenv()
from mtgscan.text import MagicRecognition
from mtgscan.ocr.azure import Azure

background_seperator = None
last_frame = None

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
        global last_frame
        last_frame = frame.copy()  # Store the last frame for potential saving
        
        process_Frame(frame)
        
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

    # Disabling this for now. Working with manual image capture as WIP.
    # handle_motion(frame, thresh)

    # Display the frame with motion rectangles
    cv2.imshow('Motion Detection', frame)
    
def handle_user_input() -> bool:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return False  # Stop running
    if key == ord('p'):
        print("Pausing for 5 seconds...")
        # time.sleep(5)
        cv2.waitKey(5000)
        ## cv2.waitKey(5000)
        return True
    if key == ord('c'):
        print("Capturing and saving current frame...")
        # Save the current frame as an image file
        if 'last_frame' in globals() and last_frame is not None:
            filename = f"capture_{int(time.time())}.png"
            cv2.imwrite(filename, last_frame)
            print(f"Saved frame as {filename}")
            deck = process_image_to_deck(filename)
            deck_filename = f"deck_{int(time.time())}.csv"
            save_deck_to_file(deck, deck_filename)
            save_deck_to_file
        else:
            print("No frame available to save.")
        return True
    # Add more key handling here if needed
    return True  # Continue running

def handle_motion(frame, thresh):
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Ignore small contours to reduce false positives
        if cv2.contourArea(contour) < 1000:  # Adjust sensitivity
            # print("Ignoring small contour")
            continue
        # Get bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw rectangle around detected motion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def process_image_to_deck(image_path: str):
    azure = Azure()
    rec = MagicRecognition(file_all_cards="all_cards.txt", file_keywords="Keywords.json")
    box_texts = azure.image_to_box_texts(image_path)
    deck = rec.box_texts_to_deck(box_texts)
    return deck

def save_deck_to_file(deck, filename):
    with open(filename, "w") as f:
        for card_name, count in deck:
            f.write(f"{card_name},{count}\n")

run()