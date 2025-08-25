import cv2
import numpy as np

# Initialize video capture (0 = default webcam)
cap = cv2.VideoCapture(0) # or 0 for webcam

# Create background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if frame not read properly

    # Apply background subtraction to get the foreground mask
    fgmask = fgbg.apply(frame)
    
    # Apply thresholding to binarize the mask
    thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
    # Remove noise with erosion
    thresh = cv2.erode(thresh, None, iterations=2)
    # Restore object size with dilation
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        print(contours.size())
        # print("Contour area:", cv2.contourArea(contour))
        # Ignore small contours to reduce false positives
        if cv2.contourArea(contour) < 1000: # Adjust sensitivity
            print("Ignoring small contour")
            continue
        # Get bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw rectangle around detected motion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with motion rectangles
    cv2.imshow('Motion Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()