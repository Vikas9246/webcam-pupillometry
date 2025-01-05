#!/usr/bin/env python3
import cv2
import numpy as np

def main():
    # Path to Haar Cascade for eyes
    eye_cascade_path = "cascades/haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to grayscale for cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes (returns list of bounding boxes)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # We'll process each eye region separately
        for (ex, ey, ew, eh) in eyes:
            # Extract region of interest (ROI) for the eye
            eye_roi = gray[ey:ey+eh, ex:ex+ew]

            # Basic thresholding to isolate the pupil (dark region)
            # You might need to tweak these params based on lighting
            _, thresholded = cv2.threshold(eye_roi, 50, 255, cv2.THRESH_BINARY_INV)

            # Optionally apply some morphological operations to reduce noise
            # kernel = np.ones((3,3), np.uint8)
            # thresholded = cv2.medianBlur(thresholded, 5)  # or cv2.GaussianBlur()
            # thresholded = cv2.erode(thresholded, kernel, iterations=2)
            # thresholded = cv2.dilate(thresholded, kernel, iterations=2)

            # Find contours in the thresholded eye region
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If we found any contours, pick the largest as the pupil
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(largest_contour)
                
                # Draw rectangle or circle around the pupil in the main frame
                # Remember to offset by ex, ey because eye_roi is a subregion
                cv2.rectangle(frame, (ex + x, ey + y), (ex + x + w, ey + y + h), (0, 255, 0), 2)
                
                # Approximate pupil size (area in px):
                pupil_area = w * h
                # You could also compute it more precisely with cv2.contourArea(largest_contour).
                cv2.putText(frame, f"Pupil area: {pupil_area}", (ex + x, ey + y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Show the thresholded eye region in a separate window (useful for debugging)
            cv2.imshow("Eye Thresholded", thresholded)

        # Display the main frame
        cv2.imshow("Webcam Pupillometry", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
