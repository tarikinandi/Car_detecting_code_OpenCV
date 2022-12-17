import cv2
import time

# Load the video
video = cv2.VideoCapture("b.mp4")

# Set up the Haar cascade classifier for detecting cars
car_cascade = cv2.CascadeClassifier("haarcascade_car.xml")

# Set up the scale factor and minimum neighbors for the classifier
scale_factor = 1.1
min_neighbors = 5

while True:
    # Read the frame
    ret, frame = video.read()

    # If the frame is not read correctly, break the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the cars in the frame
    cars = car_cascade.detectMultiScale(gray_frame, scale_factor, min_neighbors)

    # Iterate over the detected cars
    for (x, y, w, h) in cars:
        # Draw a rectangle around the car
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Calculate the distance to the car using the width of the car in the frame
        # and the focal length of the camera
        f = 0.8  # focal length of the camera in meters
        P = w * 0.5  # width of the car in pixels
        D = f * P  # distance to the car in meters

        # Display the distance on the frame
        cv2.putText(frame, f"Distance: {D:.2f} m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Wait for the user to press a key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and destroy the windows
video.release()
cv2.destroyAllWindows()