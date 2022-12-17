import cv2

# Load the video
video = cv2.VideoCapture("c.mp4")

# Create a car classifier
car_classifier = cv2.CascadeClassifier("haarcascade_car.xml")

# Set up variables for measuring distance
focal_length = 800  # The focal length of the camera in pixels
known_width = 4.8  # The width of the object in real life (in meters)

while True:
    # Read a frame from the video
    ret, frame = video.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_classifier.detectMultiScale(gray, 1.15, 5)

    # Iterate over the detected cars
    for (x, y, w, h) in cars:
        # Draw a rectangle around the car
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Calculate the distance to the car
        # Assume the car is centered horizontally in the frame
        pixel_width = w  # The width of the car in pixels
        distance = (known_width * focal_length) / pixel_width

        # Draw the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} meters", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for the user to press a key
    key = cv2.waitKey(1)
    if key == 27:  # The user pressed the "esc" key
        break

# Release the video and destroy the windows
video.release()
cv2.destroyAllWindows()