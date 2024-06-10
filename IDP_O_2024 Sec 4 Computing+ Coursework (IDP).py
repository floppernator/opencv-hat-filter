# Importing the required modules
import cv2
import numpy as np
import dlib

# List of available images for filters (assets)
hats = ['cap1.png', 'cap2.png', 'cap3.png']
glasslst = ['glasses1.png', 'glasses2.png', 'glasses3.png']

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

x = 0  # Initialize a variable for cycling through filter options

# Load the images of glasses with an alpha channel
glasses_images = [cv2.imread(filename, -1) for filename in glasslst]
current_glasses = glasses_images[0]

# Load the images of hats with an alpha channel
hat_images = [cv2.imread(filename, -1) for filename in hats]
current_hat = hat_images[0]

# Load the image of confetti with an alpha channel
confetti = cv2.imread('confati.png', -1)

# Bool to ensure that multiple of the same filter do not overlay each other
apply_glasses = False
apply_hat = False
apply_confetti = False

# Function for scrolling through different images in each filter
def get_wrapped_value(lst, x):
    return lst[x % len(lst)]

# Function for positioning glasses on the face
def overlay_glasses(frame, face_landmarks, glasses):
    # Extract the coordinates of the eyes from the facial landmarks
    eye_left = face_landmarks[36]  # Index of the left eye landmark
    eye_right = face_landmarks[45]  # Index of the right eye landmark

    # Calculate the center of the eyes
    center_x = (eye_left[0] + eye_right[0]) // 2
    center_y = (eye_left[1] + eye_right[1]) // 2

    # Calculate the maximum width and height of the glasses image based on the eye positions
    max_glasses_width = abs(eye_right[0] - eye_left[0]) * 2
    max_glasses_height = int(max_glasses_width * (glasses.shape[0] / glasses.shape[1]))

    # Resize the glasses image to fit within the bounds of the face
    glasses_resized = cv2.resize(glasses, (max_glasses_width, max_glasses_height))

    # Calculate the position for the glasses
    glasses_width = glasses_resized.shape[1]
    glasses_height = glasses_resized.shape[0]
    x1 = center_x - glasses_width // 2
    x2 = x1 + glasses_width
    y1 = center_y - glasses_height // 2

    # Ensure the position is within the frame
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1])
    y2 = min(y1 + glasses_height, frame.shape[0])  # Adjusted to ensure the overlay region is within the frame

    # Calculate the overlay region
    overlay_width = x2 - x1
    overlay_height = y2 - y1

    # Check if the shapes of the overlay region and the glasses image are compatible
    if overlay_width <= 0 or overlay_height <= 0:
        return

    # Resize the glasses image to match the overlay region
    glasses_resized = cv2.resize(glasses_resized, (overlay_width, overlay_height))

    # Create a mask for the glasses
    glasses_mask = glasses_resized[:, :, 3] / 255.0
    glasses_img = glasses_resized[:, :, 0:3]

    # Overlay the glasses onto the frame
    overlay = frame[y1:y2, x1:x2]
    for c in range(0, 3):
        overlay[:, :, c] = glasses_mask * glasses_img[:, :, c] + (1 - glasses_mask) * overlay[:, :, c]

# Main function to apply filters and display the webcam feed
def all():
    global x, current_glasses, current_hat, apply_glasses, apply_hat, apply_confetti

    # Capture frames from the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the grayscale frame
        if apply_glasses or apply_hat or apply_confetti:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                if len(landmarks.parts()) != 68:
                    continue  # Skip this face if not all landmarks are detected

                face_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
                face_landmarks = np.array(face_landmarks)

                if apply_glasses:
                    overlay_glasses(frame, face_landmarks, current_glasses)

                if apply_hat:
                    # Calculate the distance between specific landmarks to determine the width of the head
                    head_width = landmarks.part(16).x - landmarks.part(0).x  # Example: Distance between left and right corners of the head

                    # Resize the hat image based on the width of the head and make it bigger
                    scale_factor = (head_width * 1.5) / current_hat.shape[1]  # Increase the width of the hat by 50% and keep aspect ratio
                    hat_resized = cv2.resize(current_hat, None, fx=scale_factor, fy=scale_factor)

                    # Calculate the position for the hat
                    hat_x = int(landmarks.part(0).x - (hat_resized.shape[1] / 2) + (head_width * 0.6))  # Center the hat horizontally and shift it slightly to the right
                    hat_y = int(landmarks.part(19).y - hat_resized.shape[0]) + 30  # Place the hat above the forehead

                    # Ensure the position is within the frame
                    if 0 < hat_y < frame.shape[0] and 0 < hat_x < frame.shape[1]:
                        # Ensure that the hat does not extend beyond the right edge of the frame to prevent crashing
                        if hat_x + hat_resized.shape[1] < frame.shape[1]:
                            roi = frame[hat_y:hat_y + hat_resized.shape[0], hat_x:hat_x + hat_resized.shape[1]]
                            mask = hat_resized[:, :, 3] / 255.0
                            mask_inv = 1.0 - mask

                            # Resize mask_inv to match the shape of hat_resized for all color channels
                            mask_inv_resized = cv2.resize(mask_inv, (hat_resized.shape[1], hat_resized.shape[0]))

                            # Overlay the hat onto the frame
                            for c in range(0, 3):
                                roi[:, :, c] = (mask * hat_resized[:, :, c] +
                                                mask_inv_resized * roi[:, :, c])
                            frame[hat_y:hat_y + hat_resized.shape[0], hat_x:hat_x + hat_resized.shape[1]] = roi

                if apply_confetti:
                    # Resize the confetti image to match the frame size
                    confetti_resized = cv2.resize(confetti, (frame.shape[1], frame.shape[0]))

                    # Overlay the confetti image on the frame
                    overlay = frame.copy()
                    confetti_mask = confetti_resized[:, :, 3] / 255.0
                    confetti_img = confetti_resized[:, :, 0:3]
                    for c in range(0, 3):
                        overlay[:, :, c] = confetti_mask * confetti_img[:, :, c] + (1 - confetti_mask) * overlay[:, :, c]

                    frame = overlay

        # Display the frame with filters applied
        cv2.imshow('Webcam', frame)
        
        # Handle key presses to switch filters or close the camera
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            apply_glasses = not apply_glasses
        elif key == ord('e'):
            apply_confetti = not apply_confetti
        elif key == ord('r'):
            apply_hat = not apply_hat
        elif key == ord('d') and apply_hat:  # Change to a different hat design when 'd' is pressed
            x = x + 1
            current_hat = cv2.imread(get_wrapped_value(hats, x), -1)
        elif key == ord('a') and apply_hat:  # Change back to the original hat design when 'a' is pressed
            x = x - 1
            current_hat = cv2.imread(get_wrapped_value(hats, x), -1)
        elif key == ord('z') and apply_glasses:  # Change to a different glasses design when 'z' is pressed
            x = x + 1
            current_glasses = glasses_images[x % len(glasses_images)]
        elif key == ord('c') and apply_glasses:  # Change back to the original glasses design when 'c' is pressed
            x = x - 1
            current_glasses = glasses_images[x % len(glasses_images)]

    cap.release()
    cv2.destroyAllWindows()

# Open the webcam
cap = cv2.VideoCapture(1)  # Use index 0 for the default webcam, or specify a different index if you have multiple cameras
all()  # Call the main function to start applying filters
