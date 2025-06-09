import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, time, timedelta

# Load pre-trained model files for gender detection
gender_proto = "E:\Gender_Detection\Gender_Detection/gender_deploy.prototxt"
gender_model = "E:\Gender_Detection\Gender_Detection/gender_net.caffemodel"

# Load the gender model using OpenCV's dnn module
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Define model mean values for the gender model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Define gender labels
gender_list = ['Male', 'Female']

# Load OpenCV's pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Initialize SOS detection state variables
sos_start_time = None
sos_active = False
sos_frame_count = 0  # Counter for consistent SOS detection over frames
last_alert_time = None  # Variable to track the last alert time
last_snapshot_time = datetime.now()  # To handle snapshots every 10 seconds
recording = False  # To handle video recording
video_writer = None  # Video writer for recording

# Function to log alerts to a single text file
def log_alert(alert_message, male_count=0, female_count=0):
    with open("alerts_log.txt", "a") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{current_time} - {alert_message} - Males: {male_count}, Females: {female_count}\n")

# Function to detect SOS gesture (open palm above face box for 2+ seconds across multiple frames)
def detect_sos_gesture(landmarks, face_box_y):
    global sos_start_time, sos_active, sos_frame_count

    if len(landmarks) == 21:
        # Get y-coordinates of the wrist and fingertips
        wrist_y = landmarks[0][1]  # Normalized value
        index_finger_tip_y = landmarks[8][1]
        middle_finger_tip_y = landmarks[12][1]
        ring_finger_tip_y = landmarks[16][1]
        pinky_finger_tip_y = landmarks[20][1]

        # Check if the hand is above the upper line of the face box and fingers are extended (open palm)
        if wrist_y < face_box_y and all([
            index_finger_tip_y < wrist_y,
            middle_finger_tip_y < wrist_y,
            ring_finger_tip_y < wrist_y,
            pinky_finger_tip_y < wrist_y
        ]):
            # Start timer if not already started
            if sos_start_time is None:
                sos_start_time = datetime.now()
                sos_frame_count = 1
            else:
                sos_frame_count += 1  # Increment frame counter
                # Check if the SOS gesture is maintained for more than 2 seconds
                if (datetime.now() - sos_start_time).seconds >= 2:
                    sos_active = True  # SOS gesture confirmed
                    return True
        else:
            sos_start_time = None  # Reset if the gesture is not maintained
            sos_frame_count = 0
    return False

# Function to check if it's currently night-time between 12 AM and 5 AM
def is_night_time():
    current_time = datetime.now().time()
    night_start = time(0, 0)  # 12:00 AM
    night_end = time(5, 0)    # 5:00 AM
    return night_start <= current_time <= night_end

# Function to start video recording
def start_recording(frame):
    global recording, video_writer
    if not recording:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('alert_recording.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        recording = True

# Function to stop video recording
def stop_recording():
    global recording, video_writer
    if recording:
        recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None

# Function to process frames and perform gender detection
def process_frame(frame, alert_flags):
    global sos_active, last_alert_time, last_snapshot_time, recording

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    male_count = 0
    female_count = 0
    lone_woman_detected = False
    sos_detected = False

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_confidence = gender_preds[0].max()
        gender = gender_list[gender_preds[0].argmax()]

        confidence_threshold = 0.6
        if gender_confidence > confidence_threshold:
            if gender == 'Male':
                male_count += 1
                color = (255, 0, 0)  # Blue
            else:
                female_count += 1
                color = (0, 255, 0)  # Green

            label = f"{gender} ({gender_confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Process hand gestures using MediaPipe and detect SOS gesture
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # Adjust SOS gesture detection to use the face box's upper y-coordinate
                if detect_sos_gesture(landmarks, y / frame.shape[0]):  # Normalize face box y-coordinate
                    sos_detected = True
                    alert_flags["sos_detected"] = True

    cv2.putText(frame, f"Gender Distribution - Male: {male_count} Female: {female_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    lone_woman_detected = len(faces) == 1 and female_count == 1

    # Collect all alerts
    alerts = []

    if alert_flags["sos_detected"]:
        alerts.append("SOS Situation Detected!")
        log_alert("SOS Situation Detected!", male_count, female_count)
        alert_flags["sos_detected"] = False  # Reset flag after logging to avoid duplicate entries

    if lone_woman_detected and is_night_time():
        alerts.append("Lone Woman Detected at Night")
        log_alert("Lone Woman Detected at Night", male_count, female_count)

    # Show alert if male-to-female ratio is 3:1 or more
    if female_count > 0 and male_count / female_count >= 3:
        alerts.append("Alert! More Male to Female Ratio Found (3:1 or More)")
        log_alert("Alert! More Male to Female Ratio Found (3:1 or More)", male_count, female_count)

    # Update last alert time if there are any new alerts
    if alerts:
        last_alert_time = datetime.now()

    # Start or stop recording based on alert status
    if last_alert_time and (datetime.now() - last_alert_time).seconds <= 5:
        start_recording(frame)
    else:
        stop_recording()

    # Take snapshots every 10 seconds
    if (datetime.now() - last_snapshot_time).seconds >= 10:
        snapshot_filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(snapshot_filename, frame)
        last_snapshot_time = datetime.now()

    # Display the alerts within the alert box in the right corner with transparent black background
    alert_box_width = 250
    alert_box_height = 220  # Increased height for better visibility
    alert_box_x = frame.shape[1] - alert_box_width - 10
    alert_box_y = frame.shape[0] - alert_box_height - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (alert_box_x, alert_box_y), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 0), -1)
    alpha = 0.6  # Transparency factor

    # Add transparent black overlay
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display alerts in the alert box
    y_offset = alert_box_y + 20  # Start with some padding from the top
    for alert in alerts:
        cv2.putText(frame, alert, (alert_box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30  # Space between alerts

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)
alert_flags = {"sos_detected": False}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    frame = process_frame(frame, alert_flags)

    # Display the frame
    cv2.imshow('Gender Detection & SOS Alert System', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Release video writer if recording
stop_recording()