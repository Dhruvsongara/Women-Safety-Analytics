import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, time
from get_location import get_location

# Retrieve the location data
location_data = get_location()

# Load pre-trained model files for gender detection
gender_proto = "E:/Gender_Detection/Gender_Detection/gender_deploy.prototxt"
gender_model = "E:/Gender_Detection/Gender_Detection/gender_net.caffemodel"

# Load the gender model
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# SOS state variables
sos_start_time = None
sos_active = False
sos_frame_count = 0
last_alert_time = None
last_snapshot_time = datetime.now()

# Function to log alerts
def log_alert(alert_message, male_count=0, female_count=0, location=None):
    with open("alerts_log.txt", "a") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if location:
            location_str = (f"Location: {location['city']}, {location['country']} "
                            f"(Lat: {location['latitude']}, Lon: {location['longitude']}), "
                            f"IP: {location['ip_address']}")
        else:
            location_str = "Location: Unknown, IP: Unknown"

        log_file.write(f"{current_time} - {alert_message} - Males: {male_count}, "
                       f"Females: {female_count} - {location_str}\n")

# Detect SOS gesture
def detect_sos_gesture(landmarks, face_box_y):
    global sos_start_time, sos_active, sos_frame_count

    if len(landmarks) == 21:
        wrist_y = landmarks[0][1]
        finger_tips = [landmarks[i][1] for i in [8, 12, 16, 20]]

        if wrist_y < face_box_y and all(tip_y < wrist_y for tip_y in finger_tips):
            if sos_start_time is None:
                sos_start_time = datetime.now()
                sos_frame_count = 1
            else:
                sos_frame_count += 1
                if (datetime.now() - sos_start_time).seconds >= 1:
                    sos_active = True
                    return True
        else:
            sos_start_time = None
            sos_frame_count = 0
    return False

# Check night time
def is_night_time():
    current_time = datetime.now().time()
    night_start = time(0, 0)
    night_end = time(5, 0)
    return night_start <= current_time <= night_end

# Process single image
def process_frame(frame, alert_flags):
    global sos_active, last_alert_time, last_snapshot_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    male_count = 0
    female_count = 0
    lone_woman_detected = False
    sos_detected = False

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_confidence = gender_preds[0].max()
        gender = gender_list[gender_preds[0].argmax()]

        if gender_confidence > 0.6:
            color = (255, 0, 0) if gender == 'Male' else (0, 255, 0)
            male_count += 1 if gender == 'Male' else 0
            female_count += 1 if gender == 'Female' else 0

            label = f"{gender} ({gender_confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                if detect_sos_gesture(landmarks, y / frame.shape[0]):
                    sos_detected = True
                    alert_flags["sos_detected"] = True

    cv2.putText(frame, f"Gender Distribution - Male: {male_count} Female: {female_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    lone_woman_detected = len(faces) == 1 and female_count == 1

    alerts = []

    if alert_flags["sos_detected"]:
        alerts.append("SOS Situation Detected!")
        log_alert("SOS Situation Detected!", male_count, female_count, location_data)
        alert_flags["sos_detected"] = False

    if lone_woman_detected and is_night_time():
        alerts.append("Lone Woman Detected at Night")
        log_alert("Lone Woman Detected at Night", male_count, female_count, location_data)

    if female_count > 0 and male_count / female_count >= 3:
        alerts.append("Alert! More Male to Female Ratio Found (3:1 or More)")
        log_alert("Alert! More Male to Female Ratio Found (3:1 or More)", male_count, female_count, location_data)

    if alerts:
        last_alert_time = datetime.now()

    # Draw alert box
    alert_box_width = 250
    alert_box_height = 200
    alert_box_x = frame.shape[1] - alert_box_width - 10
    alert_box_y = frame.shape[0] - alert_box_height - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (alert_box_x, alert_box_y), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    y_offset = alert_box_y + 20
    max_chars_per_line = 30
    for alert in alerts:
        lines = []
        while len(alert) > max_chars_per_line:
            idx = alert.rfind(' ', 0, max_chars_per_line)
            idx = idx if idx != -1 else max_chars_per_line
            lines.append(alert[:idx])
            alert = alert[idx+1:]
        lines.append(alert)
        for line in lines:
            cv2.putText(frame, line, (alert_box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 20
        y_offset += 10

    return frame

# ------------------- Main Execution -------------------

image_path = "E:\\Gender_Detection\\Gender_Detection"  # <<<<<<<<<<<   SET YOUR IMAGE PATH HERE
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image.")
else:
    alert_flags = {"sos_detected": False}
    processed_image = process_frame(image, alert_flags)

    output_path = "output_result.jpg"
    cv2.imwrite(output_path, processed_image)
    print(f"Analysis completed. Saved output to {output_path}")

    # Display the image
    cv2.imshow('Gender Detection & SOS Alert System', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
