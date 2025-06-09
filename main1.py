import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, time

# Load pre-trained model files for gender detection
gender_proto = "E:\Gender_Detection\Gender_Detection\gender_deploy.prototxt"
gender_model = "E:\Gender_Detection\Gender_Detection\gender_net.caffemodel"

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

# Function to log alerts to a single text file
def log_alert(alert_message):
    with open("alerts_log.txt", "a") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{current_time} - {alert_message}\n")

# Function to detect SOS gesture
def detect_sos_gesture(landmarks):
    if len(landmarks) == 21:
        if landmarks[8][2] < landmarks[6][2] and landmarks[12][2] < landmarks[10][2] and landmarks[16][2] < landmarks[14][2]:
            return True
    return False

# Function to check if it's currently night-time between 12 AM and 5 AM
def is_night_time():
    current_time = datetime.now().time()
    night_start = time(0, 0)  # 12:00 AM
    night_end = time(5, 0)    # 5:00 AM
    return night_start <= current_time <= night_end

# Function to draw a semi-transparent box and display alerts
def draw_alert_box(frame, alerts, male_count, female_count):
    # Define box position and size
    box_width = 350
    box_height = (len(alerts) + 2) * 40 + 20  # Adding 2 for gender counts
    box_x = frame.shape[1] - box_width - 10  # Right side of the frame
    box_y = 10  # Top position

    # Create a semi-transparent box
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
    alpha = 0.4  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display male and female counts at the top of the box
    cv2.putText(frame, f"Male: {male_count}  Female: {female_count}", (box_x + 10, box_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Display each alert inside the box in red color
    for i, alert in enumerate(alerts):
        cv2.putText(frame, alert, (box_x + 10, box_y + 70 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

# Function to process frames and perform gender detection
def process_frame(frame, alert_flags):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    male_count = 0
    female_count = 0
    lone_woman_detected = False
    sos_detected = False

    # Initialize alert list for this frame
    alerts = []

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
                color = (255, 0, 0)  # Blue for Male
            else:
                female_count += 1
                color = (0, 255, 0)  # Green for Female

            label = f"{gender} ({gender_confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Process hand gestures using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            sos_detected = detect_sos_gesture(landmarks)
            if sos_detected:
                alert_flags["sos_detected"] = True
                break

    lone_woman_detected = len(faces) == 1 and female_count == 1

    # Display alerts if conditions are met
    if alert_flags["sos_detected"]:
        alerts.append("SOS Situation Detected!")
        log_alert("SOS Situation Detected!")
        alert_flags["sos_detected"] = False  # Reset flag after logging to avoid duplicate entries

    if lone_woman_detected and is_night_time():
        alerts.append("Lone Woman Detected at Night")
        log_alert("Lone Woman Detected at Night")

    if female_count == 1 and male_count >= 2:
        alerts.append("Alert! 1 Woman Surrounded by 2 or More Men")
        log_alert("Alert! 1 Woman Surrounded by 2 or More Men")

    if male_count >= 3 and female_count == 1:
        alerts.append("Alert! More Male to Female Ratio Found (3:1 or More)")
        log_alert("Alert! More Male to Female Ratio Found (3:1 or More)")

    # Draw the alert box with alerts and gender counts
    frame = draw_alert_box(frame, alerts, male_count, female_count)

    return frame

# Function to handle video capture from webcam or video file
def process_video(source):
    video = cv2.VideoCapture(source)
    screen_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cv2.namedWindow("Gender Detection with Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gender Detection with Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize alert flags
    alert_flags = {
        "sos_detected": False
    }

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame or end of video reached")
            break

        frame = process_frame(frame, alert_flags)
        frame = cv2.resize(frame, (int(screen_width), int(screen_height)))
        cv2.imshow("Gender Detection with Gesture Recognition", frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Process video file first (replace with the actual path to your video file)
#process_video('D:\\Gender_Detection\\test.mp4')

# Then switch to live webcam feed (using 0 as the source for the webcam)
process_video(0)
