import cv2
import winsound
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

# Function to play an alert sound
def play_alert_sound():
    winsound.Beep(1000, 1000)

# Function to detect SOS gesture
def detect_sos_gesture(landmarks):
    if len(landmarks) == 21:
        if landmarks[8][2] < landmarks[6][2] and landmarks[12][2] < landmarks[10][2] and landmarks[16][2] < landmarks[14][2]:
            return True
    return False

# Function to check if it's currently night-time
def is_night_time():
    current_time = datetime.now().time()
    night_start = time(20, 0)  # 8:00 PM
    night_end = time(6, 0)     # 6:00 AM
    return current_time >= night_start or current_time <= night_end

# Function to process frames and perform gender detection
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    male_count = 0
    female_count = 0
    lone_woman_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
            else:
                female_count += 1

            label = f"{gender} ({gender_confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Gender Distribution - Male: {male_count} Female: {female_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # Process hand gestures using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    sos_detected = False
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            sos_detected = detect_sos_gesture(landmarks)
            if sos_detected:
                break

    lone_woman_detected = len(faces) == 1 and female_count == 1

    if sos_detected and female_count >= 1:
        cv2.putText(frame, "SOS Situation Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        play_alert_sound()

    if lone_woman_detected and is_night_time():
        cv2.putText(frame, "Lone Woman Detected at Night", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    if female_count == 1 and male_count >= 2:
        cv2.putText(frame, "Alert! 1 Woman Surrounded by 2 or More Men", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        play_alert_sound()

    return frame

# Function to handle video capture from webcam or video file
def process_video(source):
    video = cv2.VideoCapture(source)
    screen_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cv2.namedWindow("Gender Detection with Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gender Detection with Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame or end of video reached")
            break

        frame = process_frame(frame)
        frame = cv2.resize(frame, (int(screen_width), int(screen_height)))
        cv2.imshow("Gender Detection with Gesture Recognition", frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()

# Process video file first (replace with the actual path to your video file)
#process_video('D:\\Gender_Detection\\test.mp4')

# Then switch to live webcam feed (using 0 as the source for the webcam)
process_video(0)

cv2.destroyAllWindows()