# 👁️‍🗨️ Women Safety Analytics – Real-time Threat Detection System

## 📖 Overview
**Women Safety Analytics** is an AI-powered computer vision system that detects potential safety threats for women. It identifies lone women at night, unbalanced gender ratios, and SOS hand gestures, while logging all alerts with timestamps and IP-based location.

---

### Screenshots
![Women Safety Analytics](./womensafety.jpg)


## ✅ Features
- 👤 Face Detection using Haar Cascade (OpenCV)
- 🧠 Gender Classification using a Caffe model
- ✋ SOS Gesture Recognition via MediaPipe (open palm above head)
- 🌙 Night-Time Monitoring (00:00 to 05:00)
- 📊 Gender Ratio Analysis (alerts if Male:Female ≥ 3:1)
- 📍 IP-Based Geolocation Logging
- 🖼️ Alert messages overlaid on processed image
- 📄 Logs saved in `alerts_log.txt`

---

## ⚙️ Technologies Used

| Component              | Tool/Library         |
|------------------------|----------------------|
| Language               | Python               |
| Face Detection         | OpenCV               |
| Gender Classification  | Caffe Model          |
| Gesture Recognition    | MediaPipe Hands      |
| Image Processing       | OpenCV, NumPy        |
| Location Detection     | IPInfo (via `requests`) |

---

## 🔧 Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/women-safety-analytics.git
   cd women-safety-analytics

2. Install required packages
   
pip install opencv-python mediapipe numpy requests

3. Download Required Model Files

⚠️ The gender detection model (gender_net.caffemodel) is larger than 25 MB and cannot be uploaded here.

👉 Download it manually from the following source:
gender_net.caffemodel – LearnOpenCV

Also download:
gender_deploy.prototxt

Place both files in the appropriate directory or update their paths in the code:

gender_model = "E:/Gender_Detection/Gender_Detection/gender_net.caffemodel"
gender_proto = "E:/Gender_Detection/Gender_Detection/gender_deploy.prototxt"

4. Add the location module

Create a get_location.py file with the following:

import requests

def get_location():
    response = requests.get("https://ipinfo.io/json")
    return response.json()

5. Usage

Update the image path in the script:

image_path = "E:\\Gender_Detection\\Gender_Detection\\your_image.jpg"
Run the script:

python women_safety_analytics.py

Output:

Image saved as output_result.jpg

Logged alerts in alerts_log.txt

Visual alerts shown on-screen

📸 Sample Output

🟦 Blue box for Male

🟩 Green box for Female

🛑 Alerts like:

SOS Situation Detected

Lone Woman Detected at Night

More Male to Female Ratio Found (3:1 or More)

🔮 Future Scope

Live CCTV/webcam integration

Mobile-triggered SOS alerts

Real-time alert dashboard & map-based hotspot detection

YOLOv5-based upgrade for faster inference

Emergency system or police API integration

