import cv2
import datetime
import os
import time
from ultralytics import YOLO
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
CAMERA_INDEX = 0
OUTPUT_DIR = "car_photos"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAR_CLASSES = [2, 3, 5, 7]        # car, truck, bus, motorcycle
CONF_THRESH = 0.5                  # YOLO confidence threshold
MIN_CONTOUR_AREA = 500             # min area for motion detection
SAVE_COOLDOWN = 10                 # seconds between photo saves

OPENVINO_MODEL_DIR = "yolov8s_openvino_model"  # your existing OpenVINO folder

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
print("Loading YOLO model...")
model = YOLO(OPENVINO_MODEL_DIR, task="detect")
print("✅ YOLO model loaded!")

# -----------------------------
# INITIALIZE CAMERA & BACKGROUND SUBTRACTOR
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

last_save_time = 0
print("Starting motion+YOLO detection. Press 'q' to quit.")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # --- Motion detection ---
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, None)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    moving_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            moving_boxes.append((x, y, w, h))

    # --- YOLO detection on moving regions ---
    car_found = False
    for (x, y, w, h) in moving_boxes:
        # crop the moving region
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue  # skip empty crops

        # resize crop for YOLO (match exported OpenVINO model input, e.g., 640x640)
        detect_size = 640
        crop_resized = cv2.resize(crop, (detect_size, detect_size))

        # YOLO detection
        results = model.predict(source=crop_resized, verbose=False)

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in CAR_CLASSES and conf >= CONF_THRESH:
                car_found = True
                # Map box back to original frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_scale = w / detect_size
                y_scale = h / detect_size
                x1 = int(x1 * x_scale + x)
                x2 = int(x2 * x_scale + x)
                y1 = int(y1 * y_scale + y)
                y2 = int(y2 * y_scale + y)
                label = f"{model.names[cls_id]} {conf*100:.1f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- Timestamp overlay ---
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (10, FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Save photo if car detected and cooldown passed ---
    if car_found and (time.time() - last_save_time > SAVE_COOLDOWN):
        filename = os.path.join(OUTPUT_DIR, f"{ts.replace(':','-')}.jpg")
        cv2.imwrite(filename, frame)
        print(f"💾 Moving car detected ({ts}) — saved to {filename}")
        last_save_time = time.time()

    # --- Show live preview ---
    cv2.imshow("Motion+YOLO Car Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("Camera stopped. Photos saved in:", OUTPUT_DIR)
