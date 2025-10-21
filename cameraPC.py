import cv2
import datetime
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
CAMERA_INDEX = 0                       # Default webcam (0)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
OUTPUT_DIR = "car_photos"              # Folder to save photos
MIN_CONTOUR_AREA = 500                 # Minimum contour area to consider a car

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# INITIALIZE CAMERA
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Background subtractor for simple car detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

car_present = False  # Track if a car is currently in frame

print("Starting camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Clean up mask
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, None)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    car_detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            car_detected = True

    # Overlay timestamp on frame
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp_str, (10, FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Save photo only when a new car enters
    if car_detected and not car_present:
        photo_filename = os.path.join(OUTPUT_DIR, f"{timestamp_str.replace(':','-')}.jpg")
        cv2.imwrite(photo_filename, frame)
        print(f"Car detected! Photo saved: {photo_filename}")
        car_present = True
    elif not car_detected and car_present:
        # Reset when car leaves
        car_present = False

    # Show live preview
    cv2.imshow('Car Camera Preview', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("Camera stopped. All photos saved.")
