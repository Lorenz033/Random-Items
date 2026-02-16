import cv2
import time
from ultralytics import YOLO

# Load model
model = YOLO(r"C:\Programming\MyProjects\RaspiYOLOv2\models\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=0.5)

    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO + FPS", annotated_frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
