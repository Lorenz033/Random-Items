import cv2
import time
from collections import deque, Counter
from ultralytics import YOLO
import pygame

# Initialize pygame mixer
pygame.mixer.init()

class_sounds = {
    "white_paper": r"C:\Programming\MyProjects\RaspiYOLOv2\CV\sounds\crumpledpaper.mp3",    #change directory
    "plastic_bottle": r"C:\Programming\MyProjects\RaspiYOLOv2\CV\sounds\waterbottle.mp3",   #change directory
    "aluminum_can": r"C:\Programming\MyProjects\RaspiYOLOv2\CV\sounds\aluminumcan.mp3",     #change directory
    "plastic_bag": r"C:\Programming\MyProjects\RaspiYOLOv2\CV\sounds\plasticbag.mp3",       #change directory
}

model = YOLO(r"C:\Programming\MyProjects\RaspiYOLOv2\CV\models\best.pt")                    #change directory

cap = cv2.VideoCapture(0)
prev_time = time.time()

FRAME_HISTORY = 5
class_history = deque(maxlen=FRAME_HISTORY)
last_played = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=0.5)
    
    detected_classes = [model.names[int(cls_id)] for cls_id in results[0].boxes.cls]

    class_history.append(detected_classes)

    all_recent = [cls for sublist in class_history for cls in sublist]
    if all_recent:
        most_common_cls, count = Counter(all_recent).most_common(1)[0]

        if count >= 3 and most_common_cls in class_sounds:
            if last_played.get(most_common_cls, 0) + 2 < time.time():  
                pygame.mixer.Sound(class_sounds[most_common_cls]).play()
                last_played[most_common_cls] = time.time()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO + FPS", frame)

    if cv2.waitKey(1) == 27: 
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
