import cv2
import time
from collections import deque, Counter
from ultralytics import YOLO
import pygame
import speech_recognition as sr
import threading

# ---------------- INIT ----------------
pygame.mixer.init()

recognizer = sr.Recognizer()
voice_command = ""

model = YOLO("/home/pi/models/best.pt")

# Camera setup (LOW RES FOR SPEED)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Sound files
class_sounds = {
    "white_paper": "/home/pi/sounds/crumpledpaper.mp3",
    "plastic_bottle": "/home/pi/sounds/waterbottle.mp3",
    "aluminum_can": "/home/pi/sounds/aluminumcan.mp3",
    "plastic_bag": "/home/pi/sounds/plasticbag.mp3",
}

# Preload sounds (FASTER)
sounds = {
    k: pygame.mixer.Sound(v)
    for k, v in class_sounds.items()
}

FRAME_HISTORY = 5
class_history = deque(maxlen=FRAME_HISTORY)

last_played = {}
last_detected_class = "unknown"

prev_time = time.time()

# Run YOLO every N frames
FRAME_SKIP = 3
frame_count = 0


# ---------------- VOICE THREAD ----------------
def listen_voice():
    global voice_command
    while True:
        with sr.Microphone() as source:
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            voice_command = text.lower()
            print("You said:", voice_command)
        except:
            voice_command = ""


threading.Thread(target=listen_voice, daemon=True).start()


# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ---- RUN YOLO ONLY EVERY FEW FRAMES ----
    if frame_count % FRAME_SKIP == 0:
        results = model(frame, conf=0.5, verbose=False)

        detected_classes = [
            model.names[int(cls_id)]
            for cls_id in results[0].boxes.cls
        ]

        class_history.append(detected_classes)

        all_recent = [
            cls for sublist in class_history for cls in sublist
        ]

        if all_recent:
            most_common_cls, count = Counter(all_recent).most_common(1)[0]
            if count >= 3:
                last_detected_class = most_common_cls

    # ---- VOICE COMMAND TRIGGER ----
    if "what item is this" in voice_command:

        if last_detected_class in sounds:
            if last_played.get(last_detected_class, 0) + 2 < time.time():
                sounds[last_detected_class].play()
                last_played[last_detected_class] = time.time()

        voice_command = ""

    # ---- LIGHT DISPLAY ONLY ----
    cv2.putText(frame, f"Item: {last_detected_class}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO AI Assistant (Raspberry Pi)", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
