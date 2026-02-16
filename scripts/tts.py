import cv2
import time
from collections import deque, Counter
from ultralytics import YOLO
import pygame
import threading
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# -------------------- INIT --------------------
pygame.mixer.init()

class_sounds = {
    "white_paper": r"C:\Programming\MyProjects\RaspiYOLOv2\sounds\crumpledpaper.mp3",
    "plastic_bottle": r"C:\Programming\MyProjects\RaspiYOLOv2\sounds\waterbottle.mp3",
    "aluminum_can": r"C:\Programming\MyProjects\RaspiYOLOv2\sounds\aluminumcan.mp3",
    "plastic_bag": r"C:\Programming\MyProjects\RaspiYOLOv2\sounds\plasticbag.mp3",
    "greeting": r"C:\Programming\MyProjects\RaspiYOLOv2\sounds\hi.mp3"  # NEW
}




model = YOLO(r"C:\Programming\MyProjects\YOLO\CV\models\best.pt")
cap = cv2.VideoCapture(0)

# -------------------- YOLO STABILITY --------------------
FRAME_HISTORY = 5
class_history = deque(maxlen=FRAME_HISTORY)
last_detected_class = "unknown"
last_played = {}

# -------------------- VOSK SETUP --------------------
voice_command = ""  
q = queue.Queue()

vosk_model = Model(r"C:\Programming\MyProjects\YOLO\CV\models\vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(vosk_model, 16000)

def audio_callback(indata, frames, time_info, status):
    q.put(bytes(indata))

def listen_voice():
    global voice_command

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        print("Voice system ready")

        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    voice_command = text.lower()
                    print("You said:", voice_command)

# Start voice thread
threading.Thread(target=listen_voice, daemon=True).start()

prev_time = time.time()

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=0.5)
    frame = results[0].plot()

    detected_classes = [model.names[int(cls_id)] for cls_id in results[0].boxes.cls]
    class_history.append(detected_classes)

    all_recent = [cls for sublist in class_history for cls in sublist]

    if all_recent:
        most_common_cls, count = Counter(all_recent).most_common(1)[0]
        if count >= 3:
            last_detected_class = most_common_cls

    # -------------------- VOICE COMMAND HANDLING --------------------
    voice_command = voice_command.lower()
    words = voice_command.split()


    # GREETING
    if "hi" in words or "hello" in words:

        print("Greeting detected")

        if last_played.get("greeting", 0) + 2 < time.time():
            pygame.mixer.Sound(class_sounds["greeting"]).play()
            last_played["greeting"] = time.time()

        voice_command = ""

    # OBJECT DETECTION QUESTION
    if "what" in voice_command:
        print("Final detected class:", last_detected_class)

        if last_detected_class in class_sounds:
            if last_played.get(last_detected_class, 0) + 2 < time.time():
                pygame.mixer.Sound(class_sounds[last_detected_class]).play()
                last_played[last_detected_class] = time.time()

        voice_command = ""

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Voice Assistant (VOSK)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
