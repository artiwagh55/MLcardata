import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

pyautogui.FAILSAFE = False

# Screen size for real mouse control
screen_w, screen_h = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Smoothing + active region inside camera frame
prev_x, prev_y = 0, 0
smoothening = 5
frameR = 100  # margin from edges

# Virtual mouse view (right window)
vm_w, vm_h = 640, 480
trail = np.zeros((vm_h, vm_w, 3), dtype=np.uint8)

# Colors for trail (BGR)
colors = [
    (255, 255, 0),   # cyan
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (255, 128, 0)    # teal-ish orange
]
color_index = 0
current_color = colors[color_index]

# color change debounce
last_color_change = 0
COLOR_COOLDOWN = 0.6  # seconds between color switches

def index_middle_up(hand):
    index_up = hand.landmark[8].y < hand.landmark[6].y
    middle_up = hand.landmark[12].y < hand.landmark[10].y
    return index_up, middle_up


cv2.namedWindow("Camera")
cv2.namedWindow("Virtual Mouse View")
cv2.moveWindow("Camera", 0, 0)
cv2.moveWindow("Virtual Mouse View", 700, 0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # draw region box on camera feed
    cv2.rectangle(frame, (frameR, frameR),
                  (w - frameR, h - frameR), (255, 0, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # fade old trail a bit -> nice glow effect
    trail = cv2.addWeighted(trail, 0.90, np.zeros_like(trail), 0.10, 0)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        index_up, middle_up = index_middle_up(handLms)

        # landmarks
        index_tip = handLms.landmark[8]
        thumb_tip = handLms.landmark[4]

        x1, y1 = int(index_tip.x * w), int(index_tip.y * h)
        x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # ------------------ MOVE MODE (index up, middle down) ------------------
        if index_up and not middle_up:
            cv2.circle(frame, (x1, y1), 10, current_color, cv2.FILLED)

            # map camera coords → screen coords
            x3 = (x1 - frameR) * screen_w / (w - 2 * frameR)
            y3 = (y1 - frameR) * screen_h / (h - 2 * frameR)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x3, y3

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening
            prev_x, prev_y = curr_x, curr_y

            pyautogui.moveTo(curr_x, curr_y)

            # also draw in virtual mouse window using current_color
            vm_x = int((x1 - frameR) * vm_w / (w - 2 * frameR))
            vm_y = int((y1 - frameR) * vm_h / (h - 2 * frameR))
            if 0 <= vm_x < vm_w and 0 <= vm_y < vm_h:
                cv2.circle(trail, (vm_x, vm_y), 10, current_color, -1)

        # ------------------ PINCH = CHANGE COLOR ------------------
        # distance between index tip & thumb tip
        distance = math.hypot(x_thumb - x1, y_thumb - y1)

        # draw line between index & thumb for feedback
        cv2.line(frame, (x1, y1), (x_thumb, y_thumb), current_color, 2)

        if distance < 35:  # pinch threshold
            now = time.time()
            if now - last_color_change > COLOR_COOLDOWN:
                # cycle color
                color_index = (color_index + 1) % len(colors)
                current_color = colors[color_index]
                last_color_change = now
                cv2.putText(frame, "COLOR CHANGE",
                            (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            current_color, 2)

    cv2.putText(
        frame,
        "Move: Index up | Pinch (Index+Thumb): Change color | q: quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    vm_display = trail.copy()

    cv2.imshow("Camera", frame)
    cv2.imshow("Virtual Mouse View", vm_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()