import cv2
import mediapipe as mp
import numpy as np
import math


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


canvas = None
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 0, 0)]  
color_index = 0
brush_size = 8
eraser_size = 50
min_brush = 2
max_brush = 50

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            index_finger = lm_list[8]
            thumb = lm_list[4]
            middle_finger = lm_list[12]
            ring_finger = lm_list[16]
            pinky = lm_list[20]

            if all(abs(lm_list[i][1] - lm_list[9][1]) < 40 for i in [4, 8, 12, 16, 20]):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

          
            for i, color in enumerate(colors):
                if 10 + i * 50 < index_finger[0] < 50 + i * 50 and 10 < index_finger[1] < 50:
                    color_index = i
                    break

            
            if (
                index_finger[1] < middle_finger[1]
                and index_finger[1] < ring_finger[1]
                and index_finger[1] < pinky[1]
                and thumb[1] < middle_finger[1] 
                and thumb[1] < pinky[1]
                and thumb[1] < ring_finger[1]
            ):
               
                distance = math.hypot(index_finger[0] - thumb[0], index_finger[1] - thumb[1])
                brush_size = int(np.interp(distance, [20, 200], [min_brush, max_brush]))
               
                cv2.line(frame, index_finger, thumb, (0, 255, 0), 2)
                cv2.putText(frame, f"Size: {brush_size}", (index_finger[0] + 20, index_finger[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            elif (
                index_finger[1] < middle_finger[1]
                and index_finger[1] < ring_finger[1]
                and index_finger[1] < pinky[1]
                and thumb[1] > index_finger[1]  
            ):
                if color_index < 5: 
                    cv2.circle(canvas, index_finger, brush_size, colors[color_index], -1)
                else: 
                    cv2.circle(canvas, index_finger, brush_size, (0, 0, 0), -1)

            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

   
    blended = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    combined_view = np.hstack((blended, canvas))

    
    for i, color in enumerate(colors):
        cv2.rectangle(combined_view, (10 + i * 50, 10), (50 + i * 50, 50), color, -1)

    cv2.imshow("Virtual Drawing App", combined_view)

   
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in range(6)]:  
        color_index = int(chr(key))
    elif key == ord('s'):
        cv2.imwrite("drawing.png", canvas)  

cap.release()
cv2.destroyAllWindows()
