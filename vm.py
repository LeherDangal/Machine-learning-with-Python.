import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)


   
      index_finger_x = int(hand_landmarks.landmark[8].x * screen_width)
      index_finger_y = int(hand_landmarks.landmark[8].y * screen_height)


      pyautogui.moveTo(index_finger_x,index_finger_y, duration=0.1)


      thumb_x,thumb_y = int(hand_landmarks.landmark[4].x*screen_width),int(hand_landmarks.landmark[4].y*screen_height)

      distance = np.sqrt((thumb_x - index_finger_x)**2 + (thumb_y-index_finger_y)**2)

      if distance < 20:
         pyautogui.click()
         pyautogui.sleep(0.2)


         middle_finger_x = int(hand_landmarks.landmark[12].x*screen_width)
         middle_finger_y = int(hand_landmarks.landmark[12].y*screen_height)


         if abs(index_finger_x - middle_finger_x)<20:
          pyautogui.mouseDown()
         else :
          pyautogui.mouseUp()

      if distance > 100:
       if index_finger_y < thumb_y:
        pyautogui.scroll(5)
       elif index_finger_y> thumb_y:
        pyautogui.scroll(-5)



      mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual mouse",frame)

    if cv2.waitKey(1) & 0XFF == ord ('e'):
     break

cap.release()
cv2.destroyAllWindows()