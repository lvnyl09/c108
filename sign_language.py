import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(1)

finger_tips =[8, 12, 16, 20]
thumb_tip= 4

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            fingertips = [lm_list[i] for i in finger_tips]
            for i in range(len(finger_tips)):
                x = int(fingertips[i].x * w)
                y = int(fingertips[i].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

            finger_fold_status = []
            for i in range(len(finger_tips)):
                if lm_list[finger_tips[i]].x < lm_list[finger_tips[i]-1].x:
                    x = int(lm_list[finger_tips[i]].x * w)
                    y = int(lm_list[finger_tips[i]].y * h)
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            if all(finger_fold_status):
                if lm_list[thumb_tip].y < lm_list[thumb_tip-1].y:
                    cv2.putText(img, "LIKE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
                else:
                    cv2.putText(img, "DISLIKE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

            mp_draw.draw_landmarks(img, hand_landmark,
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2))

    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)
