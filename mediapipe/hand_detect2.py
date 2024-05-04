import cv2
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = frame.shape[1]
            y_min = frame.shape[0]
            for lm in handLMs.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)

        # Display "Hands Detected" message
        cv2.putText(frame, 'Hands Detected', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    else:  # No hands detected
        cv2.putText(frame, 'No Hands Detected', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1) & 0xFF

    if (k == 27):
        # Break the loop.
        break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()
