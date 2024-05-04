import cv2
import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
video = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results1 = pose.process(frame_rgb)
        results = holistic.process(frame_rgb)
        if results1.pose_landmarks:
            landmarks = results1.pose_landmarks.landmark
            x_max = 0
            y_max = 0
            x_min = frame.shape[1]
            y_min = frame.shape[0]

            for lm in landmarks:

                x, y = round(lm.x * frame.shape[1]), round(lm.y * frame.shape[0])
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            '''
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # 3. Left Hand
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )'''

        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            # Break the loop.
            break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()