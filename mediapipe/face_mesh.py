import mediapipe as mp
import cv2

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize Face Mesh model
face_mesh = mp_face_mesh.FaceMesh()

# Open a video capture source (e.g., camera)
cap = cv2.VideoCapture(0)  # Change the index if using a different video source

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect facial landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Iterate through the detected landmarks
            for id, lm in enumerate(landmarks.landmark):
                # Print the coordinates of each facial landmark
                print(f"Landmark {id}: X={lm.x}, Y={lm.y}, Z={lm.z}")

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # Display the frame with landmarks
    cv2.imshow('Face Mesh', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
