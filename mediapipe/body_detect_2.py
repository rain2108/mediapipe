import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

# Initialize MediaPipe Face Detection
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection on the frame
    results = detector.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        # Break the loop.
        break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()
