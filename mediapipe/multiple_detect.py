import cv2

video = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    _, frame = video.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        Reading the Image
    framergb = cv2.imread('') # input img/vid path here

    (humans, _) = hog.detectMultiScale(frame, winStride=(10, 10),
    padding=(32, 32), scale=1.1)

    # getting no. of human detected
    print('Human Detected : ', len(humans))

    for (x, y, w, h) in humans:
       pad_w, pad_h = int(0.15 * w), int(0.01 * h)
       cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)

    # display the output image
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1) & 0xFF
    if (k == 27):
        break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()