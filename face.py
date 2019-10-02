import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    '/home/jrkjithin/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
label={"person" : 1}
with open("label.pickle","rb") as f:
	og_label = pickle.load(f)
label = {v:k for k,v in og_label.items()}
print(label)
cap = cv2.VideoCapture(0)
color = (255, 0, 0)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi = gray[y:y + h, x:x + w]
        roiframe = frame[y:y + h, x:x + w]
        img_item = "jrk2.png"
        cv2.imwrite(img_item, roiframe)
        stroke = 2
        endcord_x = x + w
        endcord_y = y + h
        cv2.rectangle(frame, (x, y), (endcord_x, endcord_y), color, stroke)
        id_,conf = recognizer.predict(roi)
        if conf>=45:
            font=cv2.FONT_HERSHEY_SIMPLEX
            name = label[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
