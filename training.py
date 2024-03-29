import os
import numpy as np
from PIL import Image
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('/home/jrkjithin/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
y_label = []
x_train = []
current_id = 0
label_id = {}

recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ","_").lower()
			#print(label)
			if not label in label_id:
				label_id[label] = current_id
				current_id += 1
			id_ =  label_id[label]
			#print(label_id)

			pil_image = Image.open(path).convert("L")
			size = (550,550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(pil_image, "uint8")
			#print(image_array)
			faces= face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_label.append(id_)

with open("label.pickle","wb") as f:
	pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")