import cv2
import os
import numpy as np
webcam = cv2.VideoCapture(0)
FaceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
(images, labels, names, id) = ([], [], {}, 0)
for subdirs, dirs, files in os.walk("datasets"):
    for subdir in dirs:
        names[id] = subdir
        path = os.path.join("datasets", subdir)
        img = os.listdir(path)
        for file in img:
            picture = path + "/" + file
            label = id
            images.append(cv2.imread(picture, 0))
            labels.append(label)
        id += 1
print(images)
print(labels)
print(names)
(images, labels) = [np.array(lis) for lis in [images, labels]]
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#training the model with data from data sets
face_recognizer.train(images, labels)
print("Camera starting")
while True:
    ret, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = FaceDetector.detectMultiScale(gray, 2, 3)
    print(face)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (240, 65, 3), 5)
        face_resize = gray[y:y+h, x:x+w]
        picture = cv2.resize(face_resize, (130,100))
        prediction = face_recognizer.predict(picture)
        if prediction[1] > 60:
            cv2.putText(img, f"{names[prediction[0]]} -{round(prediction[1], 2)}", (x - 10, y - 20), cv2.FONT_HERSHEY_PLAIN, 5, (240, 100, 10), 5)
        print(prediction)
    cv2.imshow("webcam.mp4", img)
    k = cv2.waitKey(10)
    if k == 27:
        break