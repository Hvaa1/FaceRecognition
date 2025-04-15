import cv2
import tensorflow as tf
import numpy as  np
import math
import os
import time
from FaceRecognition import load_model,verify
def draw_bounding_box(img,classifier,scaleFactor,minNeighbor,color,text):
    gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img,scaleFactor,minNeighbor)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_COMPLEX,0.8,color,1,cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords,img
def detect(img,faceCascade):
    color = {"blue":(255,0,0),"green":(0,250,0),"red":(0,0,255)}
    coords, img = draw_bounding_box(img,faceCascade,1.1,10,color["green"],"Face")
    return coords,img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
siamese_model = load_model("face_recognition_siamese.keras")
cap = cv2.VideoCapture(0)
threshold_time = 10
while True:
    res,frame = cap.read()
    coords ,frame_with_box = detect(frame,faceCascade)
    if coords:
        x, y, w, h = coords
        face = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Dự đoán giống bao nhiêu %
        current_time = time.time()
        if current_time > threshold_time:
            cv2.imwrite(os.path.join("application_data","input_images",'input_image.jpg'),face)
            results, verified = verify(siamese_model, 0.9, 0.25)
            print(verified)
    cv2.imshow("nhan dien mat",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()