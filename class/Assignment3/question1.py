import cv2 as cv
import numpy as np
from PIL import Image
import os

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

file_path = '/home/dl/Public/GSW/project/Digital-image-processing/class/Assignment3/data/orl_faces/'
path = os.listdir(file_path)
imgs = []
for i in path:
    file = os.path.join(file_path, i)
    if os.path.isdir(file):
        for j in os.listdir(file):
            img_path = os.path.join(file, j)
            imgs.append(img_path)

img = cv.imread(imgs[2])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()