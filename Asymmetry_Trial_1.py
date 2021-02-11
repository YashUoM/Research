#2021-01-19

import cv2
import numpy as np
import os

path = 'images/HAM10000-80-MEL/'

#iterate through each file to perform some action
for image in os.listdir(path):
    
    img = cv2.imread(os.path.join(path, image))
    imgcopy = img.copy()

    #hair removal & to get a bit of blur edges
    kernel = np.ones((5,5),np.uint8)
    img2 = cv2.morphologyEx(imgcopy, cv2.MORPH_CLOSE, kernel)

    #convert to gray & binary
    imgray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgray2,(5,5),0)
    ret, thresh2 = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY_INV)

    #find the maximum contour and print
    contours, heir = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)
    
    tmpArea = np.zeros(img.shape,np.uint8)
    cv2.drawContours(tmpArea,[c],0,(255, 255, 255),cv2.FILLED)
    cv2.imshow("only contour image", tmpArea)
    
    cv2.imshow("original image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()