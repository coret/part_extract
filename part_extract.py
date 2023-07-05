import numpy as np
import cv2

filename="in/NL-HtBHIC_7737_1541_0003.jpg"
img = cv2.imread(filename)

# Prepocess
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(1,1),1000)
flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea,reverse=True) 

# Select long perimeters only
perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
listindex=[i for i in range(6) if perimeters[i]>perimeters[0]/2]
numcards=len(listindex)

# warp per contour
for i in range(numcards):
    card = contours[i]
    peri = cv2.arcLength(card,True)
    approx = cv2.approxPolyDP(card,0.02*peri,True)
    h = np.array([ [0,0],[0,449],[749,449],[749,0] ],np.float32)
    approx = np.array([item for sublist in approx for item in sublist],np.float32)
    transform = cv2.getPerspectiveTransform(approx,h)
    warp = cv2.warpPerspective(img,transform,(750,450))
    filename2="out/NL-HtBHIC_7737_1541_0003_"+str(i)+".jpg"
    cv2.imwrite(filename2, warp)