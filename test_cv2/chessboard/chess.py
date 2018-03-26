import cv2
import numpy as np
import sys


# Load previously saved calibration data
with np.load('../calibration.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    print(mtx)
    print(dist)
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

video_capture = cv2.VideoCapture(0)
center = np.ndarray((1,1,2),dtype=np.float32)
while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,7),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        #center point is corners[31][0]


        cv2.drawChessboardCorners(img, (9,7), corners2,ret) 

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xff == ord('q'):
      break
    
video_capture.release()
cv2.destroyAllWindows()
