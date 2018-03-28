import logging
logging.basicConfig(level=logging.INFO)

import pyrealsense as pyrs
import cv2
import numpy as np
import sys 
from pyrealsense.constants import rs_option

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

depth_fps = 30
depth_stream = pyrs.stream.DepthStream(fps=depth_fps)

with pyrs.Service() as serv:
    with serv.Device() as dev:
        dev.apply_ivcam_preset(0)

        try:  # set custom gain/exposure values to obtain good depth image
            custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 30.0),
                              (rs_option.RS_OPTION_R200_LR_GAIN, 100.0)]
            dev.set_device_options(*zip(*custom_options))
        except pyrs.RealsenseError:
            pass  # options are not available on all devices

        while True:
            dev.wait_for_frames()
            img = dev.color
            img2 = dev.depth
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,7),None)
            print (ret)                            
            if ret == True:
                ##corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                ## print (corners)
                #center point is corners[31][0]


                cv2.drawChessboardCorners(img, (9,7), corners,ret) 

                print(dev.deproject_pixel_to_point(corners[31][0], img2[corners[31][0][0]][corners[31][0][1]]) - dev.deproject_pixel_to_point(corners[32][0], img2[corners[32][0][0]][corners[32][0][1]]))

            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xff == ord('q'):
              break

cv2.destroyAllWindows()
