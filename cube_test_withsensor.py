"""
 Augmented Reality using BNO055
 Developed by :
 Sagar Dixit
 Charles Basenga Kiyanda
 Nicholas Cierson
"""
import math, pygame
import logging
import time
import pyrealsense as pyrs

import cv2
import numpy as np
import sys

from pyrealsense.constants import rs_option
from operator import itemgetter

##Raspberry PI 3 Configuration
#from Adafruit_BNO055 import BNO055

# Create and configure the BNO sensor connection
# Raspberry Pi configuration with serial UART and RST connected to GPIO 18:
#bno = BNO055.BNO055(serial_port='/dev/ttyAMA0', rst=18)

# Enable verbose debug logging if -v is passed as a parameter.
#if len(sys.argv) == 2 and sys.argv[1].lower() == '-v':
#    logging.basicConfig(level=logging.DEBUG)
#################

##XU4 configuration
import BNO055
bno = BNO055.BNO055()
##########################




# Initialize the BNO055 and stop if something went wrong.
if not bno.begin():
    raise RuntimeError('Failed to initialize BNO055! Is the sensor connected?')

#############XU4 specific code section
time.sleep(1)
bno.setExternalCrystalUse(True)
#############

# Print system status and self test result.
status, self_test, error = bno.get_system_status()
print('System status: {0}'.format(status))
print('Self test result (0x0F is normal): 0x{0:02X}'.format(self_test))
# Print out an error if system status is in error mode.
if status == 0x01:
    print('System error: {0}'.format(error))
    print('See datasheet section 4.3.59 for the meaning.')

# Print BNO055 software revision and other diagnostic data.
sw, bl, accel, mag, gyro = bno.get_revision()
print('Software version:   {0}'.format(sw))
print('Bootloader version: {0}'.format(bl))
print('Accelerometer ID:   0x{0:02X}'.format(accel))
print('Magnetometer ID:    0x{0:02X}'.format(mag))
print('Gyroscope ID:       0x{0:02X}\n'.format(gyro))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

depth_fps = 30
depth_stream = pyrs.stream.DepthStream(fps=depth_fps)


class Point3D:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)
 
    def rotateX(self, angle):
        """ Rotates the point around the X axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        y =  self.y * cosa - self.z * sina 
        z =  self.y * sina + self.z * cosa
        return Point3D(self.x, y, z)
 
    def rotateY(self, angle):
        """ Rotates the point around the Y axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        z =  self.z * cosa - self.x * sina
        x =  self.z * sina + self.x * cosa
        return Point3D(x, self.y, z)
 
    def rotateZ(self, angle):
        """ Rotates the point around the Z axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        x =  self.x * cosa - self.y * sina
        y =  self.x * sina + self.y * cosa
        return Point3D(x, y, self.z)

    def translateX(self, dist):
        """ Translates the point along the X axis by the given distance """
        x =  self.x + dist
        return Point3D(x, self.y, self.z)

    def translateY(self, dist):
        """ Translates the point along the Y axis by the given distance """
        y =  self.y + dist
        return Point3D(self.x, y, self.z)

    def translateZ(self, dist):
        """ Translates the point along the Z axis by the given distance """
        z =  self.z + dist
        return Point3D(self.x, self.y, z)
 
    def project(self, win_width, win_height, fov, viewer_distance):
        """ Transforms this 3D point to 2D using a perspective projection. """
        factor = 9*fov / (viewer_distance + self.z)
        x = self.x * factor + win_width / 2
        y = -self.y * factor + win_height / 2
        return Point3D(x, y, self.z)

class Simulation:
    def __init__(self, x_initial = np.zeros(3), heading=0, pitch =0, roll = 0, win_width = 640, win_height = 480):

        pygame.init()

        self.screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("Concordia AR Demo: Cube")
        
        self.clock = pygame.time.Clock()

        self.vertices = [
            Point3D(-1, 1, 40),
            Point3D( 1, 1, 40),
            Point3D( 1,-1, 40),
            Point3D(-1,-1, 40),
            Point3D(-1, 1, 42),
            Point3D( 1, 1, 42),
            Point3D( 1,-1, 42),
            Point3D(-1,-1, 42)
        ]
        # Define the vertices that compose each of the 6 faces. These numbers are
        # indices to the vertices list defined above.
        self.faces  = [(0,1,2,3),(1,5,6,2),(5,4,7,6),(4,0,3,7),(0,4,5,1),(3,2,6,7)]

        # Define colors for each face
        self.colors = [(255,0,255),(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)]

        self.angle   = 0
        self.heading = heading
        self.roll    = roll
        self.pitch   = pitch

        self.heading_ref = heading
        self.roll_ref    = roll
        self.pitch_ref   = pitch


        self.xdist   = 0
        self.ydist   = 0
        self.zdist   = 0
        self.xvel    = 0
        self.yvel    = 0
        self.zvel    = 0
        self.xaccel  = 0
        self.yaccel  = 0
        self.zaccel  = 0

        self.x_now    = x_initial
        self.x_center = np.zeros(3)
        self.current_time  = time.time()
        self.previous_time = time.time()
                   
    def run(self):
            """ Main Loop """

            while 1:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                self.clock.tick(50)
                self.screen.fill((0,32,0))

                # It will hold transformed vertices.
                t = []
                
                for v in self.vertices:
                    # Translate the point about X axis, then about Y axis, and finally about Z axis.
                    r = v.rotateX(self.heading).rotateY(self.roll).rotateZ(self.pitch).translateX(self.xdist).translateY(self.ydist).translateZ(self.zdist)
                             
                   # Transform the point from 3D to 2D
                    p = r.project(self.screen.get_width(), self.screen.get_height(), 256, 4)
                    # Put the point in the list of transformed vertices
                    t.append(p)

                # Calculate the average Z values of each face.
                avg_z = []
                i = 0
                for f in self.faces:
                    z = (t[f[0]].z + t[f[1]].z + t[f[2]].z + t[f[3]].z) / 4.0
                    avg_z.append([i,z])
                    i = i + 1

                # Draw the faces using the Painter's algorithm:
                # Distant faces are drawn before the closer ones.
                for tmp in sorted(avg_z,key=itemgetter(1),reverse=True):
                    face_index = tmp[0]
                    f = self.faces[face_index]
                    pointlist = [(t[f[0]].x, t[f[0]].y), (t[f[1]].x, t[f[1]].y),
                                 (t[f[1]].x, t[f[1]].y), (t[f[2]].x, t[f[2]].y),
                                 (t[f[2]].x, t[f[2]].y), (t[f[3]].x, t[f[3]].y),
                                 (t[f[3]].x, t[f[3]].y), (t[f[0]].x, t[f[0]].y)]
                    pygame.draw.polygon(self.screen,self.colors[face_index],pointlist)
                    
                self.heading, self.roll, self.pitch = bno.read_euler()
                self.heading -= self.heading_ref
                self.pitch   -= self.pitch_ref
                self.roll    -= self.roll_ref

                self.current_time = time.time()
                self.heading = -self.heading
                self.pitch   = -self.pitch
                #print(self.heading)
           	    #print(self.pitch)
                #print(self.roll)
                #print("accelerations, x, y, z")
                #print(self.xaccel)
                #print(self.yaccel)
                #print(self.zaccel)
                #print(self.zvel)
                #print(self.zdist)
                print("time since previous measurement")
                print(self.current_time - self.previous_time)
                print(self.current_time)
                
                #self.angle += 2
                #self.zdist += 0
                #self.ydist += 0
                #self.xdist += 0
               
                dt = (self.current_time - self.previous_time)	    

                #acquire absolute position
                dev.wait_for_frames()
                img = dev.color
                img2 = dev.depth
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (9,7),None)
                         
                if ret == True:
                    self.x_center = dev.deproject_pixel_to_point(corners[31][0], img2[corners[31][0][0]][corners[31][0][1]])
                    if sum(self.x_center) == 0:
                        dist  = 0*(self.x_center - self.x_now)
                    else:
                        dist = self.x_center - self.x_now
                        self.x_now = self.x_center
                        self.x_dist = dist[0]
                        self.y_dist = dist[1]
                        self.z_dist = dist[2]
                        self.x_now = self.x_center
             
                pygame.display.flip()
                self.previous_time = self.current_time


if __name__ == "__main__":
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #objp = np.zeros((9*7,3), np.float32)
    #objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)


    with pyrs.Service() as serv:
        with serv.Device() as dev:
            dev.apply_ivcam_preset(0)
            try:  # set custom gain/exposure values to obtain good depth image
                custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 30.0),
                                      (rs_option.RS_OPTION_R200_LR_GAIN, 100.0)]
                dev.set_device_options(*zip(*custom_options))
            except pyrs.RealsenseError:
                pass  # options are not available on all devices 

            cnt = 0 
            center = np.zeros(3)
            heading_ref = 0
            roll_ref    = 0
            pitch_ref   = 0
            while cnt<100:
                dev.wait_for_frames()
                img = dev.color
                img2 = dev.depth
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (9,7),None)
            
                print (ret)                            
                if ret == True:
                    ##corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                    #center point is corners[31][0]


                    cv2.drawChessboardCorners(img, (9,7), corners,ret) 
                    
                    vec_center = dev.deproject_pixel_to_point(corners[31][0], img2[corners[31][0][0]][corners[31][0][1]])
                    heading, roll, pitch = bno.read_euler()
                
                    print(vec_center)
                    if np.sum(vec_center) == 0.0:
                        print("bad point")
                    else:
                        cnt += 1
                        center      = ((cnt-1)*center + vec_center)/cnt
                        heading_ref = ((cnt-1)*heading_ref + heading)/cnt
                        pitch_ref   = ((cnt-1)*pitch_ref   + pitch)/cnt
                        roll_ref    = ((cnt-1)*roll_ref    + roll)/cnt
                        print(center)
                ##cv2.imshow('img',img)

            ##cv2.destroyAllWindows()

            sim = Simulation(center, heading_ref, pitch_ref, roll_ref)
            sim.run()
