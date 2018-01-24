"""
 Streamlines of a flow past a sphere
 Developed by Sagar Dixit
"""
import sys, math, pygame
import logging
import time

import numpy as np

from operator import itemgetter
from Adafruit_BNO055 import BNO055

def solve_for_roots(poly_coeffs, y):
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.roots(pc)

# Create and configure the BNO sensor connection
# Raspberry Pi configuration with serial UART and RST connected to GPIO 18:
bno = BNO055.BNO055(serial_port='/dev/ttyAMA0', rst=18)

# Enable verbose debug logging if -v is passed as a parameter.
if len(sys.argv) == 2 and sys.argv[1].lower() == '-v':
    logging.basicConfig(level=logging.DEBUG)

# Initialize the BNO055 and stop if something went wrong.
if not bno.begin():
    raise RuntimeError('Failed to initialize BNO055! Is the sensor connected?')


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




class Point3D:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)
 
    def rotateX(self, angleo):
        """ Rotates the point around the X axis by the given angle in degrees. """
        rad = angleo * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        y =  self.y * cosa - self.z * sina 
        z =  self.y * sina + self.z * cosa
        return Point3D(self.x, y, z)
 
    def rotateY(self, angleo):
        """ Rotates the point around the Y axis by the given angle in degrees. """
        rad = angleo * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        z =  self.z * cosa - self.x * sina
        x =  self.z * sina + self.x * cosa
        return Point3D(x, self.y, z)
 
    def rotateZ(self, angleo):
        """ Rotates the point around the Z axis by the given angle in degrees. """
        rad = angleo * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        x =  self.x * cosa - self.y * sina
        y =  self.x * sina + self.y * cosa
        return Point3D(x, y, self.z)
 
    def project(self, win_width, win_height, fov, viewer_distance):
        """ Transforms this 3D point to 2D using a perspective projection. """
        factor = 0.23*fov / (viewer_distance + self.z)
        x = self.x * factor + win_width / 2
        y = -self.y * factor + win_height / 2
        return Point3D(x, y, self.z)

class Simulation:
    def __init__(self, win_width = 800, win_height = 480):
        pygame.init()

        self.screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("ENGR 6981 Project- Sagar Dixit")
        
        self.clock = pygame.time.Clock()

        a = 20      # radius of a sphere
        psi1 = 1    # stream function 1
        psi2 = 0.3  # stream function 2
        psi3 = 0.1  # stream function 3
        psi4 = 0.03 # stream function 4
        psi5 = 0.01 # stream function 5
        w = 50      # Velocity of a flow
  
        angles = np.linspace(0.01,np.pi-0.01,9) # range of theta
        angler = 60*np.pi/180     # angle phi 1
        angleq = 20*np.pi/180     # angle phi 2
        anglep = 0*np.pi/180      # angle phi 3 (x-y plane)

        root1 = np.zeros(len(angles))
        root2 = np.zeros(len(angles))
        root3 = np.zeros(len(angles))
        root4 = np.zeros(len(angles))
        root5 = np.zeros(len(angles))
        i = 0
        for angle in angles:
          poly_coeffs1 = np.array([1,-1.5*a, -2*psi1/(w*(np.sin(angle))**2), 0.5*a**3])
          roots1 = solve_for_roots(poly_coeffs1,0)
          root1[i] = max(roots1)

          poly_coeffs2 = np.array([1,-1.5*a, -2*psi2/(w*(np.sin(angle))**2), 0.5*a**3])
          roots2 = solve_for_roots(poly_coeffs2,0)
          root2[i] = max(roots2)

          poly_coeffs3 = np.array([1,-1.5*a, -2*psi3/(w*(np.sin(angle))**2), 0.5*a**3])
          roots3 = solve_for_roots(poly_coeffs3,0)
          root3[i] = max(roots3)

          poly_coeffs4 = np.array([1,-1.5*a, -2*psi4/(w*(np.sin(angle))**2), 0.5*a**3])
          roots4 = solve_for_roots(poly_coeffs4,0)
          root4[i] = max(roots4)

          poly_coeffs5 = np.array([1,-1.5*a, -2*psi5/(w*(np.sin(angle))**2), 0.5*a**3])
          roots5 = solve_for_roots(poly_coeffs5,0)
          root5[i] = max(roots5)
                  
          i = i +1

          x1 = root1*np.cos(angles)
          y1 = root1*np.sin(angles)*np.cos(anglep)
          z1 = root1*np.sin(angles)*np.sin(anglep)

          x2 = root2*np.cos(angles)
          y2 = root2*np.sin(angles)*np.cos(anglep)
          z2 = root2*np.sin(angles)*np.sin(anglep)

          x3 = root3*np.cos(angles)
          y3 = root3*np.sin(angles)*np.cos(anglep)
          z3 = root3*np.sin(angles)*np.sin(anglep)

          x4 = root4*np.cos(angles)
          y4 = root4*np.sin(angles)*np.cos(anglep)
          z4 = root4*np.sin(angles)*np.sin(anglep)

          x5 = root5*np.cos(angles)
          y5 = root5*np.sin(angles)*np.cos(anglep)
          z5 = root5*np.sin(angles)*np.sin(anglep)

          x6 = root1*np.cos(angles)
          y6 = root1*np.sin(angles)*np.cos(angleq)
          z6 = root1*np.sin(angles)*np.sin(angleq)

          x7 = root2*np.cos(angles)
          y7 = root2*np.sin(angles)*np.cos(angleq)
          z7 = root2*np.sin(angles)*np.sin(angleq)

          x8 = root3*np.cos(angles)
          y8 = root3*np.sin(angles)*np.cos(angleq)
          z8 = root3*np.sin(angles)*np.sin(angleq)


          x9 = root4*np.cos(angles)
          y9 = root4*np.sin(angles)*np.cos(angleq)
          z9 = root4*np.sin(angles)*np.sin(angleq)

          x10 = root5*np.cos(angles)
          y10 = root5*np.sin(angles)*np.cos(angleq)
          z10 = root5*np.sin(angles)*np.sin(angleq)

          x11 = root1*np.cos(angles)
          y11 = root1*np.sin(angles)*np.cos(angler)
          z11 = root1*np.sin(angles)*np.sin(angler)

          x12 = root2*np.cos(angles)
          y12 = root2*np.sin(angles)*np.cos(angler)
          z12 = root2*np.sin(angles)*np.sin(angler)

          x13 = root3*np.cos(angles)
          y13 = root3*np.sin(angles)*np.cos(angler)
          z13 = root3*np.sin(angles)*np.sin(angler)


          x14 = root4*np.cos(angles)
          y14 = root4*np.sin(angles)*np.cos(angler)
          z14 = root4*np.sin(angles)*np.sin(angler)

          x15 = root5*np.cos(angles)
          y15 = root5*np.sin(angles)*np.cos(angler)
          z15 = root5*np.sin(angles)*np.sin(angler) 

        self.vertices = []
        for i in range(len(x1)):
          self.vertices.append(Point3D(x1[i],y1[i],z1[i]))
          self.vertices.append(Point3D(x2[i],y2[i],z2[i]))
          self.vertices.append(Point3D(x3[i],y3[i],z3[i]))
          self.vertices.append(Point3D(x4[i],y4[i],z4[i]))
          self.vertices.append(Point3D(x5[i],y5[i],z5[i]))
          self.vertices.append(Point3D(x6[i],y6[i],z6[i]))
          self.vertices.append(Point3D(x7[i],y7[i],z7[i]))
          self.vertices.append(Point3D(x8[i],y8[i],z8[i]))
          self.vertices.append(Point3D(x9[i],y9[i],z9[i]))
          self.vertices.append(Point3D(x10[i],y10[i],z10[i]))
          self.vertices.append(Point3D(x11[i],y11[i],z11[i]))
          self.vertices.append(Point3D(x12[i],y12[i],z12[i]))
          self.vertices.append(Point3D(x13[i],y13[i],z13[i]))
          self.vertices.append(Point3D(x14[i],y14[i],z14[i]))
          self.vertices.append(Point3D(x15[i],y15[i],z15[i]))
          
        # Define the vertices that compose all set of segments of streamlines as a  faces. These numbers are indices to the vertices list defined above.
        self.faces  = [(0,0,15,15),(15,15,30,30),(30,30,45,45),(45,45,60,60),(60,60,75,75),(75,75,90,90),(90,90,105,105),(105,105,120,120),(1,1,16,16),(16,16,31,31),(31,31,46,46),(46,46,61,61),(61,61,76,76),(76,76,91,91),(91,91,106,106),(106,106,121,121),(2,2,17,17),(17,17,32,32),(32,32,47,47),(47,47,62,62),(62,62,77,77),(77,77,92,92),(92,92,107,107),(107,107,122,122),(3,3,18,18),(18,18,33,33),(33,33,48,48),(48,48,63,63),(63,63,78,78),(78,78,93,93),(93,93,108,108),(108,108,123,123),(4,4,19,19),(19,19,34,34),(34,34,49,49),(49,49,64,64),(64,64,79,79),(79,79,94,94),(94,94,109,109),(109,109,124,124),(5,5,20,20),(20,20,35,35),(35,35,50,50),(50,50,65,65),(65,65,80,80),(80,80,95,95),(95,95,110,110),(110,110,125,125),(6,6,21,21),(21,21,36,36),(36,36,51,51),(51,51,66,66),(66,66,81,81),(81,81,96,96),(96,96,111,111),(111,111,126,126),(7,7,22,22),(22,22,37,37),(37,37,52,52),(52,52,67,67),(67,67,82,82),(82,82,97,97),(97,97,112,112),(112,112,127,127),(8,8,23,23),(23,23,38,38),(38,38,53,53),(53,53,68,68),(68,68,83,83),(83,83,98,98),(98,98,113,113),(113,113,128,128),(9,9,24,24),(24,24,39,39),(39,39,54,54),(54,54,69,69),(69,69,84,84),(84,84,99,99),(99,99,114,114),(114,114,129,129),(10,10,25,25),(25,25,40,40),(40,40,55,55),(55,55,70,70),(70,70,85,85),(85,85,100,100),(100,100,115,115),(115,115,130,130),(11,11,26,26),(26,26,41,41),(41,41,56,56),(56,56,71,71),(71,71,86,86),(86,86,101,101),(101,101,116,116),(116,116,131,131),(12,12,27,27),(27,27,42,42),(42,42,57,57),(57,57,72,72),(72,72,87,87),(87,87,102,102),(102,102,117,117),(117,117,132,132),(13,13,28,28),(28,28,43,43),(43,43,58,58),(58,58,73,73),(73,73,88,88),(88,88,103,103),(103,103,118,118),(118,118,133,133),(14,14,29,29),(29,29,44,44),(44,44,59,59),(59,59,74,74),(74,74,89,89),(89,89,104,104),(104,104,119,119),(119,119,134,134)]
        # Define colors for each face
        self.colors = [(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,255,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0)]

        self.angleo  = 0
        self.heading = 0
        self.roll    = 0
        self.pitch   = 0
        
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
                # Rotate the point around X axis, then around Y axis, and finally around Z axis.
                r = v.rotateX(self.roll).rotateY(self.heading).rotateZ(self.pitch)
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
            self.heading = -self.heading
            self.pitch   = -self.pitch
	    print(self.heading)
 	    print(self.pitch)
	    print(self.roll)
            self.angleo += 0
            
            pygame.display.flip()

if __name__ == "__main__":
    Simulation().run()
