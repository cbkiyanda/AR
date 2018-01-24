"""
 Augmented Reality using BNO055
 Developed by Sagar Dixit
"""
import sys, math, pygame
import logging
import time

from operator import itemgetter
#####from Adafruit_BNO055 import BNO055

# Create and configure the BNO sensor connection
# Raspberry Pi configuration with serial UART and RST connected to GPIO 18:
#####bno = BNO055.BNO055(serial_port='/dev/ttyAMA0', rst=18)

# Enable verbose debug logging if -v is passed as a parameter.
#####if len(sys.argv) == 2 and sys.argv[1].lower() == '-v':
#####    logging.basicConfig(level=logging.DEBUG)

# Initialize the BNO055 and stop if something went wrong.
#####if not bno.begin():
#####    raise RuntimeError('Failed to initialize BNO055! Is the sensor connected?')

# Print system status and self test result.
#####status, self_test, error = bno.get_system_status()
#####print('System status: {0}'.format(status))
#####print('Self test result (0x0F is normal): 0x{0:02X}'.format(self_test))
# Print out an error if system status is in error mode.
#####if status == 0x01:
#####    print('System error: {0}'.format(error))
#####    print('See datasheet section 4.3.59 for the meaning.')

# Print BNO055 software revision and other diagnostic data.
#####sw, bl, accel, mag, gyro = bno.get_revision()
#####print('Software version:   {0}'.format(sw))
#####print('Bootloader version: {0}'.format(bl))
#####print('Accelerometer ID:   0x{0:02X}'.format(accel))
#####print('Magnetometer ID:    0x{0:02X}'.format(mag))
#####print('Gyroscope ID:       0x{0:02X}\n'.format(gyro))

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
    def __init__(self, win_width = 640, win_height = 480):
        pygame.init()

        self.screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("ENGR 6981 Project- Sagar Dixit")
        
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
        self.heading = 0
        self.roll    = 0
        self.pitch   = 0
        self.xdist   = 0
        self.ydist   = 0
        self.zdist   = 0
        self.xvel    = 0
        self.yvel    = 0
        self.zvel    = 0
        self.xaccel  = 0
        self.yaccel  = 0
        self.zaccel  = 0
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
                r = v.rotateX(self.angle).rotateY(self.angle).rotateZ(self.angle).translateX(self.xdist).translateY(self.ydist).translateZ(self.zdist)
                         
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
                
            #####self.heading, self.roll, self.pitch = bno.read_euler()
            #####self.xaccel, self.yaccel, self.zaccel = bno.read_linear_acceleration()
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
            
            self.angle += 2
            #self.zdist += 0
            #self.ydist += 0
            #self.xdist += 0
           
            dt = (self.current_time - self.previous_time)	    

            self.zdist   = self.zdist + self.zvel*dt + 0.5*(self.zaccel)*dt**2
            self.zvel    = self.zvel + self.zaccel*dt

            self.ydist   = self.ydist + self.yvel*dt + 0.5*(self.yaccel)*dt**2
            self.yvel    = self.yvel + self.yaccel*dt

            self.xdist   = self.xdist + self.xvel*dt + 0.5*(self.xaccel)*dt**2
            self.xvel    = self.xvel + self.xaccel*dt
         
            pygame.display.flip()
            self.previous_time = self.current_time


if __name__ == "__main__":
    Simulation().run()
