
# Built on top of the Simple Adafruit BNO055 sensor reading example
# Original example copyright: Adafruit Industries (Author: Tony DiCola)
# Original example license BSD 3-clause
############# Original Example License Text #######################
# Simple Adafruit BNO055 sensor reading example.  Will print the orientation
# and calibration data every second.
#
# Copyright (c) 2015 Adafruit Industries
# Author: Tony DiCola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
############# Original Example License Text #######################

import logging
import sys
import time
import matplotlib.pyplot as plt

#from Adafruit_BNO055 import BNO055
import BNO055 #simpler, alternate library that works with XU4

# Raspberry Pi configuration with serial UART and RST connected to GPIO 18:
#bno = BNO055.BNO055(serial_port='/dev/ttyAMA0', rst=18)
# Enable verbose debug logging if -v is passed as a parameter.
#if len(sys.argv) == 2 and sys.argv[1].lower() == '-v':
#    logging.basicConfig(level=logging.DEBUG)

#For XU4, initialization of sensor
bno = BNO055.BNO055()

# Initialize the BNO055 and stop if something went wrong.
if not bno.begin():
    raise RuntimeError('Failed to initialize BNO055! Is the sensor connected?')

#############XU4 specific code section
time.sleep(1)
bno.setExternalCrystalUse(True)
#############
	
# Print system status and self test result.
status, self_test, error = bno.get_system_status() ##adafruit
#status, self_test, error = bno.getSystemStatus() ##XU4
print('System status: {0}'.format(status))
print('Self test result (0x0F is normal): 0x{0:02X}'.format(self_test))
# Print out an error if system status is in error mode.
if status == 0x01:
    print('System error: {0}'.format(error))
    print('See datasheet section 4.3.59 for the meaning.')

# Print BNO055 software revision and other diagnostic data.
sw, bl, accel, mag, gyro = bno.get_revision() ##adafruit
#accel, mag, gyro, sw, bl  = bno.getRevInfo() ##XU4
print('Software version:   {0}'.format(sw))
print('Bootloader version: {0}'.format(bl))
print('Accelerometer ID:   0x{0:02X}'.format(accel))
print('Magnetometer ID:    0x{0:02X}'.format(mag))
print('Gyroscope ID:       0x{0:02X}\n'.format(gyro))

t_rel = 0

t_vec  = []
ax_vec = []
ay_vec = []
az_vec = []

print('Reading BNO055 data, press Ctrl-C to quit...')
t_start   = time.time()
accel     = 0
calibrate = False #True
t_rel = -1
while (calibrate):
    # Read the calibration status, 0=uncalibrated and 3=fully calibrated.
    sys, gyro, accel, mag = bno.get_calibration_status()
    ax,ay,az = bno.read_linear_acceleration()
    print('Ax={0:0.2F} Ay={1:0.2F} Az={2:0.2F}\tSys_cal={3} Gyro_cal={4} Accel_cal={5} Mag_cal={6}'.format(
         ax,ay,az, sys, gyro, accel, mag))
    if (accel == 3):
        t_rel = time.time() - t_start
    else:
        t_start = time.time()

    if ((accel == 3) and (t_rel >10)):
        calibrate = False
    time.sleep(0.01)

print('CALIBRATED!')
print('Calibration lasted for {0:0.2F}'.format(t_rel) )
time.sleep(10)

t_start = time.time()
t_rel   = 0
dt      = 0.05
while (t_rel<15):
    # Read the calibration status, 0=uncalibrated and 3=fully calibrated.
    # sys, gyro, accel, mag = bno.get_calibration_status()
    # Linear acceleration data (i.e. acceleration from movement, not gravity--
    # returned in meters per second squared):
    ax,ay,az = bno.read_linear_acceleration() #adafruit
    #(ax,ay,az) = bno.getVector(BNO055.BNO055.VECTOR_LINEARACCEL)
    t_rel = time.time() - t_start
    t_vec.append(t_rel)
    ax_vec.append(ax)
    ay_vec.append(ay)
    az_vec.append(az)
    # Sleep for dt  until the next reading.
    time.sleep(dt)

plt.plot(t_vec,ax_vec,'r',label='Ax')
plt.plot(t_vec,ay_vec,'g',label='Ay')
plt.plot(t_vec,az_vec,'b',label='Az')
plt.legend()
plt.title('Linear acceleration output from BNO055') 
plt.xlabel('Time in sec')
plt.ylabel('Acceleration in m/s2')
plt.savefig('Linear acceleration BNO055')
plt.show()
