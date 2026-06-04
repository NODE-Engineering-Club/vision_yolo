from rplidar import RPLidar
import time

# Replace 'COM3' with your actual serial port
PORT_NAME = 'COM8' 

try:
    lidar = RPLidar(PORT_NAME)
    print("Lidar connected successfully!")
    info = lidar.get_info()
    print(f"Lidar Info: {info}")
    health = lidar.get_health()
    print(f"Lidar Health: {health}")
    lidar.stop()
    lidar.disconnect()
    print("Lidar disconnected.")

except Exception as e:
    print(f"Error connecting to Lidar: {e}")