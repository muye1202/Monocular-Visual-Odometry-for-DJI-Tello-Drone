# Monocular Visual Odometry for Tello Drone

Author: Muye Jia

The pipeline starts with camera calibration for the RGB camera located on the front
of the drone; then the images captured will be converted to grayscale, and subsequently fed to
ORB or ShiTomasi feature extractor. Features extracted from the previous frame will be
matched in the next frame to find the image coordinates of the same set of features in the
previous frame and current frame, which can then be used to solve the epipolar constraint
equation for the essential matrix that contains the camera rotation and translation matrix
representing the pose transformation between the two frames. Next, the transformation matrices
will be stitched together to form the entire camera trajectory


## Camera Calibration

1. Calibration photos using a grided chessboard need to be taken using the target camera, from at least 10 different angles and views. Run `python3 cam_calibrate.py` with the `take_calibration_photos` function enabled, and the host computer connect to the DJI Tello drone WiFi; the script will take photos 
using Tello drone camera at at a frequency of 10Hz.

2. Next, use the photos taken and run `python3 cam_calibrate.py` with the `calibrate_camera` function enabled, the `cam_matrix` variable 
contains the intrinsic parameters of the camera.

## Visual Odometry Test

1. Using KITTI dataset as validation. Run `python3 visual_odometry.py` with `plot_KITTI` function enabled to test the VO pipeline
on the KITTI dataset, one can choose to enable the bundle-adjustment.

2. To test the VO on the drone, the user need to record images using the drone and then perform the odometry offline by running `python3 visual_odometry.py` with `plot_drone` function enabled.