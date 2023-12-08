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

## Results
Real-time visual odometry on DJI Tello Drone


https://github.com/muye1202/Monocular-Visual-Odometry-for-DJI-Tello-Drone/assets/112987403/bc1caa46-e5eb-42e7-90ea-1e7ab73334a6


The following are the trajectories obtained from real deployment on DJI Tello Drone:
![line_1turn_VO_ORB](https://github.com/muye1202/Monocular-Visual-Odometry-for-DJI-Tello-Drone/assets/112987403/cb05911d-2038-4fc3-a116-3ae319182393)
![line_1turn_VO_ShiTom](https://github.com/muye1202/Monocular-Visual-Odometry-for-DJI-Tello-Drone/assets/112987403/de288a09-934b-4071-9ce5-c46edb654935)


## Camera Calibration

1. Calibration photos using a grided chessboard need to be taken using the target camera, from at least 10 different angles and views. Run `python3 cam_calibrate.py` with the `take_calibration_photos` function enabled, and the host computer connect to the DJI Tello drone WiFi; the script will take photos 
using Tello drone camera at at a frequency of 10Hz.

2. Next, use the photos taken and run `python3 cam_calibrate.py` with the `calibrate_camera` function enabled, the `cam_matrix` variable 
contains the intrinsic parameters of the camera.

## Visual Odometry Test

1. Using KITTI dataset as validation. Run `python3 visual_odometry.py` with `plot_KITTI` function enabled to test the VO pipeline
on the KITTI dataset, one can choose to enable the bundle-adjustment.

2. To test the VO on the drone, the user need to record images using the drone and then perform the odometry offline by running `python3 visual_odometry.py` with `plot_drone` function enabled.
