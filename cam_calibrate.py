import os
import cv2
import numpy as np
from djitellopy import tello


def take_calibration_photos():
    drone = tello.Tello()
    drone.connect()
    print("Drone connected!")
    drone.streamon()
    cali_img_list = []

    while len(cali_img_list) < 300:
        img = drone.get_frame_read().frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cali_img_list.append(img)
        cv2.imshow("drone_view", img)
        save_dir = os.getenv("CALIBRATION_DIR", os.path.join("data", "drone_straight_line"))
        path_img = os.path.join(save_dir, f"{len(cali_img_list)}.png")
        cv2.imwrite(path_img, img)
        cv2.waitKey(100)

def calibrate_camera():
    CHECKER_BOARD = (7,7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    real_coord = []
    img_coord = []

    objp = np.zeros((1, CHECKER_BOARD[0] * CHECKER_BOARD[1], 3), np.float32)
    objp[0,:,:2] = 0.051 * np.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T.reshape(-1, 2)

    for i in range(9):
        calib_dir = os.getenv("CALIBRATION_IMAGES_DIR", os.path.join("data", "calibration_images"))
        img_path = os.path.join(calib_dir, f"{i+1}.jpg")
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKER_BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            refined_corner = cv2.cornerSubPix(gray, corners, (11,11), zeroZone=(-1,-1), criteria=criteria)
            # with_corner = cv2.drawChessboardCorners(img, CHECKER_BOARD, refined_corner, ret)
            # cv2.imwrite(os.path.join("data", "corner_detection", f"{i+1}_trial2.jpg"), with_corner)

            img_coord.append(refined_corner)
            real_coord.append(objp)
            # cv2.imshow("img_"+str(i), with_corner)
            # cv2.waitKey(1000)

        else:
            print("image refine failed")

    return img_coord, real_coord

if __name__ == "__main__":
    take_calibration_photos()
    # img_coord, real_coord = calibrate_camera()
    # flag, cam_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(real_coord, img_coord, (480, 480), None, None)
    # print("\nThe drone's camera matrix")
    # print(cam_matrix)