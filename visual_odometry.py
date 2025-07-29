import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import g2o
from scipy.spatial.transform import Rotation as sci_rot


KITTI_path = os.getenv("KITTI_PATH", os.path.join("data", "KITTI_sequence_1"))
# KITTI_gt_path = os.getenv("KITTI_GT_PATH", "/path/to/poses/")

optimizer = g2o.SparseOptimizer()
solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
solver = g2o.OptimizationAlgorithmLevenberg(solver)
optimizer.set_algorithm(solver)

def load_drone_images():
    img_list = []
    for i in range(1, 495):
        # the names of the images
        img_name = str(i)
        img_name = img_name + '.png'
        drone_dir = os.getenv("DRONE_IMAGE_DIR", os.path.join("data", "drone_capture"))
        path_img = os.path.join(drone_dir, img_name)
        img = cv2.imread(path_img)
        if img is not None:
            cp_img = np.zeros_like(img)
            cv2.copyTo(src=img, dst=cp_img, mask=None)
            img_list.append(cp_img)

    return img_list

def load_dataset(data_size):
    img_list = []
    for i in range(data_size):
        img_name = str(i)
        placeholder_zeros = ''
        for _ in range(1, 7-len(img_name)):
            placeholder_zeros += '0'

        img_name = placeholder_zeros + img_name + '.png'
        img_path = os.path.join(KITTI_path, "image_l", img_name)
        # img_path = KITTI_path + "00/image_0/" + img_name
        img = cv2.imread(img_path)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list.append(img)

    return img_list

def feature_extraction(img, drone=False):
    """
    Returns img coordinates of the feature points.
    """
    if drone:
        # ShiTomasi corner detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature_param = dict(maxCorners = 5000,
                            qualityLevel = 0.01,
                            minDistance = 5,
                            blockSize = 5,
                            useHarrisDetector = True)
        keypt_coord = cv2.goodFeaturesToTrack(img, mask=None, **feature_param)

    else:
        # NOTE: ORB works better for KITTI data set
        # Load the images for FAST detection
        ORB = cv2.ORB_create(nfeatures=1100, nlevels=3, scoreType=cv2.ORB_FAST_SCORE)
        features = ORB.detect(img)

        keypt_coord = []
        for keypoint in features:
            coord = keypoint.pt
            keypt_coord.append(coord)

    return None, keypt_coord

def frame2frame_tracking(old_frame, curr_frame, features):
    """
    Do tracking between two frames using detected features.

    frames need to be GRAYSCALE
    """
    lk_params = dict( winSize  = (20, 20),
                      maxLevel = 4,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    pts, st, err = cv2.calcOpticalFlowPyrLK(old_frame, curr_frame, features, None, **lk_params)

    valid_pts_new = []
    valid_pts_old = []
    if pts is not None:
        valid_pts_old = pts[st[:, 0] == 1]
        valid_pts_new = features[st[:, 0] == 1]

    return valid_pts_old, valid_pts_new, err

def find_transform(old_pts, new_pts, cam_matrix):
    """
    Return the transformation matrix between 2 frames.

    Output:
        - mask: Marks the inliers in the old and new images.
    """
    E, mask = cv2.findEssentialMat(old_pts, new_pts, cam_matrix, method=cv2.RANSAC)

    if E is not None and (E.shape[0] == 3 and E.shape[1] == 3):
        old_pts = np.ascontiguousarray(old_pts, dtype=np.float32)
        new_pts = np.ascontiguousarray(new_pts, dtype=np.float32)
        E = np.ascontiguousarray(E, dtype=np.float64)
        cam_matrix = np.ascontiguousarray(cam_matrix, dtype=np.float32)

        ret, R, t, mask = cv2.recoverPose(E, old_pts, new_pts, cam_matrix, mask=mask)

    else:
        R = np.eye(3)
        t = np.zeros((3,))
        mask = None

    return R, t, mask

def construct_T(R, t):
    transform_matrix = np.zeros((4,4))
    transform_matrix[0:3, 0:3] = R
    transform_matrix[0:3, 3] = t.reshape((3,))
    transform_matrix[3,3] = 1.0

    return transform_matrix

def bundle_adjustment(cam_matrix, principal_pt, cam_poses: list, feature_coords, img_w, img_h):
    focal_length = cam_matrix[0][0]
    cam = g2o.CameraParameters(focal_length, principal_pt, 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)

    # The camera pose vertex
    cam_poses_init = []
    for c, cam_pose in enumerate(cam_poses):
        curr_R = cam_pose[0:3, 0:3]
        curr_t = cam_pose[0:3, 3]
        pose = g2o.SE3Quat(curr_R, list(curr_t))
        curr_T = construct_T(curr_R, curr_t)
        cam_poses_init.append(curr_T)

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(c)
        v_se3.set_estimate(pose)
        optimizer.add_vertex(v_se3)

    true_pts = []
    curr_R = cam_poses[1][0:3, 0:3]
    curr_t = cam_poses[1][0:3, 3]
    for i in range(int(len(feature_coords))):
        # Image coord of the features
        img_coord = feature_coords[i]
        img_coord = np.ascontiguousarray(img_coord, dtype=np.float32).reshape((2,1))
        temp = np.ones((3,1))
        temp[0:2] = img_coord
        img_coord = temp

        # Use VO's camera R, t to get world coord
        estimate_world_coord =np.linalg.inv(curr_R) @ ((np.linalg.inv(cam_matrix) @ img_coord) - curr_t.reshape((3,1)))
        true_pts.append(estimate_world_coord)

    # The world coordinates vertex
    pt_start_id = len(cam_poses_init)
    sse = defaultdict(float)
    for i, pt_w in enumerate(true_pts):
        # Check whether the feature points is visible
        # Mark the visible points to corresponding camera
        visible = []
        for j, p in enumerate(cam_poses_init):
            cam_extrinsics = p[0:3, :]
            p_w = np.concatenate((pt_w.reshape((3,1)), np.zeros((1,1))), dtype=float)
            z = cam.cam_map(cam_extrinsics @ p_w)
            if 0 <= z[0] < img_h and 0 <= z[1] < img_w:
                visible.append((j, z))

        # Skip this point because it's invisible to both cameras
        if len(visible) == 0:
            continue
        
        # The world coordinates of feature points
        vp = g2o.VertexPointXYZ()
        vp.set_id(pt_start_id)
        vp.set_marginalized(True)
        vp.set_estimate(pt_w.reshape((3,1)))
        optimizer.add_vertex(vp)

        # k is the index of corresponding camera
        # z is the measurement of that point
        for k, z in visible:
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp)
            edge.set_vertex(1, optimizer.vertex(k))
            edge.set_measurement(z)
            edge.set_information(2.0*np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)
            # Edge defines the measurement between camera and this point

            error = vp.estimate() - true_pts[i]
            sse[0] += np.sum(error ** 2)
            # This keeps track of the cost function (error)

        pt_start_id += 1

    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(10)

    R = []
    trans = []
    temp = np.zeros((1,))
    for m in range(len(cam_poses)):
        quat = optimizer.vertex(m).estimate()
        if type(quat) != type(temp):
            quat = quat.rotation()
            rot = sci_rot.from_quat([quat.x(), quat.y(), quat.z(), quat.w()])
            R.append(rot.as_matrix())
            trans.append(np.ascontiguousarray(optimizer.vertex(m).estimate().translation(), dtype=np.float32))

    return R, trans

def draw_tracks(old_pts, new_pts, mask, frame):
    colors = np.random.randint(0, 255, (len(old_pts), 3))
    mask = np.zeros_like(frame)
    for i, (new, old) in enumerate(zip(old_pts, new_pts)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 1, colors[i].tolist(), thickness=2)

    img = cv2.add(frame, mask)
    cv2.namedWindow("resize", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("resize", 1000, 1000)
    cv2.imshow('resize', img)
    k = cv2.waitKey(0)
    while k == -1:
        pass

def plot_KITTI(ba_flag=False, draw_track=False):
    # Camera matrix for KITTI Sequence
    calib_path = os.path.join(KITTI_path, "calib.txt")
    calib_arr = np.loadtxt(calib_path, dtype=float)
    cam1 = calib_arr[0, :].reshape((3,4))
    cam1 = cam1[:, 0:3]
    cam2 = calib_arr[1, :].reshape((3,4))
    cam2 = cam2[:, 0:3]

    # Get the Ground Truth trajectory
    gt_path = os.path.join(KITTI_path, "poses.txt")
    gt_arr = np.loadtxt(fname=gt_path, dtype=float)
    data_size = gt_arr.shape[0]
    # data_size = 300 # int(data_size * 0.01)
    gt_traj = np.zeros((data_size, 3))
    for row in range(data_size):
        T_i = gt_arr[row, :].reshape((3,4))
        gt_traj[row, :] = T_i[:, -1]

    test_images = load_dataset(data_size)
    img_h, img_w, _ = test_images[0].shape
    principal_pt = (int(img_h)/2, int(img_w)/2)

    # Init R and t and features
    _, features_coord = feature_extraction(test_images[0])
    R = np.eye(3)
    t = np.zeros((3,1))
    T = construct_T(R, t)
    translation_trajectory = np.zeros((data_size, 3))

    # Calculate VO trajectory
    cam_pose_traj = [T]
    pose_index = 0
    for i in range(1, len(test_images)-1):
        old_frame = test_images[i-1]
        curr_frame = test_images[i]
        
        if len(features_coord) < 800:
            _, features_coord = feature_extraction(old_frame)

        print(str(i) + "th image")
        print("num of features: " + str(len(features_coord)))

        features_coord = np.asarray(features_coord, dtype=np.float32)
        old_pts, new_pts, _ = frame2frame_tracking(old_frame, curr_frame, features_coord)

        # NOTE: Test the BA output
        # if len(old_pts) > 0 and len(new_pts) > 0:
        #     R_new, t_new, mask = find_transform(old_pts, new_pts, cam_matrix=cam1)
        #     T_last2now = construct_T(R_new, t_new)

        #     # Compute camera trajectory
        #     T = T @ T_last2now
        #     cam_pose_traj.append(T)
        #     trans_vector = T[0:3, 3]
        #     translation_trajectory[i, :] = trans_vector.reshape((1,3))

        #     pose_index += 1

        #     if ba_flag and len(cam_pose_traj) >= 2:
        #         cam_poses = [cam_pose_traj[pose_index-1], cam_pose_traj[pose_index]]
        #         # do bundle adjustment
        #         R_list, t_list = bundle_adjustment(cam1, principal_pt, cam_poses, features_coord, img_w, img_h)
                
        #         translation_trajectory[i-1, :] = t_list[0].reshape((1,3))
        #         translation_trajectory[i, :] = t_list[1].reshape((1,3))
        #         T_last = construct_T(R_list[0], t_list[0])
        #         T_now = construct_T(R_list[1], t_list[1])
        #         cam_pose_traj[pose_index-1] = T_last
        #         cam_pose_traj[pose_index] = T_now

        # else:
        #     continue
        
        # NOTE: Original Trajectory Computation
        if len(old_pts) > 0 and len(new_pts) > 0:
            R_new, t_new, mask = find_transform(old_pts, new_pts, cam_matrix=cam1)
            T_last2now = construct_T(R_new, t_new)
            
            if ba_flag and len(cam_pose_traj) >= 2:
                # do bundle adjustment
                cam_poses = [cam_pose_traj[pose_index-1], cam_pose_traj[pose_index]]
                R_list, t_list = bundle_adjustment(cam1, principal_pt, cam_poses, features_coord, img_w, img_h)
                T_0 = construct_T(R_list[0], t_list[0])
                T_1 = construct_T(R_list[1], t_list[1])
                T_last2now = np.linalg.inv(T_0) @ T_1

        else:
            T_last2now = np.eye(4)

        # Compute camera trajectory
        T = T @ T_last2now
        cam_pose_traj.append(T)
        trans_vector = T[0:3, 3]
        translation_trajectory[i, :] = trans_vector.reshape((1,3))
        
        pose_index += 1

        # Use mask to filter out outliers in current frame
        if mask is not None and len(new_pts) > 0:
            new_pts = new_pts[mask.ravel() == 1]
            old_pts = old_pts[mask.ravel() == 1]
            features_coord = new_pts

            if draw_track:
                draw_tracks(old_pts, new_pts, mask, curr_frame)

    x_traj = translation_trajectory[:, 0]
    z_traj = translation_trajectory[:, 2]

    plt.plot(x_traj, z_traj, 'or', label="VO trajectory")
    plt.plot(gt_traj[:, 0], gt_traj[:, 2], 'ob', label="ground truth")
    plt.title("XZ position")
    plt.legend()
    plt.xlim(-30, 30)
    plt.ylim(-10, 100)
    plt.show()

def plot_drone(ba_flag=False, draw_track=False):
    # DJI Tello Drone camera matrix
    cam_matrix = np.array([[365.9666964, 0.0, 213.30875719],
                           [0.0, 496.28202321, 225.17823274],
                           [0.0, 0.0, 1.0]])
    
    # Camera Matrix for sequence_11 from TUM's Monocular VO dataset
    # The dataset website: https://cvg.cit.tum.de/data/datasets/mono-dataset
    # Camera parameters: https://github.com/JakobEngel/dso#31-dataset-format
    #                    see the "Calibration for FVO Camera" section.
    TUM_cam_matrix = np.array([
        [1280.*0.349153000000000, 0.0, 1280.*0.493140000000000],
        [0.0, 1024.*0.436593000000000 - 0.5, 1024.*0.499021000000000-0.5],
        [0., 0., 1.0]
    ])

    test_images = load_drone_images()
    img_h, img_w, _ = test_images[0].shape
    principal_pt = (int(img_h/2), int(img_w/2))

    # Init R and t and features
    _, features_coord = feature_extraction(test_images[0], drone=True)
    R = np.eye(3)
    t = np.zeros((3,1))
    T = construct_T(R, t)
    translation_trajectory = np.zeros((500, 3))

    # Calculate VO trajectory
    for i in range(1, len(test_images)-1):
        old_frame = test_images[i-1]
        curr_frame = test_images[i]

        features_coord = np.asarray(features_coord, dtype=np.float32)
        old_pts, new_pts, _ = frame2frame_tracking(old_frame, curr_frame, features_coord)

        if len(old_pts) > 0 and len(new_pts) > 0:
            R_new, t_new, mask = find_transform(old_pts, new_pts, cam_matrix=cam_matrix)
            T_last2now = construct_T(R_new, t_new)

            if ba_flag:
                # do bundle adjustment
                R_list, t_list = bundle_adjustment(cam_matrix, principal_pt, R_new, t_new, features_coord, img_w, img_h)
                T_last2now = construct_T(R_list[1], t_list[1])

            # Compute camera trajectory
            T = T @ T_last2now
            trans_vector = T[0:3, 3]
            translation_trajectory[i, :] = trans_vector.reshape((1,3))

            # Use mask to filter out outliers in current frame
            if mask is not None:
                new_pts = new_pts[mask.ravel() == 1]
                features_coord = new_pts
                
                if draw_track:
                    draw_tracks(old_pts, new_pts, mask, curr_frame)

        if i % 2 == 0:
            _, features_coord = feature_extraction(test_images[i], drone=True)

    x_traj = translation_trajectory[:, 0]
    z_traj = translation_trajectory[:, 2]
    plt.plot(x_traj, z_traj,'ob', label="VO trajectory")
    plt.title("XZ trajectory")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_KITTI(ba_flag=True)