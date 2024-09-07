#! /usr/bin/python3.8
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from threading import Thread
import pyrealsense2 as rs
import json
import open3d as o3d

class Detector:
    def __init__(self) -> None:

        self.surface_norm_X1 = 0.0  # all are in camera frame
        self.surface_norm_Y1 = 0.0
        self.surface_norm_Z1 = 0.0
        
        self.cam_frame_x1 = 0.0  # the mid point of the target, in camera frame
        self.cam_frame_y1 = 0.0
        self.cam_frame_z1 = 0.0
        
        self.target_pose = PoseStamped()
        self.detect = False
        self.t_detect = 1 / 31
        self.tStart = time()
        
        self.img = None
        self.camInfo = None
        self.best_box_pixel = []
        self.best_box_points = []
        self.latest_cameraInfo = CameraInfo()
        
        self.layer_names = None
        self.net = None
        self.cv_bridge = CvBridge()
        self.output_layers = None

        self.color_topic_name = None
        self.depth_topic_name = None
        self.camera_info_name = None
        
        self.method = None
        self.detection_method = None
        
        self.rate = None
        self.rate_send_image = None
        
        self.box_size = None
        self.depth_array = None
    
        self.pcd = o3d.geometry.PointCloud()

        self.camera_info_received = False

        self.min_search_radius = 5.0 # min search radius for KDT tree

        self.latest_center_normal = np.array([0,0,0])

        self.yaw_est_old = 0.0

        # rosnode
        self.init_pub_subs()

    

    def load_nn_parms(self,weights,config_file,names):
        """ Load neural network parameters"""
        try:
            self.net = cv2.dnn.readNet(weights, config_file)
            self.layer_names = self.net.getLayerNames()

            with open(names, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            output_layers_indices = self.net.getUnconnectedOutLayers().flatten()
            self.output_layers = [self.layer_names[i - 1] for i in output_layers_indices]
       
            rospy.loginfo(f"[gin] Neural network parameters loaded! ")

        except Exception as e:
            rospy.logerr(f"[gin] Failed to load network: {e}")

    def load_params(self):
        """Loading parameter"""
        try:
            weigths = rospy.get_param("param_file_name")
            config_file = rospy.get_param("config_file_name")
            names = rospy.get_param("names_file_name")
            self.color_topic_name = rospy.get_param("color_topic_name")
            self.depth_topic_name = rospy.get_param("depth_topic_name")
            self.camera_info_name = rospy.get_param("info_topic_name")
            self.detection_method = rospy.get_param("detection_method") # detection method
            self.box_param_file = rospy.get_param("box_param_file")
            self.default_camera_param_file = rospy.get_param("default_camera_param_file")
            self.method = rospy.get_param("method") # pose computation method
            self.rate = rospy.get_param("pub_rate", default=10)
            self.rate_send_image = rospy.get_param("pub_rate_image", default=3)
            self.load_boxes_json(self.box_param_file)

        except rospy.ROSException as e:
            rospy.logerr("Failed to some rosparam")
            raise rospy.ROSException("Some rosparam are missing")
        
        self.load_nn_parms(weigths,config_file,names)


    def load_boxes_json(self,file_path):
        """Load detected boxes sizes from json"""
        with open(file_path, "r") as file:
            box_param = json.load(file)
            if(self.detection_method != "single_layer"):
                self.box_size = np.array(
                    [
                        [
                            box_param["T_good"]["height"],
                            box_param["T_better"]["height"],
                            box_param["T_best"]["height"],
                        ],
                        [
                            box_param["T_good"]["width"],
                            box_param["T_better"]["width"],
                            box_param["T_best"]["width"],
                        ],
                    ]
                )
            else:
                self.box_size = np.array(
                    [
                        [
                            box_param["T_good"]["height"],
                            box_param["T_good"]["height"],
                            box_param["T_good"]["height"],
                        ],
                        [
                            box_param["T_good"]["width"],
                            box_param["T_good"]["width"],
                            box_param["T_good"]["width"],
                        ],
                    ]
                )
        rospy.loginfo_once("[gin] Box sizes loaded")

    def convert2relative(self, box, width, height):
        """Method to convert data from pixel to points
        -----------------------------------------------
        method of conversion: out = in/max -0.5 \n
        (0.5 so that center is 0)
        input:
            - box
                - w -> box width [pixel]
                - h -> box height [pixel]
                - x -> box x [pixel]
                - y -> box y [pixel]
            - width -> image width [pixel]
            - heigh -> image height [pixel]
        """
        x, y, w, h = box
        # ? should depend on target 0.5?
        x_rel = x / width - 0.5
        y_rel = y / height - 0.5
        w_rel = w / width
        h_rel = h / height
        return x_rel, y_rel, w_rel, h_rel
    

    def store_param_camera(self,camera_info):
        """Callback info camera"""

        rospy.loginfo_once("[gin] info callback init")

        # TODO to be called just once
        self.latest_cameraInfo = camera_info

        self.camera_info_received = True

        # rospy.loginfo(latest_cameraInfo.K)
        # rospy.loginfo(latest_cameraInfo.width)
        # rospy.loginfo(latest_cameraInfo.height)


    def convert_color_image(self,ros_image_raw):
        """Callback color topic"""
        
        rospy.loginfo_once("[gin] color callback init")

        try:
            color_image = self.cv_bridge.imgmsg_to_cv2(
                ros_image_raw, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            rospy.loginfo(e)


        self.img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        height, width, channels = self.img.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(self.img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.output_layers)
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # ? this should be a rosparam
                if confidence > 0.5:
                    # Object detected
                    self.detect = True
                    self.tStart = time()
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        if (abs(time() - self.tStart)) > self.t_detect:
            self.detect = False
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        # rospy.loginfo("boxes values: %s",(boxes))
        rospy.logdebug("indexes: %s", (indexes))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]  # output hwer are in pixel unit

                # Check if box dimensions exceed image dimensions
                if w > width:
                    w = width
                    boxes[i][2] = width
                if h > height:
                    h = height
                    boxes[i][3] = height


                label = str(self.classes[class_ids[i]])
                color = self.colors[i]
                cv2.rectangle(
                    self.img, (x, y), (x + w, y + h), color, 2
                )  # this for sure are pixel
                cv2.putText(self.img, label, (x, y + 30), font, 3, color, 3)
                cv2.circle(self.img, (int(x + w / 2), int(y + h / 2)), 1, color, thickness=10)
        
        # storing info on best label we have
        self.storing_optimal_detection(boxes, width, height, indexes)


    def storing_optimal_detection(self,boxes, width, height, indexes):
        """Method to store the best label among the ones given
        --------------------------------------------------------------
        method of selection based on the best box available
        max index => best box
        """
        if len(indexes) > 0:
            self.best_idx = max(indexes)
            if isinstance(self.best_idx, np.ndarray):
                self.best_idx = self.best_idx.item()
            self.best_box_pixel = boxes[self.best_idx]
            self.best_box_points = self.convert2relative(self.best_box_pixel, width, height)

        # rospy.loginfo("best idx %s",best_idx)


    def pnp(self,image_points_2D, figure_points_3D):
        """Implementation of pnp algorithm
        ----------------------------------------
        Goal: estimating pose and position of target wrt camera via pnp algorithm
        """
        distortion_coeffs = self.latest_cameraInfo.D
        # rospy.loginfo("[gin] pnp shapes: %s %s, %s %s",
        #               image_points_2D.shape[0], image_points_2D.shape[1],
        #               figure_points_3D.shape[0], figure_points_3D.shape[1])
        figure_points_3D = figure_points_3D.astype("float32")
        image_points_2D = image_points_2D.astype("float32")
        camera_matrix = (
            np.array(self.latest_cameraInfo.K).astype("float32").reshape((3, 3))
        )  # ensure to be 3x3
        success, vector_rotation, vector_translation = cv2.solvePnP(
            figure_points_3D,
            image_points_2D,
            camera_matrix,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        # ensure correct dimension
        vector_rotation = np.array(
            [vector_rotation[0][0], vector_rotation[1][0], vector_rotation[2][0]]
        )
        if not success:
            rospy.logwarn("[gin] pnp not sucessful")
        return vector_rotation, vector_translation
    
    def exp_movmean(self,x,x_old,alpha=0.1):
        """ Exponential movmean """
        
        return alpha * float(x) + (1 - alpha) * float(x_old)
       
    def extract_normal(self, x_start, y_start, x_end, y_end):
        """Extract normal from the image"""

        depth_region = self.depth_array[y_start:y_end,x_start:x_end]
        valid_region = (depth_region>0) & ~np.isnan(depth_region)

        if not np.any(valid_region):
            rospy.logwarn("[gin] No valid depth points found.")
            return np.array([0,0,0])
    
        valid_y, valid_x = np.where(valid_region)

        cx     = self.latest_cameraInfo.K[2]
        cy     = self.latest_cameraInfo.K[5]
        fx     = self.latest_cameraInfo.K[0]
        fy     = self.latest_cameraInfo.K[4]

        zs = depth_region[valid_y, valid_x]
        xs = (valid_x + x_start - cx) * zs / fx
        ys = (valid_y + y_start - cy) * zs / fy

        points = np.vstack((xs, ys, zs)).T
        self.pcd.points = o3d.utility.Vector3dVector(points)


        if self.pcd.is_empty() or len(self.pcd.points) == 0:
            rospy.logwarn("[gin] empty array points for normal")
            return np.array([0,0,0])

        else:
            # filter pointcloud
            self.pcd = self.pcd.voxel_down_sample(voxel_size=0.03)

            # Estimate normals
            if(len(self.pcd.points)==0 or np.isnan(np.ptp(self.pcd.get_min_bound())) or np.isnan(np.ptp(self.pcd.get_max_bound()))):
                search_radius = 0.0
            else:
                avg_density = len(self.pcd.points) / (np.ptp(self.pcd.get_min_bound()) * np.ptp(self.pcd.get_max_bound()))
                search_radius = 1.0 / avg_density  # Adjust based on density

            # prevent too small radius or too big radius values
            bounding_box = self.pcd.get_axis_aligned_bounding_box()
            max_radius = np.linalg.norm(bounding_box.get_max_bound() - bounding_box.get_min_bound())
            # rospy.loginfo("[gin] max_radius: %.3f", max_radius)

            # rospy.loginfo("[gin] search radius: %.3f",search_radius)
            search_radius = max(search_radius, self.min_search_radius)
            search_radius = min(search_radius, max_radius)

            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=120))

            # outlier filters
            self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            self.pcd.remove_radius_outlier(nb_points=16, radius=0.05)

            if len(self.pcd.normals) == 0:
                rospy.logwarn("[gin] No normals were computed for the point cloud.")
                return None
             
            #  Orient normals
            self.pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

            # Build KD-Tree
            pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

            center_point = np.array([self.cam_frame_x1,self.cam_frame_y1,self.cam_frame_z1])

            # Find k nearest neighbors
            k = 5
            [_, ind, _] = pcd_tree.search_knn_vector_3d(center_point, k)
            self.latest_center_normal = self.pcd.normals[ind[0]]

            # Normalize the normal vector
            normals = np.asarray(self.pcd.normals)[ind]
            self.latest_center_normal = np.mean(normals, axis=0)
            self.latest_center_normal /= np.linalg.norm(self.latest_center_normal)

            # Estimate yaw (angle around the Z-axis)
            yaw_est = np.arctan(self.latest_center_normal[0]/self.latest_center_normal[2])
            # yaw_est = np.pi - yaw_est


            # rospy.loginfo("[gin] yaw_est: %.1f",yaw_est)

            # filter
            yaw_est = self.exp_movmean(float(yaw_est),float(self.yaw_est_old),alpha=1.0)

            # update old value
            self.yaw_est_old = yaw_est

            return np.array([0,0,yaw_est])

    def convert_pixel_cordinate_to_camera_cordinate_simple(self,pixel_xy_arr, depth_arr):
        """Method to obtain matrix with 3D data point
        ------------------------------------------------------
        input:
        pixel_xy_arr    = 2D pixel point [Nx2]
        depth_arr       = corresponding depth points [Nx1]
        best_idx        = index best boxes (0 = T_good, 1 = T_better, 2 = T_best)
        output:
        3d_point_arr    = array with 3D points [Nx3]
        idea:
        we know that the 2D pixel points are point of a box with a certain dimension in 3D space and that box center is (0,0,b)
        """
        best_height_box = self.box_size[0, self.best_idx]
        best_width_box  = self.box_size[1, self.best_idx]
        three_D_points_arr = np.zeros((5, 3))
        three_D_points_arr[0, :] = [-best_width_box / 2, -best_height_box / 2, depth_arr[0]]
        three_D_points_arr[1, :] = [-best_width_box / 2, best_height_box / 2, depth_arr[1]]
        three_D_points_arr[2, :] = [best_width_box / 2, -best_height_box / 2, depth_arr[2]]
        three_D_points_arr[3, :] = [best_width_box / 2, best_height_box / 2, depth_arr[3]]
        three_D_points_arr[4, :] = [0, 0, depth_arr[4]]
        return three_D_points_arr


    def get_center_box_rs(self,desired_XY_pixel_arr, desired_depth_mtx):
        """Method to compute the position of the center wrt to the camera using realsense lib"""
        # find center coordinates
        if desired_XY_pixel_arr.shape[0] == 0 or desired_depth_mtx.shape[0] == 0:
            u_center = 0.0
            v_center = 0.0
            w_center = 0.0
        else:
            u_center = desired_XY_pixel_arr[-1, 0]  # last row
            v_center = desired_XY_pixel_arr[-1, 1]  # last row
            w_center = desired_depth_mtx[-1]  # last row
        # rospy.loginfo("center in image plane: %s,%s",u_center,v_center)
        # camera params
        _intrinsics = rs.intrinsics()
        _intrinsics.width   = self.latest_cameraInfo.width
        _intrinsics.height  = self.latest_cameraInfo.height
        _intrinsics.ppx     = self.latest_cameraInfo.K[2]
        _intrinsics.ppy     = self.latest_cameraInfo.K[5]
        _intrinsics.fx      = self.latest_cameraInfo.K[0]
        _intrinsics.fy      = self.latest_cameraInfo.K[4]
        _intrinsics.model = rs.distortion.none
        if self.latest_cameraInfo.D:
            _intrinsics.coeffs = [i for i in self.latest_cameraInfo.D]
        else:
            rospy.logwarn("D empty")
            _intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        # rospy.loginfo("camera info: %s, %s, %s, %s",_intrinsics.ppx, _intrinsics.ppy, _intrinsics.fx,_intrinsics.fy)
        # compute projection
        result = rs.rs2_deproject_pixel_to_point(
            _intrinsics, [u_center, v_center], w_center
        )
        # 3d position
        result_np = np.array([result[0], result[1], result[2]])
        # TODO 3d rotation
        rot = np.array([0, 0, 0])
        return rot, result_np
    

    def convert_depth_image(self,ros_image_rect_raw):
        """Callback depth topic camera"""

        rospy.loginfo_once("[gin] depth callback init")

        if not self.detect:
            return
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(
                ros_image_rect_raw, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            rospy.loginfo(e)

        self.depth_array = np.array(depth_image, dtype=np.float32)


        height, width = self.depth_array.shape  # Get height and width of the depth array
    
        # get best label store
        best_x, best_y, best_w, best_h = self.best_box_pixel
        desired_x_pixel_min = best_x
        desired_x_pixel_max = best_x + best_w
        desired_y_pixel_min = best_y
        desired_y_pixel_max = best_y + best_h

        # Saturate indices to ensure they are within bounds
        if not (0 <= desired_y_pixel_min < height and 0 <= desired_y_pixel_max < height 
                and 0 <= desired_x_pixel_max < width):
            rospy.logwarn("[gin] Indexes out of bounds for depth array. Saturation applied.")

        desired_y_pixel_min = np.clip(desired_y_pixel_min, 0, height - 1)
        desired_y_pixel_max = np.clip(desired_y_pixel_max, 0, height - 1)
        desired_x_pixel_min = np.clip(desired_x_pixel_min, 0, width - 1)
        desired_x_pixel_max = np.clip(desired_x_pixel_max, 0, width - 1)

        # pixel 2d points (vertices of box)
        desired_XY_pixel_arr = np.array(
            [
                (desired_x_pixel_min, desired_y_pixel_min),
                (desired_x_pixel_min, desired_y_pixel_max),
                (desired_x_pixel_max, desired_y_pixel_min),
                (desired_x_pixel_max, desired_y_pixel_max),
                (
                    int(desired_x_pixel_min + best_w / 2),
                    int(desired_y_pixel_min + best_h / 2),
                ),
            ]
        )  # center
        # corresponding depth point
        desired_depth_mtx = np.array(
            [
                self.depth_array[desired_y_pixel_min, desired_x_pixel_min],
                self.depth_array[desired_y_pixel_max, desired_x_pixel_min],
                self.depth_array[desired_y_pixel_min, desired_x_pixel_max],
                self.depth_array[desired_y_pixel_max, desired_x_pixel_max],
                self.depth_array[
                    int(desired_y_pixel_min + int(best_h / 2)),
                    int(desired_x_pixel_min + best_w / 2),
                ],
            ]
        )  # center
        # compute corresponding 3d points
        desired_3D_points_arr = self.convert_pixel_cordinate_to_camera_cordinate_simple(
            desired_XY_pixel_arr, desired_depth_mtx
        )
        # rospy.loginfo("sizes pixel: %s,%s",desired_XY_pixel_arr.shape,desired_3D_points_arr.shape)
        if self.best_idx == -1:
            rospy.logwarn("[gin] no frame detected")
            return
        else:
            if self.method == "pnp":
                rospy.loginfo_once("[gin] pnp detection method")
                pnp_rot, pnp_pos = self.pnp(desired_XY_pixel_arr, desired_3D_points_arr)
            elif self.method == "camera_proj":
                rospy.loginfo_once("[gin] camera_proj method")
                pnp_rot, pnp_pos = self.get_center_box_rs(
                    desired_XY_pixel_arr, desired_depth_mtx
                )
            else:
                rospy.logwarn_once(
                    "[gin] method not available, using camera_proj as default"
                )

            # extract normal
            pnp_rot = self.extract_normal(desired_x_pixel_min, desired_y_pixel_min, desired_x_pixel_max, desired_y_pixel_max)

            self.cam_frame_x1 = pnp_pos[0]
            self.cam_frame_y1 = pnp_pos[1]
            self.cam_frame_z1 = pnp_pos[2]
            # rospy.loginfo("camera pos: %s, %s, %s",cam_frame_x1,cam_frame_y1,cam_frame_z1)
            self.surface_norm_X1 = pnp_rot[0]
            self.surface_norm_Y1 = pnp_rot[1]
            self.surface_norm_Z1 = pnp_rot[2]


    def send_detection(self,event=None):
        """Publisher method target/detection"""
        self.pub_target_detection.publish(self.detect)

    def check_camera_info_status(self,event):
        """ Check params camera are published and failsafe action"""
        if not self.camera_info_received:
            rospy.logwarn("[gin] camera info are not published, loading default ...")
        
            with open(self.default_camera_param_file, "r") as file:
                camera_params = json.load(file)

                self.latest_cameraInfo.width  = camera_params["width"]
                self.latest_cameraInfo.height = camera_params["height"]
                self.latest_cameraInfo.K[2]   = camera_params["cx"]
                self.latest_cameraInfo.K[5]   = camera_params["cy"]
                self.latest_cameraInfo.K[0]   = camera_params["fx"]
                self.latest_cameraInfo.K[4]   = camera_params["fy"]
                self.latest_cameraInfo.D      = camera_params["D"]

    # TODO input (global) comes directly from pnp
    def send_localisation(self):
        """Publisher method target/pose"""
        rot = R.from_euler(
            "xyz", [self.surface_norm_X1, self.surface_norm_Y1, self.surface_norm_Z1], degrees=False
        )
        self.target_pose.header.frame_id = "camera_detection_link"
        self.target_pose.header.stamp = rospy.Time.now()
        self.target_pose.pose.position.x = 1e-3 * self.cam_frame_x1
        self.target_pose.pose.position.y = 1e-3 * self.cam_frame_y1
        self.target_pose.pose.position.z = 1e-3 * self.cam_frame_z1
        self.target_pose.pose.orientation.x = rot.as_quat()[0]
        self.target_pose.pose.orientation.y = rot.as_quat()[1]
        self.target_pose.pose.orientation.z = rot.as_quat()[2]
        self.target_pose.pose.orientation.w = rot.as_quat()[3]
        # if self.target_pose.pose.position.z != 0.0:
        self.pub_target_pose.publish(self.target_pose)
            

    def send_target_pose(self,event=None):
        rospy.loginfo_once("[gin] Publishing target pose ..")
        self.send_localisation()


    def send_image_msg(self,event=None):
        rospy.loginfo_once("[gin] Publihing image_with_detection")
        
        if self.img is not None:
            self.send_image_with_detection()


    def send_image_with_detection(self):
        image_msg = self.cv_bridge.cv2_to_imgmsg(self.img, "bgr8")
        image_msg.header.stamp = rospy.Time.now()
        self.pub_image_with_detection.publish(image_msg)

    # ! to be removed
    def calculate_surface_normal(points):
        # rospy.loginfo("point printing ",points)
        # Center the points around the origin
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_points, rowvar=False)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # Find the index of the smallest eigenvalue
        min_eigenvalue_index = np.argmin(eigenvalues)
        # The corresponding eigenvector is the surface normal
        surface_normal = eigenvectors[:, min_eigenvalue_index]
        # Normalize the vector to get a unit normal vector
        surface_normal /= np.linalg.norm(surface_normal)
        # To redirect surface normal always outwards the camera
        camera_z_axis = np.array([0, 0, 1])
        dot_product = np.dot(surface_normal, camera_z_axis)
        if dot_product < 0:  # this means surface_normal pointing inwards the camera
            aligned_surface_normal = -surface_normal
        else:
            aligned_surface_normal = surface_normal
        return aligned_surface_normal, centroid

    def init_pub_subs(self):
        rospy.init_node("image_listener")

        #  parameters
        self.load_params()

        # publishers
        self.pub_target_detection = rospy.Publisher("/target/detection", Bool, queue_size=1)
        self.pub_target_pose = rospy.Publisher("/target/pose", PoseStamped, queue_size=1)
        self.pub_image_with_detection = rospy.Publisher(
            "/target/image_with_detection", Image, queue_size=1
        )

        # subscribers
        self.sub_depth = rospy.Subscriber(
            self.depth_topic_name, Image, callback=self.convert_depth_image
        )
        self.sub_info = rospy.Subscriber(
            self.camera_info_name, CameraInfo, callback=self.store_param_camera
        )
        self.sub_color = rospy.Subscriber(
            self.color_topic_name, Image, callback=self.convert_color_image
        )

        # check camera info and failsafe action 
        rospy.Timer(rospy.Duration(10.0 * 1.0 / self.rate), self.check_camera_info_status, oneshot=True)

        rospy.loginfo("[gin] Start YOLO")
        rospy.Timer(rospy.Duration(secs = 1.0 / self.rate), self.send_detection)
        rospy.Timer(rospy.Duration(secs = 1.0 / self.rate), self.send_target_pose)
        rospy.Timer(rospy.Duration(secs = 1.0 / self.rate_send_image), self.send_image_msg)
        rospy.spin()


if __name__ == '__main__':
    target_detector = Detector()