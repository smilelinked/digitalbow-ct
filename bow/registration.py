import numpy as np
import copy
import math
import cv2

from bow.algm import get_information_json, put_information_json, get_object_prefix
from bow.s3 import put_obj


def rigid_transformation(uid, cid, moving_object_list, moving_list, fixed_list):
    
    # 将一维数组转换为矩阵
    moving_object_matrix_world = np.reshape(moving_object_list, (4, 4))
    fixed_array = np.reshape(fixed_list, (4, 3))
    moving_array = np.reshape(moving_list, (4, 3))

    # 复制移动物体的matrix_world
    transformation_rough_t0 = copy.deepcopy(moving_object_matrix_world)

    # 计算中心点
    fixed_centroid = np.mean(fixed_array, axis=0)
    moving_centroid = np.mean(moving_array, axis=0)

    # 将数组移至原点
    fixed_origin = fixed_array - fixed_centroid
    moving_origin = moving_array - moving_centroid

    # 计算平方和
    fixed_sum_squared = np.sum(fixed_origin ** 2)
    moving_sum_squared = np.sum(moving_origin ** 2)

    # 归一化数组
    fixed_normalized = np.sqrt(fixed_sum_squared)
    fixed_norm_origin = fixed_origin / fixed_normalized
    moving_normalized = np.sqrt(moving_sum_squared)
    moving_norm_origin = moving_origin / moving_normalized

    # 奇异值分解
    cov_matrix = np.matrix.transpose(moving_norm_origin) @ fixed_norm_origin
    u, s, vt = np.linalg.svd(cov_matrix)
    v = vt.T
    rotation3x3 = v @ u.T

    # 防止反射
    if np.linalg.det(rotation3x3) < 0:
        v[:, -1] *= -1
        s[-1] *= -1
        rotation3x3 = v @ u.T

    # 旋转
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3, 0:3] = rotation3x3
    moving_object_matrix_world = rotation_matrix @ moving_object_matrix_world

    # 平移
    translation_matrix = np.eye(4)
    translation_matrix[0:3, 3] = np.matrix.transpose(fixed_centroid - rotation3x3 @ moving_centroid)
    moving_object_matrix_world = translation_matrix @ moving_object_matrix_world

    # 复制 T1 变换
    transformation_rough_t1 = copy.deepcopy(moving_object_matrix_world)

    # 计算变换矩阵
    transformation_rough = transformation_rough_t1 @ np.linalg.inv(transformation_rough_t0)

    # 返回变换矩阵的一维数组
    transformation_array = transformation_rough.flatten()

    # obs标记，方便前端跳转
    put_obj(get_object_prefix(uid, cid) + "registration.pckl")

    return transformation_array


def write_pose(uid, cid, moving_object_list, moving_list, fixed_list):
    case_info = get_information_json(uid, cid)

    angle_op = case_info["angle_OP"]

    rigid_tf = rigid_transformation(uid, cid, moving_object_list, moving_list, fixed_list)
    jaw_splint_pose = np.reshape(rigid_tf, (4, 4))

    gap = [0, -2, 2]
    up_f = jaw_splint_pose[0:3, 0:3] @ [0, -2 + 5.8, 0] + gap
    translation = np.array([[1, 0, 0, up_f[0]],
                            [0, 1, 0, up_f[1]],
                            [0, 0, 1, up_f[2]],
                            [0, 0, 0, 1]])
    transform_f_up = translation @ jaw_splint_pose
    transform_t_f = np.array([[1, 0, 0, 0],
                              [0, math.cos(math.pi * (-angle_op) / 180), -math.sin(math.pi * (-angle_op) / 180), 0],
                              [0, math.sin(math.pi * (-angle_op) / 180), math.cos(math.pi * (-angle_op) / 180), 0],
                              [0, 0, 0, 1]])

    # 数据打包
    pose_info = {
        "jawSplint_pose": jaw_splint_pose.tolist(),
        "Transform_F_UP": transform_f_up.tolist(),
        "Transform_T_F": transform_t_f.tolist(),
    }
    # 存储
    case_info.update(pose_info)
    put_information_json(uid, cid, case_info)

    put_obj(get_object_prefix(uid, cid) + "pose.pckl")

    return rigid_tf.tolist()

class AprilTagDetector:
    
    def __init__(self, dictionary_type=cv2.aruco.DICT_APRILTAG_36h11):
        # 初始化ArUco字典和参数
        self.ARUCO_DICT = cv2.aruco.Dictionary_get(dictionary_type)
        self.ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()

    @staticmethod
    def determine_tag_orientation(corners):
        """
        确定AprilTag码的方向
        corners: 检测到的四个角点坐标
        返回: 'Normal' 或 'Inverted'
        """
        center = np.mean(corners[0], axis=0)
        first_corner = corners[0][0]
        relative_pos = first_corner - center
        
        if relative_pos[0] < 0 and relative_pos[1] < 0:
            return 'Normal'
        else:
            return 'Inverted'

    def detect_and_identify(self, image_path, nummarker, save_path=None):
        """处理单张图像，识别AprilTag并确定其方向"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法加载图片 {image_path}")
            return None

        # 使用 detectMarkers 检测标记
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, self.ARUCO_DICT, parameters=self.ARUCO_PARAMS)

        green_count = 0
        yellow_count = 0

        if ids is not None:
            for i in range(len(ids)):
                orientation = self.determine_tag_orientation(corners[i])
                cX, cY = tuple(corners[i][0].mean(axis=0).astype(int))
                # Convert corners[i] to the required shape
                corner_points = np.int32(corners[i]).reshape((-1, 1, 2))
                if orientation == 'Inverted':
                    cv2.polylines(frame, [corner_points], True, (0, 255, 255), 6)  # 黄色边框
                    yellow_count += 1
                else:
                    cv2.polylines(frame, [corner_points], True, (0, 255, 0), 6)  # 绿色边框
                    green_count += 1

        # 保存处理后的图像
        if save_path:
            cv2.imwrite(save_path, frame)
            print(f"图像已保存至 {save_path}")

        # 打印结果
        total_tags = green_count + yellow_count
        print(f"可识别（正常标签）数量: {green_count}")
        print(f"可识别（反向标签）数量: {yellow_count}")  

        # nummarker值根据实际情况进行调整
        damaged_tags = nummarker - total_tags 
        print(f"无法识别标识码的数量: {damaged_tags}")

        if green_count != nummarker:
            print("警告：部分标记码无法识别,请检查。")
        else:
            print("一切正常。")
        
        return frame