import numpy as np
import copy
import math

from bow.algm import get_information_json, put_information_json


def rigid_transformation(moving_list, fixed_list):
    moving_object_list = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
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

    return transformation_array.tolist()


def write_pose(uid, cid, moving_list, fixed_list):
    case_info = get_information_json(uid, cid)

    angle_op = case_info["IP"]

    rigid_tf = rigid_transformation(moving_list, fixed_list)
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

    return rigid_tf.tolist()
