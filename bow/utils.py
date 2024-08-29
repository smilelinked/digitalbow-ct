from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.spatial.transform import Rotation


#############################################################################################
# 第1部分： 一般公共处理函数
#############################################################################################
# ----------------------------------------------------------#
# 轨迹线函数：固定点赋予动作，得到轨迹线
# ----------------------------------------------------------#
def point_trajectory(matrix_list, point=None):
    # 如果没有点，计算矩阵角度的轨迹
    if point is None:
        angle_3d = []
        # 循环处理
        for i in range(len(matrix_list)):
            # 每一帧的矩阵
            matrix = np.array(matrix_list[i])
            # 计算欧拉角
            euler = Rotation.from_matrix(matrix[0:3, 0:3]).as_euler('xyz', degrees=True)
            # 保存，3D显示用
            angle_3d.append(euler)
        # 平滑处理
        angle_3d = moving_average(angle_3d)

        # 返回
        return angle_3d

    # 如果有点，计算点的轨迹
    else:
        p_3d, p_list, p_init = [], [], []
        # 循环处理
        for i in range(len(matrix_list)):
            # 每一帧的矩阵
            matrix = np.array(matrix_list[i])
            # 计算新坐标
            new_p = (matrix @ np.append(point, 1))[:-1]
            # 保存，3D显示用
            p_3d.append(new_p)

            # 相对坐标值: 以第一帧为坐标0点的运动轨迹
            if i == 0:
                p_init = new_p
            diff_p = new_p - p_init
            # 保存, 2D显示用
            p_list.append(diff_p)
        # 平滑处理
        p_3d = moving_average(p_3d)
        p_list = moving_average(p_list)

        # 返回
        return p_3d, p_list


# ----------------------------------------------------------#
# 平滑函数：采用移动平均EA算法进行平滑处理
# ----------------------------------------------------------#
def moving_average(arr, n=10, m=4):
    arr = np.array(arr)
    ret = np.cumsum(arr, axis=0, dtype=float)
    ret[n:, :] = ret[n:, :] - ret[:-n, :]
    ret[:n - 1, :] = arr[:n - 1, :] * n

    return np.around(ret / n, m)


# ----------------------------------------------------------#
# 过滤函数：过滤靠得比较近的点，排除个别比较远的点
# ----------------------------------------------------------#
def filter_close_points(points, radius=2.0):
    # 将点集转换为NumPy数组
    points = np.array(points)
    # 计算点到原点的距离
    distances = np.linalg.norm(points, axis=1)
    # 选取在半径范围内的点
    filtered_points = points[distances <= radius]

    return filtered_points


# ----------------------------------------------------------#
# 计算轨迹线的重合点
# ----------------------------------------------------------#
def coincident_point(arr, radius=0.1):
    arr = np.array(arr)
    temp = np.zeros(len(arr))
    dist = distance.squareform(distance.pdist(arr))
    for i in range(len(dist)):
        temp[i] = sum(d <= radius for d in dist[i])
    idx = np.argmax(temp)
    cp = arr[idx]

    return idx, cp


# ----------------------------------------------------------#
# 三点确定一个平面
# ----------------------------------------------------------#
def define_plane(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    a = (point3[1] - point1[1]) * (point3[2] - point1[2]) - (point2[2] - point1[2]) * (point3[1] - point1[1])
    b = (point3[0] - point1[0]) * (point2[2] - point1[2]) - (point2[0] - point1[0]) * (point3[2] - point1[2])
    c = (point2[0] - point1[0]) * (point3[1] - point1[1]) - (point3[0] - point1[0]) * (point2[1] - point1[1])
    d = -(a * point1[0] + b * point1[1] + c * point1[2])

    return a, b, c, d


# ----------------------------------------------------------#
# 照片中输入中文
# ----------------------------------------------------------#
def print_chinese(image, coordinate, word):
    # 中文字体和大小
    font_chinese = ImageFont.truetype("msyh.ttc", 32)
    # 文字颜色
    font_color = (0, 0, 0)
    # 转化为PIL库可以处理的图片格式
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    # 输出文字到图片
    draw.text(coordinate, word, font=font_chinese, fill=font_color)
    # 恢复到CV格式的图像，方便下面的输出
    new_image = np.array(img)
    return new_image


#############################################################################################
# 第2部分： CV识别算法
#############################################################################################

# ----------------------------------------------------------#
# 设置标记码
# ----------------------------------------------------------#
def setting_board():
    # 识别字典：AprilTag36h11
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

    # ARUCO检测模块的参数
    aruco_parameters = cv2.aruco.DetectorParameters_create()
    aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    aruco_parameters.aprilTagMinWhiteBlackDiff = 60

    # *************** 有上、下2个标记板，分别粘在上、下中切牙上。
    # 上板61-63，下板64-66。 下板需要颠倒过来
    # 6个标记二维码的ID
    up_board_ids = np.array([[61], [62], [63]], dtype=np.int32)
    low_board_ids = np.array([[64], [65], [66]], dtype=np.int32)

    # 标记码的角在物理世界的坐标值（相对于原点，原点是有约定的）—— 每个标记码4个角，顺时针，左上为第一个角
    # 上板
    up_board_corners = [
        np.array(
            [
                [-0.025982, 0.0189, 0.019159],
                [-0.01222, 0.0189, 0.028795],
                [-0.01222, 0.0021, 0.028795],
                [-0.025982, 0.0021, 0.019159],
            ],
            dtype=np.float32),
        np.array(
            [
                [-0.0084, 0.0189, 0.03],
                [0.0084, 0.0189, 0.03],
                [0.0084, 0.0021, 0.03],
                [-0.0084, 0.0021, 0.03],
            ],
            dtype=np.float32),
        np.array(
            [
                [0.01222, 0.0189, 0.028795],
                [0.025982, 0.0189, 0.019159],
                [0.025982, 0.0021, 0.019159],
                [0.01222, 0.0021, 0.028795],
            ],
            dtype=np.float32),
    ]

    # 下板
    low_board_corners = [
        np.array(
            [
                [-0.025982, -0.0021, 0.019159],
                [-0.01222, -0.0021, 0.028795],
                [-0.01222, -0.0189, 0.028795],
                [-0.025982, -0.0189, 0.019159],
            ],
            dtype=np.float32),
        np.array(
            [
                [-0.0084, -0.0021, 0.03],
                [0.0084, -0.0021, 0.03],
                [0.0084, -0.0189, 0.03],
                [-0.0084, -0.0189, 0.03],
            ],
            dtype=np.float32),
        np.array(
            [
                [0.01222, -0.0021, 0.028795],
                [0.025982, -0.0021, 0.019159],
                [0.025982, -0.0189, 0.019159],
                [0.01222, -0.0189, 0.028795],
            ],
            dtype=np.float32),
    ]

    # 创建上、下标记板
    up_board = cv2.aruco.Board_create(up_board_corners, aruco_dict, up_board_ids)
    low_board = cv2.aruco.Board_create(low_board_corners, aruco_dict, low_board_ids)

    # 将标记板的值传回
    return aruco_dict, aruco_parameters, up_board_ids, up_board, low_board_ids, low_board


# ----------------------------------------------------------#
# 计算姿态矩阵
# ----------------------------------------------------------#
def calculate_pose_matrix(img_gray, camera_matrix, dist_coefficient):
    # 输出结果先定义并赋空
    success, low_stable_matrix, up_cv_matrix, low_cv_matrix = [], [], [], []

    # 设置标记板
    aruco_dict, aruco_parameters, up_board_ids, up_board, low_board_ids, low_board = setting_board()

    # ********************  CV库调用，得到每张图像对应的相机外参矩阵：t,r 向量   ******************** #
    # 关键步骤1：调用detectMarkers函数，检测角和ID, ID是值在字典中的ID。一个标记4个角坐标。ID6个，角6X4=24个
    corners, ids, rejected = cv2.aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_parameters,
                                                     cameraMatrix=camera_matrix, distCoeff=dist_coefficient)

    # 按照ID来存标记码的角点
    markers_id_corners_dict = dict()

    # *************** 识别模式，有上、下2个标记板，分别粘在上下中切牙上。
    # 上板61-63，下板64-66。
    # 检测不到全部6个二维码， 则返回False
    if ids is None or len(ids) < 6:
        success = False
        return success, low_stable_matrix, up_cv_matrix, low_cv_matrix

    # 检测到全部6个二维码，
    # 所有识别到的角点
    for i in range(len(ids)):
        markers_id_corners_dict[ids[i][0]] = (list(corners))[i]

    # 上板的角点
    up_corners = [
        markers_id_corners_dict[61],
        markers_id_corners_dict[62],
        markers_id_corners_dict[63],
    ]
    # 下板的角点
    low_corners = [
        markers_id_corners_dict[64],
        markers_id_corners_dict[65],
        markers_id_corners_dict[66],
    ]

    # 关键步骤2：调用estimatePoseBoard函数， 计算并输出相机外参矩阵（旋转向量，平移向量）
    up_board_success, up_board_rvec, up_board_tvec = cv2.aruco.estimatePoseBoard(up_corners, up_board_ids, up_board,
                                                                                 camera_matrix, dist_coefficient, None,
                                                                                 None)

    low_board_success, low_board_rvec, low_board_tvec = cv2.aruco.estimatePoseBoard(low_corners, low_board_ids,
                                                                                    low_board,
                                                                                    camera_matrix, dist_coefficient,
                                                                                    None,
                                                                                    None)

    # ********************  通过旋转向量，平移向量， 计算相机外参矩阵    ******************** #
    if up_board_success and low_board_success:
        # 平移向量，米制转毫米制
        up_board_tvec = up_board_tvec * 1000
        low_board_tvec = low_board_tvec * 1000

        # 计算变换矩阵（旋转矩阵，结合平移向量） —— Up
        up_rot_mat = cv2.Rodrigues(up_board_rvec)[0]  # 罗德里格斯公式返回3X3旋转矩阵，3X9Jacobian矩阵。取第1个，[0]
        up_cv_matrix = np.c_[up_rot_mat, up_board_tvec]  # np的c_[]函数，增加一列，把3X3变成3X4
        up_cv_matrix = np.row_stack((up_cv_matrix, [0, 0, 0, 1]))  # np的函数，增加一行，把3X4变成4X4

        # 计算变换矩阵（旋转矩阵，结合平移向量） —— Low
        low_rot_mat = cv2.Rodrigues(low_board_rvec)[0]  # 罗德里格斯公式返回3X3旋转矩阵，3X9Jacobian矩阵。取第1个，[0]
        low_cv_matrix = np.c_[low_rot_mat, low_board_tvec]  # np的c_[]函数，增加一列，把3X3变成3X4
        low_cv_matrix = np.row_stack((low_cv_matrix, [0, 0, 0, 1]))  # np的函数，增加一行，把3X4变成4X4

        # 计算下板相对于上板的姿态
        # 上下板建立父子关系，上板为父，下板为子，下板相对于上板的姿态 （UpCvMatrix的逆矩阵实现从儿子到父亲的转换）
        up_board_aligned = np.eye(4)
        low_stable_matrix = up_board_aligned @ np.linalg.inv(up_cv_matrix) @ low_cv_matrix

        # 成功计算出了姿态矩阵
        success = True
    else:
        success = False

    # 返回每张图像上下板的姿态矩阵
    return success, low_stable_matrix, up_cv_matrix, low_cv_matrix
