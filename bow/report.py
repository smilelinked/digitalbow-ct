# coding=utf-8
import io
import logging
import tempfile
import zipfile

from PIL import ImageFont, ImageDraw, Image
import numpy as np
import copy
import math
import cv2
import os
import json
import dicttoxml
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from fpdf import FPDF

from bow.algm import get_information_json, get_stable_json, get_object_prefix, put_json, put_information_json, \
    get_position_json
from bow.s3 import put_obj, list_objects, get_obj_exception
from bow.utils import (print_chinese,
                       point_trajectory,
                       define_plane,
                       coincident_point,
                       moving_average)

dicttoxml.LOG.setLevel(logging.ERROR)


def process_img_tmp_dir(tmp_dir):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(False)
        image_list = [i for i in os.listdir(tmp_dir)]
        for image_path in sorted(image_list):
            pdf.add_page()
            pdf.image(os.path.join(tmp_dir, image_path), 0, 0, 210, 297)

        return pdf.output(name='', dest='S').encode('latin1')
    except Exception as e:
        print(f"process tmp dir to pdf failed with {e}")
    finally:
        pass
        # shutil.rmtree(tmp_dir)


def get_jaw_motion_json(object_prefix, case_info):
    jaw_motion = {}
    jaw_motion_frankfurt = {}

    resp = list_objects(object_prefix)
    if resp.get('ResponseMetadata').get('HTTPStatusCode') < 300 and resp.get('Contents') is not None:
        for content in resp.get('Contents'):
            key = content.get('Key')
            if key.endswith(".json"):
                motion_type = key.split("/")[-1].split(".")[0]
                temple = json.loads(get_obj_exception(key).read())

                # 用眶耳状态的运动数据
                jaw_motion_frankfurt[motion_type] = temple[0]

                if isinstance(temple, dict):
                    jaw_motion[motion_type] = temple
                else:
                    if case_info["reference"] == "frankfurt":
                        jaw_motion[motion_type] = temple[0]
                    elif case_info["reference"] == "ala-tragus":
                        jaw_motion[motion_type] = temple[1]

    return jaw_motion_frankfurt, jaw_motion


def stable_report_page1(image, case_info, stable_data):
    # 绘画的样式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    font_thickness = 2
    line_type = cv2.LINE_AA

    #####################################################################
    # -----------    病例姓名, 时间, 参考平面   ------------ #
    #####################################################################
    # 报告上写时间
    org_time = (970, 555)
    image = cv2.putText(image, str(case_info["time"]), org_time, font, font_scale, font_color, font_thickness)

    # 报告上写姓名, 中文输出要特殊处理
    org_name = (1000, 465)
    image = print_chinese(image, org_name, case_info["name"])

    # 报告上写参考平面
    if case_info["reference"] == "frankfurt":
        reference_chinese = "眶耳平面"
    elif case_info["reference"] == "ala-tragus":
        reference_chinese = "鼻翼耳屏线"
    else:
        reference_chinese = "眶耳平面"
    org_reference = (985, 745)
    image = print_chinese(image, org_reference, reference_chinese)

    #####################################################################
    # -----------    绘制点和轨迹线   ------------ #
    #####################################################################
    # 不同颜色  BGR
    color_picture_point = [197, 112, 84]
    color_stable_point = [10, 255, 0]
    color_motion_rc = [82, 77, 205]
    color_motion_lc = [2, 169, 230]
    color_motion_ip = [168, 156, 51]

    # 3个子图的坐标原点
    org_rc = (400, 1240)
    org_lc = (1700, 1240)
    org_ip = (1050, 2344)

    # 像素/毫米
    scale_rc = 55  # 图片设计时，1mm等于55像素
    scale_lc = 55
    scale_ip = 40

    # ********************     figure 1, 右髁突, ZY    ******************** #
    # 动态轨迹
    for i in range(len(stable_data["track"]["RC_list"])):
        rc_zy = (int(org_rc[0] + scale_rc * stable_data["track"]["RC_list"][i][2]),
                 int(org_rc[1] - scale_rc * stable_data["track"]["RC_list"][i][1]))
        if i > 0:
            rc_zy_pre = (int(org_rc[0] + scale_rc * stable_data["track"]["RC_list"][i - 1][2]),
                         int(org_rc[1] - scale_rc * stable_data["track"]["RC_list"][i - 1][1]))
            cv2.line(image, rc_zy_pre, rc_zy, color_motion_rc, 1, line_type)

    # 静态点
    for i in range(len(stable_data["points"]["RC"])):
        rc_zy = (int(org_rc[0] + scale_rc * stable_data["points"]["RC"][i][2]),
                 int(org_rc[1] - scale_rc * stable_data["points"]["RC"][i][1]))
        cv2.circle(image, rc_zy, 3, color_picture_point, -1)

    # ********************      figure 2, 左髁突, ZY     ******************** #
    # 动态轨迹
    for i in range(len(stable_data["track"]["LC_list"])):
        lc_zy = (int(org_lc[0] - scale_lc * stable_data["track"]["LC_list"][i][2]),
                 int(org_lc[1] - scale_lc * stable_data["track"]["LC_list"][i][1]))
        if i > 0:
            lc_zy_pre = (int(org_lc[0] - scale_lc * stable_data["track"]["LC_list"][i - 1][2]),
                         int(org_lc[1] - scale_lc * stable_data["track"]["LC_list"][i - 1][1]))
            cv2.line(image, lc_zy_pre, lc_zy, color_motion_lc, 1, line_type)

    # 静态点
    for i in range(len(stable_data["points"]["LC"])):
        lc_zy = (int(org_lc[0] - scale_lc * stable_data["points"]["LC"][i][2]),
                 int(org_lc[1] - scale_lc * stable_data["points"]["LC"][i][1]))
        cv2.circle(image, lc_zy, 3, color_picture_point, -1)

    # ********************      figure 3, 切端, XZ     ******************** #
    # 稳定点
    if len(stable_data["stable_point"]["IP"]) > 1:
        ip_xz = (int(org_ip[0] - scale_ip * stable_data["stable_point"]["IP"][0]),
                 int(org_ip[1] - scale_ip * stable_data["stable_point"]["IP"][2]))
        cv2.circle(image, ip_xz, 10, color_stable_point, -1)

    # 动态轨迹
    for i in range(len(stable_data["track"]["IP_list"])):
        ip_xz = (int(org_ip[0] - scale_ip * stable_data["track"]["IP_list"][i][0]),
                 int(org_ip[1] - scale_ip * stable_data["track"]["IP_list"][i][2]))
        if i > 0:
            ip_xz_pre = (int(org_ip[0] - scale_ip * stable_data["track"]["IP_list"][i - 1][0]),
                         int(org_ip[1] - scale_ip * stable_data["track"]["IP_list"][i - 1][2]))
            cv2.line(image, ip_xz_pre, ip_xz, color_motion_ip, 1, line_type)

    # 静态点
    for i in range(len(stable_data["points"]["IP"])):
        ip_xz = (int(org_ip[0] - scale_ip * stable_data["points"]["IP"][i][0]),
                 int(org_ip[1] - scale_ip * stable_data["points"]["IP"][i][2]))
        cv2.circle(image, ip_xz, 3, color_picture_point, -1)

    return image


def stable_report(uid, cid):
    #####################################################################
    # -----------  读入1张模板图片    ------------ #
    #####################################################################
    tmp1 = "./images/stable_page1.png"
    img_page1 = cv2.imread(tmp1)

    #####################################################################
    # -----------  读数据文件，获得数据    ------------ #
    #####################################################################
    # *************** 读病例信息文件
    case_info = get_information_json(uid, cid)

    # *************** 读stable.json文件
    temple, _ = get_stable_json(uid, cid)

    # 不同的参考平面, 不同的数据
    stable_data = []
    # 兼容旧数据
    if isinstance(temple, dict):
        stable_data = temple
    else:
        if case_info["reference"] == "frankfurt":
            stable_data = temple[0]
        elif case_info["reference"] == "ala-tragus":
            stable_data = temple[1]

    #####################################################################
    # -----------    在图片上绘制    ------------ #
    #####################################################################
    # 绘第1页
    img_page1 = stable_report_page1(img_page1, case_info, stable_data)

    #####################################################################
    # -----------    图片转成PDF保存，并删除图片    ------------ #
    #####################################################################
    # 新建临时文件夹
    tmp_dir = tempfile.mkdtemp()
    cv2.imwrite(os.path.join(tmp_dir, "stable_report_page1.png"), img_page1)
    pdf_content = process_img_tmp_dir(tmp_dir)
    pdf_file = get_object_prefix(uid, cid) + 'stable_report.pdf'
    return put_obj(pdf_file, pdf_content)


# ----------------------------------------------------------#
# 模块2的报告
# ----------------------------------------------------------#
def position_report_page1(image, case_info, position_data):
    # 绘画的样式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    font_thickness = 2

    #####################################################################
    # -----------    病例姓名, 时间, 参考平面   ------------ #
    #####################################################################
    # 报告上写时间
    org_time = (970, 555)
    image = cv2.putText(image, str(case_info["time"]), org_time, font, font_scale, font_color, font_thickness)

    # 报告上写姓名, 中文输出要特殊处理
    org_name = (1000, 465)  # 名字的坐标起点
    image = print_chinese(image, org_name, case_info["name"])

    # 报告上写参考平面
    if case_info["reference"] == "frankfurt":
        reference_chinese = "眶耳平面"
    elif case_info["reference"] == "ala-tragus":
        reference_chinese = "鼻翼耳屏线"
    else:
        reference_chinese = "眶耳平面"
    org_reference = (985, 745)
    image = print_chinese(image, org_reference, reference_chinese)

    #####################################################################
    # -----------    绘制点和轨迹线   ------------ #
    #####################################################################
    # 不同颜色  BGR
    color_points = [[166, 194, 22], [48, 59, 255], [211, 159, 0], [0, 122, 255], [255, 99, 167],
                    [126, 177, 217], [255, 122, 0], [45, 193, 251], [89, 199, 52], [146, 45, 255],
                    [88, 88, 199], [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100],
                    [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100]]

    # 4个子图的坐标原点
    org_fig1 = (620, 1480)
    org_fig2 = (1480, 1480)
    org_fig3 = (620, 2346)
    org_fig4 = (1480, 2346)

    # 像素/毫米
    scale = 80  # 图片设计时，1mm等于80像素

    # 静态画点
    for i in range(len(position_data["points"]["RC"])):
        # ********************    figure 1: 右髁突， ZY   ******************** #
        fig1_zy = (int(org_fig1[0] + scale * position_data["points"]["RC"][i][2]),
                   int(org_fig1[1] - scale * position_data["points"]["RC"][i][1]))
        cv2.circle(image, fig1_zy, 8, color_points[i], -1)

        # ********************    figure 2: 左髁突， ZY   ******************** #
        fig2_zy = (int(org_fig2[0] - scale * position_data["points"]["LC"][i][2]),
                   int(org_fig2[1] - scale * position_data["points"]["LC"][i][1]))
        cv2.circle(image, fig2_zy, 8, color_points[i], -1)

        # ********************    figure 3: 右髁突， XY   ******************** #
        fig3_xy = (int(org_fig3[0] + scale * position_data["points"]["RC"][i][0]),
                   int(org_fig3[1] - scale * position_data["points"]["RC"][i][1]))
        cv2.circle(image, fig3_xy, 8, color_points[i], -1)

        # ********************    figure 4: 左髁突， XY   ******************** #
        fig4_xy = (int(org_fig4[0] + scale * position_data["points"]["LC"][i][0]),
                   int(org_fig4[1] - scale * position_data["points"]["LC"][i][1]))
        cv2.circle(image, fig4_xy, 8, color_points[i], -1)

    return image


def position_report_page2(image, position_data):
    # 绘画的样式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (80, 80, 80)
    font_thickness = 2

    #####################################################################
    # -----------    表格输出点坐标  ------------ #
    #####################################################################
    # 不同颜色  BGR
    color_points = [[166, 194, 22], [48, 59, 255], [211, 159, 0], [0, 122, 255], [255, 99, 167],
                    [126, 177, 217], [255, 122, 0], [45, 193, 251], [89, 199, 52], [146, 45, 255],
                    [88, 88, 199], [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100],
                    [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100]]

    # 坐标的像素值
    # 左右网格起点
    org_left = (260, 526)
    org_right = (1120, 526)
    # 网格的高和宽
    grid_h = 78
    grid_w1 = 370
    grid_w2 = 116
    diff_circle = (25, grid_h / 2)  # 颜色圆相对网格左顶点的坐标
    diff_name = (50, 20)  # 名字相对网格左顶点的坐标
    diff_number = (20, 40)  # XYZ数字相对网格左顶点的坐标

    # ********************      表格中文名字     ******************** #

    font_chinese = ImageFont.truetype("msyh.ttc", 32)  # 中文字体和大小
    img = Image.fromarray(image)  # 转化为PIL库可以处理的图片格式
    draw = ImageDraw.Draw(img)

    for i in range(len(position_data["points"]["RC"])):
        # ********************    左边一栏:   右髁突    ******************** #
        name_left = (int(org_left[0] + diff_name[0]),
                     int(org_left[1] + diff_name[1] + i * grid_h))
        draw.text(name_left, position_data["points"]["position_info"][i]["position_name"], font=font_chinese,
                  fill=font_color)

        # ********************    右边一栏:   右髁突   ******************** #
        name_right = (int(org_right[0] + diff_name[0]),
                      int(org_right[1] + diff_name[1] + i * grid_h))
        draw.text(name_right, position_data["points"]["position_info"][i]["position_name"], font=font_chinese,
                  fill=font_color)

    # 恢复到CV格式的图像，方便下面的输出
    image = np.array(img)

    # ********************       表格填写      ******************** #

    for i in range(len(position_data["points"]["RC"])):
        # ********************    左边一栏:   右髁突    ******************** #
        # 画小圆
        circle_left = (int(org_left[0] + diff_circle[0]),
                       int(org_left[1] + diff_circle[1] + i * grid_h))
        cv2.circle(image, circle_left, 10, color_points[i], -1)

        # 数字
        x_left = (int(org_left[0] + grid_w1 + diff_number[0]),
                  int(org_left[1] + diff_number[1] + i * grid_h))
        image = cv2.putText(image, str(round(position_data["points"]["RC"][i][0], 1)), x_left, font, font_scale,
                            font_color, font_thickness)

        y_left = (int(org_left[0] + grid_w1 + grid_w2 + diff_number[0]),
                  int(org_left[1] + diff_number[1] + i * grid_h))
        image = cv2.putText(image, str(round(position_data["points"]["RC"][i][1], 1)), y_left, font, font_scale,
                            font_color, font_thickness)

        z_left = (int(org_left[0] + grid_w1 + grid_w2 + grid_w2 + diff_number[0]),
                  int(org_left[1] + diff_number[1] + i * grid_h))
        image = cv2.putText(image, str(round(position_data["points"]["RC"][i][2], 1)), z_left, font, font_scale,
                            font_color, font_thickness)

        # ********************    右边一栏:   左髁突    ******************** #
        # 画小圆
        circle_right = (int(org_right[0] + diff_circle[0]),
                        int(org_right[1] + diff_circle[1] + i * grid_h))
        cv2.circle(image, circle_right, 10, color_points[i], -1)

        # 数字
        x_right = (int(org_right[0] + grid_w1 + diff_number[0]),
                   int(org_right[1] + diff_number[1] + i * grid_h))
        image = cv2.putText(image, str(round(position_data["points"]["LC"][i][0], 1)), x_right, font, font_scale,
                            font_color, font_thickness)

        y_right = (int(org_right[0] + grid_w1 + grid_w2 + diff_number[0]),
                   int(org_right[1] + diff_number[1] + i * grid_h))
        image = cv2.putText(image, str(round(position_data["points"]["LC"][i][1], 1)), y_right, font, font_scale,
                            font_color, font_thickness)

        z_right = (int(org_right[0] + grid_w1 + grid_w2 + grid_w2 + diff_number[0]),
                   int(org_right[1] + diff_number[1] + i * grid_h))
        image = cv2.putText(image, str(round(position_data["points"]["LC"][i][2], 1)), z_right, font, font_scale,
                            font_color, font_thickness)

    # 返回
    return image


def position_report(uid, cid):
    #####################################################################
    # -----------  读入1张模板图片    ------------ #
    #####################################################################
    source_dir = "./images/"

    tmp1 = source_dir + "position_page1.png"
    tmp2 = source_dir + "position_page2.png"

    img_page1 = cv2.imread(tmp1)
    img_page2 = cv2.imread(tmp2)

    #####################################################################
    # -----------  读数据文件，获得数据    ------------ #
    #####################################################################
    # *************** 读病例信息文件
    case_info = get_information_json(uid, cid)

    # *************** 读json文件
    temple, _ = get_position_json(uid, cid)

    # 不同的参考平面, 不同的数据
    position_data = []
    if case_info["reference"] == "frankfurt":
        position_data = temple[0]
    elif case_info["reference"] == "ala-tragus":
        position_data = temple[1]

    #####################################################################
    # -----------    在图片上绘制    ------------ #
    #####################################################################
    # 绘第1页
    img_page1 = position_report_page1(img_page1, case_info, position_data)

    # 绘第2页
    img_page2 = position_report_page2(img_page2, position_data)

    #####################################################################
    # -----------    图片转成PDF保存，并删除图片    ------------ #
    #####################################################################
    # 新建临时文件夹
    tmp_dir = tempfile.mkdtemp()

    cv2.imwrite(os.path.join(tmp_dir, "position_report_page1.png"), img_page1)
    cv2.imwrite(os.path.join(tmp_dir, "position_report_page2.png"), img_page2)

    pdf_content = process_img_tmp_dir(tmp_dir)
    pdf_file = get_object_prefix(uid, cid) + 'position_report.pdf'
    return put_obj(pdf_file, pdf_content)


# ---------------------------------------------------------- #
# 模块3：标准运动轨迹模块的报告
# ---------------------------------------------------------- #
def matrix_pin(uid, cid):
    # *************** 读病例信息文件
    case_info = get_information_json(uid, cid)

    if case_info.get("Matrix_Pin_Processed"):
        return case_info

    # *************** 获得初始数据
    bp = case_info["BP"]
    gap = np.array(case_info["gap"])  # 变成数组格式，后面有数组加法
    angle_op = case_info["angle_OP"]
    jaw_splint_pose = np.array(case_info["jawSplint_pose"])
    lc = case_info["LC"]
    articulator = case_info["articulator"]

    #####################################################################
    # ----------- Matrix:  报告模块（PDF,XML等）需要的变换矩阵  ------------ #
    #####################################################################
    #  被动旋转：物体不动，只是从不坐标系来看
    #  不同厂商，用不同的坐标系，所以需要进行转换
    #  被动旋转需要特别注意的是，顺时针旋转是正角度。为了以示区分，用大写表示。
    #  被动旋转与主动旋转是互逆的

    ##########################################
    # -----     P - > X 坐标变换     -------- #
    ##########################################
    # P->X: 没有旋转，只有平移（原点变为BP）。
    transform_x_p = np.array([[1, 0, 0, -bp[0]],
                              [0, 1, 0, -bp[1]],
                              [0, 0, 1, -bp[2]],
                              [0, 0, 0, 1]])

    # ***********    眶耳平面下   F//L    ********** #
    ##########################################
    # ---------   P - > F 坐标变换  --------- #
    ##########################################
    # P->F:  gap， 颌插板粘胶的高度(凹下去1mm, 粘胶1mm)，颌插板姿态
    # 颌插板原点P在F坐标系下坐标
    p_f = jaw_splint_pose[0:3, 0:3] @ [0, -2, 0] + gap
    translation = np.array([[1, 0, 0, p_f[0]],
                            [0, 1, 0, p_f[1]],
                            [0, 0, 1, p_f[2]],
                            [0, 0, 0, 1]])
    transform_f_p = translation @ jaw_splint_pose

    #########################################
    # ---------   F - > X 坐标变换  --------- #
    #########################################
    transform_x_f = transform_x_p @ np.linalg.inv(transform_f_p)

    #########################################
    # ---------  A - > F 坐标变换  --------- #
    #########################################
    # 第一步：
    # F->A: 顺时针转90度，还需要平移。
    # 下颌中切牙（IP点）在A坐标系下的坐标值， 坐标值就是平移向量。
    ip_a_x = 0
    ip_a_y = articulator["condyle"][1] - np.abs(lc[2])
    ip_a_z = articulator["condyle"][2] - np.abs(lc[1])
    transform_a_f = np.array([[1, 0, 0, ip_a_x],
                              [0, math.cos(math.pi * 90 / 180), -math.sin(math.pi * 90 / 180), ip_a_y],
                              [0, math.sin(math.pi * 90 / 180), math.cos(math.pi * 90 / 180), ip_a_z],
                              [0, 0, 0, 1]])
    # 第二步
    transform_f_a = np.linalg.inv(transform_a_f)

    #########################################
    # ------    F - > XA 坐标变换     ------ #
    #########################################
    # F->XA:  先平移原点， 在围绕左右髁突X轴顺时针旋转7.5度 (这里的顺序要注意)
    # 下颌中切牙（IP点）在XA坐标系下的坐标值，坐标值就是平移向量。
    ip_xa_x = 0
    ip_xa_y = - lc[1]
    ip_xa_z = - lc[2]
    translation = np.array([[1, 0, 0, ip_xa_x],
                            [0, 1, 0, ip_xa_y],
                            [0, 0, 1, ip_xa_z],
                            [0, 0, 0, 1]])
    angle_xa = articulator["angle"]
    rotation = np.array([[1, 0, 0, 0],
                         [0, math.cos(math.pi * angle_xa / 180), -math.sin(math.pi * angle_xa / 180), 0],
                         [0, math.sin(math.pi * angle_xa / 180), math.cos(math.pi * angle_xa / 180), 0],
                         [0, 0, 0, 1]])
    transform_xa_f = rotation @ translation

    #########################################
    # ------   P - > XA 坐标变换矩阵   ------ #
    #########################################
    transform_xa_p = transform_xa_f @ transform_f_p

    ##########################################
    # -------   XA - > A  坐标变换   -------- #
    ##########################################
    # XA->A: XA -> F, F -> A
    transform_a_xa = transform_a_f @ np.linalg.inv(transform_xa_f)

    #########################################
    # ------   P - > A  坐标变换    -------- #
    #########################################
    # ***********  眶耳平面下  ************** #
    # 实体合架：眶耳平面实际可能不水平，会上仰，取决于颌架。比如Artex颌架上仰7.5度
    # 主动旋转变换， 绕X轴顺时针旋转颌架角
    angle_a = articulator["angle"]
    matrix_a = np.array([[1, 0, 0, 0],
                         [0, math.cos(math.pi * (-angle_a) / 180), -math.sin(math.pi * (-angle_a) / 180), 0],
                         [0, math.sin(math.pi * (-angle_a) / 180), math.cos(math.pi * (-angle_a) / 180), 0],
                         [0, 0, 0, 1]])
    # 相乘得P->A
    transform_a_p = transform_a_xa @ matrix_a @ transform_xa_f @ transform_f_p

    # ***********  鼻翼耳屏线下  T (夹OP角) F ************** #
    # 实体合架：整个头颅绕髁突顺时针旋转OP角+颌架角，使咬合面水平，下颌中切牙（IP点）会跟着旋转、移动。
    # 主动旋转变换， 绕X轴顺时针旋转OP角+颌架角
    angle_a_op = angle_a + angle_op
    matrix_a_op = np.array([[1, 0, 0, 0],
                            [0, math.cos(math.pi * (-angle_a_op) / 180), -math.sin(math.pi * (-angle_a_op) / 180), 0],
                            [0, math.sin(math.pi * (-angle_a_op) / 180), math.cos(math.pi * (-angle_a_op) / 180), 0],
                            [0, 0, 0, 1]])
    # 相乘得P->A
    transform_a_p_tragus = transform_a_xa @ matrix_a_op @ transform_xa_f @ transform_f_p

    # 保存数据
    case_info["Transform_X_P"] = transform_x_p.tolist()
    case_info["Transform_F_P"] = transform_f_p.tolist()
    case_info["Transform_F_A"] = transform_f_a.tolist()
    case_info["Transform_X_F"] = transform_x_f.tolist()
    case_info["Transform_XA_F"] = transform_xa_f.tolist()
    case_info["Transform_XA_P"] = transform_xa_p.tolist()
    case_info["Transform_A_P"] = transform_a_p.tolist()
    case_info["Transform_A_P_Tragus"] = transform_a_p_tragus.tolist()

    #####################################################################
    # -----------     Pin: 计算合架切道针的轨迹线     ------------ #
    #####################################################################
    resp = list_objects(get_object_prefix(uid, cid) + 'standard')
    print(f"resp: {resp}")
    if resp.get('ResponseMetadata').get('HTTPStatusCode') < 300 and resp.get('Contents') is not None:
        for content in resp.get('Contents'):
            key = content.get('Key')
            if key.endswith(".json"):
                track_data = json.loads(get_obj_exception(key).read())

                # pin点的轨迹线,  眶耳平面,  F坐标系
                _, pin_list = point_trajectory(track_data[0]["Matrix_list"], articulator["pin"])
                # pin点的轨迹线,  鼻翼耳屏线,  T坐标系
                _, pin_list_tragus = point_trajectory(track_data[1]["Matrix_list"], articulator["pin"])

                # 保存数据
                track_data[0]["pin_list"] = pin_list.tolist()
                track_data[1]["pin_list"] = pin_list_tragus.tolist()
                put_json(key, track_data)

    #####################################################################
    # -----------    修改状态，并保持information     ------------ #
    #####################################################################
    # 修改状态
    case_info["Matrix_Pin_Processed"] = True

    # 存储
    put_information_json(uid, cid, case_info)

    return case_info


def get_motion_info(jaw_motion):
    #####################################################################
    # -----------  注意事项 ------------ #
    #####################################################################
    # 跟下颌切点相关的3个角度：即 左侧方角度，右侧方角度，前伸角度，要根据切导针特殊处理

    #####################################################################
    # -----------    轨迹线要计算的12个指标  ------------ #
    #####################################################################
    #  定义12个指标
    info = {
        # 左侧方运动
        "left_angle": -10000,
        "left_bennett_shift_length": -10000,
        # 右侧方运动
        "right_angle": -10000,
        "right_bennett_shift_length": -10000,
        # 前伸运动
        "forward_angle": -10000,
        "rc_forward_angle": -10000,
        "lc_forward_angle": -10000,
        # 后退运动
        "backward_length": -10000,
        # 最大开口运动
        "maxopen_angle": -10000,
        "maxopen_length": -10000,
        # 左侧方运动，前伸运动，同时存在
        "right_bennett_angle": -10000,
        # 右侧方运动，前伸运动，同时存在
        "left_bennett_angle": -10000,
    }

    #####################################################################
    # -----------    寻找峰值点的函数  ------------ #
    #####################################################################
    def find_peak_index(axis_1, axis_2, index):
        # 通过绝对值来比较大小
        if index == 1:  # 根据第1个轴的数据寻找峰值
            tmp = list(map(abs, axis_1))
        else:  # 根据第2个轴的数据寻找峰值
            tmp = list(map(abs, axis_2))
        # 寻找端点的序号
        peak_index = np.argmax(tmp)

        return peak_index

    #####################################################################
    # -----------    需要记录的17个峰值点  ------------ #
    #####################################################################
    left_ip_xy_peak_index = []
    left_rc_xz_peak_index = []
    left_lc_xz_peak_index = []
    left_pin_xy_peak_index = []  # 特殊处理，左侧角度要从合架切导针pin点去算角度，而不是IP点

    right_ip_xy_peak_index = []
    right_rc_xz_peak_index = []
    right_lc_xz_peak_index = []
    right_pin_xy_peak_index = []  # 特殊处理，右侧角度要从合架切导针pin点去算角度，而不是IP点

    forward_ip_zy_peak_index = []
    forward_rc_xz_peak_index = []
    forward_lc_xz_peak_index = []
    forward_rc_zy_peak_index = []
    forward_lc_zy_peak_index = []
    forward_pin_zy_peak_index = []  # 特殊处理，前伸角度要从合架切导针pin点去算角度，而不是IP点

    backward_ip_zy_peak_index = []

    maxopen_ip_xy_peak_index = []
    maxopen_ip_zy_peak_index = []

    #####################################################################
    # -----------   计算运动轨迹的角度和长度   ------------ #
    #####################################################################
    # ********************   每个轨迹，求峰值, 17个峰值（计算12报告数据）  ******************** #
    for i in range(len(jaw_motion)):
        # motion的类型
        motion_type = list(jaw_motion.keys())[i]

        # 获取本次motion的 IP，LC, RC
        ip_list = np.asarray(jaw_motion[motion_type]["IP_list"])
        lc_list = np.asarray(jaw_motion[motion_type]["LC_list"])
        rc_list = np.asarray(jaw_motion[motion_type]["RC_list"])
        pin_list = np.asarray(jaw_motion[motion_type]["pin_list"])

        # 计算峰值
        if motion_type == "standard_left":
            left_ip_xy_peak_index = find_peak_index(ip_list[:, 0], ip_list[:, 1], 1)
            left_rc_xz_peak_index = find_peak_index(rc_list[:, 0], rc_list[:, 2], 2)
            left_lc_xz_peak_index = find_peak_index(lc_list[:, 0], lc_list[:, 2], 1)
            left_pin_xy_peak_index = find_peak_index(pin_list[:, 0], pin_list[:, 1], 1)

        elif motion_type == "standard_right":
            right_ip_xy_peak_index = find_peak_index(ip_list[:, 0], ip_list[:, 1], 1)
            right_rc_xz_peak_index = find_peak_index(rc_list[:, 0], rc_list[:, 2], 1)
            right_lc_xz_peak_index = find_peak_index(lc_list[:, 0], lc_list[:, 2], 2)
            right_pin_xy_peak_index = find_peak_index(pin_list[:, 0], pin_list[:, 1], 1)

        elif motion_type == "standard_forward":
            forward_ip_zy_peak_index = find_peak_index(ip_list[:, 1], ip_list[:, 2], 2)
            forward_rc_xz_peak_index = find_peak_index(rc_list[:, 0], rc_list[:, 2], 2)
            forward_lc_xz_peak_index = find_peak_index(lc_list[:, 0], lc_list[:, 2], 2)
            forward_pin_zy_peak_index = find_peak_index(pin_list[:, 1], pin_list[:, 2], 2)

            forward_rc_zy_peak_index = find_peak_index(rc_list[:, 1], rc_list[:, 2], 2)
            forward_lc_zy_peak_index = find_peak_index(lc_list[:, 1], lc_list[:, 2], 2)

        elif motion_type == "standard_backward":
            backward_ip_zy_peak_index = find_peak_index(ip_list[:, 1], ip_list[:, 2], 2)

        elif motion_type == "standard_maxopen":
            maxopen_ip_xy_peak_index = find_peak_index(ip_list[:, 0], ip_list[:, 1], 2)
            maxopen_ip_zy_peak_index = find_peak_index(ip_list[:, 1], ip_list[:, 2], 1)

    # ********************  计算角度和长度, 并绘制图像  ******************** #
    # left运动存在
    if left_ip_xy_peak_index:
        # 左侧方运动，角度
        # x = math.fabs(jawMotion["standard_left"]["IP_list"][left_ip_xy_peak_index][0])
        # y = math.fabs(jawMotion["standard_left"]["IP_list"][left_ip_xy_peak_index][1])
        # motion_info["left_angle"] = math.atan(y / x) * 180 / math.pi

        # 特殊处理，左侧角度要从合架切导针pin点去算角度，而不是IP点
        x = math.fabs(jaw_motion["standard_left"]["pin_list"][left_pin_xy_peak_index][0])
        y = math.fabs(jaw_motion["standard_left"]["pin_list"][left_pin_xy_peak_index][1])
        info["left_angle"] = math.atan(y / x) * 180 / math.pi

        # 左侧方运动，瞬时侧移距离iss
        info["left_bennett_shift_length"] = math.fabs(
            jaw_motion["standard_left"]["LC_list"][left_lc_xz_peak_index][0])

    # right运动存在
    if right_ip_xy_peak_index:
        # 右侧方运动，角度
        # x = math.fabs(jawMotion["standard_right"]["IP_list"][right_ip_xy_peak_index][0])
        # y = math.fabs(jawMotion["standard_right"]["IP_list"][right_ip_xy_peak_index][1])
        # motion_info["right_angle"] = math.atan(y / x) * 180 / math.pi

        # 特殊处理，右侧角度要从合架切导针的角度去算角度，而不是IP点
        x = math.fabs(jaw_motion["standard_right"]["pin_list"][right_pin_xy_peak_index][0])
        y = math.fabs(jaw_motion["standard_right"]["pin_list"][right_pin_xy_peak_index][1])
        info["right_angle"] = math.atan(y / x) * 180 / math.pi

        # 右侧方运动，瞬时侧移距离
        info["right_bennett_shift_length"] = math.fabs(
            jaw_motion["standard_right"]["RC_list"][right_rc_xz_peak_index][0])

    # forward运动存在
    if forward_ip_zy_peak_index:
        # 前伸运动，角度
        # y = math.fabs(jawMotion["standard_forward"]["IP_list"][forward_ip_zy_peak_index][1])
        # z = math.fabs(jawMotion["standard_forward"]["IP_list"][forward_ip_zy_peak_index][2])
        # motion_info["forward_angle"] = math.atan(y / z) * 180 / math.pi

        # 特殊处理，前伸角度要从合架切导针的角度去算角度，而不是IP点
        y = math.fabs(jaw_motion["standard_forward"]["pin_list"][forward_pin_zy_peak_index][1])
        z = math.fabs(jaw_motion["standard_forward"]["pin_list"][forward_pin_zy_peak_index][2])
        info["forward_angle"] = math.atan(y / z) * 180 / math.pi

        # 前伸运动，右髁道斜度
        y = math.fabs(jaw_motion["standard_forward"]["RC_list"][forward_rc_zy_peak_index][1])
        z = math.fabs(jaw_motion["standard_forward"]["RC_list"][forward_rc_zy_peak_index][2])
        info["rc_forward_angle"] = math.atan(y / z) * 180 / math.pi

        # 前伸运动，左髁道斜度
        y = math.fabs(jaw_motion["standard_forward"]["LC_list"][forward_lc_zy_peak_index][1])
        z = math.fabs(jaw_motion["standard_forward"]["LC_list"][forward_lc_zy_peak_index][2])
        info["lc_forward_angle"] = math.atan(y / z) * 180 / math.pi

    # backward运动存在
    if backward_ip_zy_peak_index:
        # 后退运动，距离
        info["backward_length"] = math.fabs(
            jaw_motion["standard_backward"]["IP_list"][backward_ip_zy_peak_index][2])

    # maxopen运动存在
    if maxopen_ip_xy_peak_index:
        # 最大开口运动，角度
        x = math.fabs(jaw_motion["standard_maxopen"]["IP_list"][maxopen_ip_xy_peak_index][0])
        y = math.fabs(jaw_motion["standard_maxopen"]["IP_list"][maxopen_ip_xy_peak_index][1])
        info["maxopen_angle"] = math.atan(x / y) * 180 / math.pi

        # 最大开口运动，开口距离
        y = math.fabs(jaw_motion["standard_maxopen"]["IP_list"][maxopen_ip_zy_peak_index][1])
        z = math.fabs(jaw_motion["standard_maxopen"]["IP_list"][maxopen_ip_zy_peak_index][2])
        info["maxopen_length"] = pow(pow(y, 2) + pow(z, 2), 0.5)

    # 左侧方运动，前伸运动，同时存在
    if left_rc_xz_peak_index and forward_rc_xz_peak_index:
        # 右贝内特角
        x = math.fabs(jaw_motion["standard_left"]["RC_list"][left_rc_xz_peak_index][0])
        z = math.fabs(jaw_motion["standard_left"]["RC_list"][left_rc_xz_peak_index][2])
        theta1 = math.atan(x / z) * 180 / math.pi

        x = math.fabs(jaw_motion["standard_forward"]["RC_list"][forward_rc_xz_peak_index][0])
        z = math.fabs(jaw_motion["standard_forward"]["RC_list"][forward_rc_xz_peak_index][2])
        theta2 = math.atan(x / z) * 180 / math.pi

        if (jaw_motion["standard_left"]["RC_list"][left_rc_xz_peak_index][0] *
            jaw_motion["standard_forward"]["RC_list"][forward_rc_xz_peak_index][0]) > 0:
            info["right_bennett_angle"] = math.fabs(theta1 - theta2)
        else:
            info["right_bennett_angle"] = theta1 + theta2

    # 右侧方运动，前伸运动，同时存在
    if right_lc_xz_peak_index and forward_lc_xz_peak_index:
        #  左贝内特角
        x = math.fabs(jaw_motion["standard_right"]["LC_list"][right_lc_xz_peak_index][0])
        z = math.fabs(jaw_motion["standard_right"]["LC_list"][right_lc_xz_peak_index][2])
        theta1 = math.atan(x / z) * 180 / math.pi

        x = math.fabs(jaw_motion["standard_forward"]["LC_list"][forward_lc_xz_peak_index][0])
        z = math.fabs(jaw_motion["standard_forward"]["LC_list"][forward_lc_xz_peak_index][2])
        theta2 = math.atan(x / z) * 180 / math.pi

        if (jaw_motion["standard_right"]["LC_list"][right_lc_xz_peak_index][0] *
            jaw_motion["standard_forward"]["LC_list"][forward_lc_xz_peak_index][0]) > 0:
            info["left_bennett_angle"] = math.fabs(theta1 - theta2)
        else:
            info["left_bennett_angle"] = theta1 + theta2

    return info


def get_platform_info(case_info):
    # 输入数据
    bp = case_info["BP"]
    lp = case_info["LP"]
    rp = case_info["RP"]
    transform_ap = []
    # 不同的参考平面, 计算方式不同
    if case_info["reference"] == "frankfurt":
        transform_ap = np.array(case_info["Transform_A_P"])
    elif case_info["reference"] == "ala-tragus":
        transform_ap = np.array(case_info["Transform_A_P_Tragus"])

    #####################################################################
    # -----------   计算转移平台的信息    ------------ #
    #####################################################################
    # A坐标系Z=0水平面上3个点, 形成一个水平面方程（水平面就是后面的UV平面）          —— A坐标系下
    p1_uv = [1, 0, 0]
    p2_uv = [-1, 0, 0]
    p3_uv = [0, 1, 0]
    plane_uv = define_plane(p1_uv, p2_uv, p3_uv)
    norm_plane_uv = np.array([plane_uv[0], plane_uv[1], plane_uv[2]])

    # 颌插板平面：BP， LP, RP 形成的平面 (但是需要把BP,LP,RP换成A坐标系下的坐标)    —— A坐标系下
    # 选择不同的参考平面，颌插板;  在颌架中，为了适配机械面弓，髁突和颌插板需要前向平移一个量
    bp_a = (transform_ap @ np.append(bp, 1))[:-1] + case_info["articulator"]["translation"]
    lp_a = (transform_ap @ np.append(lp, 1))[:-1] + case_info["articulator"]["translation"]
    rp_a = (transform_ap @ np.append(rp, 1))[:-1] + case_info["articulator"]["translation"]
    plane_jaw_splint = define_plane(bp_a, lp_a, rp_a)
    norm_plane_jaw_splint = np.array([plane_jaw_splint[0], plane_jaw_splint[1], plane_jaw_splint[2]])

    # 两个平面的夹角：法线的夹角就是平面的夹角
    cos_theta = np.vdot(norm_plane_uv, norm_plane_jaw_splint) / (
            np.linalg.norm(norm_plane_uv) * np.linalg.norm(norm_plane_jaw_splint))

    # 计算长度后，需要根据观察平面，调整下杆上的刻度值，颌插板0点平面距离观察平面有3mm的高度，
    pin_length_center = abs(bp_a[2] / cos_theta) + 3
    pin_length_left = abs(lp_a[2] / cos_theta) + 3
    pin_length_right = abs(rp_a[2] / cos_theta) + 3

    # BP沿法线延伸后与UV水平面相交，求交点                                      ——  A坐标系下
    norm_bp = np.array([plane_jaw_splint[0] + bp_a[0], plane_jaw_splint[1] + bp_a[1], plane_jaw_splint[2] + bp_a[2]])
    p1_d = np.vdot(norm_bp, norm_plane_uv) + plane_uv[3]
    p2_d = np.vdot(bp_a - norm_bp, norm_plane_uv)
    n = p1_d / p2_d
    pin_position_center = norm_bp - n * (bp_a - norm_bp)

    # LP沿法线延伸后与UV水平面相交，求交点                                    ——  A坐标系下
    norm_lp = np.array([plane_jaw_splint[0] + lp_a[0], plane_jaw_splint[1] + lp_a[1], plane_jaw_splint[2] + lp_a[2]])
    p1_d = np.vdot(norm_lp, norm_plane_uv) + plane_uv[3]
    p2_d = np.vdot(lp_a - norm_lp, norm_plane_uv)
    n = p1_d / p2_d
    pin_position_left = norm_lp - n * (lp_a - norm_lp)

    # RP沿法线延伸后与UV水平面相交，求交点                                    ——  A坐标系下
    norm_rp = np.array([plane_jaw_splint[0] + rp_a[0], plane_jaw_splint[1] + rp_a[1], plane_jaw_splint[2] + rp_a[2]])
    p1_d = np.vdot(norm_rp, norm_plane_uv) + plane_uv[3]
    p2_d = np.vdot(rp_a - norm_rp, norm_plane_uv)
    n = p1_d / p2_d
    pin_position_right = norm_rp - n * (rp_a - norm_rp)

    #####################################################################
    # -----------    保存数据   ------------ #
    #####################################################################
    # 数据结构
    info = {
        # 定位支撑杆的长度
        "pinLength_center": pin_length_center,
        "pinLength_left": pin_length_left,
        "pinLength_right": pin_length_right,

        # 坐标纸上的点坐标
        "pinPosition_center": pin_position_center,
        "pinPosition_left": pin_position_left,
        "pinPosition_right": pin_position_right,
    }

    return info


def standard_report_page1(image, case_info, motion_info):
    # 绘画的样式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    font_thickness = 2

    #####################################################################
    # -----------    病例姓名，时间, 参考平面，合架品牌   ------------ #
    #####################################################################
    # 报告上写时间
    org_time = (970, 555)
    image = cv2.putText(image, str(case_info["time"]), org_time, font, font_scale, font_color, font_thickness)

    # 报告上写姓名, 中文输出要特殊处理
    org_name = (1000, 465)  # 名字的坐标起点
    image = print_chinese(image, org_name, case_info["name"])

    # 报告上写参考平面
    if case_info["reference"] == "frankfurt":
        reference_chinese = "眶耳平面"
    elif case_info["reference"] == "ala-tragus":
        reference_chinese = "鼻翼耳屏线"
    else:
        reference_chinese = "眶耳平面"
    org_reference = (985, 745)
    image = print_chinese(image, org_reference, reference_chinese)

    # 报告上写颌架品牌
    org_brand = (1000, 835)
    image = cv2.putText(image, str(case_info["articulator"]["brand"]), org_brand, font, font_scale, font_color,
                        font_thickness)

    #####################################################################
    # -----------    表格中写值   ------------ #
    #####################################################################
    left_chinese = (710, 2138)
    right_chinese = (1750, 2138)
    grid_h = 100
    grid_w = 300

    # ********************    左列6个   ******************** #
    if motion_info["left_angle"] != -10000:  # 左1
        org_left_angle = (630, 2170)
        image = cv2.putText(image, str(round(motion_info["left_angle"], 1)), org_left_angle, font, font_scale,
                            font_color, font_thickness)
        coordinate_chinese = left_chinese
        image = print_chinese(image, coordinate_chinese, "°")

    if motion_info["right_angle"] != -10000:  # 左2
        org_right_angle = (630, 2270)
        image = cv2.putText(image, str(round(motion_info["right_angle"], 1)), org_right_angle, font, font_scale,
                            font_color, font_thickness)
        coordinate_chinese = (left_chinese[0], left_chinese[1] + grid_h)
        image = print_chinese(image, coordinate_chinese, "°")

    if motion_info["forward_angle"] != -10000:  # 左3
        org_forward_angle = (630, 2370)
        image = cv2.putText(image, str(round(motion_info["forward_angle"], 1)), org_forward_angle, font,
                            font_scale, font_color, font_thickness)
        coordinate_chinese = (left_chinese[0], left_chinese[1] + 2 * grid_h)
        image = print_chinese(image, coordinate_chinese, "°")

    if motion_info["backward_length"] != -10000:  # 左4
        org_backward_length = (630, 2470)
        image = cv2.putText(image, str(round(motion_info["backward_length"], 1)) + " mm", org_backward_length, font,
                            font_scale, font_color, font_thickness)

    if motion_info["maxopen_length"] != -10000:  # 左5
        org_maxopen_length = (630, 2570)
        image = cv2.putText(image, str(round(motion_info["maxopen_length"], 1)) + " mm", org_maxopen_length, font,
                            font_scale, font_color, font_thickness)

    if motion_info["maxopen_angle"] != -10000:  # 左6
        org_maxopen_angle = (630, 2670)
        image = cv2.putText(image, str(round(motion_info["maxopen_angle"], 1)), org_maxopen_angle, font,
                            font_scale, font_color, font_thickness)
        coordinate_chinese = (left_chinese[0], left_chinese[1] + 5 * grid_h)
        image = print_chinese(image, coordinate_chinese, "°")

    # ********************     右列6个    ******************** #
    if motion_info["lc_forward_angle"] != -10000:  # 右1
        org_lc_forward_angle = (1670, 2170)
        image = cv2.putText(image, str(round(motion_info["lc_forward_angle"], 1)), org_lc_forward_angle, font,
                            font_scale, font_color, font_thickness)
        coordinate_chinese = right_chinese
        image = print_chinese(image, coordinate_chinese, "°")

    if motion_info["rc_forward_angle"] != -10000:  # 右2
        org_rc_forward_angle = (1670, 2270)
        image = cv2.putText(image, str(round(motion_info["rc_forward_angle"], 1)), org_rc_forward_angle, font,
                            font_scale, font_color, font_thickness)
        coordinate_chinese = (right_chinese[0], right_chinese[1] + grid_h)
        image = print_chinese(image, coordinate_chinese, "°")

    if motion_info["left_bennett_angle"] != -10000:  # 右3
        org_left_bennett_angle = (1670, 2370)
        image = cv2.putText(image, str(round(motion_info["left_bennett_angle"], 1)), org_left_bennett_angle, font,
                            font_scale, font_color, font_thickness)
        coordinate_chinese = (right_chinese[0], right_chinese[1] + 2 * grid_h)
        image = print_chinese(image, coordinate_chinese, "°")

    if motion_info["right_bennett_angle"] != -10000:  # 右4
        org_right_bennett_angle = (1670, 2470)
        image = cv2.putText(image, str(round(motion_info["right_bennett_angle"], 1)), org_right_bennett_angle,
                            font, font_scale, font_color, font_thickness)
        coordinate_chinese = (right_chinese[0], right_chinese[1] + 3 * grid_h)
        image = print_chinese(image, coordinate_chinese, "°")

    if motion_info["left_bennett_shift_length"] != -10000:  # 右5
        org_left_bennett_shift_length = (1670, 2570)
        image = cv2.putText(image, str(round(motion_info["left_bennett_shift_length"], 1)) + " mm",
                            org_left_bennett_shift_length, font, font_scale, font_color, font_thickness)

    if motion_info["right_bennett_shift_length"] != -10000:  # 右6
        org_right_bennett_shift_length = (1670, 2670)
        image = cv2.putText(image, str(round(motion_info["right_bennett_shift_length"], 1)) + " mm",
                            org_right_bennett_shift_length, font, font_scale, font_color, font_thickness)

    return image


def standard_report_page2(image, platform_info):
    # 文字的格式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    font_thickness = 2

    #####################################################################
    # -----------  需要用到的数据   ------------ #
    #####################################################################
    pin_length_center = platform_info["pinLength_center"]
    pin_length_left = platform_info["pinLength_left"]
    pin_length_right = platform_info["pinLength_right"]

    pin_position_center = platform_info["pinPosition_center"]
    pin_position_left = platform_info["pinPosition_left"]
    pin_position_right = platform_info["pinPosition_right"]

    #####################################################################
    # -----------  1/2： 表格中写UV值   ------------ #
    #####################################################################
    # 表格信息
    org_table = (400, 894)
    height_table = 120
    width_table = 150

    # 像素/毫米
    scale = 10  # 图片设计时，1mm等于10像素   采用254dpi： 254像素/25.4毫米 = 10像素/毫米

    # 后点（center）
    u_p1 = (org_table[0] + 25, org_table[1] + 65)
    image = cv2.putText(image, str(round(pin_position_center[0], 1)), u_p1, font, font_scale, font_color,
                        font_thickness)
    v_p1 = (u_p1[0], u_p1[1] + height_table - 18)
    image = cv2.putText(image, str(round(pin_position_center[1], 1)), v_p1, font, font_scale, font_color,
                        font_thickness)
    l_p1 = (v_p1[0], v_p1[1] + height_table - 18)
    image = cv2.putText(image, str(round(pin_length_center, 1)), l_p1, font, font_scale, font_color, font_thickness)

    # 左点
    u_p2 = (u_p1[0] + width_table, u_p1[1])
    image = cv2.putText(image, str(round(pin_position_left[0], 1)), u_p2, font, font_scale, font_color,
                        font_thickness)
    v_p2 = (v_p1[0] + width_table, v_p1[1])
    image = cv2.putText(image, str(round(pin_position_left[1], 1)), v_p2, font, font_scale, font_color,
                        font_thickness)
    l_p2 = (l_p1[0] + width_table, l_p1[1])
    image = cv2.putText(image, str(round(pin_length_left, 1)), l_p2, font, font_scale, font_color, font_thickness)

    # 右点
    u_p3 = (u_p2[0] + width_table, u_p2[1])
    image = cv2.putText(image, str(round(pin_position_right[0], 1)), u_p3, font, font_scale, font_color,
                        font_thickness)
    v_p3 = (v_p2[0] + width_table, v_p2[1])
    image = cv2.putText(image, str(round(pin_position_right[1], 1)), v_p3, font, font_scale, font_color,
                        font_thickness)
    l_p3 = (l_p2[0] + width_table, l_p2[1])
    image = cv2.putText(image, str(round(pin_length_right, 1)), l_p3, font, font_scale, font_color, font_thickness)

    #####################################################################
    # -----------  2/2： UV坐标中绘点    ------------ #
    #####################################################################

    origin_uv = [1390, 1467]  # UV原点在page3中的位置  （ UV坐标图中：往下95mm,往右45mm）

    # center点
    axis_c = (int(origin_uv[0] + scale * pin_position_center[0]), int(origin_uv[1] - scale * pin_position_center[1]))
    color_c = (86, 255, 72)
    cv2.circle(image, axis_c, 10, color_c, -1)

    # left点 (左点LP 要 画在UV坐标系的右边， 因为颌插板是反过来看的)
    axis_l = (int(origin_uv[0] + scale * pin_position_left[0]), int(origin_uv[1] - scale * pin_position_left[1]))
    color_l = (118, 113, 255)
    cv2.circle(image, axis_l, 10, color_l, -1)

    # right点 (右点RP 要 画在UV坐标系的左边， 因为颌插板是反过来看的)
    axis_r = (int(origin_uv[0] + scale * pin_position_right[0]), int(origin_uv[1] - scale * pin_position_right[1]))
    color_r = (255, 199, 43)
    cv2.circle(image, axis_r, 10, color_r, -1)

    return image


def standard_report_page3(image, jaw_motion):
    # 绘画的样式
    line_type = cv2.LINE_AA

    #####################################################################
    # -----------      绘制彩色轨迹  ------------ #
    #####################################################################
    # 不同轨迹的颜色  BGR
    color_left = [52, 46, 193]
    color_maxleft = [215, 72, 134]
    color_right = [0, 182, 230]
    color_maxright = [102, 199, 52]
    color_forward = [217, 152, 0]
    color_forwardmax = [29, 130, 43]
    color_chew = [42, 100, 180]
    color_maxopen = [168, 156, 51]
    color_backward = [170, 94, 0]

    # 3个子图的坐标原点
    org_fig1 = (622, 475)
    org_fig2 = (1328, 475)
    org_fig3 = (1051, 2227)

    # 像素/毫米
    scale_fig1 = 10
    scale_fig2 = 10
    scale_fig3 = 20

    # ********************      每个motion，循环画轨迹    ******************** #
    for i in range(len(jaw_motion)):
        # motion的类型
        motion_type = list(jaw_motion.keys())[i]

        # 不同轨迹，画不同颜色
        if motion_type == 'standard_left':
            motion_color = color_left
        elif motion_type == 'standard_maxleft':
            motion_color = color_maxleft
        elif motion_type == 'standard_right':
            motion_color = color_right
        elif motion_type == 'standard_maxright':
            motion_color = color_maxright
        elif motion_type == 'standard_forward':
            motion_color = color_forward
        elif motion_type == 'standard_forwardmax':
            motion_color = color_forwardmax
        elif motion_type == 'standard_chew':
            motion_color = color_chew
        elif motion_type == 'standard_maxopen':
            motion_color = color_maxopen
        elif motion_type == 'standard_backward':
            motion_color = color_backward
        else:
            motion_color = [0, 0, 0]

        # 获取本次motion的 IP
        ip_list = np.asarray(jaw_motion[motion_type]["IP_list"])

        # ********************    3子图中画轨迹（画点，然后和前面的点连线）   ******************** #
        for i in range(len(ip_list)):
            # figure 1,  IP点的XY
            fig1_xy = (int(org_fig1[0] + scale_fig1 * ip_list[i][0]), int(org_fig1[1] - scale_fig1 * ip_list[i][1]))
            if i > 0:
                fig1_xy_pre = (
                    int(org_fig1[0] + scale_fig1 * ip_list[i - 1][0]),
                    int(org_fig1[1] - scale_fig1 * ip_list[i - 1][1]))
                cv2.line(image, fig1_xy_pre, fig1_xy, motion_color, 1, line_type)

            # figure 2,  IP点的YZ
            fig2_yz = (int(org_fig2[0] - scale_fig2 * ip_list[i][2]), int(org_fig2[1] - scale_fig2 * ip_list[i][1]))
            if i > 0:
                fig2_yz_pre = (
                    int(org_fig2[0] - scale_fig2 * ip_list[i - 1][2]),
                    int(org_fig2[1] - scale_fig2 * ip_list[i - 1][1]))
                cv2.line(image, fig2_yz_pre, fig2_yz, motion_color, 1, line_type)

            # figure 3, IP点的XZ
            fig3_xz = (int(org_fig3[0] + scale_fig3 * ip_list[i][0]), int(org_fig3[1] + scale_fig3 * ip_list[i][2]))
            if i > 0:
                fig3_xz_pre = (
                    int(org_fig3[0] + scale_fig3 * ip_list[i - 1][0]),
                    int(org_fig3[1] + scale_fig3 * ip_list[i - 1][2]))
                cv2.line(image, fig3_xz_pre, fig3_xz, motion_color, 1, line_type)

    return image


def standard_report_page4(image, jaw_motion):
    # 绘画的样式
    line_type = cv2.LINE_AA

    #####################################################################
    # -----------      绘制彩色轨迹  ------------ #
    #####################################################################
    # 不同轨迹的颜色  BGR
    color_left = [52, 46, 193]
    color_maxleft = [215, 72, 134]
    color_right = [0, 182, 230]
    color_maxright = [102, 199, 52]
    color_forward = [217, 152, 0]
    color_forwardmax = [29, 130, 43]
    color_chew = [42, 100, 180]
    color_maxopen = [168, 156, 51]
    color_backward = [170, 94, 0]

    # 6个子图的坐标原点
    org_fig1 = (622, 475)
    org_fig2 = (1478, 475)
    org_fig3 = (622, 1275)
    org_fig4 = (1478, 1275)
    org_fig5 = (422, 2225)
    org_fig6 = (1678, 2225)

    # 像素/毫米
    scale = 10

    # ********************   每个motion，循环画轨迹，求峰值   ******************** #
    for i in range(len(jaw_motion)):
        # motion的类型
        motion_type = list(jaw_motion.keys())[i]

        # 不同轨迹，画不同颜色
        if motion_type == 'standard_left':
            motion_color = color_left
        elif motion_type == 'standard_maxleft':
            motion_color = color_maxleft
        elif motion_type == 'standard_right':
            motion_color = color_right
        elif motion_type == 'standard_maxright':
            motion_color = color_maxright
        elif motion_type == 'standard_forward':
            motion_color = color_forward
        elif motion_type == 'standard_forwardmax':
            motion_color = color_forwardmax
        elif motion_type == 'standard_chew':
            motion_color = color_chew
        elif motion_type == 'standard_maxopen':
            motion_color = color_maxopen
        elif motion_type == 'standard_backward':
            motion_color = color_backward
        else:
            motion_color = [0, 0, 0]

        # 获取本次motion的RC， LC
        rc_list = np.asarray(jaw_motion[motion_type]["RC_list"])
        lc_list = np.asarray(jaw_motion[motion_type]["LC_list"])

        # ********************    6副子图中画轨迹（画点，然后和前面的点连线）   ******************** #
        for i in range(len(rc_list)):
            # figure 1, RC点的XY
            fig1_xy = (int(org_fig1[0] + scale * rc_list[i][0]), int(org_fig1[1] - scale * rc_list[i][1]))
            if i > 0:
                fig1_xz_pre = (
                    int(org_fig1[0] + scale * rc_list[i - 1][0]), int(org_fig1[1] - scale * rc_list[i - 1][1]))
                cv2.line(image, fig1_xz_pre, fig1_xy, motion_color, 1, line_type)

            # figure 3, RC点的XZ
            fig3_xz = (int(org_fig3[0] + scale * rc_list[i][0]), int(org_fig3[1] + scale * rc_list[i][2]))
            if i > 0:
                fig3_xz_pre = (
                    int(org_fig3[0] + scale * rc_list[i - 1][0]), int(org_fig3[1] + scale * rc_list[i - 1][2]))
                cv2.line(image, fig3_xz_pre, fig3_xz, motion_color, 1, line_type)

            # figure 5, RC点的ZY
            fig5_yz = (int(org_fig5[0] + scale * rc_list[i][2]), int(org_fig5[1] - scale * rc_list[i][1]))
            if i > 0:
                fig5_yz_pre = (
                    int(org_fig5[0] + scale * rc_list[i - 1][2]), int(org_fig5[1] - scale * rc_list[i - 1][1]))
                cv2.line(image, fig5_yz_pre, fig5_yz, motion_color, 1, line_type)

            # figure 2, LC点的XY
            fig2_xy = (int(org_fig2[0] + scale * lc_list[i][0]), int(org_fig2[1] - scale * lc_list[i][1]))
            if i > 0:
                fig2_xz_pre = (
                    int(org_fig2[0] + scale * lc_list[i - 1][0]), int(org_fig2[1] - scale * lc_list[i - 1][1]))
                cv2.line(image, fig2_xz_pre, fig2_xy, motion_color, 1, line_type)

            # figure 4, LC点的XZ
            fig4_xz = (int(org_fig4[0] + scale * lc_list[i][0]), int(org_fig4[1] + scale * lc_list[i][2]))
            if i > 0:
                fig4_xz_pre = (
                    int(org_fig4[0] + scale * lc_list[i - 1][0]), int(org_fig4[1] + scale * lc_list[i - 1][2]))
                cv2.line(image, fig4_xz_pre, fig4_xz, motion_color, 1, line_type)

            # figure 6, LC点的ZY
            fig6_yz = (int(org_fig6[0] - scale * lc_list[i][2]), int(org_fig6[1] - scale * lc_list[i][1]))
            if i > 0:
                fig6_yz_pre = (
                    int(org_fig6[0] - scale * lc_list[i - 1][2]), int(org_fig6[1] - scale * lc_list[i - 1][1]))
                cv2.line(image, fig6_yz_pre, fig6_yz, motion_color, 1, line_type)

    return image


def standard_report(uid, cid):
    #####################################################################
    # -----------  读入3张模板图片    ------------ #
    #####################################################################
    source_dir = "./images/"

    tmp1 = source_dir + "standard_page1.png"
    tmp2 = source_dir + "standard_page2.png"
    tmp3 = source_dir + "standard_page3.png"
    tmp4 = source_dir + "standard_page4.png"

    img_page1 = cv2.imread(tmp1)
    img_page2 = cv2.imread(tmp2)
    img_page3 = cv2.imread(tmp3)
    img_page4 = cv2.imread(tmp4)

    #####################################################################
    # -----------   准备工作  ------------ #
    #####################################################################
    # 病例信息文件
    case_info = matrix_pin(uid, cid)

    #####################################################################
    # -----------  读数据文件，获得数据    ------------ #
    #####################################################################
    # *************** 读运动轨迹json，获取运动信息
    _, jaw_motion = get_jaw_motion_json(get_object_prefix(uid, cid) + 'standard', case_info)

    if len(jaw_motion) == 0:
        raise Exception(f"case {cid} has no standard motion")

    #####################################################################
    # -----------    注意事项       ------------ #
    # -----------  重合点数据处理    ------------ #
    #####################################################################
    # 合架报告部分，不需要进行重合点计算； 动态轨迹线部分，需要进行重合点计算
    jaw_motion_cp = copy.deepcopy(jaw_motion)

    # 存在3种以上的运动，重合点做原点
    if len(jaw_motion_cp) >= 3:

        # *************** 数据准备
        # 方便求轨迹的重合点
        ip_all, lc_all, rc_all = [], [], []
        for i in range(len(jaw_motion_cp)):
            # 运动的轨迹信息
            motion_type = list(jaw_motion_cp.keys())[i]
            # 为计算重合点装填数据
            ip_all += jaw_motion_cp[motion_type]["IP_list"]
            if motion_type != "standard_left":
                lc_all += jaw_motion_cp[motion_type]["LC_list"]
            if motion_type != "standard_right":
                rc_all += jaw_motion_cp[motion_type]["RC_list"]

        # *************** 计算各自的重合点
        # 半径
        radius = 0.25
        # 计算IP的重合点
        cp_ip = coincident_point(ip_all, radius)[1]
        # 计算LC的重合点
        cp_lc = coincident_point(lc_all, radius)[1]
        # 计算RC的重合点
        cp_rc = coincident_point(rc_all, radius)[1]

        # *************** 重合点做原点： 调整IP, LC, RC的坐标
        for i in range(len(jaw_motion_cp)):
            # 运动的轨迹信息
            motion_type = list(jaw_motion_cp.keys())[i]
            jaw_motion_cp[motion_type]["IP_list"] -= cp_ip
            jaw_motion_cp[motion_type]["LC_list"] -= cp_lc
            jaw_motion_cp[motion_type]["RC_list"] -= cp_rc

    #####################################################################
    # -----------    在图片上绘制    ------------ #
    #####################################################################
    # 绘第1页: 运动轨迹的12个参数， 用不带重合点的数据
    motion_info = get_motion_info(jaw_motion)
    img_page1 = standard_report_page1(img_page1, case_info, motion_info)

    # 绘第2页: 合架转移平台的参数
    platform_info = get_platform_info(case_info)
    img_page2 = standard_report_page2(img_page2, platform_info)

    # 绘第3页: 画动态轨迹线， 用带重合点的数据
    img_page3 = standard_report_page3(img_page3, jaw_motion_cp)

    # 绘第4页: 画动态轨迹线， 用带重合点的数据
    img_page4 = standard_report_page4(img_page4, jaw_motion_cp)

    #####################################################################
    # -----------    图片转成PDF保存，并删除图片    ------------ #
    #####################################################################
    # 新建临时文件夹
    tmp_dir = tempfile.mkdtemp()

    cv2.imwrite(tmp_dir + "/myPage1.png", img_page1)
    cv2.imwrite(tmp_dir + "/myPage2.png", img_page2)
    cv2.imwrite(tmp_dir + "/myPage3.png", img_page3)
    cv2.imwrite(tmp_dir + "/myPage4.png", img_page4)

    pdf_content = process_img_tmp_dir(tmp_dir)
    pdf_file = get_object_prefix(uid, cid) + 'standard_report.pdf'
    return put_obj(pdf_file, pdf_content)


def standard_xml(uid, cid, show_range):
    # 病例信息文件
    case_info = matrix_pin(uid, cid)

    #####################################################################
    # -----------  读数据文件，获得数据     ------------ #
    #####################################################################
    # *************** 读运动轨迹json，获取运动信息
    plate_movement, data_parameter_va = get_jaw_motion_json(get_object_prefix(uid, cid) + 'standard', case_info)

    if len(plate_movement) == 0:
        raise Exception(f"case {cid} has no standard motion")

    # *************** 读取数据
    bp = case_info["BP"]
    lp = case_info["LP"]
    rp = case_info["RP"]
    transform_x_p = np.array(case_info["Transform_X_P"])
    transform_f_p = np.array(case_info["Transform_F_P"])
    transform_x_f = np.array(case_info["Transform_X_F"])
    transform_xa_f = np.array(case_info["Transform_XA_F"])
    transform_xa_p = np.array(case_info["Transform_XA_P"])
    transform_t_f = np.array(case_info["Transform_T_F"])

    #####################################################################
    # -----------  特殊字符处理     ------------ #
    #####################################################################
    # 特别的处理
    movement_str = {
        "standard_left": "movement",  # 0空格，
        "standard_maxleft": "movement ",  # 1空格，
        "standard_right": "movement  ",  # 2空格，
        "standard_maxright": "movement   ",  # 3空格，
        "standard_forward": "movement    ",  # 4空格，
        "standard_forwardmax": "movement     ",  # 5空格，
        "standard_chew": "movement      ",  # 6空格，
        "standard_maxopen": "movement       ",  # 7空格，
        "standard_backward": "movement        ",  # 8空格，
    }

    chinese_str = {
        "standard_left": "左侧方运动",
        "standard_maxleft": "最大左侧方运动",
        "standard_right": "右侧方运动",
        "standard_maxright": "最大右侧方运动",
        "standard_forward": "前伸运动",
        "standard_forwardmax": "前伸最大运动",
        "standard_chew": "咀嚼运动",
        "standard_maxopen": "最大开口运动",
        "standard_backward": "后退运动",
    }

    #####################################################################
    # -----------  第1个XML   ------------ #
    #####################################################################
    # 第1个XML， 记录相对运动， 颌插板是理论坐标， 采用X坐标系（BP是原点）
    xml_dict = {
        "link": "http://www.teethlink.cn",
        "program": "ARSS-AI智能面弓系统",
        "program_version": "2.3.0",
        "patient": case_info["name"],
        "measured": case_info["time"],
        "description": {},
        "coordinate_system": "jaw_splint",
        "upper_position": {},
        "movements": {},
    }

    # ********************    upper_position 字段    ******************** #
    # BP, LP, RP 理论值在X坐标系的坐标                                       —— X坐标系下
    bp_x = (transform_x_p @ np.append(bp, 1))[:-1]
    lp_x = (transform_x_p @ np.append(lp, 1))[:-1]
    rp_x = (transform_x_p @ np.append(rp, 1))[:-1]

    xml_dict["upper_position"] = {
        "type": "jaw_splint",
        "points": {
            "mark_1": {"x": bp_x[0], "y": bp_x[1], "z": bp_x[2]},
            "mark_2": {"x": lp_x[0], "y": lp_x[1], "z": lp_x[2]},
            "mark_3": {"x": rp_x[0], "y": rp_x[1], "z": rp_x[2]},
        }
    }

    # ********************    movements 字段   ******************** #
    # *************** BP, LP, RP 运动轨迹
    movements = {}
    for i in range(len(plate_movement)):
        # 某次运动
        movement_type = list(plate_movement.keys())[i]

        # 本次运动， 轨迹的矩阵列表
        matrix_list = plate_movement[movement_type]["Matrix_list"]

        # 本次运动， BP, LP, RP 的列表
        bp_list, lp_list, rp_list = [], [], []

        for i in range(len(matrix_list)):
            matrix = np.array(matrix_list[i])

            # X坐标系下的坐标值
            bp = (transform_x_f @ matrix @ transform_f_p @ np.append(bp, 1))[:-1]
            lp = (transform_x_f @ matrix @ transform_f_p @ np.append(lp, 1))[:-1]
            rp = (transform_x_f @ matrix @ transform_f_p @ np.append(rp, 1))[:-1]

            # 存成数组，准备平滑
            bp_list.append(bp)
            lp_list.append(lp)
            rp_list.append(rp)

        # 平滑处理
        bp_list = moving_average(bp_list)
        lp_list = moving_average(lp_list)
        rp_list = moving_average(rp_list)

        # 存储
        plate_movement[movement_type]["BP_list"] = bp_list.tolist()
        plate_movement[movement_type]["LP_list"] = lp_list.tolist()
        plate_movement[movement_type]["RP_list"] = rp_list.tolist()

    # *************** 计算重合点
    # 存在3种以上的运动，重合点做原点
    if len(plate_movement) >= 3:
        # 方便求轨迹的重合点
        bp_all, lp_all, rp_all = [], [], []

        # 装填数据
        for i in range(len(plate_movement)):
            # 某次运动
            movement_type = list(plate_movement.keys())[i]
            # 为计算重合点装填数据
            bp_all += plate_movement[movement_type]["BP_list"]
            if movement_type != "standard_left":
                lp_all += plate_movement[movement_type]["LP_list"]
            if movement_type != "standard_right":
                rp_all += plate_movement[movement_type]["RP_list"]

        # 半径的球体
        radius = 0.25
        # 计算BP的重合点
        cp_bp = coincident_point(bp_all, radius)[1]
        # 计算LP的重合点
        cp_lp = coincident_point(lp_all, radius)[1]
        # 计算RP的重合点
        cp_rp = coincident_point(rp_all, radius)[1]

        # 重合点做原点
        for i in range(len(plate_movement)):
            # 运动的轨迹信息
            movement_type = list(plate_movement.keys())[i]
            plate_movement[movement_type]["BP_list"] -= cp_bp
            plate_movement[movement_type]["LP_list"] -= cp_lp
            plate_movement[movement_type]["RP_list"] -= cp_rp

    # *************** 得到movements字典
    for i in range(len(plate_movement)):
        # 运动的轨迹信息
        movement_type = list(plate_movement.keys())[i]

        # 加上起始点
        bp_list = plate_movement[movement_type]["BP_list"] + bp_x
        lp_list = plate_movement[movement_type]["LP_list"] + lp_x
        rp_list = plate_movement[movement_type]["RP_list"] + rp_x

        # 根据EXO格式要求，把BP, LP, RP按要求重新做成list
        bp_list_exo, lp_list_exo, rp_list_exo = [], [], []

        # 计算用户需要展示的起始帧和结束帧
        begin, end = 0, plate_movement[movement_type]["size"]
        show_range_key = movement_type[9:]
        if show_range.get(show_range_key):
            begin, end = show_range.get(show_range_key)

        for i in range(len(bp_list)):
            if i < begin or i > end:
                continue
            bp_list_exo.append({"x": bp_list[i][0], "y": bp_list[i][1], "z": bp_list[i][2]})
            lp_list_exo.append({"x": lp_list[i][0], "y": lp_list[i][1], "z": lp_list[i][2]})
            rp_list_exo.append({"x": rp_list[i][0], "y": rp_list[i][1], "z": rp_list[i][2]})

        # 本次运动，BP, LP, RP轨迹打包
        bp_track = {
            "type": "",
            "id": "mark_1",
            "size": plate_movement[movement_type]["size"],
            "frequency": plate_movement[movement_type]["frequency"],
            "quants": bp_list_exo,
        }
        lp_track = {
            "type": "",
            "id": "mark_2",
            "size": plate_movement[movement_type]["size"],
            "frequency": plate_movement[movement_type]["frequency"],
            "quants": lp_list_exo,
        }
        rp_track = {
            "type": "",
            "id": "mark_3",
            "size": plate_movement[movement_type]["size"],
            "frequency": plate_movement[movement_type]["frequency"],
            "quants": rp_list_exo,
        }
        # movement特殊空格
        movement_name = movement_str[movement_type]
        movements[movement_name] = {
            "type": movement_type,
            "id": chinese_str[movement_type],
            "tracks": {"track": bp_track, "track ": lp_track, "track  ": rp_track},  # 空格是有用的，能区分3个track，不然后面覆盖前面
        }
    xml_dict["movements"] = movements

    #####################################################################
    # -----------  第2个XML   ------------ #
    #####################################################################
    # 第2个XML， 记录相对运动， 颌插板是实际姿态的坐标， 采用XA坐标系（髁突中点是原点）， 虚拟颌架信息也要包含
    xml_va_dict = {
        "link": "http://www.teethlink.cn",
        "program": "ARSS-AI智能面弓系统",
        "program_version": "2.3.0",
        "patient": case_info["name"],
        "measured": case_info["time"],
        "description": {},
        "coordinate_system": "axis_orbital",
        "upper_position": {},
        "articulator_settings": {},
        "roms": {},
        "movements": {},
    }

    # ********************   upper_position 字段   ******************** #

    # # 主动旋转变换， 绕X轴逆时针旋转7.5度（合架设计的问题）
    # matrix = np.array([[1, 0, 0, 0],
    #                    [0, math.cos(math.pi * 7.5 / 180), -math.sin(math.pi * 7.5 / 180), 0],
    #                    [0, math.sin(math.pi * 7.5 / 180), math.cos(math.pi * 7.5 / 180), 0],
    #                    [0, 0, 0, 1]])

    # 虚拟合架下面，不同参考平面会有旋转
    rotation = []
    if case_info["reference"] == "frankfurt":
        rotation = np.eye(4)
    elif case_info["reference"] == "ala-tragus":
        rotation = transform_t_f

    # BP, LP, RP                                                       —— XA坐标系下
    bp_xa = (rotation @ transform_xa_p @ np.append(bp, 1))[:-1]
    lp_xa = (rotation @ transform_xa_p @ np.append(lp, 1))[:-1]
    rp_xa = (rotation @ transform_xa_p @ np.append(rp, 1))[:-1]

    xml_va_dict["upper_position"] = {
        "type": "jaw_splint",
        "points": {
            "mark_1": {"x": bp_xa[0], "y": bp_xa[1], "z": bp_xa[2]},
            "mark_2": {"x": lp_xa[0], "y": lp_xa[1], "z": lp_xa[2]},
            "mark_3": {"x": rp_xa[0], "y": rp_xa[1], "z": rp_xa[2]},
        }
    }

    # ********************   articulator_settings 字段   ******************** #
    # 运动轨迹的12个参数， 用不带重合点的数据
    motion_info = get_motion_info(data_parameter_va)
    amann_girrbach_artex = {
        "left": {
            "bennett_angle": motion_info["left_bennett_angle"],
            "iss": motion_info["left_bennett_shift_length"],
            "inclination": motion_info["lc_forward_angle"],
            "fti": motion_info["left_angle"]
        },
        "right": {
            "bennett_angle": motion_info["right_bennett_angle"],
            "iss": motion_info["right_bennett_shift_length"],
            "inclination": motion_info["rc_forward_angle"],
            "fti": motion_info["right_angle"]
        },
        "sagittal_fti": motion_info["forward_angle"],
    }
    xml_va_dict["articulator_settings"] = {"amann_girrbach_artex": amann_girrbach_artex, }

    # ********************    roms 字段   ******************** #
    # xml_va_dict["roms"] = {
    #
    # }

    # ********************     movements 字段     ******************** #
    movements = {}
    for i in range(len(plate_movement)):
        # 某次运动
        movement_type = list(plate_movement.keys())[i]

        # 这5种运动不需要
        if movement_type == "standard_maxleft" or movement_type == "standard_maxright" \
                or movement_type == "standard_forwardmax" or movement_type == "standard_chew" \
                or movement_type == "standard_maxopen":
            continue

        # 本次运动， 轨迹的矩阵列表
        matrix_list = plate_movement[movement_type]["Matrix_list"]

        # 本次运动， BP, LP, RP 的列表
        bp_list, lp_list, rp_list = [], [], []

        for i in range(len(matrix_list)):
            matrix = np.array(matrix_list[i])

            # XA坐标系下的坐标值
            bp = (rotation @ transform_xa_f @ matrix @ transform_f_p @ np.append(bp, 1))[:-1]
            lp = (rotation @ transform_xa_f @ matrix @ transform_f_p @ np.append(lp, 1))[:-1]
            rp = (rotation @ transform_xa_f @ matrix @ transform_f_p @ np.append(rp, 1))[:-1]

            # 存成数组，准备平滑
            bp_list.append(bp)
            lp_list.append(lp)
            rp_list.append(rp)

        # 平滑处理
        bp_list = moving_average(bp_list)
        lp_list = moving_average(lp_list)
        rp_list = moving_average(rp_list)

        # 加上起始点
        bp_list = bp_list + bp_xa
        lp_list = lp_list + lp_xa
        rp_list = rp_list + rp_xa

        # 根据EXO格式要求，把BP, LP, RP按要求重新做成list
        bp_list_exo, lp_list_exo, rp_list_exo = [], [], []

        # 计算用户需要展示的起始帧和结束帧
        begin, end = 0, plate_movement[movement_type]["size"]
        show_range_key = movement_type[9:]
        if show_range.get(show_range_key):
            begin, end = show_range.get(show_range_key)

        for i in range(len(matrix_list)):
            if i < begin or i > end:
                continue
            bp_list_exo.append({"x": bp_list[i][0], "y": bp_list[i][1], "z": bp_list[i][2]})
            lp_list_exo.append({"x": lp_list[i][0], "y": lp_list[i][1], "z": lp_list[i][2]})
            rp_list_exo.append({"x": rp_list[i][0], "y": rp_list[i][1], "z": rp_list[i][2]})

        # 本次运动，BP, LP, RP轨迹打包
        bp_track = {
            "type": "",
            "id": "mark_1",
            "size": plate_movement[movement_type]["size"],
            "frequency": plate_movement[movement_type]["frequency"],
            "quants": bp_list_exo,
        }
        lp_track = {
            "type": "",
            "id": "mark_2",
            "size": plate_movement[movement_type]["size"],
            "frequency": plate_movement[movement_type]["frequency"],
            "quants": lp_list_exo,
        }
        rp_track = {
            "type": "",
            "id": "mark_3",
            "size": plate_movement[movement_type]["size"],
            "frequency": plate_movement[movement_type]["frequency"],
            "quants": rp_list_exo,
        }
        # movement特殊空格
        movement_name = movement_str[movement_type]
        movements[movement_name] = {
            "type": movement_type,
            "id": chinese_str[movement_type],
            "tracks": {"track": bp_track, "track ": lp_track, "track  ": rp_track},  # 空格是有用的，能区分3个track，不然后面覆盖前面
        }

    xml_va_dict["movements"] = movements

    #####################################################################
    # ----------- 生成XML文件    ------------ #
    #####################################################################
    # *************** 第1个XML
    # DICT -> XML
    track_binary = dicttoxml.dicttoxml(xml_dict, custom_root='dental_measurement', attr_type=False)
    xml = track_binary.decode('utf-8')
    dom = parseString(xml)
    track_output = dom.toprettyxml(encoding='utf-8')

    # 保存文件
    track_buffer = io.BytesIO()
    track_buffer.write(track_output)
    track_buffer.seek(0)

    # 修改标签： item -> quant
    tree = ET.parse(track_buffer)
    root = tree.getroot()
    for node in root.iter('item'):
        node.tag = 'quant'
    # 重新保存
    memory_file = io.BytesIO()
    track_xml_buffer = io.BytesIO()
    tree.write(track_xml_buffer, encoding='utf-8', xml_declaration=True)

    # *************** 第2个XML
    # DICT -> XML
    track_va_binary = dicttoxml.dicttoxml(xml_va_dict, custom_root='dental_measurement', attr_type=False)
    va_xml = track_va_binary.decode('utf-8')
    dom = parseString(va_xml)
    track_va_output = dom.toprettyxml(encoding='utf-8')
    # 保存文件
    track_va_buffer = io.BytesIO()
    track_va_buffer.write(track_va_output)
    track_va_buffer.seek(0)

    # 修改标签： item -> quant
    va_tree = ET.parse(track_va_buffer)
    va_root = va_tree.getroot()
    for node in va_root.iter('item'):
        node.tag = 'quant'
    # 重新保存
    track_va_xml_buffer = io.BytesIO()
    va_tree.write(track_va_xml_buffer, encoding='utf-8', xml_declaration=True)

    # *************** ZIP
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("standard_xml.xml", track_xml_buffer.getvalue())
        zf.writestr("standard_xml_va.xml", track_va_xml_buffer.getvalue())

    memory_file.seek(0)
    res = memory_file.getvalue()
    zip_file = get_object_prefix(uid, cid) + "standard.zip"
    return put_obj(zip_file, res)


# ----------------------------------------------------------#
# 模块4：自定义运动轨迹模块的报告
# ----------------------------------------------------------#
def custom_report_page1(image, case_info):
    # 绘画的样式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    font_thickness = 2

    #####################################################################
    # -----------    病例姓名,  时间, 参考平面    ------------ #
    #####################################################################
    # 报告上写时间
    org_time = (970, 555)
    image = cv2.putText(image, str(case_info["time"]), org_time, font, font_scale, font_color, font_thickness)

    # 报告上写姓名, 中文输出要特殊处理
    org_name = (1000, 465)  # 名字的坐标起点
    image = print_chinese(image, org_name, case_info["name"])

    # 报告上写参考平面
    if case_info["reference"] == "frankfurt":
        reference_chinese = "眶耳平面"
    elif case_info["reference"] == "ala-tragus":
        reference_chinese = "鼻翼耳屏线"
    else:
        reference_chinese = "眶耳平面"
    org_reference = (985, 745)
    image = print_chinese(image, org_reference, reference_chinese)

    return image


def custom_report_page2(image, case_info, jaw_motion):
    # 绘画的样式
    line_type = cv2.LINE_AA

    #####################################################################
    # -----------     绘制彩色轨迹    ------------ #
    #####################################################################
    # 不同颜色  BGR
    # 坐标原点颜色
    color_org = [166, 194, 22]

    # 曲线的颜色
    colors = [[52, 46, 193], [0, 182, 230], [217, 152, 0], [29, 130, 43], [170, 94, 0], [168, 156, 51],
              [48, 59, 255], [211, 159, 0], [0, 122, 255], [255, 99, 167], [126, 177, 217],
              [255, 122, 0], [45, 193, 251], [89, 199, 52], [146, 45, 255], [88, 88, 199],
              [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100],
              [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100]]

    # 像素/毫米
    scale = 8

    # ********************    坐标原点     ******************** #

    # 选择不同的参考平面
    matrix = []
    if case_info["reference"] == "frankfurt":
        matrix = np.eye(4)
    elif case_info["reference"] == "ala-tragus":
        matrix = np.array(case_info["Transform_T_F"])

    # *************** figure 1
    # IP
    org_fig1 = (1050, 1035)
    cv2.circle(image, org_fig1, 10, color_org, -1)

    # RC
    rc = case_info["RC"]
    new_rc = (matrix @ np.append(rc, 1))[:-1]
    org_rc_fig1 = (int(org_fig1[0] + scale * new_rc[0]), int(org_fig1[1] - scale * new_rc[1]))  # XY
    cv2.circle(image, org_rc_fig1, 10, color_org, -1)

    # LC
    lc = case_info["LC"]
    new_lc = (matrix @ np.append(lc, 1))[:-1]
    org_lc_fig1 = (int(org_fig1[0] + scale * new_lc[0]), int(org_fig1[1] - scale * new_lc[1]))  # XY
    cv2.circle(image, org_lc_fig1, 10, color_org, -1)

    # *************** figure 2
    # IP
    org_fig2 = (1050, 2477)
    cv2.circle(image, org_fig2, 10, color_org, -1)

    # RC
    org_rc_fig2 = (int(org_fig2[0] + scale * new_rc[0]), int(org_fig2[1] + scale * new_rc[2]))  # XZ
    cv2.circle(image, org_rc_fig2, 10, color_org, -1)

    # LC
    org_lc_fig2 = (int(org_fig2[0] + scale * new_lc[0]), int(org_fig2[1] + scale * new_lc[2]))  # XZ
    cv2.circle(image, org_lc_fig2, 10, color_org, -1)

    # ********************    每个motion，循环画轨迹     ******************** #
    for i in range(len(jaw_motion)):
        # motion的类型
        motion_type = list(jaw_motion.keys())[i]

        # motion的颜色
        motion_color = colors[i]

        # 获取本次motion的 IP，LC, RC
        ip_list = np.array(jaw_motion[motion_type]["IP_list"])
        rc_list = np.array(jaw_motion[motion_type]["RC_list"])
        lc_list = np.array(jaw_motion[motion_type]["LC_list"])

        # ********************     2副子图中画轨迹（画点，然后和前面的点连线）   ******************** #
        for i in range(len(ip_list)):
            # *************** figure 1
            # IP点的XY
            fig1_ip = (int(org_fig1[0] + scale * ip_list[i][0]), int(org_fig1[1] - scale * ip_list[i][1]))
            if i > 0:
                fig1_ip_pre = (
                    int(org_fig1[0] + scale * ip_list[i - 1][0]), int(org_fig1[1] - scale * ip_list[i - 1][1]))
                cv2.line(image, fig1_ip_pre, fig1_ip, motion_color, 1, line_type)

            # RC点的XY
            fig1_rc = (int(org_rc_fig1[0] + scale * rc_list[i][0]), int(org_rc_fig1[1] - scale * rc_list[i][1]))
            if i > 0:
                fig1_rc_pre = (
                    int(org_rc_fig1[0] + scale * rc_list[i - 1][0]), int(org_rc_fig1[1] - scale * rc_list[i - 1][1]))
                cv2.line(image, fig1_rc_pre, fig1_rc, motion_color, 1, line_type)

            # LC点的XY
            fig1_lc = (int(org_lc_fig1[0] + scale * lc_list[i][0]), int(org_lc_fig1[1] - scale * lc_list[i][1]))
            if i > 0:
                fig1_lc_pre = (
                    int(org_lc_fig1[0] + scale * lc_list[i - 1][0]), int(org_lc_fig1[1] - scale * lc_list[i - 1][1]))
                cv2.line(image, fig1_lc_pre, fig1_lc, motion_color, 1, line_type)

            # *************** figure 2
            # IP点的XZ
            fig2_ip = (int(org_fig2[0] + scale * ip_list[i][0]), int(org_fig2[1] + scale * ip_list[i][2]))
            if i > 0:
                fig2_ip_pre = (
                    int(org_fig2[0] + scale * ip_list[i - 1][0]), int(org_fig2[1] + scale * ip_list[i - 1][2]))
                cv2.line(image, fig2_ip_pre, fig2_ip, motion_color, 1, line_type)

            # RC点的XZ
            fig2_rc = (int(org_rc_fig2[0] + scale * rc_list[i][0]), int(org_rc_fig2[1] + scale * rc_list[i][2]))
            if i > 0:
                fig2_rc_pre = (
                    int(org_rc_fig2[0] + scale * rc_list[i - 1][0]), int(org_rc_fig2[1] + scale * rc_list[i - 1][2]))
                cv2.line(image, fig2_rc_pre, fig2_rc, motion_color, 1, line_type)

            # LC点的XZ
            fig2_lc = (int(org_lc_fig2[0] + scale * lc_list[i][0]), int(org_lc_fig2[1] + scale * lc_list[i][2]))
            if i > 0:
                fig2_lc_pre = (
                    int(org_lc_fig2[0] + scale * lc_list[i - 1][0]), int(org_lc_fig2[1] + scale * lc_list[i - 1][2]))
                cv2.line(image, fig2_lc_pre, fig2_lc, motion_color, 1, line_type)

    return image


def custom_report_page3(image, case_info, jaw_motion):
    # 绘画的样式
    line_type = cv2.LINE_AA

    #####################################################################
    # -----------     绘制彩色轨迹    ------------ #
    #####################################################################
    # 坐标原点颜色
    color_org = [166, 194, 22]

    # 曲线的颜色
    colors = [[52, 46, 193], [0, 182, 230], [217, 152, 0], [29, 130, 43], [170, 94, 0], [168, 156, 51],
              [48, 59, 255], [211, 159, 0], [0, 122, 255], [255, 99, 167], [126, 177, 217],
              [255, 122, 0], [45, 193, 251], [89, 199, 52], [146, 45, 255], [88, 88, 199],
              [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100],
              [255, 210, 100], [255, 210, 100], [255, 210, 100], [255, 210, 100]]

    # 像素/毫米
    scale = 8

    # ********************     坐标原点     ******************** #

    # 选择不同的参考平面
    matrix = []
    if case_info["reference"] == "frankfurt":
        matrix = np.eye(4)
    elif case_info["reference"] == "ala-tragus":
        matrix = np.array(case_info["Transform_T_F"])

    # *************** figure 1
    # IP
    org_fig1 = (1370, 1035)
    cv2.circle(image, org_fig1, 10, color_org, -1)

    # RC
    rc = case_info["RC"]
    new_rc = (matrix @ np.append(rc, 1))[:-1]
    org_rc_fig1 = (int(org_fig1[0] + scale * new_rc[2]), int(org_fig1[1] - scale * new_rc[1]))  # ZY
    cv2.circle(image, org_rc_fig1, 10, color_org, -1)

    # *************** figure 2
    # IP
    org_fig2 = (730, 2237)
    cv2.circle(image, org_fig2, 10, color_org, -1)

    # LC
    lc = case_info["LC"]
    new_lc = (matrix @ np.append(lc, 1))[:-1]
    org_lc_fig2 = (int(org_fig2[0] - scale * new_lc[2]), int(org_fig2[1] - scale * new_lc[1]))  # ZY
    cv2.circle(image, org_lc_fig2, 10, color_org, -1)

    # ********************      每个motion，循环画轨迹     ******************** #
    for i in range(len(jaw_motion)):
        # motion的类型
        motion_type = list(jaw_motion.keys())[i]

        # motion的颜色
        motion_color = colors[i]

        # 获取本次motion的 IP，LC, RC
        ip_list = np.array(jaw_motion[motion_type]["IP_list"])
        rc_list = np.array(jaw_motion[motion_type]["RC_list"])
        lc_list = np.array(jaw_motion[motion_type]["LC_list"])

        # ********************     2副子图中画轨迹（画点，然后和前面的点连线）    ******************** #
        for i in range(len(ip_list)):
            # *************** figure 1
            # IP点的ZY
            fig1_ip = (int(org_fig1[0] + scale * ip_list[i][2]), int(org_fig1[1] - scale * ip_list[i][1]))
            if i > 0:
                fig1_ip_pre = (
                    int(org_fig1[0] + scale * ip_list[i - 1][2]), int(org_fig1[1] - scale * ip_list[i - 1][1]))
                cv2.line(image, fig1_ip_pre, fig1_ip, motion_color, 1, line_type)

            # RC点的ZY
            fig1_rc = (int(org_rc_fig1[0] + scale * rc_list[i][2]), int(org_rc_fig1[1] - scale * rc_list[i][1]))
            if i > 0:
                fig1_rc_pre = (
                    int(org_rc_fig1[0] + scale * rc_list[i - 1][2]), int(org_rc_fig1[1] - scale * rc_list[i - 1][1]))
                cv2.line(image, fig1_rc_pre, fig1_rc, motion_color, 1, line_type)

            # *************** figure 2
            # IP点的ZY
            fig2_ip = (int(org_fig2[0] - scale * ip_list[i][2]), int(org_fig2[1] - scale * ip_list[i][1]))
            if i > 0:
                fig2_ip_pre = (
                    int(org_fig2[0] - scale * ip_list[i - 1][2]), int(org_fig2[1] - scale * ip_list[i - 1][1]))
                cv2.line(image, fig2_ip_pre, fig2_ip, motion_color, 1, line_type)

            # LC点的ZY
            fig2_lc = (int(org_lc_fig2[0] - scale * lc_list[i][2]), int(org_lc_fig2[1] - scale * lc_list[i][1]))
            if i > 0:
                fig2_lc_pre = (
                    int(org_lc_fig2[0] - scale * lc_list[i - 1][2]), int(org_lc_fig2[1] - scale * lc_list[i - 1][1]))
                cv2.line(image, fig2_lc_pre, fig2_lc, motion_color, 1, line_type)

    return image


def custom_report(uid, cid):
    #####################################################################
    # -----------  读入3张模板图片    ------------ #
    #####################################################################
    source_dir = "./images/"

    tmp1 = source_dir + "custom_page1.png"
    tmp2 = source_dir + "custom_page2.png"
    tmp3 = source_dir + "custom_page3.png"

    img_page1 = cv2.imread(tmp1)
    img_page2 = cv2.imread(tmp2)
    img_page3 = cv2.imread(tmp3)

    # *************** 读病例信息文件
    case_info = get_information_json(uid, cid)

    # *************** 读运动轨迹json，获取运动信息
    _, jaw_motion = get_jaw_motion_json(get_object_prefix(uid, cid) + 'custom', case_info)

    if len(jaw_motion) == 0:
        raise Exception(f"case {cid} has no custom motion")

    #####################################################################
    # -----------    在图片上绘制    ------------ #
    #####################################################################
    img_page1 = custom_report_page1(img_page1, case_info)
    img_page2 = custom_report_page2(img_page2, case_info, jaw_motion)
    img_page3 = custom_report_page3(img_page3, case_info, jaw_motion)

    #####################################################################
    # -----------    图片转成PDF保存，并删除图片    ------------ #
    #####################################################################
    # 新建临时文件夹
    tmp_dir = tempfile.mkdtemp()

    cv2.imwrite(tmp_dir + "/myPage1.png", img_page1)
    cv2.imwrite(tmp_dir + "/myPage2.png", img_page2)
    cv2.imwrite(tmp_dir + "/myPage3.png", img_page3)

    pdf_content = process_img_tmp_dir(tmp_dir)
    pdf_file = get_object_prefix(uid, cid) + 'custom_report.pdf'
    return put_obj(pdf_file, pdf_content)
