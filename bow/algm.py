import asyncio
import copy
import gc
import io
import logging

import numpy as np
import math
import cv2
import os
import requests
from scipy.spatial.transform import Rotation
import pickle
import json
import redis.asyncio as redis
from tornado.iostream import StreamClosedError
from tornado.websocket import WebSocketClosedError

from bow.s3 import get_obj_exception, generate_signed_url, has_obj, put_obj, del_objects_by_prefix, \
    list_objects, list_all_files, move_file
from bow.utils import (calculate_pose_matrix,
                       moving_average,
                       filter_close_points,
                       coincident_point,
                       point_trajectory
                       )

DEFAULT_CALIBRATION_PKL = 'digitalbow/calibration/nova10/calibration.pckl'
CALIBRATION_PKL = 'calibration.pckl'
INFORMATION_JSON = 'information.json'
STABLE_JSON = 'stable.json'
STABLE_VIDEO = 'stable.mp4'
POSITION_JSON = 'position.json'

STABLE_STEP1_QUEUE_SUFFIX = 'step1'
STABLE_STEP3_QUEUE_SUFFIX = 'step3'
CANCEL_SIGNAL_VALUE = 'cancel'
STOP_SIGNAL_VALUE = 'stop'
STOP_WS_CODE = 1001
MOTION_RUNNING_CASES = set()

connection_pool = redis.ConnectionPool(
    host=os.getenv('REDIS_HOST', '121.36.209.14'),
    port=os.getenv('REDIS_PORT', 6379),
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True,
)


async def get_redis_connection():
    """
    获取一个 Redis 连接，该连接从全局连接池中获取，确保多个连接之间互不影响。
    """
    # 使用连接池创建 Redis 实例
    return redis.Redis(connection_pool=connection_pool)


def get_object_prefix(uid, cid):
    return f"doctor/{uid}/digitalbow-ct/{cid}/"


def get_calibration_pkl(uid, cid):
    cali_file = get_object_prefix(uid, cid) + CALIBRATION_PKL
    if not has_obj(cali_file):
        move_file(DEFAULT_CALIBRATION_PKL, cali_file)
    return pickle.loads(get_obj_exception(cali_file).read())


def get_information_json(uid, cid):
    info_file = get_object_prefix(uid, cid) + INFORMATION_JSON
    return json.loads(get_obj_exception(info_file).read())


def put_information_json(uid, cid, data):
    info_file = get_object_prefix(uid, cid) + INFORMATION_JSON
    return put_json(info_file, data)


# json file format:
# {
#     "Transform_LP_F": {},
#     "pic1_point": {"IP": [], "LC": [], "RC": []},
#     "points": {"IP_3D": [], "LC_3D": [], "RC_3D": [], "IP": [], "LC": [], "RC": []},
#     "track": {"frame": 0, "frequency": [], "size": [],
#               "IP_3D": [], "LC_3D": [], "RC_3D": [], "Angle_3D": [],
#               "IP_list": [], "LC_list": [], "RC_list": []},
#     "stable_point": {"IP_3D": [], "IP": []},
# }
def get_stable_json(uid, cid):
    stable_file = get_object_prefix(uid, cid) + STABLE_JSON
    if has_obj(stable_file):
        return json.loads(get_obj_exception(stable_file).read()), False
    else:
        child_dict = {
            "pic1_point": {
                "IP": [], "LC": [], "RC": []
            },
            "points": {
                "IP_3D": [], "LC_3D": [], "RC_3D": [], "IP": [], "LC": [], "RC": []
            },
            "track": {
                "frame": [], "frequency": [], "size": [],
                "IP_3D": [], "LC_3D": [], "RC_3D": [], "Angle_3D": [],
                "IP_list": [], "LC_list": [], "RC_list": []
            },
            "stable_point": {"IP_3D": [], "IP": []},
        }
        return [child_dict for _ in range(2)], True


def put_json(file, data):
    io_buffer = io.BytesIO(json.dumps(data).encode())
    res = io_buffer.getvalue()
    put_obj(file, res)


def put_stable_json(uid, cid, data):
    stable_file = get_object_prefix(uid, cid) + STABLE_JSON
    put_json(stable_file, data)


def get_stable_video(uid, cid):
    video_file = get_object_prefix(uid, cid) + STABLE_VIDEO
    return generate_signed_url(video_file)


def get_position_json(uid, cid):
    position_file = get_object_prefix(uid, cid) + POSITION_JSON
    if has_obj(position_file):
        return json.loads(get_obj_exception(position_file).read()), False
    else:
        child_dict = {
            "pic1_point": {
                "IP": [], "LC": [], "RC": []
            },
            "points": {
                "IP_3D": [], "LC_3D": [], "RC_3D": [], "IP": [], "LC": [], "RC": [], "position_info": []
            },
        }
        return [copy.deepcopy(child_dict) for _ in range(2)], True


def put_position_json(uid, cid, data):
    position_file = get_object_prefix(uid, cid) + POSITION_JSON
    put_json(position_file, data)


async def clear_stable_related_resource_but_json(uid, cid):
    await del_stable_step1_queue(cid)
    await del_stable_step3_queue(cid)
    del_objects_by_prefix(get_object_prefix(uid, cid) + "stableTarget")


async def clear_stable_related_resource(uid, cid):
    await del_stable_step1_queue(cid)
    await del_stable_step3_queue(cid)
    del_objects_by_prefix(get_object_prefix(uid, cid) + "stable")


async def clear_position_related_resource(uid, cid):
    await del_position_queue(cid)
    del_objects_by_prefix(get_object_prefix(uid, cid) + "position")


def get_stable_step1_queue(cid):
    return f"case:{cid}:stable:{STABLE_STEP1_QUEUE_SUFFIX}"


async def del_stable_step1_queue(cid):
    r = await get_redis_connection()
    await r.delete(get_stable_step1_queue(cid))


def get_stable_step3_queue(cid):
    return f"case:{cid}:stable:{STABLE_STEP3_QUEUE_SUFFIX}"


async def del_stable_step3_queue(cid):
    r = await get_redis_connection()
    await r.delete(get_stable_step3_queue(cid))


def get_position_queue(cid):
    return f"case:{cid}:position"


async def del_position_queue(cid):
    r = await get_redis_connection()
    await r.delete(get_position_queue(cid))


def get_motion_queue(cid):
    return f"case:{cid}:motion"


async def del_motion_queue(cid):
    r = await get_redis_connection()
    await r.delete(get_motion_queue(cid))


async def stable_step1(uid, cid, conns):
    camera_matrix, dist_coeffs = get_calibration_pkl(uid, cid)

    case_info = get_information_json(uid, cid)
    ip = case_info["IP"]
    lc = case_info["LC"]
    rc = case_info["RC"]
    reference = case_info["reference"]
    transform_f_up = np.array(case_info["Transform_F_UP"])
    transform_t_f = np.array(case_info["Transform_T_F"])

    stable_data, is_first_pic = get_stable_json(uid, cid)

    r = await get_redis_connection()

    try:
        while True:
            _, pic_name = await r.brpop([get_stable_step1_queue(cid)])
            logging.info(f"[Stable step1]: case {cid} get a new pic named: {pic_name}")
            if pic_name == CANCEL_SIGNAL_VALUE:
                await del_stable_step1_queue(cid)
                ws = conns.get(cid)
                if ws:
                    conns.pop(cid)
                logging.info(f"case {cid} closed by client")
                return
            elif pic_name == STOP_SIGNAL_VALUE:
                queue_len = await r.llen(get_stable_step1_queue(cid))
                if queue_len == 0:
                    await clear_stable_related_resource_but_json(uid, cid)
                    ws = conns.get(cid)
                    if ws:
                        conns.pop(cid)
                        ws.close(STOP_WS_CODE, "close normally")
                        logging.info(f"[Stable step1]: case {cid} close normally with code {STOP_WS_CODE}")
                    return
                else:
                    gc.collect()
                    logging.warning(f"[Stable step1]: case {cid} closes failed because it is not empty")
                    continue

            # 读取静态图片
            resp = arr = None
            try:
                pic_key = get_object_prefix(uid, cid) + pic_name
                pic_resp = generate_signed_url(pic_key)
                # TODO: try aiohttp
                resp = requests.get(pic_resp)
                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(arr, -1)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logging.error(
                    f"[Stable step1]: case {cid} read {pic_name} failed with resp: {resp}; arr.length: {len(arr)}; "
                    f"exception: {e}")
                continue

            # CV识别姿态矩阵
            success, low_stable_matrix, _, _ = calculate_pose_matrix(img_gray, camera_matrix, dist_coeffs)
            if not success:
                logging.warning(f"[Stable step1]: case {cid} calculate_pose_matrix failed with picture name {pic_name}")
                continue

            # 第1张照片，记录了粘歪的情况， 要进行修正
            if is_first_pic:
                # LP - > F 反映下板的粘歪情况:  计算分2步实现， LP - > UP - > F
                transform_f_lp = transform_f_up @ low_stable_matrix
                # 逆运算得 F - > LP
                transform_lp_f = np.linalg.inv(transform_f_lp)
                stable_data[0]["Transform_LP_F"] = transform_lp_f.tolist()

            # 最终的矩阵
            matrix = transform_f_up @ low_stable_matrix @ np.array(stable_data[0]["Transform_LP_F"])

            # 眶耳平面,  F坐标系
            # IP, LC, RC的实际坐标值
            new_ip = np.around((matrix @ np.append(ip, 1))[:-1], 4)
            new_lc = np.around((matrix @ np.append(lc, 1))[:-1], 4)
            new_rc = np.around((matrix @ np.append(rc, 1))[:-1], 4)
            stable_data[0]["points"]["IP_3D"].append(new_ip.tolist())
            stable_data[0]["points"]["LC_3D"].append(new_lc.tolist())
            stable_data[0]["points"]["RC_3D"].append(new_rc.tolist())

            # IP, LC, RC的相对坐标值: 10张照片与第一张照片的差值 （前端绘图以第一张照片为原点）
            if is_first_pic:
                stable_data[0]["pic1_point"]["IP"] = new_ip.tolist()
                stable_data[0]["pic1_point"]["LC"] = new_lc.tolist()
                stable_data[0]["pic1_point"]["RC"] = new_rc.tolist()

            diff_ip = np.around(new_ip - stable_data[0]["pic1_point"]["IP"], 4)
            diff_lc = np.around(new_lc - stable_data[0]["pic1_point"]["LC"], 4)
            diff_rc = np.around(new_rc - stable_data[0]["pic1_point"]["RC"], 4)
            stable_data[0]["points"]["IP"].append(diff_ip.tolist())
            stable_data[0]["points"]["LC"].append(diff_lc.tolist())
            stable_data[0]["points"]["RC"].append(diff_rc.tolist())

            # 鼻翼耳屏线,  T坐标系
            # IP, LC, RC的实际坐标值
            new_ip_tragus = np.around((transform_t_f @ matrix @ np.append(ip, 1))[:-1], 4)
            new_lc_tragus = np.around((transform_t_f @ matrix @ np.append(lc, 1))[:-1], 4)
            new_rc_tragus = np.around((transform_t_f @ matrix @ np.append(rc, 1))[:-1], 4)
            stable_data[1]["points"]["IP_3D"].append(new_ip_tragus.tolist())
            stable_data[1]["points"]["LC_3D"].append(new_lc_tragus.tolist())
            stable_data[1]["points"]["RC_3D"].append(new_rc_tragus.tolist())

            # IP, LC, RC的相对坐标值: 10张照片与第一张照片的差值 （前端绘图以第一张照片为原点）
            if is_first_pic:
                stable_data[1]["pic1_point"]["IP"] = new_ip_tragus.tolist()
                stable_data[1]["pic1_point"]["LC"] = new_lc_tragus.tolist()
                stable_data[1]["pic1_point"]["RC"] = new_rc_tragus.tolist()
                is_first_pic = False

            diff_ip_tragus = np.around(new_ip_tragus - stable_data[1]["pic1_point"]["IP"], 4)
            diff_lc_tragus = np.around(new_lc_tragus - stable_data[1]["pic1_point"]["LC"], 4)
            diff_rc_tragus = np.around(new_rc_tragus - stable_data[1]["pic1_point"]["RC"], 4)
            stable_data[1]["points"]["IP"].append(diff_ip_tragus.tolist())
            stable_data[1]["points"]["LC"].append(diff_lc_tragus.tolist())
            stable_data[1]["points"]["RC"].append(diff_rc_tragus.tolist())

            # 保存数据文件
            put_stable_json(uid, cid, stable_data)

            # 发送数据到前端
            web_ip, web_lc, web_rc = [], [], []
            if reference == "frankfurt":
                web_ip, web_lc, web_rc = diff_ip.tolist(), diff_lc.tolist(), diff_rc.tolist()
            elif reference == "ala-tragus":
                web_ip, web_lc, web_rc = diff_ip_tragus.tolist(), diff_lc_tragus.tolist(), diff_rc_tragus.tolist()

            try:
                content = {
                    "IP": web_ip,
                    "LC": web_lc,
                    "RC": web_rc,
                }
                ws = conns.get(cid)
                if ws:
                    await ws.write_message(json.dumps(content))
                else:
                    await asyncio.sleep(0.01)
            except (StreamClosedError, WebSocketClosedError):
                ws = conns.get(cid)
                if ws and ws.close_code == 1000:
                    conns.pop(cid)
                    break
                conns[cid] = None
            except Exception as e:
                logging.error(f"[Stable step1]: case {cid} unexpected exception: {e}")
                ws = conns.get(cid)
                if ws:
                    conns.pop(cid)
                    ws.write_message(json.dumps({"err_code": 1500}))

    except asyncio.CancelledError:
        pass
    finally:
        logging.info(f"[Stable step1]: case {cid} finish step1...")
        await r.close()


async def stable_step2(uid, cid, conns):
    try:
        camera_matrix, dist_coeffs = get_calibration_pkl(uid, cid)

        case_info = get_information_json(uid, cid)
        ip = case_info["IP"]
        lc = case_info["LC"]
        rc = case_info["RC"]
        reference = case_info["reference"]
        transform_f_up = np.array(case_info["Transform_F_UP"])
        transform_t_f = np.array(case_info["Transform_T_F"])

        stable_data, _ = get_stable_json(uid, cid)

        # 需要存储的数据
        keys = [
            'ip_3d', 'lc_3d', 'rc_3d', 'angle_3d',
            'ip_list', 'lc_list', 'rc_list',
            'ip_3d_tragus', 'lc_3d_tragus', 'rc_3d_tragus', 'angle_3d_tragus',
            'ip_list_tragus', 'lc_list_tragus', 'rc_list_tragus'
        ]
        contents = {key: [] for key in keys}

        effective_frame = 0

        video_url = get_stable_video(uid, cid)
        cap = cv2.VideoCapture(video_url)
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logging.info(f"number of frame: {num_frame}")

        for count in range(num_frame):
            # 获取1帧, BGR彩色图， read函数返回2个值，1个是否成功，1个图像数据
            success, img = cap.read()

            # TODO 异常处理: 读图失败
            if not success:
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            success, low_stable_matrix, _, _ = calculate_pose_matrix(img_gray, camera_matrix, dist_coeffs)

            # TODO 异常处理： 上下板姿态检测失败
            if not success:
                continue

            # 最终的矩阵
            # 参照静态照片第1张。  静态 动态 起始点会有细微差别
            matrix = transform_f_up @ low_stable_matrix @ np.array(stable_data[0]["Transform_LP_F"])

            # 眶耳平面,  F坐标系
            # IP, LC, RC的实际坐标值
            new_ip = (matrix @ np.append(ip, 1))[:-1]
            new_lc = (matrix @ np.append(lc, 1))[:-1]
            new_rc = (matrix @ np.append(rc, 1))[:-1]
            euler = Rotation.from_matrix(matrix[0:3, 0:3]).as_euler('xyz', degrees=True)
            # 保存, 3D显示用
            contents['ip_3d'].append(new_ip)
            contents['lc_3d'].append(new_lc)
            contents['rc_3d'].append(new_rc)
            contents['angle_3d'].append(euler)

            # *************** IP, LC, RC的相对坐标值: 以静态第1张为参考系的运动轨迹
            # 保留4位小数
            diff_ip = new_ip - stable_data[0]["pic1_point"]["IP"]
            diff_lc = new_lc - stable_data[0]["pic1_point"]["LC"]
            diff_rc = new_rc - stable_data[0]["pic1_point"]["RC"]
            # 保存, 2D显示用
            contents['ip_list'].append(diff_ip)
            contents['lc_list'].append(diff_lc)
            contents['rc_list'].append(diff_rc)

            # 鼻翼耳屏线,  T坐标系
            # IP, LC, RC的实际坐标值
            # 保留4位小数
            new_ip_tragus = (transform_t_f @ matrix @ np.append(ip, 1))[:-1]
            new_lc_tragus = (transform_t_f @ matrix @ np.append(lc, 1))[:-1]
            new_rc_tragus = (transform_t_f @ matrix @ np.append(rc, 1))[:-1]
            euler_tragus = transform_t_f[0:3, 0:3] @ np.array(euler)
            # 保存, 3D显示用
            contents['ip_3d_tragus'].append(new_ip_tragus)
            contents['lc_3d_tragus'].append(new_lc_tragus)
            contents['rc_3d_tragus'].append(new_rc_tragus)
            contents['angle_3d_tragus'].append(euler_tragus)

            # *************** IP, LC, RC的相对坐标值: 以静态第1张为参考系的运动轨迹
            # 保留4位小数
            diff_ip_tragus = new_ip_tragus - stable_data[1]["pic1_point"]["IP"]
            diff_lc_tragus = new_lc_tragus - stable_data[1]["pic1_point"]["LC"]
            diff_rc_tragus = new_rc_tragus - stable_data[1]["pic1_point"]["RC"]
            # 保存, 2D显示用
            contents['ip_list_tragus'].append(diff_ip_tragus)
            contents['lc_list_tragus'].append(diff_lc_tragus)
            contents['rc_list_tragus'].append(diff_rc_tragus)

            # 有效帧数自加1
            effective_frame += 1

            # 与前端通信
            web_ip, web_lc, web_rc = [], [], []
            if reference == "frankfurt":
                web_ip, web_lc, web_rc = diff_ip.tolist(), diff_lc.tolist(), diff_rc.tolist()
            elif reference == "ala-tragus":
                web_ip, web_lc, web_rc = diff_ip_tragus.tolist(), diff_lc_tragus.tolist(), diff_rc_tragus.tolist()

            averaged_cord = (web_ip, web_lc, web_rc)
            queue = CircularQueue(10)
            if not queue.is_full():
                queue.push((web_ip, web_lc, web_rc))
            else:
                queue.pop()
                queue.push((web_ip, web_lc, web_rc))

            if queue.is_full():
                averaged_cord = queue.get_averaged()

            try:
                content = {
                    "total": num_frame,
                    "current": effective_frame,
                    "IP": averaged_cord[0],
                    "LC": averaged_cord[1],
                    "RC": averaged_cord[2],
                }
                ws = conns.get(cid)
                if ws:
                    await ws.write_message(json.dumps(content))
                else:
                    await asyncio.sleep(0.01)
            except (StreamClosedError, WebSocketClosedError):
                ws = conns.get(cid)
                if ws and ws.close_code == 1000:
                    conns.pop(cid)
                    break
                conns[cid] = None
            except Exception as e:
                logging.error(f"[Stable step2]: case {cid} unexpected exception: {e}")
                ws = conns.get(cid)
                if ws:
                    conns.pop(cid)
                    ws.write_message(json.dumps({"err_code": 1500}))

        # 全为不合格帧保护
        if len(contents['ip_3d']) < 10:
            ws = conns.get(cid)
            if ws:
                await ws.write_message(json.dumps({"err_code": 1500, "err_message": "视频质量过低，请重做"}))
                conns.pop(cid)
            return

        # 轨迹线平滑处理
        moving_average_contents = {key: moving_average(content) for key, content in contents.items()}

        # 计算稳定点
        # 眶耳平面下
        # 10张静态照片点的重合点
        origin = stable_data[0]["pic1_point"]["IP"]
        static_point = filter_close_points(stable_data[0]["points"]["IP_3D"], origin=origin, radius=2.0)
        cp_ip_static = coincident_point(static_point, 0.1)[1]

        # 动态轨迹线的重合点
        cp_ip_motion = coincident_point(moving_average_contents['ip_3d'], 0.1)[1]

        # 稳定点：静态 与 动态 的中点
        sp_ip = (cp_ip_static + cp_ip_motion) / 2

        # 稳定点做原点，2D数据需要更新
        stable_data[0]["points"]["IP"] = np.around(stable_data[0]["points"]["IP"] - sp_ip, 4).tolist()
        moving_average_contents['ip_list'] = np.around(moving_average_contents['ip_list'] - sp_ip, 4)

        # 鼻翼耳屏线下
        # 10张静态照片点的重合点
        static_point_tragus = filter_close_points(stable_data[1]["points"]["IP_3D"], origin=origin, radius=2.0)
        cp_ip_static_tragus = coincident_point(static_point_tragus, 0.1)[1]

        # 动态轨迹线的重合点
        cp_ip_motion_tragus = coincident_point(moving_average_contents['ip_3d_tragus'], 0.1)[1]

        # 稳定点：静态 与 动态 的中点
        sp_ip_tragus = (cp_ip_static_tragus + cp_ip_motion_tragus) / 2

        # 稳定点做原点，2D数据需要更新
        stable_data[1]["points"]["IP"] = np.around(stable_data[1]["points"]["IP"] - sp_ip_tragus, 4).tolist()
        moving_average_contents['ip_list_tragus'] = np.around(moving_average_contents['ip_list_tragus'] - sp_ip_tragus,
                                                              4)

        # 保存数据
        common_track = {
            "frame": num_frame,
            "frequency": fps,
            "size": effective_frame,
        }
        oe_track = {
            "IP_3D": moving_average_contents['ip_3d'].tolist(),
            "LC_3D": moving_average_contents['lc_3d'].tolist(),
            "RC_3D": moving_average_contents['rc_3d'].tolist(),
            "Angle_3D": moving_average_contents['angle_3d'].tolist(),
            "IP_list": moving_average_contents['ip_list'].tolist(),
            "LC_list": moving_average_contents['lc_list'].tolist(),
            "RC_list": moving_average_contents['rc_list'].tolist(),
        }
        oe_stable_points = {
            "IP_3D": sp_ip.tolist(),
            "IP": (sp_ip - sp_ip).tolist()
        }
        tr_track = {
            "IP_3D": moving_average_contents['ip_3d_tragus'].tolist(),
            "LC_3D": moving_average_contents['lc_3d_tragus'].tolist(),
            "RC_3D": moving_average_contents['rc_3d_tragus'].tolist(),
            "Angle_3D": moving_average_contents['angle_3d_tragus'].tolist(),
            "IP_list": moving_average_contents['ip_list_tragus'].tolist(),
            "LC_list": moving_average_contents['lc_list_tragus'].tolist(),
            "RC_list": moving_average_contents['rc_list_tragus'].tolist(),
        }
        tr_stable_points = {
            "IP_3D": sp_ip_tragus.tolist(),
            "IP": (sp_ip_tragus - sp_ip_tragus).tolist()
        }
        stable_data[0]["track"].update(common_track | oe_track)
        stable_data[0]["stable_point"].update(oe_stable_points)
        stable_data[1]["track"].update(common_track | tr_track)
        stable_data[1]["stable_point"].update(tr_stable_points)

        put_stable_json(uid, cid, stable_data)
        put_obj(get_object_prefix(uid, cid) + "stable.pckl")

        del img, img_gray
        cap.release()
        gc.collect()
    except Exception as e:
        logging.exception(f"[Stable step2]: case {cid} running failed with {e}")
        # close websocket.
        if cid in conns:
            ws = conns.get(cid)
            if ws:
                ws.close(STOP_WS_CODE, "close normally")
            conns.pop(cid)


async def stable_step3(uid, cid, conns):
    camera_matrix, dist_coeffs = get_calibration_pkl(uid, cid)

    case_info = get_information_json(uid, cid)
    ip = case_info["IP"]
    reference = case_info["reference"]
    transform_f_up = np.array(case_info["Transform_F_UP"])
    transform_t_f = np.array(case_info["Transform_T_F"])

    stable_data, is_first_pic = get_stable_json(uid, cid)

    r = await get_redis_connection()

    try:
        while True:
            _, pic_name = await r.brpop([get_stable_step3_queue(cid)])
            logging.info(f"[Stable step3]: case {cid} get a new pic named: {pic_name}")
            if pic_name == CANCEL_SIGNAL_VALUE:
                await del_stable_step3_queue(cid)
                ws = conns.get(cid)
                if ws:
                    conns.pop(cid)
                logging.info(f"case {cid} closed by client")
                return
            elif pic_name == STOP_SIGNAL_VALUE:
                queue_len = await r.llen(get_stable_step3_queue(cid))
                if queue_len == 0:
                    await clear_stable_related_resource_but_json(uid, cid)
                    ws = conns.get(cid)
                    if ws:
                        conns.pop(cid)
                        ws.close(STOP_WS_CODE, "close normally")
                        logging.info(f"[Stable step3]: case {cid} close normally with code {STOP_WS_CODE}")
                else:
                    gc.collect()
                    logging.warning(f"[Stable step3]: case {cid} closes failed because it is not empty")
                    continue

            # 读取静态图片
            resp = arr = None
            try:
                pic_key = get_object_prefix(uid, cid) + pic_name
                pic_resp = generate_signed_url(pic_key)
                # TODO: try aiohttp
                resp = requests.get(pic_resp)
                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(arr, -1)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logging.error(
                    f"[Stable step3]: case {cid} read {pic_name} failed with resp: {resp}; arr.length: {len(arr)}; "
                    f"exception: {e}")
                continue

            # CV识别姿态矩阵
            success, low_stable_matrix, _, _ = calculate_pose_matrix(img_gray, camera_matrix, dist_coeffs)
            if not success:
                logging.warning(f"[Stable step3]: case {cid} calculate_pose_matrix failed with picture name {pic_name}")
                try:
                    content = {
                        "IP": [10, 10, 10],
                        "alarm": False,
                    }
                    ws = conns.get(cid)
                    if ws:
                        await ws.write_message(json.dumps(content))
                    else:
                        await asyncio.sleep(0.01)
                except (StreamClosedError, WebSocketClosedError):
                    ws = conns.get(cid)
                    if ws and ws.close_code == 1000:
                        conns.pop(cid)
                        break
                    conns[cid] = None
                except Exception as e:
                    logging.error(f"[Stable step3]: case {cid} unexpected exception: {e}")
                    ws = conns.get(cid)
                    if ws:
                        conns.pop(cid)
                        ws.write_message(json.dumps({"err_code": 1500}))
                continue

            # 新照片的点和稳定点进行对比，距离门限是0.2mm（根据IP点报警）
            threshold = 0.2

            # 最终的矩阵
            matrix = transform_f_up @ low_stable_matrix @ np.array(stable_data[0]["Transform_LP_F"])

            # 根据选择的参考平面，计算
            projection, point = [], []
            if reference == "frankfurt":
                projection = np.eye(4)
                point = stable_data[0]["stable_point"]["IP_3D"]
            elif reference == "ala-tragus":
                projection = transform_t_f
                point = stable_data[1]["stable_point"]["IP_3D"]

            # IP 的实际坐标值
            new_ip = (projection @ matrix @ np.append(ip, 1))[:-1]

            # 与稳定点对比
            diff_ip = (new_ip - point).tolist()

            # 与稳定点的空间距离
            dist = math.sqrt(pow(diff_ip[0], 2) + pow(diff_ip[1], 2) + pow(diff_ip[2], 2))
            if dist <= threshold:
                alarm = "true"
            else:
                alarm = "false"

            # 发送数据到前端
            try:
                content = {
                    "IP": diff_ip,
                    "alarm": alarm,
                }
                ws = conns.get(cid)
                if ws:
                    await ws.write_message(json.dumps(content))
                else:
                    await asyncio.sleep(0.01)
            except (StreamClosedError, WebSocketClosedError):
                ws = conns.get(cid)
                if ws and ws.close_code == 1000:
                    conns.pop(cid)
                    break
                conns[cid] = None
            except Exception as e:
                logging.error(f"[Stable step3]: case {cid} unexpected exception: {e}")
                ws = conns.get(cid)
                if ws:
                    conns.pop(cid)
                    ws.write_message(json.dumps({"err_code": 1500}))

    except asyncio.CancelledError:
        pass
    finally:
        logging.info(f"[Stable step3]: case {cid} finish step3...")
        await r.close()


async def position(uid, cid, conns):
    camera_matrix, dist_coeffs = get_calibration_pkl(uid, cid)

    case_info = get_information_json(uid, cid)
    ip = case_info["IP"]
    lc = case_info["LC"]
    rc = case_info["RC"]
    reference = case_info["reference"]
    transform_f_up = np.array(case_info["Transform_F_UP"])
    transform_t_f = np.array(case_info["Transform_T_F"])

    position_data, is_first_pic = get_position_json(uid, cid)

    r = await get_redis_connection()

    try:
        while True:
            _, pic_info = await r.brpop([get_position_queue(cid)])
            position_info = json.loads(pic_info)
            logging.info(f"[Position]: case {cid} get a new pic info: {position_info}")

            position_id = position_info.get("position_id")
            position_name = position_info.get("position_name")
            pic_name = position_info.get("picture_name")

            if pic_name == CANCEL_SIGNAL_VALUE:
                await del_position_queue(cid)
                ws = conns.get(cid)
                if ws:
                    conns.pop(cid)
                logging.info(f"case {cid} closed by client")
                return
            elif pic_name == STOP_SIGNAL_VALUE:
                queue_len = await r.llen(get_position_queue(cid))
                if queue_len == 0:
                    await del_position_queue(cid)
                    ws = conns.get(cid)
                    if ws:
                        conns.pop(cid)
                        ws.close(STOP_WS_CODE, "close normally")
                        logging.info(f"[Position]: case {cid} close normally with code {STOP_WS_CODE}")
                else:
                    gc.collect()
                    logging.warning(f"[Position]: case {cid} closes failed because it is not empty")
                    continue

            # 读取静态图片
            resp = arr = None
            try:
                pic_key = get_object_prefix(uid, cid) + pic_name
                pic_resp = generate_signed_url(pic_key)
                # TODO: try aiohttp
                resp = requests.get(pic_resp)
                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(arr, -1)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logging.error(
                    f"[Position]: case {cid} read {pic_name} failed with resp: {resp}; arr.length: {len(arr)}; "
                    f"exception: {e}")
                continue

            # CV识别姿态矩阵
            success, low_stable_matrix, _, _ = calculate_pose_matrix(img_gray, camera_matrix, dist_coeffs)
            if not success:
                logging.warning(f"[Position]: case {cid} calculate_pose_matrix failed with picture name {pic_name}")

            # 第1张照片，记录了粘歪的情况， 要进行修正
            if is_first_pic:
                # LP - > F 反映下板的粘歪情况:  计算分2步实现， LP - > UP - > F
                transform_f_lp = transform_f_up @ low_stable_matrix
                # 逆运算得 F - > LP
                transform_lp_f = np.linalg.inv(transform_f_lp)
                # 存起来
                position_data[0]["Transform_LP_F"] = transform_lp_f.tolist()

            # *************** 最终的矩阵
            matrix = transform_f_up @ low_stable_matrix @ np.array(position_data[0]["Transform_LP_F"])

            # ********************      眶耳平面,  F坐标系     ******************** #

            # *************** IP, LC, RC的实际坐标值
            # 保留4位小数
            new_ip = np.around((matrix @ np.append(ip, 1))[:-1], 4)
            new_lc = np.around((matrix @ np.append(lc, 1))[:-1], 4)
            new_rc = np.around((matrix @ np.append(rc, 1))[:-1], 4)
            # 保存，3D显示用
            position_data[0]["points"]["IP_3D"].append(new_ip.tolist())
            position_data[0]["points"]["LC_3D"].append(new_lc.tolist())
            position_data[0]["points"]["RC_3D"].append(new_rc.tolist())

            # *************** IP, LC, RC的相对坐标值: 照片与第一张照片的差值 （前端绘图以第一张照片为原点）
            if is_first_pic:
                position_data[0]["pic1_point"]["IP"] = new_ip.tolist()
                position_data[0]["pic1_point"]["LC"] = new_lc.tolist()
                position_data[0]["pic1_point"]["RC"] = new_rc.tolist()
            # 保留4位小数
            diff_ip = np.around(new_ip - position_data[0]["pic1_point"]["IP"], 4)
            diff_lc = np.around(new_lc - position_data[0]["pic1_point"]["LC"], 4)
            diff_rc = np.around(new_rc - position_data[0]["pic1_point"]["RC"], 4)
            # 保存, 2D显示用
            position_data[0]["points"]["IP"].append(diff_ip.tolist())
            position_data[0]["points"]["LC"].append(diff_lc.tolist())
            position_data[0]["points"]["RC"].append(diff_rc.tolist())

            # ********************      鼻翼耳屏线,  T坐标系    ******************** #

            # *************** IP, LC, RC的实际坐标值
            # 保留4位小数
            new_ip_tragus = np.around((transform_t_f @ matrix @ np.append(ip, 1))[:-1], 4)
            new_lc_tragus = np.around((transform_t_f @ matrix @ np.append(lc, 1))[:-1], 4)
            new_rc_tragus = np.around((transform_t_f @ matrix @ np.append(rc, 1))[:-1], 4)
            # 保存，3D显示用
            position_data[1]["points"]["IP_3D"].append(new_ip_tragus.tolist())
            position_data[1]["points"]["LC_3D"].append(new_lc_tragus.tolist())
            position_data[1]["points"]["RC_3D"].append(new_rc_tragus.tolist())

            # *************** IP, LC, RC的相对坐标值: 照片与第一张照片的差值 （前端绘图以第一张照片为原点）
            if is_first_pic:
                position_data[1]["pic1_point"]["IP"] = new_ip_tragus.tolist()
                position_data[1]["pic1_point"]["LC"] = new_lc_tragus.tolist()
                position_data[1]["pic1_point"]["RC"] = new_rc_tragus.tolist()
                is_first_pic = False
            # 保留4位小数
            diff_ip_tragus = np.around(new_ip_tragus - position_data[1]["pic1_point"]["IP"], 4)
            diff_lc_tragus = np.around(new_lc_tragus - position_data[1]["pic1_point"]["LC"], 4)
            diff_rc_tragus = np.around(new_rc_tragus - position_data[1]["pic1_point"]["RC"], 4)
            # 保存, 2D显示用
            position_data[1]["points"]["IP"].append(diff_ip_tragus.tolist())
            position_data[1]["points"]["LC"].append(diff_lc_tragus.tolist())
            position_data[1]["points"]["RC"].append(diff_rc_tragus.tolist())

            # 名称
            position_settings = [
                {"position_id": "0", "position_name": "参考位置"},
                {"position_id": "1", "position_name": "牙尖交错位"},
                {"position_id": "2", "position_name": "后退接触位"},
                {"position_id": "3", "position_name": "下颌姿势位"},
            ]
            num = len(position_data[0]["points"]["IP"]) - 1
            position_data[0]["points"]["position_info"].append(position_settings[num])
            position_data[1]["points"]["position_info"].append(position_settings[num])

            put_position_json(uid, cid, position_data)

            # 发送数据到前端
            web_ip, web_lc, web_rc = [], [], []
            if reference == "frankfurt":
                web_ip, web_lc, web_rc = diff_ip.tolist(), diff_lc.tolist(), diff_rc.tolist()
            elif reference == "ala-tragus":
                web_ip, web_lc, web_rc = diff_ip_tragus.tolist(), diff_lc_tragus.tolist(), diff_rc_tragus.tolist()

            try:
                content = {
                    "IP": web_ip,
                    "LC": web_lc,
                    "RC": web_rc,
                    "position_name": position_name,
                    "position_id": position_id,
                }
                ws = conns.get(cid)
                if ws:
                    await ws.write_message(json.dumps(content))
                else:
                    await asyncio.sleep(0.01)
            except (StreamClosedError, WebSocketClosedError):
                ws = conns.get(cid)
                if ws and ws.close_code == 1000:
                    conns.pop(cid)
                    break
                conns[cid] = None

            except Exception as e:
                logging.error(f"[Position]: case {cid} unexpected exception: {e}")
                ws = conns.get(cid)
                if ws:
                    conns.pop(cid)
                    ws.write_message(json.dumps({"err_code": 1500}))

    except asyncio.CancelledError:
        pass
    finally:
        logging.info(f"[Position]: case {cid} finish position...")
        await r.close()


# CV识别运动
async def motion(uid, cid):
    camera_matrix, dist_coeffs = get_calibration_pkl(uid, cid)

    case_info = get_information_json(uid, cid)

    r = await get_redis_connection()

    try:
        while True:
            video_element = await r.brpop([get_motion_queue(cid)], timeout=10)
            if video_element is None:
                logging.warning(f"[Motion]: case {cid} did not get a video in 10s, breaking...")
                break

            MOTION_RUNNING_CASES.add(cid)

            video_info = json.loads(video_element[1])
            logging.info(f"[Motion]: case {cid} get a new video info: {video_info}")
            video_type = video_info.get("video_type") or video_info.get("video_name").removesuffix(".mp4")
            if video_type == STOP_SIGNAL_VALUE:
                queue_len = await r.llen(get_motion_queue(cid))
                if queue_len == 0:
                    break
                else:
                    gc.collect()
                    logging.warning(f"[Motion]: case {cid} closes failed because it is not empty")
                    continue

            # 读视频文件
            object_key = f"{get_object_prefix(uid, cid)}{video_type}.mp4"
            if not has_obj(object_key):
                logging.warning(f"[Motion]: case {cid} read file {object_key} failed: NOT FOUND!!!")
                continue
            video_resp = generate_signed_url(object_key)
            cap = cv2.VideoCapture(video_resp)
            num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # 读取数据
            transform_f_up = np.array(case_info["Transform_F_UP"])
            transform_t_f = np.array(case_info["Transform_T_F"])

            # 需要存储的数据
            matrix_list = []
            matrix_list_tragus = []

            # 临时变量
            effective_frame = 0
            transform_lp_f = []

            # 读视频，循环处理每帧
            for count in range(num_frame):
                success, img = cap.read()  # 获取1帧, BGR彩色图， read函数返回2个值，1个是否成功，1个图像数据

                # TODO 异常处理: 读图失败
                if not success:
                    continue

                # 转灰度图
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # CV识别姿态矩阵
                success, low_stable_matrix, _, _ = calculate_pose_matrix(img_gray, camera_matrix, dist_coeffs)

                # TODO 异常处理: 上下标记板检测失败
                if not success:
                    continue

                # *************** 下板粘歪处理
                # 第1帧很关键，记录粘歪的情况
                if effective_frame == 0:
                    # LP - > F 反映下板的粘歪情况:  计算分2步实现， LP - > UP - > F
                    transform_f_lp = transform_f_up @ low_stable_matrix
                    # 逆运算得 F - > LP
                    transform_lp_f = np.linalg.inv(transform_f_lp)

                # *************** 最终的矩阵
                # 上下板粘歪的情况修正： Transform_F_UP 修正上板的粘歪，Transform_LP_F 修正下板的粘歪
                matrix = transform_f_up @ low_stable_matrix @ transform_lp_f

                # ********************     眶耳平面,  F坐标系    ******************** #
                # 把每一帧运动的矩阵存起来
                matrix_list.append(matrix.tolist())

                # ********************     鼻翼耳屏线,  T坐标系    ******************** #
                # 把每一帧运动的矩阵存起来
                matrix_tragus = transform_t_f @ matrix
                matrix_list_tragus.append(matrix_tragus.tolist())

                # 有效帧数自加1
                effective_frame += 1

                await asyncio.sleep(0.05)

            # 数据打包
            track_data = [
                # 眶耳平面下
                {
                    "frame": num_frame,
                    "frequency": fps,
                    "size": effective_frame,
                    "Matrix_list": matrix_list,
                },
                # 鼻翼耳屏线下
                {
                    "frame": num_frame,
                    "frequency": fps,
                    "size": effective_frame,
                    "Matrix_list": matrix_list_tragus,
                }
            ]

            if not has_obj(object_key):
                logging.info(f"[Motion]: {cid} delete video {video_type}, so do not put track to obs")
                continue
            track_file = f"{get_object_prefix(uid, cid)}{video_type}.track"
            put_json(track_file, track_data)
            logging.info(f"[Motion]: put {video_type} track to obs success...")

            motion_trajectory(uid, cid, video_type, case_info, track_data)

            # 释放内存
            del img, img_gray
            cap.release()
            gc.collect()

    except asyncio.CancelledError:
        pass
    finally:
        logging.info(f"[Motion]: case {cid} finish recognition...")
        MOTION_RUNNING_CASES.remove(cid)
        await r.close()


# 计算轨迹线
def motion_trajectory(uid, cid, video_type, case_info, track_data=None):
    if track_data is None:
        track_file = f"{get_object_prefix(uid, cid)}{video_type}.track"
        track_data = json.loads(get_obj_exception(track_file).read())

    ip = case_info["IP"]
    lc = case_info["LC"]
    rc = case_info["RC"]

    matrix_list = track_data[0]["Matrix_list"]
    matrix_list_tragus = track_data[1]["Matrix_list"]

    # ********************     眶耳平面,  F坐标系    ******************** #
    angle_3d = point_trajectory(matrix_list)
    ip_3d, ip_list = point_trajectory(matrix_list, ip)
    lc_3d, lc_list = point_trajectory(matrix_list, lc)
    rc_3d, rc_list = point_trajectory(matrix_list, rc)

    # ********************     鼻翼耳屏线,  T坐标系    ******************** #
    angle_3d_tragus = point_trajectory(matrix_list_tragus)
    ip_3d_tragus, ip_list_tragus = point_trajectory(matrix_list_tragus, ip)
    lc_3d_tragus, lc_list_tragus = point_trajectory(matrix_list_tragus, lc)
    rc_3d_tragus, rc_list_tragus = point_trajectory(matrix_list_tragus, rc)

    #####################################################################
    # -----------   保存数据文件    ------------ #
    #####################################################################
    # 眶耳平面下
    track_data[0]["Angle_3D"] = angle_3d.tolist()  # 必须tolist,不然json存不了（json存不了二维数组）
    track_data[0]["IP_3D"] = ip_3d.tolist()
    track_data[0]["LC_3D"] = lc_3d.tolist()
    track_data[0]["RC_3D"] = rc_3d.tolist()
    track_data[0]["IP_list"] = ip_list.tolist()
    track_data[0]["LC_list"] = lc_list.tolist()
    track_data[0]["RC_list"] = rc_list.tolist()

    # 鼻翼耳屏线下
    track_data[1]["Angle_3D"] = angle_3d_tragus.tolist()
    track_data[1]["IP_3D"] = ip_3d_tragus.tolist()
    track_data[1]["LC_3D"] = lc_3d_tragus.tolist()
    track_data[1]["RC_3D"] = rc_3d_tragus.tolist()
    track_data[1]["IP_list"] = ip_list_tragus.tolist()
    track_data[1]["LC_list"] = lc_list_tragus.tolist()
    track_data[1]["RC_list"] = rc_list_tragus.tolist()

    video_file = f"{get_object_prefix(uid, cid)}{video_type}.mp4"
    json_file = f"{get_object_prefix(uid, cid)}{video_type}.json"
    if not has_obj(video_file):
        logging.info(f"[Motion]: {cid} delete video {video_type}, so do not put json to obs")
        return
    put_json(json_file, track_data)
    logging.info(f"[Motion]: put {video_type} json to obs success...")

    gc.collect()


# 更新轨迹线
def motion_renew(uid, cid):
    case_info = get_information_json(uid, cid)
    for t in ("standard", "custom"):
        resp = list_objects(get_object_prefix(uid, cid) + t)
        if resp.get('ResponseMetadata').get('HTTPStatusCode') > 300 or resp.get('Contents') is None:
            return

        for content in resp.get('Contents'):
            key = content.get('Key')
            if key.endswith(".mp4"):
                motion_type = key.split("/")[-1].split(".")[0]
                logging.info(f"[Motion]: {cid} renew {motion_type}")
                motion_trajectory(uid, cid, motion_type, case_info)


def get_case_status(uid, cid):
    file_mapping = {
        "calibration.pckl": "calibration_pckl_exist",
        "parameter.pckl": "parameter_pckl_exist",
        "information.json": "information_json_exist",
        "validation.pckl": "validation_pckl_exist",
        "calibration.mp4": "calibration_video_exist",
        "validation.jpg": "validation_image_exist",
        "stable.json": "stable_json_exist",
        "position.json": "position_json_exist",
        "stable.mp4": "stable_video_exist",
        "standard_left.json": "left_json_exist",
        "standard_maxleft.json": "maxleft_json_exist",
        "standard_right.json": "right_json_exist",
        "standard_maxright.json": "maxright_json_exist",
        "standard_forward.json": "forward_json_exist",
        "standard_forwardmax.json": "forwardmax_json_exist",
        "standard_chew.json": "chew_json_exist",
        "standard_maxopen.json": "maxopen_json_exist",
        "standard_backward.json": "backward_json_exist",
        "standard_left.mp4": "left_video_exist",
        "standard_maxleft.mp4": "maxleft_video_exist",
        "standard_right.mp4": "right_video_exist",
        "standard_maxright.mp4": "maxright_video_exist",
        "standard_forward.mp4": "forward_video_exist",
        "standard_forwardmax.mp4": "forwardmax_video_exist",
        "standard_chew.mp4": "chew_video_exist",
        "standard_maxopen.mp4": "maxopen_video_exist",
        "standard_backward.mp4": "backward_video_exist",
    }

    content = {value: False for value in file_mapping.values()}

    object_prefix = get_object_prefix(uid, cid)
    items = list_all_files(object_prefix)
    for item in items:
        relative_key = item.replace(object_prefix, "")
        if not relative_key.startswith('models') and relative_key in file_mapping:
            content[file_mapping[relative_key]] = True

    return content


def app_ready(uid, cid):
    put_obj(get_object_prefix(uid, cid) + "app_ready.pckl")


def code_detect(uid, cid, module, need_valid_count=6):
    image_prefix = f"{get_object_prefix(uid, cid)}detect/{module}"
    try:
        pic_resp = generate_signed_url(image_prefix + '.jpg')
        # TODO: try aiohttp
        resp = requests.get(pic_resp)
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.exception(f"get detect image failed with resp: {e}")
        raise ShowingException('读取图片失败')

    # 使用 detectMarkers 检测标记
    corners, ids, rejected = cv2.aruco.detectMarkers(
        frame,
        cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11),
        parameters=cv2.aruco.DetectorParameters_create()
    )

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

    green_count = 0
    yellow_count = 0

    if ids is not None:
        for i in range(len(ids)):
            orientation = determine_tag_orientation(corners[i])
            # cX, cY = tuple(corners[i][0].mean(axis=0).astype(int))
            # Convert corners[i] to the required shape
            corner_points = np.int32(corners[i]).reshape((-1, 1, 2))
            if orientation == 'Inverted':
                cv2.polylines(frame, [corner_points], True, (0, 255, 255), 6)  # 黄色边框
                yellow_count += 1
            else:
                cv2.polylines(frame, [corner_points], True, (0, 255, 0), 6)  # 绿色边框
                green_count += 1

    resp_data = {
            'green_count': green_count,
            'yellow_count': yellow_count,
        }
    if green_count == need_valid_count:
        return True, resp_data

    ret, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    put_obj(image_prefix + '_result.jpg', image_bytes, 'image/jpeg')
    return False, resp_data


def finish_detect(uid, cid):
    del_objects_by_prefix(get_object_prefix(uid, cid) + 'detect')


def modify_information(uid, cid, entries):
    information = get_information_json(uid, cid)

    # 判断传入的键是否合法
    allowed_keys = set(information.keys())
    actual_keys = set(entries.keys())
    if not actual_keys.issubset(allowed_keys):
        raise Exception(f"allowed modified keys are {allowed_keys} but actual keys are {actual_keys}")

    information.update(entries)

    return put_information_json(uid, cid, information)


class CircularQueue(object):
    def __init__(self, size):
        self.size = size  # 定义队列长度
        self.queue = []  # 存储队列 列表

    def __str__(self):
        # 返回对象的字符串表达式，方便查看
        return str(self.queue)

    def push(self, n):
        # 入队
        if self.is_full():
            return -1
        self.queue.append(n)  # 列表末尾添加新的对象

    def pop(self):
        # 出队
        if self.is_empty():
            return -1
        first_element = self.queue[0]  # 删除队头元素
        self.queue.remove(first_element)  # 删除队操作
        return first_element

    def delete(self, n):
        # 删除某元素
        element = self.queue[n]
        self.queue.remove(element)

    def set(self, n, m):
        # 插入某元素 n代表列表当前的第n位元素 m代表传入的值
        self.queue[n] = m

    def size(self):
        # 获取当前长度
        return len(self.queue)

    def get(self, n):
        # 获取某个元素
        element = self.queue[n]
        return element

    def is_empty(self):
        # 判断是否为空
        if len(self.queue) == 0:
            return True
        return False

    def is_full(self):
        # 判断队列是否满
        if len(self.queue) == self.size:
            return True
        return False

    def get_averaged(self):
        sum0 = np.array([0, 0, 0])
        sum1 = np.array([0, 0, 0])
        sum2 = np.array([0, 0, 0])
        num = len(self.queue)
        for item in self.queue:
            sum0 = sum0 + np.array(item[0])
            sum1 = sum1 + np.array(item[1])
            sum2 = sum2 + np.array(item[2])

        result = (sum0 / num, sum1 / num, sum2 / num)
        return result[0].tolist(), result[1].tolist(), result[2].tolist()


class ShowingException(Exception):
    """显示异常
    :param status_code: 错误代码
    :param content: 错误信息
    """

    def __init__(self, content, status_code=500):
        self.status_code = status_code
        self.content = content
        super().__init__(f'{self.status_code} - {self.content}')
