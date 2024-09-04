import asyncio
import functools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tornado.websocket
from tornado import httputil
from tornado.concurrent import run_on_executor
from tornado.ioloop import IOLoop

from bow.algm import ShowingException, stable_step1, stable_step2, stable_step3, r, get_stable_step1_queue, \
    CANCEL_SIGNAL_VALUE, get_stable_step3_queue, clear_stable_related_resource, position, get_position_queue, \
    clear_position_related_resource, motion, get_motion_queue, get_case_status, modify_information, motion_renew, \
    STOP_WS_CODE, MOTION_RUNNING_CASES
from bow.registration import rigid_transformation, write_pose
from bow.report import stable_report, position_report, standard_report, standard_xml, custom_report

ErrMap = {
    "CalibrationHandler": "视频解析失败",
    "ParameterHandler": "确定参数失败",
    "ValidationHandler": "𬌗平面解析失败",
    "FinalReportHandler": "生成报告失败",
    "XMLReportHandler": "导出轨迹数据失败"
}
ErrSuccess = {"code": 200, "message": "成功"}
ErrMissingParam = {"code": 100004, "message": "缺少参数"}
ErrPermissionDenied = {"code": 100409, "message": "没有操作权限"}
ErrResourceOverload = {"code": 100429, "message": "资源超载"}


class BaseHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(20)

    def response(self, code, message):
        self.set_status(code)
        self.write(message)


def async_authenticated(method):
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        uid = self.request.headers.get("Uid", None)
        if uid is None:
            return self.response(401, ErrPermissionDenied)
        self._current_user = uid

        try:
            return await method(self, *args, **kwargs)
        except ShowingException as e:
            self.response(e.status_code, e.content)
        except Exception as e:
            logging.exception(e)
            class_name = self.__class__.__name__
            res_data = {
                "code": 100002,
                "message": ErrMap.get(class_name, "服务器内部错误")
            }
            self.response(500, res_data)

    return wrapper


def authenticated(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        uid = self.request.headers.get("Uid", None)
        if uid is None:
            return self.response(401, ErrPermissionDenied)
        self._current_user = uid
        try:
            return method(self, *args, **kwargs)
        except ShowingException as e:
            self.response(e.status_code, e.content)
        except Exception as e:
            logging.exception(e)
            class_name = self.__class__.__name__
            res_data = {
                "code": 100002,
                "message": ErrMap.get(class_name, "服务器内部错误")
            }
            self.response(500, res_data)

    return wrapper


def ws_authenticated(method):
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        uid = self.request.headers.get("Uid", None)
        if uid is None:
            return self.response(401, ErrPermissionDenied)
        self._current_user = uid

        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logging.exception(e)
            self.response(500, "websocket server error")

    return wrapper


class IllCaseStatusHandler(BaseHandler):
    @authenticated
    def get(self, cid):
        print("?")
        content = get_case_status(self._current_user, cid)
        self.write(json.dumps(content))


class InformationHandler(BaseHandler):
    @authenticated
    def post(self, cid):
        if not self.request.body:
            return self.write(ErrMissingParam)

        entries = json.loads(self.request.body.decode('utf-8'))
        modify_information(self._current_user, cid, entries)
        self.write(ErrSuccess)


class RegistrationHandler(BaseHandler):
    @authenticated
    def post(self, cid):
        post_body = self.request.body
        if not post_body:
            return self.write(ErrMissingParam)

        post_data = json.loads(post_body.decode('utf-8'))

        moving_object_list = post_data.get("moving_object_list")
        moving_list = post_data.get("moving_list")
        fixed_list = post_data.get("fixed_list")

        data = rigid_transformation(moving_object_list, moving_list, fixed_list).tolist()
        self.write({"code": 200, "message": "成功", "data": data})


class WritePoseHandler(BaseHandler):
    @authenticated
    def post(self, cid):
        post_body = self.request.body
        if not post_body:
            return self.write(ErrMissingParam)

        post_data = json.loads(post_body.decode('utf-8'))

        moving_object_list = post_data.get("moving_object_list")
        moving_list = post_data.get("moving_list")
        fixed_list = post_data.get("fixed_list")

        data = write_pose(self._current_user, cid, moving_object_list, moving_list, fixed_list)
        self.write({"code": 200, "message": "成功", "data": data})


class StableConnectHandler(tornado.websocket.WebSocketHandler):
    connections = dict()

    def __init__(
            self,
            application: tornado.web.Application,
            request: httputil.HTTPServerRequest,
            **kwargs: Any
    ):
        super().__init__(application, request, **kwargs)
        self.case_id = None
        self.step = None

    def check_origin(self, origin):
        return True

    @ws_authenticated
    def open(self, cid):
        self.case_id = cid

        step_funcs = {
            'step1': stable_step1,
            'step2': stable_step2,
            'step3': stable_step3,
        }
        step = self.get_argument("step")
        if step not in step_funcs:
            self.write('invalid step')
            self.close()
        logging.info(f"case {cid} step: {step}")
        self.step = step

        logging.info(f"connections1: {StableConnectHandler.connections}")
        if cid in StableConnectHandler.connections:
            StableConnectHandler.connections[cid] = self
        else:
            StableConnectHandler.connections[cid] = self
            logging.info(f"connections2: {StableConnectHandler.connections}")

            IOLoop.current().spawn_callback(step_funcs[step], self._current_user, cid, StableConnectHandler.connections)

    def on_close(self):
        if self.close_code == STOP_WS_CODE:
            return
        IOLoop.current().spawn_callback(self.send_cancel)
        self.close()

    async def send_cancel(self):
        logging.info(f"client actively closes the connection: {self.case_id} in {self.step}")
        # if self.case_id in StableConnectHandler.connections and StableConnectHandler.connections[self.case_id] != self:
        #     return
        if self.step == 'step1':
            await r.rpush(get_stable_step1_queue(self.case_id), CANCEL_SIGNAL_VALUE)
        elif self.step == 'step3':
            await r.rpush(get_stable_step1_queue(self.case_id), CANCEL_SIGNAL_VALUE)

    def on_message(self, message):
        pass


class StableHandler(BaseHandler):
    @async_authenticated
    async def post(self, cid):
        post_data = self.request.body
        if post_data:
            post_data = self.request.body.decode('utf-8')
            post_data = json.loads(post_data)

        step_state = post_data.get("step")
        pic_name = post_data.get("picture_name")

        if step_state == "step1":
            await r.lpush(get_stable_step1_queue(cid), pic_name)
            queue_len = await r.llen(get_stable_step1_queue(cid))
            logging.info(f"case {cid} accept pic: {pic_name}, queue length: {queue_len}")
            self.write({"code": 200})
        elif step_state == "step3":
            await r.lpush(get_stable_step3_queue(cid), pic_name)
            queue_len = await r.llen(get_stable_step3_queue(cid))
            logging.info(f"case {cid} accept pic: {pic_name}, queue length: {queue_len}")
            self.write({"code": 200})
        elif step_state == "clear":
            await clear_stable_related_resource(self._current_user, cid)
            self.write({"code": 200})
        else:
            self.write("bad content")


class PositionConnectHandler(tornado.websocket.WebSocketHandler):
    connections = dict()

    def __init__(
            self,
            application: tornado.web.Application,
            request: httputil.HTTPServerRequest,
            **kwargs: Any
    ):
        super().__init__(application, request, **kwargs)
        self.case_id = None

    def check_origin(self, origin):
        return True

    @ws_authenticated
    def open(self, cid):
        logging.info(f"connections1: {PositionConnectHandler.connections}")
        self.case_id = cid
        if cid in PositionConnectHandler.connections:
            PositionConnectHandler.connections[cid] = self
        else:
            PositionConnectHandler.connections[cid] = self
            logging.info(f"connections2: {PositionConnectHandler.connections}")
            IOLoop.current().spawn_callback(position, self._current_user, cid, PositionConnectHandler.connections)

    def on_close(self):
        if self.close_code == STOP_WS_CODE:
            return
        IOLoop.current().spawn_callback(self.send_cancel)
        self.close()

    async def send_cancel(self):
        logging.info(f"client actively closes the connection: {self.case_id} in position")
        await r.rpush(get_position_queue(self.case_id), json.dumps({
            "picture_name": CANCEL_SIGNAL_VALUE
        }))

    def on_message(self, message):
        pass


class PositionHandler(BaseHandler):
    @async_authenticated
    async def post(self, cid):
        post_data = self.request.body
        if post_data:
            post_data = self.request.body.decode('utf-8')
            post_data = json.loads(post_data)

        # position_id = post_data.get("position_id")
        # position_name = post_data.get("position_name")
        pic_name = post_data.get("picture_name")

        if pic_name == "clear":
            await clear_position_related_resource(self._current_user, cid)
        else:
            position_info = {
                "position_id": post_data.get("position_id"),
                "position_name": post_data.get("position_name"),
                "picture_name": pic_name,
            }
            await r.lpush(get_position_queue(cid), json.dumps(position_info))
            queue_len = await r.llen(get_position_queue(cid))
            logging.info(f"case {cid} accept pic: {pic_name}, queue length: {queue_len}")
        self.write({"code": 200})


class StandardHandler(BaseHandler):
    @async_authenticated
    async def post(self, cid):
        post_body = self.request.body
        if post_body:
            post_data = json.loads(post_body.decode('utf-8'))
            # check post data validity
            if post_data.get("video_type") is None:
                self.write({"code": 400, "message": "缺少 video_type 参数"})
                return

        if cid not in MOTION_RUNNING_CASES:
            IOLoop.current().spawn_callback(motion, self._current_user, cid)
        await r.lpush(get_motion_queue(cid), post_body)

        self.write({"code": 200})


class CustomHandler(BaseHandler):
    @async_authenticated
    async def post(self, cid):
        post_body = self.request.body
        if post_body:
            post_data = self.request.body.decode('utf-8')
            post_data = json.loads(post_data)
            # check post data validity
            need_params = ["custom_id", "custom_name", "video_name"]
            for param in need_params:
                if param not in post_data:
                    self.write({"code": 400, "message": f"缺少 {param} 参数"})
                    return

        if cid not in MOTION_RUNNING_CASES:
            IOLoop.current().spawn_callback(motion, self._current_user, cid)
        await r.lpush(get_motion_queue(cid), post_body)

        self.write({"code": 200})


class MotionRenewHandler(BaseHandler):
    @authenticated
    def post(self, cid):
        motion_renew(self._current_user, cid)
        self.write(ErrSuccess)


class StableReportHandler(BaseHandler):
    @async_authenticated
    async def get(self, cid):
        loop = asyncio.get_event_loop()
        await self._callback(cid, loop)

    @run_on_executor
    def _callback(self, cid, loop):
        asyncio.set_event_loop(loop)
        resp = stable_report(self._current_user, cid)
        self.write(resp)


class PositionReportHandler(BaseHandler):
    @async_authenticated
    async def get(self, cid):
        loop = asyncio.get_event_loop()
        await self._callback(cid, loop)

    @run_on_executor
    def _callback(self, cid, loop):
        asyncio.set_event_loop(loop)
        resp = position_report(self._current_user, cid)
        self.write(resp)


class StandardReportHandler(BaseHandler):
    @async_authenticated
    async def get(self, cid):
        loop = asyncio.get_event_loop()
        await self._callback(cid, loop)

    @run_on_executor
    def _callback(self, cid, loop):
        asyncio.set_event_loop(loop)
        resp = standard_report(self._current_user, cid)
        self.write(resp)


class StandardXMLHandler(BaseHandler):
    @async_authenticated
    async def post(self, cid):
        show_range = {}
        if self.request.body:
            show_range = json.loads(self.request.body.decode('utf-8'))
        loop = asyncio.get_event_loop()
        await self._callback(cid, show_range, loop)

    @run_on_executor
    def _callback(self, cid, show_range, loop):
        asyncio.set_event_loop(loop)
        resp = standard_xml(self._current_user, cid, show_range)
        self.write(resp)


class CustomReportHandler(BaseHandler):
    @async_authenticated
    async def get(self, cid):
        loop = asyncio.get_event_loop()
        await self._callback(cid, loop)

    @run_on_executor
    def _callback(self, cid, loop):
        asyncio.set_event_loop(loop)
        resp = custom_report(self._current_user, cid)
        self.write(resp)
