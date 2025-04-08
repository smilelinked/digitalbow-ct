import asyncio

import tornado.web
from tornado.log import enable_pretty_logging
from tornado.options import parse_command_line

from bow import handler


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


def make_app():
    settings = {
        "secret_key": "42wqTE23123wffLU94342wgldgFs",
    }

    return tornado.web.Application(
        [
            (r"/", MainHandler),

            # 病例状态
            (r"/case_status/(?P<cid>[0-9]+)", handler.IllCaseStatusHandler),

            # 病例准备
            (r"/app_ready/(?P<cid>[0-9]+)", handler.AppReadyHandler),
            (r"/code_detect/(?P<cid>[0-9]+)", handler.CodeDetectHandler),
            (r"/finish_detect/(?P<cid>[0-9]+)", handler.FinishDetectHandler),
            (r"/information/(?P<cid>[0-9]+)", handler.InformationHandler),

            # 配准
            (r"/registration/(?P<cid>[0-9]+)", handler.RegistrationHandler),
            (r"/pose/(?P<cid>[0-9]+)", handler.WritePoseHandler),

            # 算法功能：5个模块
            (r"/stable_dignose/(?P<cid>[0-9]+)", handler.StableConnectHandler),  # 功能模块1：稳定重复位
            (r"/stable/(?P<cid>[0-9]+)", handler.StableHandler),
            (r"/position_dignose/(?P<cid>[0-9]+)", handler.PositionConnectHandler),  # 功能模块2：髁突位置关系
            (r"/position/(?P<cid>[0-9]+)", handler.PositionHandler),
            (r"/standard/(?P<cid>[0-9]+)", handler.StandardHandler),
            (r"/custom/(?P<cid>[0-9]+)", handler.CustomHandler),
            (r"/motion_renew/(?P<cid>[0-9]+)", handler.MotionRenewHandler),

            # 输出文件报告
            (r"/stable_report/(?P<cid>[0-9]+)", handler.StableReportHandler),
            (r"/position_report/(?P<cid>[0-9]+)", handler.PositionReportHandler),
            (r"/standard_report/(?P<cid>[0-9]+)", handler.StandardReportHandler),
            (r"/standard_xml/(?P<cid>[0-9]+)", handler.StandardXMLHandler),
            (r"/custom_report/(?P<cid>[0-9]+)", handler.CustomReportHandler),
            # (r"/custom_xml/(?P<cid>[0-9]+)", handler.CustomXMLHandler),

            # 更新自定义点轨迹线
            (r"/custom_motion_trajectory/(?P<cid>[0-9]+)", handler.CustomMotionTrajectory),

        ],
        **settings
    )


async def main():
    app = make_app()
    app.listen(8288)
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parse_command_line()
    enable_pretty_logging()

    asyncio.run(main())
