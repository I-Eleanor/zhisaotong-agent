import csv
import os
import random
from datetime import datetime, timedelta

from agent.services.interfaces import (
    DeviceLogService,
    DeviceStatusService,
    ExternalDataService,
    LocationService,
    UserIdService,
    WeatherService,
)
from utils.config_handler import agent_conf
from utils.config_validator import validate_before_use
from utils.exceptions import ServiceUnavailableError
from utils.logger_handler import log_safe_text, log_safe_value, logger
from utils.path_tool import get_abs_path


class MockWeatherService(WeatherService):
    def get_weather(self, city: str) -> str:
        logger.info({"event": "mock_weather", "city": log_safe_text(city)})
        return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


class MockLocationService(LocationService):
    def get_user_location(self) -> str:
        location = random.choice(["深圳", "合肥", "杭州"])
        logger.info({"event": "mock_location", "location": location})
        return location


class MockUserIdService(UserIdService):
    def get_user_id(self) -> str:
        user_id = random.choice(["1001", "1002", "1003", "1004", "1005",
                                  "1006", "1007", "1008", "1009", "1010"])
        logger.info({"event": "mock_user_id", "user_id": user_id})
        return user_id


class CsvExternalDataService(ExternalDataService):
    FIELD_MAP = {
        "用户ID": "user_id",
        "特征": "feature",
        "清洁效率": "efficiency",
        "耗材": "consumables",
        "对比": "comparison",
        "时间": "time",
    }

    def __init__(self):
        self._data: dict[str, dict[str, dict[str, str]]] = {}

    def _load_data(self):
        if self._data:
            return

        validate_before_use("external_data")
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapped = {}
                for cn_key, en_key in self.FIELD_MAP.items():
                    mapped[en_key] = row.get(cn_key, "").strip().strip('"')

                user_id = mapped.get("user_id", "")
                time_val = mapped.get("time", "")

                if not user_id or not time_val:
                    logger.warning({
                        "event": "csv_skip_row",
                        "reason": "缺失字段",
                        "row": log_safe_value(row),
                    })
                    continue

                if user_id not in self._data:
                    self._data[user_id] = {}

                self._data[user_id][time_val] = {
                    "特征": mapped.get("feature", ""),
                    "效率": mapped.get("efficiency", ""),
                    "耗材": mapped.get("consumables", ""),
                    "对比": mapped.get("comparison", ""),
                }

        logger.info({"event": "csv_loaded", "user_count": len(self._data)})

    def fetch_data(self, user_id: str, month: str) -> dict[str, str]:
        self._load_data()

        try:
            return self._data[user_id][month]
        except KeyError:
            logger.warning({
                "event": "external_data_not_found",
                "user_id": log_safe_text(user_id),
                "month": log_safe_text(month),
            })
            return {}

    def available_months(self, user_id: str) -> list[str]:
        """返回该用户已有数据的月份列表（升序）。"""
        self._load_data()
        return sorted(self._data.get(user_id, {}).keys())

    def latest_month(self, user_id: str) -> str:
        months = self.available_months(user_id)
        return months[-1] if months else ""


class CsvDeviceStatusService(DeviceStatusService):
    """基于 CSV 数据的设备状态服务，复用 CsvExternalDataService 的数据加载与缓存。"""

    def __init__(self, external_data_service: CsvExternalDataService | None = None):
        self._external = external_data_service or CsvExternalDataService()

    def get_status(self, user_id: str, month: str = "") -> str:
        """查询设备状态；数据不可用时抛 ServiceUnavailableError（不再返回错误字符串，
        避免上层把"未查询到数据"误当成成功的查询结果）。"""
        if not user_id:
            raise ServiceUnavailableError("未提供用户ID，无法查询设备状态")

        target_month = month or self._external.latest_month(user_id)

        if not target_month:
            logger.warning({"event": "device_status_not_found", "user_id": log_safe_text(user_id)})
            raise ServiceUnavailableError(f"未查询到用户{user_id}的设备运行数据")

        record = self._external.fetch_data(user_id, target_month)

        if not record:
            raise ServiceUnavailableError(f"未查询到用户{user_id}在{target_month}的设备运行数据")

        logger.info({
            "event": "device_status_success",
            "user_id": log_safe_text(user_id),
            "month": log_safe_text(target_month),
        })

        lines = [f"用户ID：{user_id}", f"数据月份：{target_month}"]
        for key, value in record.items():
            value = str(value).replace("\\n", "；")
            lines.append(f"{key}：{value}")

        return "\n".join(lines)


class MockDeviceLogService(DeviceLogService):
    """模拟设备运行日志服务。

    基于 CSV 中的设备特征数据派生出结构化运行日志，字段为
    {timestamp, level, event, device_id, message}，用于诊断 Agent 与日志 MCP Server。
    """

    LOG_TEMPLATES = [
        ("INFO", "clean_start", "开始定时清扫任务，模式：{mode}"),
        ("INFO", "clean_finish", "清扫完成，覆盖面积{area}㎡，耗时{minutes}分钟"),
        ("WARNING", "obstacle_stuck", "行进受阻，疑似被{obstacle}缠绕，已自动脱困"),
        ("WARNING", "consumable_low", "耗材提醒：{part}接近更换阈值"),
        ("ERROR", "dock_fail", "回充失败，未检测到充电座红外信号"),
        ("INFO", "dock_success", "已返回充电座，电量{battery}%"),
    ]

    MODES = ["标准清扫", "强力清扫", "静音清扫", "边角清扫"]
    OBSTACLES = ["电线", "地毯流苏", "袜子", "宠物玩具"]
    PARTS = ["主刷", "边刷", "HEPA滤网", "拖布"]

    def __init__(self, external_data_service: CsvExternalDataService | None = None, seed: int = 20260806):
        self._external = external_data_service or CsvExternalDataService()
        self._seed = seed

    def get_logs(self, user_id: str, days: int = 7) -> str:
        if not user_id:
            return "未提供用户ID，无法查询设备日志"

        try:
            days = max(1, min(int(days), 30))
        except (TypeError, ValueError):
            days = 7

        # 以 user_id 派生固定随机种子，保证同一用户多次查询结果稳定、可复现
        rng = random.Random(f"{self._seed}-{user_id}")
        base_date = datetime.now()

        lines = []
        for day_offset in range(days - 1, -1, -1):
            log_date = base_date - timedelta(days=day_offset)
            for _ in range(rng.randint(2, 4)):
                level, event, template = rng.choice(self.LOG_TEMPLATES)
                message = template.format(
                    mode=rng.choice(self.MODES),
                    area=rng.randint(20, 90),
                    minutes=rng.randint(15, 75),
                    obstacle=rng.choice(self.OBSTACLES),
                    part=rng.choice(self.PARTS),
                    battery=rng.randint(80, 100),
                )
                timestamp = log_date.replace(
                    hour=rng.randint(7, 22), minute=rng.randint(0, 59), second=rng.randint(0, 59)
                )
                lines.append(
                    f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {level:<7} | {event:<16} | "
                    f"device-{user_id} | {message}"
                )

        lines.sort()

        logger.info({
            "event": "device_logs_success",
            "user_id": log_safe_text(user_id),
            "days": days,
            "log_count": len(lines),
        })

        header = f"设备 device-{user_id} 最近{days}天运行日志（共{len(lines)}条）："
        return header + "\n" + "\n".join(lines)
