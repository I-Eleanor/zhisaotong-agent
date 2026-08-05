import csv
import os
import random
from utils.logger_handler import logger
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.config_validator import validate_before_use
from agent.services.interfaces import WeatherService, LocationService, UserIdService, ExternalDataService


class MockWeatherService(WeatherService):
    def get_weather(self, city: str) -> str:
        logger.info({"event": "mock_weather", "city": city})
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
        self._data: dict = {}

    def _load_data(self):
        if self._data:
            return

        validate_before_use("external_data")
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapped = {}
                for cn_key, en_key in self.FIELD_MAP.items():
                    mapped[en_key] = row.get(cn_key, "").strip().strip('"')

                user_id = mapped.get("user_id", "")
                time_val = mapped.get("time", "")

                if not user_id or not time_val:
                    logger.warning({"event": "csv_skip_row", "reason": "缺失字段", "row": row})
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

    def fetch_data(self, user_id: str, month: str) -> str:
        self._load_data()

        try:
            return self._data[user_id][month]
        except KeyError:
            logger.warning({"event": "external_data_not_found", "user_id": user_id, "month": month})
            return ""