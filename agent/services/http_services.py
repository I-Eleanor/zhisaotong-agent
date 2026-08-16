import requests

from agent.services.interfaces import ExternalDataService, LocationService, UserIdService, WeatherService
from utils.logger_handler import logger


class HttpWeatherService(WeatherService):
    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url
        self.api_key = api_key

    def get_weather(self, city: str) -> str:
        logger.info({"event": "http_weather_request", "city": city})
        try:
            params = {"city": city}
            if self.api_key:
                params["key"] = self.api_key

            resp = requests.get(self.api_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            weather = data.get("weather", "未知")
            temp = data.get("temp", "未知")
            humidity = data.get("humidity", "未知")
            wind = data.get("wind", "未知")
            aqi = data.get("aqi", "未知")
            rain_prob = data.get("rain_prob", "未知")

            result = f"城市{city}天气为{weather}，气温{temp}摄氏度，空气湿度{humidity}%，{wind}，AQI{aqi}，最近6小时降雨概率{rain_prob}"
            logger.info({"event": "http_weather_success", "city": city})
            return result

        except requests.RequestException as e:
            logger.error({"event": "http_weather_error", "city": city, "error": str(e)})
            return f"获取{city}天气信息失败，请稍后重试"


class HttpLocationService(LocationService):
    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url
        self.api_key = api_key

    def get_user_location(self) -> str:
        logger.info({"event": "http_location_request"})
        try:
            params = {}
            if self.api_key:
                params["key"] = self.api_key

            resp = requests.get(self.api_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            location = data.get("city", "未知")
            logger.info({"event": "http_location_success", "location": location})
            return location

        except requests.RequestException as e:
            logger.error({"event": "http_location_error", "error": str(e)})
            return "获取用户位置失败，请稍后重试"


class HttpUserIdService(UserIdService):
    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url
        self.api_key = api_key

    def get_user_id(self) -> str:
        logger.info({"event": "http_user_id_request"})
        try:
            params = {}
            if self.api_key:
                params["key"] = self.api_key

            resp = requests.get(self.api_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            user_id = data.get("user_id", "")
            logger.info({"event": "http_user_id_success", "user_id": user_id})
            return user_id

        except requests.RequestException as e:
            logger.error({"event": "http_user_id_error", "error": str(e)})
            return "获取用户ID失败，请稍后重试"


class HttpExternalDataService(ExternalDataService):
    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url
        self.api_key = api_key

    def fetch_data(self, user_id: str, month: str) -> str:
        logger.info({"event": "http_external_data_request", "user_id": user_id, "month": month})
        try:
            params = {"user_id": user_id, "month": month}
            if self.api_key:
                params["key"] = self.api_key

            resp = requests.get(self.api_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                logger.warning({"event": "http_external_data_empty", "user_id": user_id, "month": month})
                return ""

            logger.info({"event": "http_external_data_success", "user_id": user_id, "month": month})
            return str(data)

        except requests.RequestException as e:
            logger.error({"event": "http_external_data_error", "user_id": user_id, "month": month, "error": str(e)})
            return "获取外部数据失败，请稍后重试"
