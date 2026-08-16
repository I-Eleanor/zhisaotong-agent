from agent.services.http_services import (
    HttpExternalDataService,
    HttpLocationService,
    HttpUserIdService,
    HttpWeatherService,
)
from agent.services.interfaces import (
    DeviceLogService,
    DeviceStatusService,
    ExternalDataService,
    LocationService,
    UserIdService,
    WeatherService,
)
from agent.services.mock_services import (
    CsvDeviceStatusService,
    CsvExternalDataService,
    MockDeviceLogService,
    MockLocationService,
    MockUserIdService,
    MockWeatherService,
)
from utils.logger_handler import logger


def create_weather_service(mode: str = "mock", **kwargs) -> WeatherService:
    if mode == "http":
        return HttpWeatherService(api_url=kwargs.get("api_url", ""), api_key=kwargs.get("api_key", ""))
    logger.info({"event": "service_init", "service": "weather", "mode": "mock"})
    return MockWeatherService()


def create_location_service(mode: str = "mock", **kwargs) -> LocationService:
    if mode == "http":
        return HttpLocationService(api_url=kwargs.get("api_url", ""), api_key=kwargs.get("api_key", ""))
    logger.info({"event": "service_init", "service": "location", "mode": "mock"})
    return MockLocationService()


def create_user_id_service(mode: str = "mock", **kwargs) -> UserIdService:
    if mode == "http":
        return HttpUserIdService(api_url=kwargs.get("api_url", ""), api_key=kwargs.get("api_key", ""))
    logger.info({"event": "service_init", "service": "user_id", "mode": "mock"})
    return MockUserIdService()


def create_external_data_service(mode: str = "csv", **kwargs) -> ExternalDataService:
    if mode == "http":
        return HttpExternalDataService(api_url=kwargs.get("api_url", ""), api_key=kwargs.get("api_key", ""))
    logger.info({"event": "service_init", "service": "external_data", "mode": "csv"})
    return CsvExternalDataService()


def create_device_status_service(mode: str = "csv", **kwargs) -> DeviceStatusService:
    """设备运行状态服务；目前基于 CSV 数据，预留 http 扩展。"""
    if mode == "http":
        logger.warning({"event": "service_init", "service": "device_status",
                        "mode": "http", "note": "http 实现待接入，回退 csv"})
    logger.info({"event": "service_init", "service": "device_status", "mode": "csv"})
    return CsvDeviceStatusService()


def create_device_log_service(mode: str = "mock", **kwargs) -> DeviceLogService:
    """设备运行日志服务；mock 模式由用户 ID 派生可复现日志，预留 http 扩展。"""
    if mode == "http":
        logger.warning({"event": "service_init", "service": "device_log",
                        "mode": "http", "note": "http 实现待接入，回退 mock"})
    logger.info({"event": "service_init", "service": "device_log", "mode": "mock"})
    return MockDeviceLogService()
