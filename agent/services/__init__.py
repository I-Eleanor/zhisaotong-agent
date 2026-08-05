from agent.services.interfaces import WeatherService, LocationService, UserIdService, ExternalDataService
from agent.services.mock_services import MockWeatherService, MockLocationService, MockUserIdService, CsvExternalDataService
from agent.services.http_services import HttpWeatherService, HttpLocationService, HttpUserIdService, HttpExternalDataService
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