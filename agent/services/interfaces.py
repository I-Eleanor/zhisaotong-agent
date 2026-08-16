from abc import ABC, abstractmethod


class WeatherService(ABC):
    @abstractmethod
    def get_weather(self, city: str) -> str:
        pass


class LocationService(ABC):
    @abstractmethod
    def get_user_location(self) -> str:
        pass


class UserIdService(ABC):
    @abstractmethod
    def get_user_id(self) -> str:
        pass


class ExternalDataService(ABC):
    @abstractmethod
    def fetch_data(self, user_id: str, month: str) -> str:
        pass


class DeviceStatusService(ABC):
    """设备运行状态查询服务（覆盖率、清洁效率、耗材状态等），供诊断 Agent 使用。"""

    @abstractmethod
    def get_status(self, user_id: str, month: str = "") -> str:
        """查询指定用户的设备状态；month 为空时返回最近一期数据。"""


class DeviceLogService(ABC):
    """设备运行日志查询服务，供诊断 Agent 与日志 MCP Server 使用。"""

    @abstractmethod
    def get_logs(self, user_id: str, days: int = 7) -> str:
        """查询指定用户设备最近 days 天的运行日志。"""
