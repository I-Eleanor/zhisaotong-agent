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