import os
from abc import ABC, abstractmethod
from typing import Dict, Optional


class SecretsManager(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def get_dict(self, key: str) -> Optional[Dict[str, str]]:
        pass

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        pass


class EnvSecretsManager(SecretsManager):
    def __init__(self, prefix: str = "CLYPTQ_"):
        self.prefix = prefix
        self._cache: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        env_key = f"{self.prefix}{key.upper()}"

        if env_key in self._cache:
            return self._cache[env_key]

        value = os.getenv(env_key)

        if value:
            self._cache[env_key] = value

        return value

    def get_dict(self, key: str) -> Optional[Dict[str, str]]:
        base_key = f"{self.prefix}{key.upper()}"
        result = {}

        for env_key, env_value in os.environ.items():
            if env_key.startswith(base_key):
                suffix = env_key[len(base_key) + 1 :]
                if suffix:
                    result[suffix.lower()] = env_value

        return result if result else None

    def set(self, key: str, value: str) -> None:
        env_key = f"{self.prefix}{key.upper()}"
        os.environ[env_key] = value
        self._cache[env_key] = value

    def clear_cache(self) -> None:
        self._cache.clear()
