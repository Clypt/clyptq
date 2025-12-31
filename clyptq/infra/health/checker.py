from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime


@dataclass
class HealthCheckResult:
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in self.components
            ],
        }


class HealthChecker:
    def __init__(self):
        self._components: Dict[str, ComponentHealth] = {}
        self._startup_time = datetime.now(timezone.utc)

    def register_component(self, name: str) -> None:
        self._components[name] = ComponentHealth(
            name=name,
            status=HealthStatus.HEALTHY,
            message="Component registered",
            timestamp=datetime.now(timezone.utc),
        )

    def update_component(
        self, name: str, status: HealthStatus, message: str = ""
    ) -> None:
        if name not in self._components:
            self.register_component(name)

        self._components[name] = ComponentHealth(
            name=name,
            status=status,
            message=message,
            timestamp=datetime.now(timezone.utc),
        )

    def check_liveness(self) -> HealthCheckResult:
        timestamp = datetime.now(timezone.utc)
        uptime = (timestamp - self._startup_time).total_seconds()

        component = ComponentHealth(
            name="process",
            status=HealthStatus.HEALTHY,
            message=f"Process alive, uptime: {uptime:.1f}s",
            timestamp=timestamp,
        )

        return HealthCheckResult(
            status=HealthStatus.HEALTHY, components=[component], timestamp=timestamp
        )

    def check_readiness(self) -> HealthCheckResult:
        timestamp = datetime.now(timezone.utc)
        components = list(self._components.values())

        if not components:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                components=[
                    ComponentHealth(
                        name="system",
                        status=HealthStatus.UNHEALTHY,
                        message="No components registered",
                        timestamp=timestamp,
                    )
                ],
                timestamp=timestamp,
            )

        overall_status = HealthStatus.HEALTHY
        for component in components:
            if component.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif component.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED

        return HealthCheckResult(
            status=overall_status, components=components, timestamp=timestamp
        )
