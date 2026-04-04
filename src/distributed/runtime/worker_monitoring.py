from __future__ import annotations

from threading import Lock

from codecarbon import EmissionsTracker


class WorkerEmissionsMonitor:
    def __init__(self) -> None:
        self._lock = Lock()
        self._tracker: EmissionsTracker | None = None
        self._active = False

    def start(self) -> None:
        with self._lock:
            if self._active:
                self._stop_tracker()

            tracker = EmissionsTracker(
                measure_power_secs=1,
                log_level="critical",
            )
            tracker.start()
            self._tracker = tracker
            self._active = True

    def stop(self) -> tuple[float | None, float | None]:
        with self._lock:
            if not self._active or self._tracker is None:
                return None, None

            tracker = self._tracker
            self._stop_tracker()

            emissions_data = tracker._prepare_emissions_data()
            carbon_kg = emissions_data.emissions
            energy_kwh = emissions_data.energy_consumed
            return (
                float(carbon_kg) if carbon_kg is not None else None,
                float(energy_kwh) if energy_kwh is not None else None,
            )

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def _stop_tracker(self) -> None:
        if self._tracker is not None:
            self._tracker.stop()
        self._tracker = None
        self._active = False
