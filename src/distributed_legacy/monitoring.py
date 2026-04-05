from threading import Lock


class WorkerEmissionsMonitor(object):
    def __init__(self):
        self._lock = Lock()
        self._tracker = None
        self._active = False
        self._import_error = None

    def start(self):
        with self._lock:
            if self._active:
                self._stop_tracker()

            try:
                from codecarbon import EmissionsTracker
            except Exception as exc:
                self._import_error = exc
                self._tracker = None
                self._active = False
                return

            tracker = EmissionsTracker(
                measure_power_secs=1,
                log_level="critical",
            )
            tracker.start()
            self._tracker = tracker
            self._active = True
            self._import_error = None

    def stop(self):
        with self._lock:
            if not self._active or self._tracker is None:
                return None, None

            tracker = self._tracker
            self._stop_tracker()

            emissions_data = tracker._prepare_emissions_data()
            carbon_kg = emissions_data.emissions
            energy_kwh = emissions_data.energy_consumed

            return carbon_kg, energy_kwh

    def is_active(self):
        with self._lock:
            return self._active

    def _stop_tracker(self):
        if self._tracker is not None:
            self._tracker.stop()
        self._tracker = None
        self._active = False
