"""Optional CPU wall-clock boundary around native transfer startup.

This includes native startup work for deferred STORE as well as LOAD. It is
neither CUDA copy time nor a complete connector/model phase decomposition.
The caller labels service steps and drain; no CUDA call or scheduler change.
"""


class NativeTransferStartProbe:
    def __init__(self, connector, clock, enabled=False):
        self.connector, self.clock, self.enabled = connector, clock, enabled
        self.records = []
        self.phase, self.step_index = 'outside_service', None

    def service_step(self, index):
        self.phase, self.step_index = 'service', index

    def drain(self):
        self.phase, self.step_index = 'drain', None

    def __enter__(self):
        if not self.enabled:
            return self
        self.had_instance_method = 'start_load_kv' in self.connector.__dict__
        self.saved_instance_method = self.connector.__dict__.get('start_load_kv')
        self.original = self.connector.start_load_kv

        def observed(*args, **kwargs):
            phase, step = self.phase, self.step_index
            returned = False
            start = self.clock()
            try:
                result = self.original(*args, **kwargs)
                returned = True
                return result
            finally:
                end = self.clock()
                self.records.append(dict(phase=phase, step_index=step,
                    start_s=start, end_s=end, returned_normally=returned))

        self.connector.start_load_kv = observed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            if self.had_instance_method:
                self.connector.start_load_kv = self.saved_instance_method
            else:
                del self.connector.start_load_kv
        return False
