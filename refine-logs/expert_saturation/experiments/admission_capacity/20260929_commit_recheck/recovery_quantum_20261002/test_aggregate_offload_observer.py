"""One counter/behavior fixture; no native runtime or performance claim."""
import inspect
from types import SimpleNamespace as NS
import unittest
from aggregate_offload_observer import install


class Scheduler:
    def update_connector_output(self,connector_output):
        if getattr(connector_output,'fail',False):raise connector_output.fail
        return connector_output.result


class OffloadingConnector:
    def __init__(self):self.connector_scheduler=Scheduler()


def output(load_bytes,store_bytes):
    return NS(result=object(),kv_connector_worker_meta=NS(completed_jobs={7:1},
        transfer_stats=NS(load=NS(bytes=load_bytes,time=.2,sizes=[load_bytes] if load_bytes else []),
                          store=NS(bytes=store_bytes,time=.3,sizes=[store_bytes] if store_bytes else []))))


class CounterFixture(unittest.TestCase):
    def test_capture_drain_and_native_behavior(self):
        connector=OffloadingConnector();target=connector.connector_scheduler
        signature=inspect.signature(target.update_connector_output)
        data,mark,undo=install(connector)
        self.assertEqual(inspect.signature(target.update_connector_output),signature)
        first=output(0,4096)
        self.assertIs(target.update_connector_output(connector_output=first),first.result)
        mark();second=output(8192,0)
        self.assertIs(target.update_connector_output(second),second.result)
        # An observer extraction failure cannot mask a valid native return.
        malformed=NS(result=object(),kv_connector_worker_meta=object())
        self.assertIs(target.update_connector_output(malformed),malformed.result)
        failure=ValueError('native failure');bad=output(1,1);bad.fail=failure
        with self.assertRaises(ValueError) as caught:target.update_connector_output(bad)
        self.assertIs(caught.exception,failure)
        undo()
        self.assertEqual(data['capture']['store']['bytes'],4096)
        self.assertEqual(data['post_capture_drain']['load']['bytes'],8192)
        self.assertEqual(data['capture_plus_drain']['completed_job_records'],2)
        self.assertEqual(data['capture_plus_drain']['load']['transfer_records'],1)
        self.assertEqual(data['status'],'PARTIAL')
        self.assertNotIn('update_connector_output',vars(target))
        self.assertIs(target.update_connector_output(second),second.result)


if __name__=='__main__':unittest.main()
