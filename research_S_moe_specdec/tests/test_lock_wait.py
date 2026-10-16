"""Two real-process temporary-file lock checks; no GPU/controller interaction."""
import fcntl
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest


WORKER = r'''
import fcntl,json,signal,sys,time
sys.path.insert(0,sys.argv[1])
from lock_wait import acquire_exclusive
with open(sys.argv[2],'a') as lock:
    before=signal.getsignal(signal.SIGALRM)
    print('ready',flush=True)
    start=time.monotonic()
    acquired=acquire_exclusive(lock,float(sys.argv[3]))
    print(json.dumps({'acquired':acquired,'elapsed':time.monotonic()-start,
        'restored':signal.getsignal(signal.SIGALRM)==before,
        'timer_off':signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0)}),flush=True)
    sys.stdin.readline()  # Stay alive so the parent can inspect lock ownership.
    if acquired:fcntl.flock(lock,fcntl.LOCK_UN)
'''


class LockWaitTests(unittest.TestCase):
    def start_waiter(self,path,timeout):
        process=subprocess.Popen([sys.executable,'-u','-c',WORKER,
            str(Path(__file__).resolve().parents[1]/'code'),str(path),str(timeout)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.addCleanup(self.stop_waiter,process)
        self.assertEqual(self.read_line(process).strip(),'ready')
        return process

    @staticmethod
    def stop_waiter(process):
        if process.poll() is None:
            try:
                process.communicate('\n',timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        for stream in (process.stdin,process.stdout,process.stderr):
            if stream and not stream.closed:stream.close()

    def read_line(self,process):
        ready,_,_=select.select([process.stdout],[],[],3)
        self.assertTrue(ready,'lock worker did not respond within 3 seconds')
        line=process.stdout.readline()
        self.assertTrue(line,'lock worker exited without a result')
        return line

    def test_release_wakes_waiter_and_transfers_exclusive_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'temporary.lock'
            with path.open('a') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX)
                process=self.start_waiter(path,5)
                # Give the worker time to enter the blocking system call.
                self.assertEqual(select.select([process.stdout],[],[],.15)[0],[])
                released=time.monotonic()
                fcntl.flock(owner,fcntl.LOCK_UN)
                result=json.loads(self.read_line(process))
                self.assertTrue(result['acquired'])
                self.assertLess(time.monotonic()-released,2)
                self.assertTrue(result['restored'] and result['timer_off'])
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.stop_waiter(process)
                self.assertEqual(process.returncode,0)
                fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_deadline_returns_without_holding_or_later_acquiring_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'temporary.lock'
            with path.open('a') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX)
                process=self.start_waiter(path,.2)
                result=json.loads(self.read_line(process))
                self.assertFalse(result['acquired'])
                self.assertGreaterEqual(result['elapsed'],.18)
                self.assertLess(result['elapsed'],2)
                self.assertTrue(result['restored'] and result['timer_off'])
                self.assertIsNone(process.poll())
                fcntl.flock(owner,fcntl.LOCK_UN)
                with path.open('a') as observer:
                    fcntl.flock(observer,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    self.stop_waiter(process)
                    self.assertEqual(process.returncode,0)


if __name__=='__main__':
    unittest.main()
