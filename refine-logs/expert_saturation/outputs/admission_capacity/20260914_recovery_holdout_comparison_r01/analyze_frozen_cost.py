"""Reuse the frozen T actual-schedule cost check on eight holdout runs."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'20260914_restore_token_reservation_r01/analyze_frozen_cost.py'
EXPECTED = 'a367cab6edcbfdf417c50c553956185895dd0e1e6b01d34bbbf0e76dc9fb9b46'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
text = SOURCE.read_text()
changes = {
    "len(primary['cells']) == 6": "len(primary['cells']) == 8",
    "independently executed T schedules": "independently executed holdout schedules",
    "for a,b in ((1,2),(0,2),(4,3),(5,3)):":
        "for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3),"
        "(7,6),(7,5),(7,4),(6,5),(6,4),(5,4)):",
}
for old, new in changes.items():
    assert text.count(old) == 1
    text = text.replace(old, new)
namespace = dict(__name__='frozen_cost_source', __file__=str(SOURCE))
exec(compile(text, str(SOURCE)+':eight-cell-adapter', 'exec'), namespace)
namespace['ROOT'] = ROOT
if __name__ == '__main__':
    namespace['main']()
