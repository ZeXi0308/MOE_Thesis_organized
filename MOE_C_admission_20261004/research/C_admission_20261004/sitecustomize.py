"""Install the C-only import repair in Python children as well as the runner.

launch.sh puts this directory on inherited PYTHONPATH. Python imports this module
before vLLM's registry-inspection subprocess can import the stale Torch kernels.
"""
from runtime_bootstrap import apply

apply()
