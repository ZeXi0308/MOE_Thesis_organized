#!/bin/bash
set -eu
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -m venv /root/autodl-tmp/wisp-runtime-0112-r01/venv
/root/autodl-tmp/wisp-runtime-0112-r01/venv/bin/python -m pip install --no-cache-dir 'vllm==0.11.2' 'transformers==4.57.1'
