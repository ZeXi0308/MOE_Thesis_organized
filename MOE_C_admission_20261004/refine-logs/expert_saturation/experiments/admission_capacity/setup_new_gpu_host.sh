#!/bin/bash
# Bring a fresh AutoDL RTX 5090 host to the state the campaigns expect.
#
# Measured on connect.westc.seetacloud.com, 2026-09-13, pulling the OLMoE
# shards from hf-mirror:
#
#     no acceleration, 1 connection      2.2 MB/s     13 GB in ~98 min
#     /etc/network_turbo, 1 connection   4.4 MB/s     13 GB in ~49 min
#     /etc/network_turbo, 4 connections   10 MB/s     13 GB in ~22 min
#
# The first run of this campaign used the slowest of the three and also wasted
# roughly 19 GB of transfer: the retry loop restarted `hf download` while its
# own workers were still writing, leaving duplicate `.incomplete` blobs behind
# (interface counter read 40.4 GB for a 13 GB model plus ~8 GB of wheels).
#
# Four things this script does differently:
#   1. sources /etc/network_turbo before any HuggingFace traffic
#   2. raises --max-workers to 8
#   3. checks for a complete snapshot before retrying, so a finished download is
#      never restarted, and clears stale locks and partial blobs between tries
#   4. keeps pip on the Tsinghua mirror in a separate shell, because the
#      acceleration proxy makes pip slower (stated in /etc/network_turbo itself)
#
# Usage:  bash setup_new_gpu_host.sh          # runs both jobs in the background
#         tail -f /root/autodl-tmp/install.log /root/autodl-tmp/dl.log

set -u
BASE=/root/autodl-tmp
VENV=$BASE/expert-saturation/vllm-0.26
REV=6d84c48581ece794365f2b8e9cfb043c68ade9c5
MODEL=allenai/OLMoE-1B-7B-0924

mkdir -p "$BASE/expert-saturation" /root/.pip

# pip: Tsinghua. The Aliyun default stalls on the 303 MB vLLM wheel, and the
# academic proxy is explicitly slower for pip, so this stays unaccelerated.
cat > /root/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
retries = 10
EOF

if [ ! -x "$VENV/bin/python" ]; then
  /root/miniconda3/bin/python -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip setuptools wheel
fi

cat > "$BASE/install_vllm.sh" <<EOF
#!/bin/bash
V=$VENV/bin
for i in \$(seq 1 6); do
  echo "=== vllm attempt \$i \$(date +%T) ==="
  \$V/pip install vllm==0.26.0 transformers==5.15.1 && { echo INSTALL_OK; break; }
  sleep 10
done
\$V/python -c "import vllm,torch,transformers;print('VERIFY',vllm.__version__,torch.__version__,transformers.__version__)"
EOF

cat > "$BASE/dl.sh" <<EOF
#!/bin/bash
# Academic acceleration roughly doubles per-connection throughput to hf-mirror.
source /etc/network_turbo 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1          # xet transport returns 401 on this mirror
export HF_HOME=$BASE/hf-cache
SNAP=$BASE/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/$REV

complete() {
  for f in model-00001-of-00003.safetensors model-00002-of-00003.safetensors \\
           model-00003-of-00003.safetensors config.json tokenizer.json; do
    [ -e "\$SNAP/\$f" ] || return 1
  done
  return 0
}

for i in \$(seq 1 40); do
  if complete; then echo DOWNLOAD_OK; break; fi
  echo "=== dl attempt \$i \$(date +%T) \$(du -sh $BASE/hf-cache 2>/dev/null | cut -f1) ==="
  # A previous attempt killed mid-flight leaves locks that make the next one
  # block for ~500 s, and partial blobs that are re-fetched from scratch.
  rm -rf $BASE/hf-cache/hub/.locks
  find $BASE/hf-cache -name '*.incomplete' -delete 2>/dev/null
  /root/miniconda3/bin/hf download $MODEL --revision $REV --max-workers 8
  sleep 3
done
complete && echo DOWNLOAD_OK || echo DOWNLOAD_FAILED
EOF

chmod +x "$BASE/install_vllm.sh" "$BASE/dl.sh"
/root/miniconda3/bin/pip install -q -U "huggingface_hub[cli]"

setsid nohup "$BASE/install_vllm.sh" > "$BASE/install.log" 2>&1 < /dev/null &
setsid nohup "$BASE/dl.sh"           > "$BASE/dl.log"      2>&1 < /dev/null &

echo "started; expect vllm in ~10 min and the 13 GB model in ~20-25 min"
echo "  tail -f $BASE/install.log $BASE/dl.log"
echo
echo "Runtime environment required by every campaign script:"
echo "  export HF_HOME=$BASE/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0"
echo "  (without the sampler flag vLLM 0.26 aborts with 'FlashInfer requires sm75 or higher')"
