# Credentials intentionally omitted. Commands executed from the isolated remote repo.
python3 -B -m unittest discover -v -s docs/ideas/bcrd/experiments -p 'test_*.py'
python3 -B -m unittest discover -v -s docs/ideas/energy_slo/routeslack/experiments -p 'test_*.py'
python3 docs/ideas/bcrd/experiments/capture_native_routes.py --model llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M --model-key llmjp --model-revision 1d5983076dfc67aee4a77ec06a27027f5bab6055 --dataset builtin --split test --samples 1 --seq-len 16 --decode-steps 2 --dtype bfloat16 --phase decode --offline --output <RUN_DIR>/llmjp/routes.csv
python3 docs/ideas/bcrd/experiments/capture_native_routes.py --model allenai/OLMoE-1B-7B-0924 --model-key olmoe --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --dataset builtin --split test --samples 1 --seq-len 16 --decode-steps 2 --dtype bfloat16 --phase decode --offline --output <RUN_DIR>/olmoe/routes.csv
