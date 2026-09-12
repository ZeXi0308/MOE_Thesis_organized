# Midpoint static-budget probe: PREPARED / GPU UNRUN

4 measured episodes +4 warmups across forward/reverse fresh engines; mixed512 versus mixed1024 only.
3/3 targeted CPU tests passed; both prepare-only plans checked without torch/vllm import.
Runtime, mixed inputs and launcher remain byte-identical to the executed source archive; only runner/test changes appear in midpoint.patch.

Execution archive: 54280 bytes, 12 members.
SHA256: `e3902dcaaed44ee8b92ed6ac303c9d0c93776799ad2f6a00a8005434999ce747`.

Run one engine at a time: `python -u launch_block.py forward`, then after complete readback `python -u launch_block.py reverse`.
No GPU or remote operation was performed during preparation. No new method claim.
