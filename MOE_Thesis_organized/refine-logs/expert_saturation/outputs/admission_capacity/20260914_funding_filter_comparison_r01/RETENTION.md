# Full-raw retention

The six original raw files and original readback archive are retained read-only. Git omits only the large raw JSON and archive; configs, captured decisions, metrics, logs, execution receipts and analysis are retained in the commit.

Durable local bundle: `/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_funding_filter_comparison_r01`. Original remote: `/root/autodl-tmp/moe-funding-filter-comparison-20260914-r01`. The isolated worktree has an additional complete copy. No favorable-only selection and no raw overwrite.

| Raw cell | Bytes | SHA256 |
|---|---:|---|
| funding-block0-most_output | 299562026 | 888ce52a677bd9fb2dbbc902290265e9996a309e3ddd013f51ae6970039c2018 |
| funding-block0-least_progress | 299633199 | 2e8ebf8660c2e63f5fb8be28aa2d2084a06b0f99ffea35fcc84ae89cd43c402b |
| funding-block0-least_feasible | 299716567 | f223b0b70e57f7e242ff560377c68f35f1ffe6f7384225b3eba1ff99211a4c2e |
| funding-block1-least_feasible | 299684386 | 3f2d1c1013262b07d36198a38871e4b3aa5684359d7fd1dbb993d67cc43c2c7a |
| funding-block1-least_progress | 299635195 | c8ebfd4feca819cf46fe1b07ef0887034a86f5652ca087d180dec8eb1a073a6a |
| funding-block1-most_output | 299550930 | d12c5207ed05b40a4cca5b93b11b9636fe1ebfaa68eed44a8662a302f4a4bb41 |

`execution/readback.tar.gz`: 133554053 bytes, SHA256 `7ad278ed5df66db7729338f534cd522c60f76958800ec60fc9ec52b5cf3a8329`. Original staged receipts and UNRUN analysis are preserved alongside COMPLETE execution/recovery receipts.
