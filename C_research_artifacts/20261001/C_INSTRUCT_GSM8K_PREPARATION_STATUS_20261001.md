# Instruct preparation, fixed16 and native128 COMPLETE — 2026-10-01T15:12Z

All13,838,721,960weight bytes are verified on the spare. The three preparation receipts are retained; persistentRAM cache remains, and primary retains its separate first-shard copy. No weight preparation remains pending.

After one GPU_DEFERRED attempt, the next natural-breakpoint launch acquired the shared lock. Session55796 ended0 at14:37:12UTC: all16 requests complete, child5683reaped, GPU2MiB/no compute, private stageREMOVED, lockreleased. Complete original archive34780bytes SHAea8a8f5c92b7ff12f81081a1c41dd38f5586fa5f65584977bec53df9888a9de5 copied and matched locally; remote expanded/archive retained.

Frozen analysis:5/16correct,16EOS,0cap,0preemptions,1571tokens,length49–129. Scorerissues=[]; all16outputs retained. See C_INSTRUCT_GSM8K_RESULT_20261001.md and instruct_gsm8k_spare_qualification_v1.json. No general model-quality, capacity-policy or performance contribution is established.

Native128 completed at15:11:34UTC, session91137 exit0, child9116 reaped, GPU2MiB/no compute, stageREMOVED. All128 completed:57correct,126EOS,2cap,0preemptions,13989tokens. Peak932/4096 observed blocks; all128running before anycompletion. The preregistered no-pressure stop rule is triggered; no further GSM8K capacity unit is scheduled. See C_INSTRUCT_NATIVE128_RESULT_20261001.md and the quality/capacity/timeline/output-diagnostics JSONs. Archive208101bytes SHA35fb6e6672c8c6f6f56506bb2cbafdb75a28ec6f48f9ccb13f28433669472cf7 is matched local+remote. Persistent cache remains countedRAM. NoCjob/heldlock/queuedretry. Entire paper goal remains ACTIVE/INCOMPLETE; next is CPU-only feasibility checking of a distinct long-document QA domain, with no selected new controller.
