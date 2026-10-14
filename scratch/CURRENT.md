# Current task

Action: P0 binary NSFW trên deepghs trước; local pilot rồi Modal full sau review. VIMD/3-class tạm không phải blocker P0.
Status: p0_full_local_completed; held_out_test_not_opened; LoRA_not_started.
Result: `EXP-005/full-01` completed 1,200/1,200 local steps, gated by `EXP-003/pilot-01`. Validation Macro-F1 `0.9524997031`, accuracy `0.9525`, 400 validation images (200/lớp), finite gradients, artifact and checkpoint present. Train took `527.86 s`; peak allocated VRAM `0.397 GB`. P0 remains binary preliminary work, not the thesis 3-class result or held-out test result.
Evidence: `guide/`; `experiments/EXP-005/full-01/run.json`; `experiments/EXP-005/full-01/metrics.json`; `experiments/EXP-005/full-01/artifact/`; `WORKLOG.md`.
Next: user chooses whether to (a) run a LoRA comparison before opening test, or (b) lock this frozen P0 baseline and run held-out test once. Do not call `Non-NSFW` `Safe`.
Constraints: P0 là result sơ bộ binary; Modal chỉ được submit sau pilot và khi có budget/repo upload scope rõ ràng. VIMD chưa có license và có missing XML image references.

Latest handoff: đọc `guide/00-hieu-train-tu-con-so-0.md` trước, sau đó `WORKLOG.md` và record của `EXP-005/full-01` để nắm lý thuyết, lịch sử và evidence P0 hiện tại.
