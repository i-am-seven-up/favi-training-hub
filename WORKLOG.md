# FAVI current state

Milestone: M0 scaffold v1 hoàn tất; nghiên cứu đang ở M1 dataset composition.
Objective: P0 binary NSFW trên deepghs: local GPU pilot 5–10% → review → Modal calibration/full; Kaggle fallback. Sau P0 mới quay lại canonical 3-class dataset composition.
Last action: chuyển pilot mặc định sang RTX 4060 local, cài PyTorch CUDA và lấy inspection subset của deepghs.
Evidence: `uv run favi doctor` thấy RTX 4060 8 GB; 40 tests pass; `outputs/reviews/deepghs_nsfw/` có inventory/contact sheet 50 ảnh cân bằng.
Next: người dùng review contact sheet; tạo evidence JSON, rồi xác minh nguồn Violence và Safe trước canonical freeze/baseline pilot.
Running remote jobs: none.
Blockers for real training: deepghs mới có subset; chưa có Violence/Safe hợp lệ, canonical manifest/splits và backbone revision.

## 2026-10-11 — local pilot migration

- Action: thêm `run local`, local resume, CUDA doctor và local/Kaggle pilot gate; đổi micro-batch 2 × accumulation 4.
- Result/evidence: `torch 2.8.0+cu128`, CUDA 12.8, RTX 4060 Laptop 8 GB, phép tính CUDA đạt; `uv run ruff check .` đạt; `uv run pytest -q` đạt 40/40.
- Action: khóa deepghs revision, đăng nhập gated repo, đọc ZIP bằng HTTP range và lấy 10 ảnh cho mỗi nhãn gốc.
- Result/evidence: 50 ảnh, 0 unreadable; chỉ là source-inspection subset, không phải canonical dataset hay training evidence.
- Next: không train 3-class từ deepghs đơn lẻ; cần Safe đã xác minh non-sexual AND non-violent và Violence đã review.

## 2026-10-11 — advisor report draft

- Action: tổng hợp hub, CUDA/local-pilot evidence và deepghs inspection evidence thành báo cáo 15 phút.
- Result/evidence: `REPORT_DRAFT_15MIN.md`; không nêu metric training chưa tồn tại là kết quả nghiên cứu.
- Next: trình bày draft, lấy xác nhận scope/dataset policy; sau đó validate VIMD và WeaponDetection.

## 2026-10-12 — VIMD validation start

- Action: xác định repo chính chủ, clone raw VIMD, parse XML và tạo subset review cân bằng 25 fighting/25 normal.
- Result/evidence: commit `79fef289d093658ba2caeaead4a5243d7755353a`; 2,421 XML, 1,972 cặp XML/ảnh hợp lệ; `outputs/reviews/vimd/` có inventory/contact sheet, 0 unreadable. Repo không khai báo license; 444 ảnh XML tham chiếu bị thiếu.
- Next: không dùng VIMD để freeze/train research khi chưa có permission/license và policy group/video; tiếp tục xác minh nguồn có license cho Violence và Safe.

## 2026-10-12 — priority corrected: P0 binary NSFW

- Action: đặt lại thứ tự thực thi theo yêu cầu người dùng.
- Result/evidence: P0 là deepghs binary NSFW (`hentai/porn/sexy` vs `neutral/drawings`), local pilot trước, Modal full sau review; record này tách với thesis 3-class.
- Next: tạo experiment record, chạy local GPU pilot 10% optimizer steps và lưu checkpoint/metrics để review trước khi quyết định Modal.

## 2026-10-12 — P0 dataset frozen

- Action: tải archive deepghs revision đã khóa, dựng manifest binary cân bằng và freeze dataset P0.
- Result/evidence: `data/manifests/nsfw-p0-v1.jsonl` có 4,000 ảnh (2,000 `NSFW`, 2,000 `Non-NSFW`), split group-level cố định 80/10/10; `data/manifests/nsfw-p0-v1.frozen.json`; fingerprint `36c7b9b4dc6c45488c2c10ff59209e8555f540bec83d13a3f6cd3ba40b8844f4`; audit đạt, không phát hiện near-duplicate trong manifest.
- Interpretation: đây là input đã versioned cho P0, chưa phải metric hay kết quả train. `Non-NSFW` không được diễn giải là lớp `Safe` của bài toán khoá luận 3 lớp.
- Action: tạo experiment `EXP-001` cho baseline frozen binary P0.
- Result/evidence: config hash `82bd7d3ddd7af94a6ac34b27b259b46c63e99685c4914c3a28ef4a1d09e511e4`; code hash `2c16b4d8432703da7b8530339a861b45794a4e656a20c8f5f1f3f1dbc27149b2`.
- Next: chạy local GPU pilot 10% optimizer steps và review checkpoint/metrics/VRAM/time trước Modal.

## 2026-10-12 — P0 pilot environment failure diagnosed

- Action: chạy `EXP-001/pilot-01` trên RTX 4060 local.
- Result/evidence: attempt dừng trước optimizer step; `experiments/EXP-001/pilot-01/run.json` ghi `failed` / `RuntimeError`, `history.csv` chỉ có header, không có metric/checkpoint. Traceback xác định cuBLAS cần `CUBLAS_WORKSPACE_CONFIG` khi protocol bật deterministic algorithms trên CUDA 10.2+.
- Action: đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước khi import PyTorch trong trainer; chạy quality gate.
- Result/evidence: `uv run ruff check .` đạt; `uv run pytest -q` đạt 42/42. Đây là sửa môi trường reproducibility, không đổi dataset/label/training hyperparameter.
- Next: tạo experiment mới vì code hash đã thay đổi, sau đó chạy local pilot mới. Không ghi đè failed attempt.

## 2026-10-12 — Phase gate: local pilot paused by user

- Action: tạo `EXP-002` sau reproducibility fix và khởi chạy `EXP-002/pilot-01`.
- Result/evidence: process local đã bị dừng theo yêu cầu người dùng sau 99 optimizer steps; `history.csv` có partial train-loss rows nhưng chưa đến evaluation/checkpoint ở step 100, nên không có `metrics.json` và không phải pilot hoàn tất.
- Action: thêm CLI `favi run cancel --run ... --reason ...` để ghi intentional stop thay vì để attempt kẹt trạng thái `running`.
- Result/evidence: `EXP-002/pilot-01` là `cancelled` với reason đã ghi trong `run.json`; `uv run ruff check .` đạt; `uv run pytest -q` đạt 43/43.
- Next: dừng tại phase gate để giải thích local pilot. Khi người dùng bảo tiếp tục, prepare experiment mới (code hash đã đổi do thêm CLI cancel) rồi chạy pilot từ đầu, không dùng partial run làm metric.

## 2026-10-12 — P0 local pilot completed

- Action: tạo `EXP-003` và chạy `EXP-003/pilot-01` đến hết 120/120 optimizer steps trên RTX 4060 local.
- Result/evidence: completed real pilot, non-mock; `experiments/EXP-003/pilot-01/metrics.json`; validation Macro-F1 `0.8696055568`, accuracy `0.87`, 400 validation images (200/lớp), finite gradients và gradient flow verified; checkpoint + local artifact đã lưu.
- Resource evidence: train `59.12 s`, setup `15.34 s`, peak allocated VRAM `0.397 GB`, peak reserved VRAM `0.439 GB`. Đây là phép đo local pilot, không dùng để ngoại suy thẳng chi phí/thời gian Modal.
- Interpretation: baseline frozen encoder + linear head có tín hiệu tốt trên P0 binary validation. Đây không phải held-out test, không phải kết quả 3-class, và chưa đủ để tuyên bố generalization.
- Next: project owner chọn full P0 local. Tạo experiment mới, gate nó với `EXP-003/pilot-01`, rồi train đủ 1.200 steps. Modal không dùng cho frozen P0.

## 2026-10-13 — P0 full-local policy and learning guide

- Action: project owner chọn full P0 chạy thuần local sau pilot, thay vì Modal.
- Result/evidence: `guide/` giải thích concepts, phases, metrics và P0 hiện tại bằng ngôn ngữ cho Software Developer. CLI local được mở rộng để full-local bắt buộc tham chiếu một pilot comparable đã completed; không nới lỏng Modal gate.
- Verification: `uv run ruff check .` đạt; `uv run pytest -q` đạt 44/44.
- Next: prepare full-local experiment mới và chạy với pilot `EXP-003/pilot-01` làm gate.

## 2026-10-13 — P0 frozen baseline full-local completed

- Action: chạy `EXP-005/full-01` local từ đầu, gated bởi completed pilot `EXP-003/pilot-01`.
- Result/evidence: completed real full run, 1,200/1,200 steps; `experiments/EXP-005/full-01/metrics.json`; validation Macro-F1 `0.9524997031`, accuracy `0.9525`, 400 validation images (200/lớp), full coverage, finite gradients và gradient flow verified. Artifact + checkpoint đã lưu local.
- Resource evidence: train `527.86 s`, setup `16.78 s`, peak allocated VRAM `0.397 GB`, peak reserved VRAM `0.439 GB` trên RTX 4060 Laptop 8 GB. Full frozen P0 xác nhận Modal không cần thiết cho track này.
- Interpretation: metric này là validation của full P0 binary, không phải held-out test, không phải 3-class thesis result và không được dùng để kết luận generalization ngoài source/protocol này.
- Next: user review full validation evidence. Nếu chốt frozen P0 artifact, test 400 ảnh held-out có thể được mở một lần; nếu muốn compare LoRA thì train LoRA trước khi mở test.

## 2026-10-15 — beginner training explanation consolidated

- Action: viết lại giải thích toàn bộ flow train, benchmark, metric, frozen baseline, LoRA và P0 bằng ngôn ngữ tự nhiên cho Software Developer chưa có nền tảng AI.
- Result/evidence: `guide/00-hieu-train-tu-con-so-0.md`; được liên kết đầu tiên từ `guide/README.md`. `guide/04-p0-hien-tai.md` cập nhật full P0 đã completed thay vì mô tả kế hoạch cũ.
- Next: nếu người dùng chọn comparison LoRA, tạo config LoRA P0 comparable rồi chỉ chạy local pilot trước; test vẫn chưa mở.
