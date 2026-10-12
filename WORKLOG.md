# FAVI current state

Milestone: M0 scaffold v1 hoàn tất; nghiên cứu đang ở M1 dataset composition.
Objective: local GPU pilot 5–10% → review; Kaggle fallback; Modal calibration/full sau gate.
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
