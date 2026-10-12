# Current task

Action: chuyển Kaggle-first thành local RTX 4060 pilot; lấy nhanh deepghs inspection subset.
Status: local_cuda_ready; source_inspection_done; research_training_blocked_by_dataset.
Result: PyTorch CUDA nhận RTX 4060 8 GB; CLI `run local` + resume và gate đã test. Deepghs revision
`ac763cb1e1557225168be3b6b6b1ee864c17bc36`; 50 ảnh cân bằng nằm ở
`data/raw/deepghs_nsfw_subset/`, inventory/contact sheet ở `outputs/reviews/deepghs_nsfw/`.
Next: người dùng review contact sheet; không map neutral/drawings thành Safe. Xác minh Violence + Safe,
freeze canonical manifest, khóa CLIP revision, rồi chạy frozen baseline local pilot trước LoRA.
Constraints: subset không phải training evidence; chưa chạy training/remote job, chưa upload dữ liệu.

Latest handoff: `REPORT_DRAFT_15MIN.md` đã sẵn sàng cho báo cáo giảng viên 15 phút.
