# Review evidence

Registry chỉ ghi nhận findings trong handoff, chưa xác minh lại dataset tại repo này.
Mỗi review JSON cần `reviewer`, `reviewed_at`, `decision` (pass/pass_with_limitation/reject),
`source_revision`, `sample_ids`, `notes`. Ghi limitations và annotation examples còn mơ hồ.
Giữ contact sheets/ảnh trong `outputs/` (ignored). Không tự map neutral/drawings thành Safe.
Sau khi review, điền `review_evidence`, license evidence và revision trong sources.yaml.
