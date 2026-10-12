# FAVI Training Hub

- Tư duy bằng tiếng Anh, trao đổi với người dùng bằng tiếng Việt.
- Người dùng là Software Developer và chưa có nền tảng AI/ML. Xem các khái niệm dataset, split,
  checkpoint, metric, baseline, LoRA, GPU training và experiment tracking là kiến thức mới.
- Khi hướng dẫn, luôn nói **bước này nhằm mục đích gì** trước, sau đó mới đưa lệnh hoặc file cần sửa.
  Giải thích thuật ngữ ở lần đầu xuất hiện bằng ngôn ngữ đơn giản và liên hệ với khái niệm software
  engineering quen thuộc như test fixture, build artifact, immutable version, CI gate và deployment.
- Hướng dẫn theo từng bước nhỏ có thứ tự. Mỗi lần ưu tiên một action kế tiếp, cho biết input, output mong
  đợi, cách nhận biết thành công và lỗi thường gặp. Không đưa một danh sách lệnh dài thiếu ngữ cảnh.
- Không giả định người dùng biết notebook, Hugging Face, Kaggle, Modal, CUDA, VRAM hoặc quy trình ML.
  Nếu một thao tác cần tài khoản, token, GPU hay có thể tốn tiền, nói rõ trước khi thực hiện.
- Ưu tiên mặc định an toàn và dễ học. Khi có nhiều lựa chọn kỹ thuật, đề xuất một lựa chọn chính kèm
  lý do ngắn; chỉ mở rộng các lựa chọn khác khi chúng ảnh hưởng trực tiếp tới quyết định hiện tại.
- Khi báo kết quả, phân biệt rõ: scaffold/test giả lập, pilot, validation và kết quả nghiên cứu cuối.
  Không dùng từ “train thành công” nếu mới chỉ kiểm tra code bằng fixture/mock.
- Đọc `WORKLOG.md`, `scratch/CURRENT.md`, sau đó đọc record được tham chiếu trước khi làm việc.
- Nguồn nghiệp vụ: `materials/FAVI_DATASET_HANDOFF_final.md`; thiết kế: `SCAFFOLD_PLAN.md`.
- Bài toán single-label Safe=0, Sexual=1, Violence=2. Baseline là frozen image encoder + trained linear head; LoRA dùng cùng encoder/head/protocol.
- Không sửa material/raw. Dataset chưa review, license/provenance chưa xác minh không được freeze để nghiên cứu.
- Không suy ra Safe từ non-NSFW/non-violence; weapon presence không đủ chứng minh Violence. Nhãn overlap/ambiguous cần review.
- Split cố định theo group và duplicate cluster; test không dùng tuning, pilot hay review gate.
- Mỗi training experiment cần pilot 5–10% optimizer steps trên GPU local và review đúng fingerprint trước Modal full. Kaggle chỉ là fallback khi GPU local 8 GB không đủ VRAM hoặc môi trường local bị chặn.
- Thay training semantics phải tạo experiment mới; retry tạo attempt mới. Không ghi đè kết quả.
- Modal cần budget rõ ràng và calibration GPU đích. Không ngoại suy thời gian local/Kaggle theo tỷ lệ đơn giản thành thời gian Modal.
- Không tự chạy GPU trả phí hoặc upload dữ liệu ngoài budget/phạm vi đã được giao.
- Không bịa metric/license/review/job status. Mock/fixture không phải evidence nghiên cứu.
- Mỗi task để lại action → result/evidence → next. Cập nhật WORKLOG và scratch tại mốc quan trọng/trước kết thúc.
- Dùng `uv run pytest` và `uv run ruff check .` cho core; model tests dùng extra train. Báo riêng offline/dry-run/live checks.
- Giữ hub gọn: CLI + files; chưa cần web UI, database, W&B hay scheduler.
