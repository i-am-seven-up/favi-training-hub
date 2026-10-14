# Các giai đoạn train và lý do tồn tại

## 1. Review source và freeze dataset

Trước train, hub kiểm tra source, license/provenance, nhãn, duplicate và split. Sau đó tạo frozen manifest + fingerprint.

**Mục đích:** biết chắc input không bị thay đổi âm thầm.
**Output cần có:** frozen manifest, fingerprint, audit report.
**Không được nói:** “data tốt” nếu chưa có evidence review.

## 2. Prepare experiment

Hub snapshot config và code hash rồi tạo ID như `EXP-004`.

**Mục đích:** kết quả sau này gắn với đúng code/config.
**Output cần có:** `experiments/EXP-xxx/config.yaml` và `experiment.json`.

## 3. Local pilot

Pilot chạy 5–10% optimizer steps, không phải train cuối.

**Mục đích:** phát hiện sớm lỗi data path, CUDA, OOM, gradient, loss, VRAM hoặc metric trước khi bỏ nhiều thời gian/chi phí.
**Input:** frozen dataset + prepared experiment.
**Output cần có:** `metrics.json`, `history.csv`, checkpoint, VRAM/time.
**Thành công nghĩa là:** run completed, gradient hữu hạn, validation coverage đủ lớp. Nó không có nghĩa là model đã là kết quả cuối.

## 4. Review gate

Đọc pilot trước khi chạy lớn hơn. Câu hỏi thực tế:

- Data fingerprint có đúng không?
- Loss/gradient có bình thường không?
- Model nhầm lớp nào?
- VRAM có đủ không?
- Có lý do khoa học/kỹ thuật để train full không?

Quyết định có thể là tiếp tục, sửa config/dataset rồi tạo experiment mới, hoặc dừng.

## 5. Full train

Full run chạy toàn bộ số steps đã chốt. Với P0 frozen baseline hiện tại, full là 1.200 steps và có thể chạy local vì pilot cho thấy RTX 4060 8 GB còn rất nhiều VRAM.

**Output:** model artifact cuối, checkpoint, history, validation metrics, confusion matrix và resource evidence.

Full train không tự biến validation thành test. Nó chỉ tạo ra ứng viên model đủ nghiêm túc để chốt.

## 6. Chọn model và test

Sau khi không còn định chỉnh model dựa trên validation, ta khóa lựa chọn model. Sau đó chạy artifact đã chọn lên test set một lần.

**Vì sao?** Nếu xem test rồi quay lại đổi learning rate/số steps/model để tăng điểm test, test đã vô tình trở thành validation. Khi đó con số cuối kém đáng tin.

**Output cuối:** test metrics + confusion matrix + liên kết tới artifact/config/dataset fingerprint.

## Khi nào cần Modal?

Modal là GPU cloud, không phải một bước bắt buộc. Nó phù hợp khi local không đủ VRAM, quá chậm, hoặc cần chạy nhiều thí nghiệm. Với P0 frozen, local đủ. LoRA có thể cần Modal sau local pilot nếu 8 GB không đủ.
