# FAVI Training Hub — đọc từ đây

Folder này là “bản đồ tinh thần” của hub. Nó dành cho một Software Developer bắt đầu làm AI/ML: không cần biết trước notebook, CUDA hay fine-tuning.

## Hub này đang giải quyết việc gì?

Hub biến một lần train thành chuỗi bằng chứng có thể kể lại:

```text
Nguồn ảnh → dataset đã khóa → train → validation → chọn model → test một lần → báo cáo
```

Mỗi mũi tên đều để lại file. Vì vậy khi giảng viên hỏi “con số này đến từ đâu?”, câu trả lời không phải là “em chạy notebook”, mà là dataset version nào, code/config nào và model artifact nào.

## Đọc theo thứ tự này

1. [Giải thích train từ số 0](00-hieu-train-tu-con-so-0.md) — tài liệu ưu tiên: toàn bộ flow, thuật ngữ, ví dụ và P0 hiện tại.
2. [Khái niệm cốt lõi](01-khai-niem-cot-loi.md) — bản ngắn về dataset, label, model, frozen và LoRA.
3. [Các giai đoạn train](02-cac-giai-doan-train.md) — từng phase làm gì, input/output nào phải xuất hiện.
4. [Cách đọc một kết quả](03-doc-ket-qua.md) — Macro-F1, confusion matrix và cách không hiểu sai một con số.
5. [P0 hiện tại](04-p0-hien-tai.md) — những gì đã làm thật trong repo này và bước tiếp theo.

## Quy ước quan trọng

- Không có file evidence thì không coi là kết quả nghiên cứu.
- `Non-NSFW` của P0 **không phải** lớp `Safe` trong khóa luận 3 lớp.
- Test giữ kín cho đến khi model/config đã chốt. Validation là nơi dùng để xem hướng train có ổn hay không.
- Không ghi đè attempt cũ: retry tạo attempt mới, như giữ lại CI build log thất bại.
