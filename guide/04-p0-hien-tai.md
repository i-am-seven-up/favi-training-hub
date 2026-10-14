# P0 hiện tại: baseline NSFW local

## P0 trả lời câu hỏi gì?

P0 hỏi một câu hẹp: frozen CLIP + linear head có phân biệt được `NSFW` và `Non-NSFW` trên deepghs hay không?

P0 giúp kiểm tra hub end-to-end và tạo baseline trước khi đi vào luận văn chính ba lớp `Safe / Sexual / Violence`.

## Data đã dùng

- 4.000 ảnh deepghs đã freeze.
- Train: 3.200 ảnh, cân bằng 1.600 mỗi lớp.
- Validation: 400 ảnh, cân bằng 200 mỗi lớp.
- Test: 400 ảnh, chưa mở.
- Frozen fingerprint: `36c7b9b4dc6c45488c2c10ff59209e8555f540bec83d13a3f6cd3ba40b8844f4`.

`Non-NSFW` là negative của NSFW, **không được gọi là Safe**.

## Pilot đã đo thật

`EXP-003/pilot-01` hoàn thành 120/120 steps trên RTX 4060 Laptop 8 GB:

- Validation Macro-F1: 0,8696.
- Accuracy: 0,87.
- Peak VRAM: 0,397 GB.
- Train time: khoảng 59 giây, chưa gồm setup.

Đây là tín hiệu tốt để chạy full local, không phải test cuối.

## Full frozen baseline đã hoàn thành

`EXP-005/full-01` đã train 1.200/1.200 steps local:

- Validation Macro-F1: 0,9525.
- Accuracy: 0,9525.
- Peak VRAM: 0,397 GB.
- Train time: khoảng 8 phút 48 giây, chưa gồm setup.

Đây vẫn là validation P0, chưa phải held-out test.

## Bước tiếp theo

Nếu mục tiêu là so sánh LoRA, tạo LoRA P0 dùng cùng dataset/split, chạy pilot local trước để đo VRAM, sau đó full LoRA. Test 400 ảnh được giữ kín cho đến khi frozen và LoRA đều đã chốt.
