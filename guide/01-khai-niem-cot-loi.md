# Khái niệm cốt lõi, không cần AI background

## Dataset, label và split

**Dataset** là bộ ảnh dùng cho một thí nghiệm. Mỗi ảnh có một **label** — câu trả lời đúng đã có sẵn, ví dụ `NSFW` hoặc `Non-NSFW`.

Hub chia dataset thành ba phần cố định, giống chia test fixture thành các vai trò khác nhau:

| Phần | Model có được học từ ảnh này không? | Mục đích |
|---|---:|---|
| Train | Có | Để model cập nhật tham số |
| Validation | Không | Để xem hướng train/config có ổn không |
| Test | Không | Chỉ đo kết quả cuối một lần |

Việc “khóa” manifest và fingerprint giống lockfile/dependency version: sau này vẫn xác định đúng tập ảnh nào đã tạo ra một metric.

## Model trong P0

P0 nhận một ảnh và trả về một trong hai nhãn:

```text
Ảnh → CLIP encoder → linear head → NSFW / Non-NSFW
```

- **CLIP encoder** là phần lớn của model, đã pretrained để biến ảnh thành đặc trưng số. Có thể xem nó như framework/library đã có năng lực chung.
- **Linear head** là một lớp nhỏ nhận đặc trưng đó và trả ra nhãn. Nó giống phần business rule nhỏ đặt trên một dependency lớn.

## Frozen baseline

**Frozen** nghĩa là không đổi CLIP encoder; chỉ học linear head. Nó rẻ, ít VRAM, dễ kiểm soát và là mốc chuẩn tối thiểu (**baseline**).

Nếu sau này train LoRA, câu hỏi không phải “LoRA có số đẹp không?” mà là “LoRA có tốt hơn frozen baseline theo cùng data/split/metric không?”. Nếu không hơn, LoRA không đáng tăng độ phức tạp.

## LoRA là gì?

LoRA không train lại toàn bộ CLIP. Nó gắn thêm các ma trận nhỏ vào một số phần của encoder và chỉ train phần thêm đó. Nó thường mạnh hơn frozen baseline nhưng tốn VRAM/thời gian hơn và có thêm hyperparameter phải kiểm soát.

## Optimizer step, epoch và checkpoint

- **Batch**: một nhóm nhỏ ảnh đưa vào GPU cùng lúc. P0 dùng batch size 2 để an toàn cho VRAM.
- **Optimizer step**: một lần model cập nhật tham số sau khi xem batch/nhóm batch. Đây là đơn vị ngân sách train đáng tin hơn “bao lâu”.
- **Epoch**: model đi qua gần hết train set một lần.
- **Checkpoint**: ảnh chụp trạng thái model + optimizer, giống build artifact trung gian; dùng để resume hoặc kiểm tra đúng model nào được chọn.

## Artifact, experiment và attempt

- **Experiment**: định nghĩa một ý tưởng có config/code fingerprint, ví dụ frozen P0 với dataset P0 v1.
- **Attempt/run**: một lần thực thi experiment. Nếu lỗi hay bị dừng thì giữ nguyên evidence và tạo attempt mới.
- **Artifact**: model weights, processor và config đủ để chạy lại dự đoán.

Ba khái niệm này giúp tránh câu mơ hồ “em train model rồi”; thay vào đó có thể nói rõ “artifact này thuộc attempt này của experiment này”.
