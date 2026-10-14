# Hiểu việc train AI từ con số 0

> Đọc file này trước nếu bạn là Software Developer và muốn hiểu hub đang làm gì mà không cần nền tảng AI.

## Một câu duy nhất: hub đang làm gì?

Ta đưa cho máy nhiều ảnh **đã có đáp án**, để nó học quy tắc dự đoán đáp án cho ảnh mới.

Trong P0 hiện tại, câu hỏi chỉ là:

```text
Ảnh này là NSFW hay Non-NSFW?
```

Một ảnh `porn` đã được gắn nhãn `NSFW`. Một ảnh `neutral` đã được gắn nhãn `Non-NSFW`. Đây là **label**: đáp án dùng để dạy và kiểm tra model.

Không có “AI tự biết đáp án”. AI chỉ dự đoán; ta biết nó đúng/sai bằng cách so dự đoán với label đã có.

## Hình dung như viết phần mềm

| Trong Software Engineering | Trong hub AI |
|---|---|
| Input fixture | Ảnh + label |
| Hàm cần implement | Model dự đoán nhãn ảnh |
| Unit test / CI check | Validation trên ảnh model chưa được học |
| Release candidate | Full-trained model artifact |
| Acceptance test | Held-out test set, chỉ chạy khi mọi lựa chọn đã chốt |
| Build artifact + lockfile | Checkpoint/model artifact + dataset fingerprint/config hash |

Điểm khác lớn nhất: thay vì tự viết rule `if/else`, ta chọn cấu trúc model và để nó tự điều chỉnh các con số bên trong khi nhìn nhiều ví dụ.

## Một ảnh đi qua model như thế nào?

```text
Ảnh RGB
  → resize/crop thành 224 × 224
  → CLIP encoder: biến ảnh thành dãy số mô tả nội dung ảnh
  → classifier head: chọn nhãn NSFW hoặc Non-NSFW
```

**CLIP** là model pretrained: trước khi vào project, nó đã học cách biểu diễn đặc trưng hình ảnh từ dữ liệu rất lớn. Nó giống một framework mạnh đã có sẵn.

**Classifier head** là lớp nhỏ cuối cùng. Nó nhận đặc trưng của CLIP và trả về hai điểm số; điểm cao hơn quyết định nhãn dự đoán.

## “Train” thực sự thay đổi cái gì?

Mỗi lần model nhìn một số ảnh train, nó dự đoán. Hệ thống so sánh dự đoán với label đúng, tính mức sai (**loss**) rồi điều chỉnh các tham số được phép train để lần sau sai ít hơn.

Một lần điều chỉnh gọi là **optimizer step**.

Ví dụ đơn giản:

```text
Ảnh thật là NSFW
Model ban đầu đoán Non-NSFW
→ loss cao
→ điều chỉnh tham số trainable
→ về sau gặp ảnh tương tự, cơ hội đoán NSFW cao hơn
```

Không phải mọi phần model đều được phép thay đổi. Đây là chỗ frozen và LoRA khác nhau.

## Frozen baseline là gì?

```text
Ảnh → CLIP encoder (khóa nguyên) → classifier head (được train) → nhãn
```

- **Frozen** nghĩa là giữ nguyên phần CLIP đã pretrained.
- Chỉ train classifier head cuối.
- Tên kỹ thuật chính xác: **linear probing** hoặc **linear-probe baseline**.

Nó không phải “không train”. Nó vẫn train thật, nhưng chỉ train phần rất nhỏ: P0 hiện train 1.538 tham số trong khi CLIP có khoảng 87 triệu tham số.

Mục đích của frozen baseline là tạo mốc tối thiểu: “nếu chỉ dùng khả năng có sẵn của CLIP rồi train lớp cuối thì đạt bao nhiêu?”

## LoRA là gì?

```text
Ảnh → CLIP encoder + LoRA adapters (được train một phần) → head (được train) → nhãn
```

LoRA giữ phần lớn CLIP nguyên, nhưng gắn các module nhỏ gọi là **adapter** vào vài chỗ trong attention của CLIP. Khi train, adapter và classifier head được thay đổi.

Nói kiểu software:

- Frozen: dùng nguyên dependency, chỉ viết tầng business logic cuối.
- Full fine-tuning: sửa cả dependency rất lớn.
- LoRA: giữ dependency, nhưng gắn plugin nhỏ vào vài điểm để nó thích nghi với bài toán.

LoRA thường học được nhiều hơn frozen, nhưng tốn VRAM/thời gian hơn. Không được giả định LoRA luôn tốt hơn: phải chạy và đo.

## Vì sao cần cả frozen và LoRA?

Đây là so sánh đúng:

```text
A. Frozen / linear probe      = mốc đơn giản
B. Standard LoRA              = phương pháp chính
C. LoRA + cải tiến (tùy chọn) = đề xuất cần chứng minh
```

Không thay A bằng B. Nếu chỉ chạy LoRA và được điểm 0,95, ta không biết frozen có thể đã đạt 0,95 hay chưa. So sánh B với A trả lời: “LoRA có thực sự đáng đổi thêm độ phức tạp không?”

Nếu sau này có C, phải so C với B để trả lời: “cải tiến của mình có thật sự giúp LoRA không?”

## Dataset được chia như thế nào?

Một dataset không được chia riêng cho từng method. Frozen và LoRA phải dùng cùng ảnh để so sánh công bằng.

```text
Dataset đã freeze
├─ Train       → cả frozen và LoRA đều được học từ đây
├─ Validation  → cả frozen và LoRA đều được đo ở đây
└─ Test        → cả frozen và LoRA đều chỉ được đo cuối cùng ở đây
```

### Train

Model nhìn ảnh và được cập nhật tham số. Không dùng score train để tuyên bố model tốt, vì model đã nhìn các ảnh này rồi.

### Validation

Model không được cập nhật tham số. Nó dự đoán các ảnh chưa từng train; ta so với label. Validation dùng để quyết định: số epoch có ổn không, LoRA có fit VRAM không, method nào đáng tiếp tục.

### Test

Test cũng gồm ảnh chưa từng train, nhưng bị giữ kín cho đến cuối. Lý do: nếu nhìn test rồi chỉnh learning rate/rank/epoch để tăng điểm, test không còn độc lập nữa.

Test không phải ảnh “không có label”; test vẫn có label, chỉ là label không được dùng để ra quyết định kỹ thuật trước đó.

## Metric: model tốt hay không được đo bằng gì?

Model dự đoán nhãn cho validation/test. Hệ thống so sánh từng dự đoán với label đúng.

### Accuracy

**Accuracy** = tổng số ảnh đoán đúng / tổng số ảnh.

Ví dụ 381/400 ảnh đúng thì accuracy là 95,25%.

### Precision và recall

Với lớp NSFW:

- **Precision**: trong các ảnh model gọi là NSFW, có bao nhiêu ảnh thật sự NSFW?
- **Recall**: trong các ảnh thật sự NSFW, model tìm ra được bao nhiêu ảnh?

Một model có thể gọi rất nhiều ảnh là NSFW để không bỏ sót, recall cao nhưng precision thấp. F1 cân bằng hai điều này.

### F1 và Macro-F1

- **F1**: điểm cân bằng precision và recall cho một lớp.
- **Macro-F1**: lấy F1 của từng lớp rồi trung bình đều.

Macro-F1 phù hợp khi không muốn lớp có nhiều ảnh lấn át lớp ít ảnh. Với bài toán 3 lớp cuối, nó quan trọng hơn accuracy đơn thuần.

### Confusion matrix

Confusion matrix là bảng “model nhầm kiểu gì”, không chỉ là một con số.

Ví dụ P0 full hiện tại, trên 400 ảnh validation:

```text
                         Model đoán Non-NSFW   Model đoán NSFW
Thực tế Non-NSFW                  191                   9
Thực tế NSFW                       10                 190
```

Tức là model nhầm 9 ảnh Non-NSFW thành NSFW và bỏ sót 10 ảnh NSFW.

## Benchmark là gì?

**Benchmark** là quy tắc để so sánh công bằng, không phải chỉ là một con số đẹp.

Một benchmark tốt đảm bảo frozen và LoRA dùng cùng:

- dataset version và fingerprint,
- label mapping,
- train/validation/test split,
- CLIP backbone + revision,
- preprocessing,
- seed,
- metric.

Khi đó khác biệt metric có thể được quy cho method nhiều hơn, thay vì do vô tình dùng data khác.

## P0 hiện tại đã tạo ra cái gì?

P0 là bài thử binary riêng, **không phải** kết quả khóa luận `Safe / Sexual / Violence`.

| Thành phần | Giá trị thực tế |
|---|---|
| Dataset P0 | 4.000 ảnh deepghs đã freeze, 2.000 mỗi lớp |
| Train / validation / test | 3.200 / 400 / 400 ảnh |
| Model đã train | Frozen linear probe, CLIP ViT-B/32 + linear head |
| Full run | `EXP-005/full-01`, 1.200/1.200 steps local |
| Validation Macro-F1 | 0,9525 |
| Validation accuracy | 0,9525 |
| VRAM peak | 0,397 GB trên RTX 4060 Laptop 8 GB |
| Test | Chưa mở |

Diễn giải đúng:

> Frozen baseline P0 đạt Macro-F1 0,9525 trên 400 ảnh validation của P0.

Không được diễn giải thành “model đạt 95% trong thực tế” hoặc “kết quả 3 lớp của khóa luận đạt 95%”.

## LoRA config là gì và tại sao cần nó?

LoRA config là file ghi cách chạy LoRA; nó giống build configuration. Ví dụ:

```yaml
method: lora
lora:
  r: 8
  alpha: 16
  target_modules: [q_proj, v_proj]
training:
  batch_size: 1
  accumulation_steps: 8
```

- `r` (**rank**): kích thước/capacity của adapter. Rank lớn hơn thường nhiều tham số hơn và tốn VRAM hơn.
- `alpha`: độ mạnh scale của adapter.
- `target_modules`: các phần của attention được gắn adapter.
- `batch_size: 1`: 8 GB VRAM chỉ xử lý một ảnh một lượt để an toàn.
- `accumulation_steps: 8`: gom gradient của 8 ảnh trước khi update, giữ effective batch gần với baseline.

Đây là điểm bắt đầu, không phải chân lý. Local pilot sẽ chứng minh cấu hình có chạy/fit VRAM thật không.

## Nếu muốn “cải tiến” thì làm gì?

Không cần bỏ LoRA hoặc phát minh thuật toán mới. Cải tiến là một thay đổi có lý do, được so với standard LoRA.

Ví dụ hợp lý cho dataset 3 lớp sau khi đã biết data:

```text
Frozen baseline
→ Standard LoRA
→ LoRA + class-balanced loss
```

**Class-balanced loss** chỉ phù hợp nếu Safe/Sexual/Violence bị lệch số lượng. Nó cho lỗi ở lớp ít ảnh trọng số lớn hơn để model không chỉ ưu tiên lớp nhiều ảnh.

Một hướng khác là so LoRA rank/target modules. Đây gọi là **ablation**: thay đổi đúng một thành phần để biết nó có ích hay không.

Đừng làm nhiều thay đổi cùng lúc. Nếu đồng thời đổi data, rank, loss và model, một điểm số tốt hơn sẽ không cho biết chính xác thứ nào đã tạo ra cải thiện.

## Thứ tự làm việc nên nhớ

```text
1. Freeze dataset
2. Frozen baseline
3. Standard LoRA pilot
4. Standard LoRA full
5. (Tùy hypothesis) LoRA + một cải tiến
6. So sánh trên validation
7. Chốt config/method
8. Mở test đúng một lần cho các method đã chốt
9. Báo cáo kết quả và giới hạn
```

## Từ vựng ngắn gọn

| Từ | Nghĩa ngắn |
|---|---|
| Label | Đáp án đúng đi kèm ảnh |
| Dataset | Bộ ảnh + label dùng cho experiment |
| Split | Chia dataset thành train/validation/test |
| Model | Hàm biến ảnh thành dự đoán nhãn |
| Pretrained | Đã học kiến thức chung trước project này |
| Train | Điều chỉnh tham số để giảm lỗi trên train data |
| Loss | Mức sai dùng để hướng dẫn model học |
| Step | Một lần optimizer cập nhật tham số |
| Epoch | Đi qua train set một lượt |
| Frozen | Khóa một phần model, không cho nó thay đổi |
| Linear probe | Chỉ train classifier head cuối |
| LoRA | Train adapter nhỏ gắn vào model pretrained |
| Baseline | Mốc đơn giản để so sánh |
| Benchmark | Protocol so sánh công bằng |
| Validation | Dữ liệu kiểm tra để chọn hướng/config |
| Test | Dữ liệu giữ kín để báo kết quả cuối |
| Metric | Con số đo chất lượng, ví dụ Macro-F1 |
| Artifact | Model/config/processor đã lưu để chạy lại |
| Checkpoint | Ảnh chụp trạng thái train để resume/audit |

## Bước tiếp theo của P0

Nếu mục tiêu là so sánh LoRA, chưa mở test frozen. Tạo LoRA P0 dùng đúng dataset/split này, chạy pilot local trước để đo VRAM, sau đó mới quyết định full run. Khi frozen và LoRA đều đã chốt, test set 400 ảnh sẽ được dùng một lần cho cả hai.
