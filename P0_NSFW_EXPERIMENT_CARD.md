# P0 — Binary NSFW: Experiment Card

## Mục tiêu dễ nói với giảng viên

P0 trả lời một câu hỏi hẹp trước: **một CLIP image encoder đóng băng, chỉ train linear head, có phân biệt được ảnh NSFW và Non-NSFW trên deepghs hay không?**

P0 được làm trước để xác minh data/train/evaluation pipeline bằng một nguồn đã được project owner kiểm tra. Đây là bước tạo baseline và kinh nghiệm vận hành, trước track luận văn chính `Safe / Sexual / Violence`.

## P0 không phải là gì

- `Non-NSFW` không phải nhãn `Safe` của khóa luận: chưa xác minh độc lập là non-violent.
- P0 không thay thế so sánh LoRA với frozen baseline cho bài toán 3-class.
- Metric P0 không được trích dẫn như Macro-F1 cuối cùng của khóa luận 3-class.

## Dataset và rule nhãn

| Original folder deepghs | Nhãn P0 | Lý do |
|---|---|---|
| `hentai`, `porn`, `sexy` | `NSFW` | Positive NSFW theo source đã review |
| `neutral`, `drawings` | `Non-NSFW` | Negative của khái niệm NSFW; không gọi là Safe |

- Source revision: `ac763cb1e1557225168be3b6b6b1ee864c17bc36`
- License reported by source: MIT.
- Dataset P0 dự kiến: 4.000 ảnh cân bằng, 2.000 mỗi lớp.
- Split cố định: 80% train / 10% validation / 10% test, seed 42.
- Split/fingerprint được tạo trước train; test không dùng để chỉnh config.

## Model và cách so sánh

| Thành phần | P0 lựa chọn | Ý nghĩa đơn giản |
|---|---|---|
| Backbone | `openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | Model nhìn ảnh pretrained, version được khóa như dependency lock |
| Baseline | frozen encoder + linear head 2 outputs | Chỉ học classifier cuối, rẻ và là mốc so sánh |
| Primary metric | Macro-F1 | Trung bình F1 hai lớp, không để lớp phổ biến lấn át |
| Secondary evidence | per-class precision/recall/F1, confusion matrix, VRAM, training time | Biết model sai kiểu gì và tốn tài nguyên thế nào |

## Flow execution

```text
full source → manifest + audit + frozen split
→ local pilot (10% optimizer steps)
→ review metric/VRAM/time/fingerprint
→ explicit Modal budget + calibration
→ Modal full run
→ held-out test evaluation + report
```

| Bước | Nó chứng minh gì | Artifact cần xem |
|---|---|---|
| Audit/freeze | Nhãn, source revision, split và duplicate guard đã khóa | frozen manifest + fingerprint |
| Local pilot | Pipeline/model gradient chạy với data thật; đo VRAM/thời gian sớm | `metrics.json`, `history.csv`, checkpoint |
| Review | Con người quyết định có nên chi tiền Modal không | `review.json` |
| Modal calibration | Ước lượng full run trên đúng GPU và chặn vượt budget | `calibration.json` |
| Modal full | Kết quả train đủ budget | artifact + `metrics.json` |
| Test evaluation | Đo kết quả cuối một lần sau khi config đã khóa | test metrics + report |

## Trạng thái hiện tại

| Hạng mục | Trạng thái | Evidence |
|---|---|---|
| CUDA local | Ready | RTX 4060 Laptop, 8 GB VRAM; PyTorch CUDA 12.8 |
| Full archive deepghs | Ready | `data/raw/deepghs_nsfw_full/` (Git-ignore) |
| Binary model/config support | Ready | `configs/nsfw_protocol.yaml`, `configs/nsfw_frozen.yaml` |
| Frozen manifest | Ready | `data/manifests/nsfw-p0-v1.frozen.json`, fingerprint `36c7b9b4…8844f4` |
| Local pilot | Completed | `experiments/EXP-003/pilot-01/metrics.json`: validation Macro-F1 0.8696; 120/120 steps; local artifact + checkpoint |
| Full frozen baseline | Completed local | `experiments/EXP-005/full-01/metrics.json`: validation Macro-F1 0.9525; 1,200/1,200 steps; artifact + checkpoint |
| Modal full | Chưa chạy | Chờ pilot review + budget/upload authorization |

## Câu nói ngắn khi báo cáo

> P0 là một binary NSFW track riêng để xác minh pipeline end-to-end trên nguồn dữ liệu đã review. Tôi khóa source revision, split, backbone revision và metric trước khi train. Kết quả P0 giúp quyết định pipeline có đáng đẩy sang Modal không; nó không được đánh đồng với kết quả 3-class cuối của khóa luận.
