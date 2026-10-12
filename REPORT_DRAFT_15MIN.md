# Báo cáo tiến độ FAVI Training Hub

**Mốc báo cáo:** 11/10/2026  
**Trạng thái nghiên cứu:** M1 — xây dựng canonical dataset; chưa có kết quả huấn luyện để báo cáo.

## 1. Mục tiêu khóa luận

FAVI nghiên cứu phân loại ảnh nhạy cảm theo đúng một nhãn trong ba nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| Safe | Ảnh đã xác minh vừa không Sexual vừa không Violence |
| Sexual | Nội dung tình dục/NSFW |
| Violence | Bạo lực trong ngữ cảnh hành vi, thương tích hoặc máu me |

Câu hỏi nghiên cứu: **LoRA có cải thiện Macro-F1 so với frozen pretrained image encoder + linear head, khi cùng dataset và evaluation protocol hay không?**

Baseline trước là frozen encoder + trained linear head. Sau đó mới so sánh LoRA trên cùng backbone, split, preprocessing và metric. Không bắt đầu bằng LoRA để tránh không biết kết quả đến từ model hay từ data/protocol.

## 2. Vai trò của FAVI Training Hub

Training Hub là repo CLI/file-based để biến từng thực nghiệm thành chuỗi có thể kiểm tra:

```text
dataset review → audit/freeze split → local pilot → review → Modal full run → locked test evaluation
```

Mỗi action để lại artifact: config snapshot, fingerprint dataset, run ID, metrics, checkpoint và review record. Vai trò của nó tương tự CI + build artifact trong software engineering: một metric chỉ có giá trị khi truy được đúng data, code và config tạo ra nó.

Các guardrail chính:

- Safe không được suy ra từ “không NSFW” hoặc “không Violence”.
- Có group/duplicate check để tránh ảnh cùng video hoặc ảnh gần giống rơi vào cả train và test.
- Test set bị giữ lại; không dùng cho pilot hoặc chỉnh model.
- Đổi training semantics tạo experiment mới; không ghi đè kết quả cũ.
- Modal full run chỉ được phép sau một pilot thật đã review và budget/calibration hợp lệ.

## 3. Quy trình thực nghiệm và sử dụng tài nguyên

```text
Local RTX 4060 pilot (5–10% optimizer steps)
       ↓ review metric + VRAM + thời gian
Kaggle fallback nếu local thiếu VRAM
       ↓ approved pilot
Modal calibration trên GPU đích → full run trong budget
```

Lý do: local pilot là cách nhanh, miễn phí để phát hiện sai data/model/config trước khi dùng cloud. Kaggle không còn là dependency bắt buộc. Modal chỉ dành cho full run sau khi có bằng chứng pilot.

## 4. Tiến độ kỹ thuật và bằng chứng

| Hạng mục | Trạng thái | Evidence |
|---|---|---|
| CLI/workflow scaffold | Hoàn thành local | `README.md`, `src/favi_hub/` |
| GPU local | Sẵn sàng | PyTorch 2.8.0+cu128 nhìn thấy RTX 4060 Laptop, 8 GB VRAM |
| Local pilot + resume | Đã dựng và test | `favi run local`, checkpoint identity guard |
| Quality gates | Đã dựng | data audit, fingerprint, review gate, Modal budget/calibration gate |
| Regression checks | Đạt | `ruff check .`: pass; `pytest -q`: **40/40 pass** |
| Remote execution | Chưa live-run | Không có Kaggle/Modal job nào submit |

Đây là bằng chứng hạ tầng/reproducibility; **không phải kết quả Macro-F1 của khóa luận**.

## 5. Tiến độ chuẩn bị dữ liệu

Nguồn Sexual đã được handoff đánh giá PASS là `deepghs/nsfw_detect` (reported 28.000 ảnh, 5 nhãn gốc). Đã khóa revision:

```text
ac763cb1e1557225168be3b6b6b1ee864c17bc36
```

Để kiểm nhanh thay vì tải ZIP 1.68 GiB, Hub đọc archive qua HTTP range và lấy subset xác định được bằng seed:

| Nhãn gốc | Ảnh kiểm tra |
|---|---:|
| drawings | 10 |
| hentai | 10 |
| neutral | 10 |
| porn | 10 |
| sexy | 10 |
| **Tổng** | **50** |

Kết quả kỹ thuật: 50/50 ảnh đọc được. Artifact review nằm ở:

- `outputs/reviews/deepghs_nsfw/inventory.json`
- `outputs/reviews/deepghs_nsfw/contact-sheet.jpg`

Quan sát quan trọng: nhãn Sexual có signal rõ; nhưng `neutral + drawings` mới là negative của NSFW, **chưa đủ điều kiện gọi là Safe** vì chưa chứng minh non-violent. Nguồn này cũng không có Violence. Vì vậy chưa được freeze thành canonical dataset.

## 6. Hạng mục chưa thực hiện và lý do

Chưa train frozen baseline hoặc LoRA. Đây là quyết định có chủ đích, không phải thiếu hạ tầng:

1. Chưa có thành phần Violence đã review đủ chắc: VIMD và WeaponDetection còn là candidate; WeaponDetection phải loại ảnh chỉ có súng/dao mà không có ngữ cảnh bạo lực.
2. Chưa có Safe được xác minh đồng thời non-sexual và non-violent.
3. Chưa có canonical manifest, split cố định, duplicate/group audit, model revision bất biến.

Nếu huấn luyện ngay với deepghs đơn lẻ hoặc tự gọi non-NSFW là Safe, Macro-F1 có thể xuất hiện nhanh nhưng không trả lời được câu hỏi nghiên cứu 3-class một cách bảo vệ được.

## 7. Kế hoạch thực hiện tiếp theo

| Thứ tự | Action | Deliverable / điều kiện dừng |
|---:|---|---|
| 1 | Review và lưu evidence cho subset deepghs | PASS WITH LIMITATION; chỉ dùng làm Sexual/negative candidate |
| 2 | Validate VIMD | 20–50 sample review, count, provenance/video-group, PASS/REJECT |
| 3 | Validate WeaponDetection subset theo labels ngữ cảnh | Rule lọc tái lập được; không dùng weapon-only làm Violence |
| 4 | Bổ sung Safe + Violence thiếu coverage | Manifest 3-class với provenance và review status |
| 5 | Audit, deduplicate, freeze split | Canonical dataset fingerprint cố định |
| 6 | Chạy frozen baseline local pilot | Validation Macro-F1, per-class metrics, VRAM, training time |
| 7 | Review pilot rồi mới train LoRA | So sánh công bằng với baseline |

Mốc “kết quả ML đầu tiên” là **frozen baseline local pilot sau canonical freeze**, không phải metric trên subset inspection.

## 8. Nội dung cần xin ý kiến giảng viên

1. Đồng ý giữ single-label `Safe / Sexual / Violence` và không mở rộng sang multi-label ở giai đoạn này?
2. Đồng ý đặt tiêu chuẩn Safe = non-sexual **và** non-violent, dù điều này làm chậm bước chuẩn bị data?
3. Với Violence, có chấp nhận composition theo subtype (fight, weapon-context aggression, blood/gore, crowd violence) nhưng vẫn báo cáo một nhãn Violence duy nhất?
4. Ưu tiên đánh đổi nào khi nguồn thực tế còn thiếu: giữ scope 3-class chặt chẽ, hay thu hẹp nhiệm vụ có ghi rõ thay đổi research question?

## Kết luận

Hub và local GPU đã sẵn sàng, có guardrails để kết quả tái lập và không lẫn smoke test với evidence nghiên cứu. Tiến độ đang bị chặn đúng chỗ: chất lượng/cấu thành dataset, không phải compute. Bước tiếp theo là hoàn tất evidence cho Safe và Violence, freeze canonical dataset, rồi chạy frozen baseline pilot trước LoRA.
