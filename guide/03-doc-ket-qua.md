# Cách đọc một kết quả train

## Metric không phải lời phán quyết tuyệt đối

Metric là phép so sánh dự đoán model với label có sẵn trên một split. Nó chỉ đáng tin trong phạm vi: dataset, nhãn, split và protocol đang dùng.

Ví dụ: “Macro-F1 0,87 trên validation P0” nghĩa là model khớp khá tốt với nhãn `NSFW/Non-NSFW` của 400 ảnh validation P0. Nó không có nghĩa là model đạt 87% trên Internet, và không nói gì trực tiếp về `Safe/Sexual/Violence`.

## Accuracy và Macro-F1

- **Accuracy**: tỉ lệ tổng số ảnh model đoán đúng.
- **Precision của NSFW**: các ảnh model gọi là NSFW có bao nhiêu ảnh thực sự NSFW.
- **Recall của NSFW**: trong ảnh thực sự NSFW, model tìm được bao nhiêu ảnh.
- **F1**: cân bằng giữa precision và recall.
- **Macro-F1**: lấy F1 của từng lớp rồi trung bình đều. Nó không để lớp nhiều ảnh lấn át lớp ít ảnh.

Trong P0 hai lớp cân bằng, Macro-F1 dễ kể: model có làm ổn với cả `NSFW` lẫn `Non-NSFW` không?

## Confusion matrix

Confusion matrix trả lời model nhầm như thế nào. Với P0:

```text
                    Dự đoán Non-NSFW   Dự đoán NSFW
Thực tế Non-NSFW              TN                FP
Thực tế NSFW                  FN                TP
```

- `FP` cao: ảnh không NSFW bị báo NSFW quá nhiều.
- `FN` cao: ảnh NSFW bị bỏ sót quá nhiều.

Đây thường hữu ích hơn chỉ nói “điểm 0,87”.

## Cách nói đúng trong báo cáo

Có thể nói:

> Trên validation P0 gồm 400 ảnh, frozen baseline đạt Macro-F1 X. Dataset version, source revision, model revision và code/config fingerprint đều được lưu trong artifact.

Chỉ sau test mới nói:

> Trên held-out test gồm 400 ảnh chưa dùng để chọn cấu hình, model đã khóa đạt Macro-F1 Y.

Không nói “model chính xác Y% trong thực tế” nếu chỉ mới đo trên dataset P0.
