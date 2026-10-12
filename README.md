# FAVI Training Hub

Repo này điều phối nghiên cứu phân loại ảnh nhạy cảm `Safe / Sexual / Violence`:

```text
canonical dataset + frozen split
→ frozen encoder + trained linear head
→ local GPU pilot 5–10% optimizer steps
→ review đúng fingerprint
→ Modal calibration + full training
→ locked validation selection
→ held-out test evaluation
```

Nguồn nghiệp vụ nằm ở `materials/FAVI_DATASET_HANDOFF_final.md`; quyết định scaffold và giới hạn
nằm ở `SCAFFOLD_PLAN.md`. `WORKLOG.md` là điểm vào cho session mới.

## Trạng thái hiện tại

Scaffold local đã có; nghiên cứu thật vẫn ở M1 dataset composition. Chưa có canonical dataset/split,
model revision, canonical dataset hay Modal budget được khóa. Fixture và smoke result không phải bằng
chứng luận văn. Không submit cloud cho đến khi `doctor` và các gate research đạt.

## Setup

Yêu cầu Python 3.11 và `uv`.

```powershell
uv sync --group dev
uv run favi doctor
uv run favi run smoke
uv run ruff check .
uv run pytest -q --ignore=tests/test_models.py
```

Để cài môi trường training trên NVIDIA GPU local và kiểm tra model integration:

```powershell
uv sync --extra train --extra remote --group dev
uv run favi doctor
uv run pytest -q tests/test_models.py
```

`doctor` phải báo `local_pilot_ready: true`, tên GPU và VRAM. Project dùng PyTorch CUDA 12.8 trên
Windows/Linux; driver NVIDIA có thể mới hơn CUDA runtime đi kèm wheel. Nếu `available: false`, chưa chạy
pilot: đó thường là PyTorch CPU hoặc driver chưa nhận GPU.

`uv.lock` được commit để local, Kaggle và Modal dùng cùng dependency set. `.env.example` chỉ liệt kê
tên secret; đặt giá trị trong shell hoặc secret manager, không ghi vào repo.

## Dataset workflow

1. Điền `local_path`, immutable revision, license evidence và provenance vào `data/sources.yaml`.
2. Inspect và lấy seeded sample để review:

```powershell
uv run favi data inspect --source vimd
uv run favi data sample --source vimd --count 50 --seed 42
```

Với archive Hugging Face lớn, có thể lấy inspection subset qua HTTP range thay vì tải toàn bộ ZIP.
Ví dụ deepghs (sau `hf auth login`) lấy 10 ảnh mỗi nhãn vào thư mục Git-ignore:

```powershell
uv run python scripts/fetch_hf_zip_subset.py `
  --repo deepghs/nsfw_detect `
  --revision ac763cb1e1557225168be3b6b6b1ee864c17bc36 `
  --file nsfw_dataset_v1.zip --labels drawings hentai neutral porn sexy `
  --per-label 10 --seed 42 --out data/raw/deepghs_nsfw_subset
```

Subset chỉ phục vụ inspect/review source; không được coi là canonical training dataset.

3. Lưu review text/JSON trong `data/reviews/`, rồi cập nhật source thành `pass` hoặc
   `pass_with_limitation`. Contact sheet và ảnh ở `outputs/`, đã ignore.
4. Tạo JSONL theo contract trong `data/manifests/README.md`, audit rồi freeze:

```powershell
uv run favi data audit --manifest data/manifests/canonical-v1.jsonl
uv run favi data freeze --manifest data/manifests/canonical-v1.jsonl `
  --out data/manifests/canonical-v1.frozen.json --version canonical-v1 --seed 42
```

Safe cần xác minh cả `non_sexual=true` và `non_violent=true`. Video frames, exact duplicate và
duplicate cluster phải ở cùng split. Test set đã freeze không tham gia pilot, tuning hay selection.

## Chuẩn bị experiment

Khóa `configs/protocol.yaml` trước: seed, split, dataset fingerprint, model ID, immutable model commit
và input resolution. Với LoRA, inspect checkpoint rồi chọn target modules rõ ràng.

```powershell
uv run favi run prepare --config configs/frozen.yaml
uv run favi run prepare --config configs/lora.yaml
```

`prepare` snapshot config/code hash. Nếu đổi code hoặc training semantics, tạo experiment mới. Retry
cùng config tạo run attempt mới; không sửa/ghi đè result cũ.

## Local pilot (mặc định)

Pilot là một lần chạy ngắn bằng 5–10% số bước tối ưu của full run. Mục đích của nó giống integration
test có dữ liệu thật: xác nhận model học đúng tham số, validation chạy đủ ba lớp và đo thời gian/VRAM
trước khi trả tiền cho Modal. Nó không phải kết quả cuối của luận văn.

Sau khi `prepare` trả về experiment ID, chạy:

```powershell
uv run favi run local --experiment EXP-001 --data-root D:\favi-data
```

`--data-root` là thư mục gốc chứa các ảnh được nhắc bởi `image_uri` trong frozen manifest. Lệnh tạo
`pilot-01`, train trên CUDA, rồi ghi `run.json`, `history.csv`, `metrics.json`, checkpoint và artifact
trong `experiments/EXP-001/pilot-01/`. Thành công khi `run.json` có `status: completed`; lỗi vẫn được
giữ thành attempt `failed`, không bị ghi đè.

Nếu tiến trình bị ngắt sau khi đã có checkpoint cùng identity:

```powershell
uv run favi run local --resume-run EXP-001/pilot-01 --data-root D:\favi-data
```

Nếu RTX 4060 8 GB hết VRAM, trước tiên tạo experiment mới với `batch_size: 1` và tăng
`accumulation_steps` để giữ effective batch. Nếu vẫn không đủ, dùng Kaggle fallback; không âm thầm đổi
config của experiment đã snapshot.

## Kaggle pilot (fallback)

Kaggle chỉ dùng khi GPU local không đủ VRAM hoặc môi trường local bị chặn. Runner dùng script kernel
qua CLI, không yêu cầu mở notebook mỗi lần. Thiết lập một lần:

- `kaggle auth login` hoặc `KAGGLE_API_TOKEN`.
- Secret `HF_TOKEN` trong Kaggle và cho kernel quyền dùng secret.
- Dataset Hub repo/revision private phù hợp với license.
- `remote.kaggle_owner`, accelerator, HF dataset repo/revision trong config trước khi `prepare`.

Kiểm tra trước khi submit:

```powershell
uv run favi run submit --backend kaggle --experiment EXP-001 --dry-run
uv run favi run submit --backend kaggle --experiment EXP-001
uv run favi run status --run EXP-001/pilot-01
uv run favi run collect --run EXP-001/pilot-01
```

Nếu submit bị ngắt, run chuyển thành `submission_unknown`; dùng `status`/Kaggle UI để reconcile trước
khi tạo job khác. Collect kiểm tra run/config/code/dataset identity và chạy lặp lại an toàn.

Review sau pilot local hoặc Kaggle:

```powershell
uv run favi run review --run EXP-001/pilot-01 `
  --decision ready_for_modal --reviewer "Tên reviewer" `
  --reason "Pipeline, gradient, validation coverage và resource evidence đạt"
```

Pilot thấp hơn baseline chưa tự động bị loại; có thể là `inconclusive` nếu còn warmup hoặc thiếu signal.
Pilot không dùng test set.

## Modal full run

Thiết lập Modal token và tạo private secret `favi-hf` chứa `HF_TOKEN`. Điền target GPU, calibration
steps, `max_hours`, `max_cost_usd`, tổng estimated hourly rate và ngày kiểm tra rate vào config trước
khi `prepare`. Rate là guard do người dùng cung cấp; calibration đo throughput/VRAM trên GPU đích.

```powershell
uv run favi run submit --backend modal --experiment EXP-001 --dry-run
uv run favi run submit --backend modal --experiment EXP-001
uv run favi run status --run EXP-001/full-01
uv run favi run collect --run EXP-001/full-01
```

Modal chạy calibration từ initial state, kiểm budget rồi reset model/optimizer/RNG trước full run.
Checkpoint nằm trên persistent Volume; runner không resume pilot local/Kaggle sang Modal.

## Evaluation và report

Khi frozen và LoRA full runs đã có validation evidence tương thích, khóa selection trước khi đọc test:

```powershell
uv run favi finalize --runs EXP-001/full-01 EXP-002/full-01 `
  --reviewer "Tên reviewer" --reason "Chọn theo protocol validation đã khóa"
uv run favi evaluate --run EXP-002/full-01 --split test `
  --data-root D:\favi-data --artifact D:\favi-artifact
uv run favi report
```

Report được sinh từ run records vào `experiments/index.csv` và `results/comparison.md`. Không nhập
metric thủ công vào nhiều nơi. Một seed không đủ để tuyên bố statistical significance.

## Recovery

- Đọc `WORKLOG.md` và `scratch/CURRENT.md`.
- Với remote job dở, đọc `run.json` và gọi `status`; không submit lại khi trạng thái chưa rõ.
- Với local pilot bị ngắt, dùng `run local --resume-run`; checkpoint phải có identity trùng.
- Giá trị chưa đo được để null/vắng mặt có lý do; không dùng 0 thay thế.

Remote integration hiện mới được kiểm tra bằng static CLI/dry-run và mocks. Chỉ ghi “live verified”
sau lần chạy thật có account, dataset, artifact và budget hợp lệ.
