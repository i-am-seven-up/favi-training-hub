# FAVI Training Hub — kế hoạch scaffold

Ngày: 2026-10-11. Trạng thái: **scaffold v1 đã triển khai; remote live chưa kiểm chứng**.

Nguồn nghiệp vụ: `materials/FAVI_DATASET_HANDOFF_final.md`.
Flow đã thống nhất: local smoke → local GPU pilot 5–10% → review → Modal full training → thu kết quả. Kaggle là fallback khi GPU local không đủ VRAM hoặc môi trường local bị chặn.
Tài liệu này là thiết kế của scaffold hiện tại; không đồng nghĩa cho phép khởi chạy GPU hoặc upload dataset ngay.

## 1. Kết quả cần đạt

Một repo Python gọn, dùng cùng code train/evaluate trên local, Kaggle và Modal. Mỗi action để lại kết quả có thể kiểm chứng. Agent mới đọc file trong repo là biết mục tiêu, đang ở bước nào, dữ liệu ở đâu, job nào đang chạy và action tiếp theo.

Scaffold được nghiệm thu khi luồng local bằng fixture chạy được, các gate chặn đúng trường hợp chưa sẵn sàng, và runner remote có thể chuẩn bị job/đọc cấu hình mà không cần GPU. Remote integration chỉ được ghi là đã kiểm chứng sau khi có lần chạy thật.

Không làm web dashboard, hệ thống serving, retrieval, sweep tự động hoặc full fine-tuning trong scaffold v1.

## 2. Những quyết định lấy từ handoff

| Nội dung | Quyết định |
|---|---|
| Bài toán | Phân loại ảnh, single-label: Safe / Sexual / Violence |
| Backbone | Một pretrained VLM thuộc họ CLIP hoặc SigLIP; checkpoint cụ thể chưa chọn |
| Baseline chính | Frozen pretrained image encoder + linear classification head được train |
| Phương pháp nghiên cứu | LoRA trên cùng encoder + cùng kiến trúc classification head |
| Primary metric | Macro-F1; kết quả chính là Macro-F1(LoRA) − Macro-F1(Frozen) |
| Secondary metrics | Precision/recall/F1/support từng lớp, confusion matrix, accuracy |
| Efficiency | Tổng/trainable parameters, tỷ lệ trainable, peak VRAM, thời gian train |
| Dataset | Mục tiêu thực dụng khoảng 12k–15k ảnh; chất lượng ưu tiên hơn số lượng |
| Tiến độ thực tế | M1: hoàn thiện thành phần Violence/Safe; chưa có canonical dataset/split frozen được xác minh trong repo |

Không thay baseline thành zero-shot. Không coi frozen baseline có chi phí train bằng 0: classification head vẫn phải được train. Full fine-tuning là tùy chọn nghiên cứu về sau, không phải điều kiện bắt buộc trước LoRA.

Không chốt model ID, target modules, license, dataset URI, split ratio, budget hoặc metric threshold bằng phỏng đoán. Những trường chưa có bằng chứng được để trống có trạng thái rõ ràng.

## 3. Cấu trúc đích

```text
favi-training-hub/
├─ AGENTS.md                      # Luật ngắn, đường dẫn đọc đầu phiên
├─ README.md                      # Setup và các action thường dùng
├─ SCAFFOLD_PLAN.md                # Kế hoạch này; checklist triển khai
├─ WORKLOG.md                      # Current stage/action/result/next/blockers
├─ .env.example                   # Tên biến; không chứa token
├─ .gitignore
├─ pyproject.toml                  # Dependencies core/train/remote/dev
├─ uv.lock                        # Khóa dependency sau khi resolve được
├─ materials/
│  └─ FAVI_DATASET_HANDOFF_final.md # Nguồn gốc, giữ nguyên
├─ data/
│  ├─ sources.yaml                # Nguồn, trạng thái review, URI, license
│  ├─ reviews/                    # Biên bản duyệt source/sample
│  ├─ manifests/                  # Schema, manifest canonical, split IDs
│  └─ sample/                     # Fixture phi nhạy cảm, không phải research data
├─ configs/
│  ├─ protocol.yaml               # Label IDs, metric, seed, data/model revision
│  ├─ frozen.yaml                 # Cấu hình baseline dự kiến
│  └─ lora.yaml                   # Cấu hình LoRA dự kiến
├─ src/favi_hub/
│  ├─ __init__.py
│  ├─ __main__.py                 # CLI: python -m favi_hub
│  ├─ config.py                   # Validate/resolve/hash cấu hình
│  ├─ data.py                     # Inspect, manifest, sampling, leakage audit
│  ├─ models.py                   # Image encoder, head, LoRA attachment
│  ├─ train.py                    # Train chung; pilot/full khác run budget
│  ├─ evaluate.py                 # Một metric implementation dùng chung
│  ├─ runs.py                     # Run records, state, gate, checkpoint metadata
│  └─ report.py                   # Sinh index và comparison từ result
├─ runners/
│  ├─ kaggle/                     # Entry script + metadata template + CLI wrapper
│  └─ modal_app.py                # Remote wrapper + environment + volume
├─ experiments/
│  ├─ index.csv                   # Bảng tổng hợp sinh từ run records
│  └─ EXP-001/
│     ├─ config.yaml             # Snapshot cấu hình đã resolve, bất biến
│     ├─ pilot-01/               # run.json, metrics.json, history.csv, review.json
│     └─ full-01/                # run.json, metrics.json, history.csv
├─ results/
│  └─ comparison.md               # Bảng + claim gắn experiment/artifact
├─ scratch/
│  └─ CURRENT.md                  # Nháp task đang làm, có action/result/next
└─ tests/
   ├─ fixtures/                   # Synthetic records/ảnh phi nhạy cảm
   ├─ test_data.py
   ├─ test_metrics.py
   ├─ test_gate.py
   └─ test_workflow.py
```

Các thư mục EXP/run trong cây là cấu trúc khi có experiment, không tạo sẵn kết quả giả. Scaffold khởi tạo registry rỗng và template/example ghi rõ là example.

Raw dataset, ảnh review/contact sheet, cache model, checkpoints, predictions lớn, logs lớn nằm trong storage/cache ngoài Git hoặc thư mục local được ignore. Metadata nhỏ và bằng chứng review dạng text được version trong repo. Không cần DVC, database hay W&B ở v1.

## 4. Nguồn sự thật và bộ nhớ

| Cần biết gì | Đọc ở đâu |
|---|---|
| Mục tiêu và phạm vi nghiên cứu | Handoff; tóm tắt vận hành trong README |
| Protocol đang dùng | `configs/protocol.yaml` |
| Cần dataset nào, lấy ở đâu, duyệt chưa | `data/sources.yaml` và review được trỏ tới |
| Dataset/split chính xác | Manifest version + split IDs + checksums |
| Hiện đang làm gì | `WORKLOG.md` |
| Task bị ngắt cần tiếp tục | `scratch/CURRENT.md` và run record được tham chiếu |
| Job remote đang ở đâu | `run.json`: backend, remote ID, status, artifact URI |
| Vì sao cho chạy Modal | `review.json` gắn đúng pilot/config/protocol |
| Số liệu gốc | `metrics.json`, `history.csv` của từng run |
| So sánh hiện tại | `experiments/index.csv`, `results/comparison.md` |

`WORKLOG.md` giữ ngắn: milestone hiện tại, last action, result/evidence path, next command/action, running jobs, blockers và quyết định còn hiệu lực. Không chép transcript session.

`scratch/CURRENT.md` chỉ giữ kế hoạch tạm, phát hiện có bằng chứng, lệnh đã chạy và bước tiếp theo. Cập nhật trước/sau bước dài hoặc submit remote job. Khi hoàn thành, chuyển kết quả bền vững về run record/WORKLOG rồi rút gọn nháp. Không lưu suy luận nội bộ dài.

Agent bắt đầu bằng AGENTS → WORKLOG → scratch nếu có task dở → record được trỏ tới. Đọc handoff khi tiếp nhận repo hoặc khi cần kiểm tra phạm vi/semantics. File thiếu được báo rõ; không dựa vào session cũ để tự điền kết quả.

## 5. Dataset readiness trước training

### 5.1. Khởi tạo source registry từ handoff

| Source/nhóm | Trạng thái ban đầu | Việc tiếp theo |
|---|---|---|
| deepghs/nsfw_detect | PASS được báo cáo trong handoff; 28.000 ảnh, 5 lớp × 5.600 | Xác minh revision/license/location; duyệt nhãn canonical |
| onullusoy/harmful-contents | REJECT làm nguồn Violence chính | Chỉ dùng bổ sung nếu có curated subset được duyệt riêng |
| NeuralShell/Gore-Blood-Dataset-v1.0 | PASS WITH LIMITATION được báo cáo; khoảng 398 ảnh stylized | Lưu limitation; xác minh ZIP/version, trùng lặp giữa các archive |
| VIMD | CANDIDATE | Xác định đúng nguồn, fetch/inspect/count, review khoảng 50 ảnh |
| Subh775/WeaponDetection | CANDIDATE | Xác minh schema, lọc violent-context candidates, review khoảng 50 ảnh |
| Real-world injury/blood; crowd violence | MISSING | Tìm nguồn và review |
| Non-Violence; hard negatives hai nhóm | MISSING / NEED ENRICHMENT | Tìm nguồn và duyệt canonical labels |

Mỗi nguồn ghi: ID, URL/URI, revision/checksum, license/evidence, original task/labels, counts, limitations, grouped/video origin, review status, review evidence path. Phân biệt `reported_in_handoff` với `verified_in_this_repo`; không đánh dấu tải/review xong khi chưa làm.

### 5.2. Data actions và đầu ra

1. Inspect source → inventory, counts, schema và danh sách lỗi đọc ảnh.
2. Sample 20–50 ảnh cố định theo seed → danh sách image IDs + contact sheet local để con người review.
3. Record review → PASS / PASS_WITH_LIMITATION / REJECT, reviewer, thời điểm, lý do, giới hạn được chấp nhận.
4. Build manifest → chỉ lấy sample có mapping/review phù hợp; giữ provenance.
5. Audit → exact duplicates, near-duplicate candidates, group leakage, phân bố label/source/subtype.
6. Freeze split → lưu sample IDs/checksums, dataset fingerprint và audit report; các action sau dùng lại split này.

Manifest mỗi ảnh: `sample_id`, `image_uri`, `sha256`, `canonical_label`, `original_label`, `source_id`, `source_revision`, `group_id`, `subtype`, `review_status`, `split`.

Canonical IDs đề xuất để khóa trong protocol: Safe=0, Sexual=1, Violence=2. Không suy ra IDs từ thứ tự folder. Với nguồn frame/video, thiếu group provenance phải được giải quyết hoặc loại khỏi canonical split; không random-split frame.

Exact duplicates và các near-duplicate đã xác nhận phải nằm cùng một nhóm trước split. Near-duplicate detection tạo candidate để review; không tuyên bố tự động phát hiện hết cropped/resized/reuploaded images. Ghi giới hạn audit.

Safe phải được xác minh đồng thời non-sexual và non-violent. `neutral/drawings` hoặc `non-violence` không tự động map thành Safe. Weapon-context annotations chỉ dùng lấy candidates, không thay thế review semantics.

Ảnh có cả Sexual và Violence hoặc nhãn mơ hồ: đưa vào hàng chờ review, chưa đưa vào train; không tự tạo quy tắc ưu tiên lớp. Ranh giới `sexy` với swimwear/fashion, medical injury với violence, combat sports với fighting cần annotation examples được chốt trước canonical freeze. Không tự mở rộng sang self-harm, suicide hay corpse-only imagery.

Readiness gate yêu cầu đủ 3 lớp, review/license có căn cứ, split không rỗng với coverage cần thiết, audit đạt và dataset fingerprint đã khóa. Không ép đạt đúng 4k–5k/lớp bằng dữ liệu kém chất lượng.

## 6. Protocol training và pilot local

### 6.1. Cách hiểu 5–10%

Mặc định đề xuất: **pilot chạy 10% số optimizer steps dự kiến của full run**, cấu hình được trong khoảng 5–10%. Giữ dataset train và sampler của full run; ghi số ảnh duy nhất, số tokens/pixels nếu phù hợp, số batches và optimizer steps thực tế đã xử lý.

Không đồng thời giảm train data còn 10% rồi giảm steps thêm 10%: như vậy pilot chỉ còn khoảng 1% khối lượng. Nếu cần chế độ subset vì I/O, chọn 5–10% train có stratification theo label và kiểm tra source/subtype coverage, giữ lịch epoch đã định; ghi rõ `pilot_basis=data_fraction`. Chế độ mặc định là `pilot_basis=optimizer_steps`.

Pilot mặc định giữ lịch learning-rate/warmup tính theo full run rồi dừng sớm, để quan sát phần đầu của training dự kiến. Nếu pilot kết thúc trong warmup hoặc chưa đủ validation signal, kết luận `inconclusive`; không coi đó là bằng chứng phương pháp không hiệu quả. Bất kỳ schedule thay thế cho pilot phải được ghi riêng và giới hạn suy luận rõ.

Pilot và full dùng cùng code, checkpoint revision, canonical mapping, image preprocessing/resolution, train/val/test version, kiến trúc head, seed, LoRA modules và metric implementation. Backend/GPU có thể khác; microbatch được thay đổi khi cần nhưng phải ghi gradient accumulation/effective batch và kiểm tra tương thích. Thay effective batch/precision/augmentation/schedule có ý nghĩa học thuật thì phải review lại.

Validation dùng subset validation cố định đủ coverage hoặc toàn validation; luôn ghi IDs/fingerprint và kích thước. **Không đọc test set trong pilot, review hoặc chọn cấu hình.**

### 6.2. Frozen baseline và LoRA

- Frozen: đóng băng image encoder; optimizer chỉ cập nhật head. Encoder mode/augmentation phải được kiểm soát và ghi rõ.
- LoRA: cùng image features và head architecture; optimizer cập nhật adapter + head theo config. Text encoder không là biến nghiên cứu ở v1.
- Kiểm tra trainable parameter names/counts và gradient flow; fail nếu adapter không attach hoặc vô tình unfreeze toàn backbone.
- Chọn một backbone/checkpoint khi triển khai integration thật. Scaffold có interface chung nhưng không xây đầy đủ hai stack CLIP và SigLIP cùng lúc.
- Baseline head và LoRA head dùng cùng quy tắc khởi tạo/seed. Mặc định mỗi full run bắt đầu từ pretrained backbone + head initialization, không âm thầm kế thừa head đã train từ baseline.
- Cả baseline có train head và từng cấu hình LoRA mới đều qua local pilot. Kaggle được chấp nhận như fallback có cùng protocol. Pure evaluation/report không cần một training pilot mới.

### 6.3. Result tối thiểu của pilot

`run.json`: experiment/run ID, stage, backend, timestamps/status, config/protocol/code/data fingerprints, seed, model revision, environment versions, GPU, remote job ID, artifact URI, error nếu có.

`metrics.json`: split/fingerprint, số mẫu evaluate, macro-F1, per-class precision/recall/F1/support, confusion matrix đúng label order, accuracy, total/trainable parameters, peak CUDA allocated/reserved VRAM, train wall time, setup/download time tách riêng, throughput images/s, steps/s, loss đầu/cuối, completed steps, finite-gradient/NaN/OOM status.

`history.csv`: step, learning rate, train loss; validation loss/metrics tại các mốc evaluate. Giá trị chưa đo là null hoặc vắng mặt có lý do; không đặt 0 để lấp chỗ trống. Baseline frozen vẫn có VRAM và train time.

Ước tính full run dựa trên steady-state throughput sau warmup, số steps còn lại, evaluation/checkpoint/I/O overhead. Cung cấp khoảng ước lượng và assumptions. **Không nhân thời gian local/Kaggle theo tỷ lệ đơn giản thành thời gian Modal** khi GPU khác nhau.

### 6.4. Review gate

`review.json` ghi reviewer, thời điểm, decision, bằng chứng, unresolved concerns, fingerprint của pilot/config/protocol, target GPU, resource estimate và budget áp dụng.

| Decision | Điều kiện/ý nghĩa |
|---|---|
| `revise` | Pipeline lỗi, leakage, label sai, NaN, adapter không học, OOM hoặc config cần sửa |
| `inconclusive` | Chưa đủ tín hiệu: pilot ngắn, thiếu lớp, mới warmup, learning curve chưa rõ |
| `ready_for_modal` | Pipeline/data gate đạt, learning signal hợp lý, resources khả thi, có budget đã xác định |
| `stop` | Có bằng chứng không nên tiếp tục experiment theo mục tiêu/nguồn lực hiện tại |

Không bắt buộc pilot LoRA vượt baseline full để được tiếp tục: khác training budget không phải so sánh công bằng. Có thể tham chiếu baseline pilot trên cùng validation protocol và nêu rõ mức độ bất định.

Review hợp lệ gắn với đúng fingerprint. Thay dữ liệu, model, LoRA config hoặc training semantics làm approval cũ hết hiệu lực và phải chạy lại pilot phù hợp.

Trước full training, runner Modal chạy đoạn calibration ngắn trên GPU đích để đo VRAM/throughput, cập nhật estimate. Nếu vượt budget hoặc sai khác lớn so với assumptions, dừng tại gate để revise. Budget thiếu thì không tự submit full run.

Full run mặc định train mới từ initialization đã khóa. Calibration không được âm thầm trở thành một phần training: reset model/optimizer/RNG trước full run hoặc lưu và mô tả rõ cơ chế tiếp tục có kiểm chứng.

Checkpoint resume sau interruption phải phục hồi adapter/head, optimizer, scheduler, scaler nếu có, RNG, bước hiện tại và data-order state phù hợp. Adapter dùng inference riêng không đủ để cam kết resume training chính xác. Không mặc định resume pilot local/Kaggle sang full Modal.

## 7. Remote execution và storage

Local runner là pilot path mặc định: prepare experiment → chạy CUDA local → review các artifact trong run folder. Kaggle runner là fallback CLI-first: prepare package/metadata → submit → status → collect. Khi dùng fallback, đối chiếu Kaggle CLI/API chính thức và kiểm tra tài khoản để xác nhận accelerator, Internet, secrets, quyền dataset và quota.

Modal wrapper chỉ phụ trách environment/GPU/volume/secrets và gọi code chung. Có dry-run để inspect request và gate trước submit. Tham số/API Modal cụ thể được xác minh với docs lúc triển khai; không xem các tên command trong kế hoạch là API có sẵn.

| Nơi lưu | Nội dung |
|---|---|
| Git/repo local | Code, lockfile, config, manifests nhỏ, review, results nhỏ, WORKLOG |
| HF private dataset/model repos | Dataset đã được phép upload, adapter + classification head + processor/config; revision cụ thể |
| Kaggle input/output | Chỉ cho pilot fallback: input có version và output mapping trở lại dataset fingerprint |
| Modal Volume | Cache dataset/model và checkpoint đang train; commit/persist theo API được xác minh |
| Local ignored cache/output | Downloads, contact sheets, predictions lớn, temporary bundles |

Không upload lại toàn dataset mỗi run; sử dụng revision/cache đã xác minh. V1 chỉ tích hợp backend storage cần thiết, không dựng abstraction S3/Drive nhiều nhánh. Dataset nhạy cảm chỉ chuyển tới nơi được phép theo provenance/license và quyền truy cập.

Token lấy qua môi trường hoặc secret manager của backend. `.env.example` chỉ nêu tên biến, README chỉ dẫn cấu hình. Không ghi token vào notebook metadata, CLI output, result hoặc Git.

Submit ghi remote ID vào run record ngay; session sau `status/collect` job cũ trước khi submit mới. Collect có thể chạy lại, xác minh IDs/fingerprints, không ghi đè run khác. Ghi rõ timeout/failed/cancelled, giữ partial evidence; không tự retry training vô hạn.

Checkpoint định kỳ lên persistent storage; final artifact gồm adapter, head, label mapping, preprocessing/model revisions và resolved config. Với frozen baseline, lưu trained head và backbone revision. Có checksum/artifact URI để reload/evaluate.

## 8. State và báo cáo

Research milestones:

```text
M0 scaffold → M1 canonical dataset + frozen split
→ M2 frozen baseline → M3 LoRA → M4 locked final evaluation + comparison
```

Experiment lifecycle:

```text
planned → pilot_running → pilot_completed → reviewed
reviewed(revise/inconclusive) → new pilot attempt
reviewed(ready_for_modal) → modal_calibration → full_running
→ full_completed → validation_evaluated → selected/archived
selected → final_test_evaluated → reported
```

Mọi stage có nhánh failed/cancelled/blocked. Đổi hypothesis/config đáng kể tạo EXP mới và `parent_experiment_id`; retry cùng config tạo attempt mới. Không ghi đè result cũ. Final-test dùng cho phương pháp đã khóa sau validation, không dùng để quyết định rank/alpha/epoch tiếp theo.

`experiments/index.csv` đề xuất một dòng mỗi run: experiment_id, run_id, method, stage, backend, status, config_hash, dataset_version, eval_split, macro_f1, trainable_params, peak_vram_gb, train_seconds, artifact_uri, review_decision. Các kết quả pilot/validation/test được tách cột/filter rõ ràng.

Index và comparison được sinh lại từ run records, không nhập metric bằng tay vào nhiều file. Comparison kiểm tra protocol compatibility trước tính delta; phần hardware/time khác máy được chú thích, không gộp thành kết luận hiệu suất tuyệt đối. Claims gắn run IDs; không tuyên bố statistical significance từ một seed. Multi-seed/CI là mở rộng sau, khi budget cho phép.

## 9. CLI dự kiến và action-result contract

Đây là giao diện cần xây, **chưa phải lệnh đang chạy được**. Dùng Python CLI trên Windows local và Linux remote; ưu tiên tránh shell-specific scripts.

| Action dự kiến | Output kiểm chứng |
|---|---|
| `python -m favi_hub doctor` | Environment/config readiness; thiếu token/data hiển thị rõ, không in secret |
| `python -m favi_hub data inspect --source SOURCE` | Inventory/count/schema report |
| `python -m favi_hub data sample --source SOURCE --count 50` | Seeded sample IDs + local review sheet |
| `python -m favi_hub data audit --manifest PATH` | Duplicate/group/label/source coverage report |
| `python -m favi_hub data freeze --manifest PATH` | Frozen split manifest + fingerprint; chặn unresolved issues |
| `python -m favi_hub run prepare --config PATH` | EXP ID + resolved config + fingerprints |
| `python -m favi_hub run smoke --config PATH` | Local fixture pipeline report |
| `python -m favi_hub run local --experiment EXP --data-root PATH` | Local CUDA pilot metrics/checkpoint/artifact; persisted state |
| `python -m favi_hub run submit --backend kaggle --experiment EXP` | Fallback remote pilot job ID; persisted state |
| `python -m favi_hub run status --run RUN` | Reconciled remote status |
| `python -m favi_hub run collect --run RUN` | Verified result/artifact references |
| `python -m favi_hub run review --run RUN` | Review record với bằng chứng/decision |
| `python -m favi_hub run submit --backend modal --experiment EXP` | Gate result → calibration → full job khi đủ điều kiện |
| `python -m favi_hub evaluate --run RUN --split validation` | Metrics cùng schema; test yêu cầu finalization gate |
| `python -m favi_hub report` | Regenerated index.csv + comparison.md |

Các action submit hỗ trợ `--dry-run`. Doctor/smoke/tests core không cần token, network, pretrained weights hay CUDA. Fixture/mock mode được đánh dấu rõ và bị loại khỏi research comparison/gate approval.

## 10. Trình tự triển khai sau khi được yêu cầu execute

| Bước | Việc cần xây | Điều kiện nghiệm thu |
|---|---|---|
| S1 | AGENTS, README, WORKLOG, scratch, config/schema, pyproject, ignore rules | Đọc repo biết đang M1 dataset; không có metric hoặc trạng thái hoàn thành giả |
| S2 | Source registry lấy từ handoff; inspect/sample/audit/freeze | Fixture chứng minh Safe không map tắt; duplicate/video-group không lọt giữa splits |
| S3 | CLI, run IDs, config hashing, results, gate, report | Local fixture đi hết action-result; invalid/stale/missing pilot chặn Modal |
| S4 | Một model backend; frozen/head/LoRA train/evaluate/checkpoint | Unit/integration nhỏ chứng minh parameter freezing, head/adapter gradients, reload nhất quán |
| S5 | Local pilot command + Kaggle fallback wrapper | Local CUDA run có state/resume; fallback dry-run hợp lệ; live checks ghi riêng |
| S6 | Modal wrapper, persistent cache/checkpoint, calibration, budget guard | Dry-run không submit; calibration/budget gates được test; live test ghi riêng |
| S7 | README hướng dẫn local smoke → local pilot → remote, dependency lock, report và recovery | Clean local setup chạy core checks; liệt kê rõ external prerequisites chưa xác minh |

Chọn dependency tối thiểu: Python, package manager/lock, YAML/config validation, Pillow/data utilities, metrics, PyTorch + Transformers/PEFT cho model integration, Kaggle/Modal/HF clients cho remote. Phiên bản tương thích được kiểm tra và pin lúc execute; tách core/dev khỏi train/remote để local checks nhẹ.

Không cài CUDA hoặc tải pretrained weights để chạy core smoke test. Test model nhỏ có thể dùng cấu hình khởi tạo ngẫu nhiên phi nghiên cứu để kiểm tra gradient/checkpoint; performance metric từ test đó tuyệt đối không đưa vào báo cáo luận văn.

## 11. Bộ kiểm nghiệm scaffold bắt buộc

- Manifest từ chối nhãn ngoài taxonomy, URI/sample ID trùng sai quy tắc, missing required provenance và unresolved canonical labels.
- Duplicates/group frames không xuất hiện chéo splits; sampling pilot không lấy validation/test.
- Metric fixture có kết quả tính tay, cố định label order và xử lý missing class/zero division có cảnh báo coverage.
- Frozen model chỉ update head; LoRA chỉ update adapter/head; trainable count đúng và adapter có gradient.
- Pilot budget 5–10% không bị giảm hai lần; validation coverage/schedule status được ghi vào review inputs.
- Gate từ chối missing/failed/mock/stale pilot, incompatible fingerprint, dataset chưa ready và budget chưa xác định.
- Loss finite nhưng metric thấp không bị tự động đánh đồng với failed pilot; hỗ trợ inconclusive với lý do.
- Checkpoint reload/resume có kiểm nghiệm tương xứng; không ghi claim exact resume nếu state chưa được phục hồi đầy đủ.
- Remote failed/cancelled/collect lại không tạo result done giả hoặc submit job trùng.
- Report tách pilot và full, validation và test; delta chỉ tính giữa run tương thích, thiếu metric giữ null.
- Core end-to-end fixture chạy offline trên local. Remote dry-run, mock integration và live integration được báo riêng.

Không cần test cho từng dòng tài liệu; kiểm nghiệm tập trung ở data leakage, model updates, gate và khả năng khôi phục run.

## 12. Ruleset AGENTS cần sinh khi execute

1. Trò chuyện tiếng Việt; đọc WORKLOG và task dở trước khi hành động.
2. Bám handoff: 3 lớp, frozen encoder + trained head là baseline; LoRA trên cùng encoder/head.
3. Mỗi task có action, result/evidence path, next action; cập nhật sau mốc quan trọng và trước kết thúc.
4. Giữ nguyên material/raw; nguồn chưa review không tự tích hợp vào canonical data.
5. Không tự map concept-negative thành Safe; không dùng weapon presence thay violent context.
6. Dataset/split/protocol có fingerprint; test không tham gia tuning/gate.
7. Mỗi training experiment phải có local pilot 5–10% và review hợp lệ trước Modal full; Kaggle chỉ là fallback.
8. Không suy số giờ Modal trực tiếp từ local/Kaggle; dùng calibration và budget rõ ràng.
9. Thay training semantics làm review cũ hết hiệu lực; giữ lịch sử attempts/results.
10. Không chạy remote trả phí vượt phạm vi/budget đã được người dùng giao.
11. Không bịa metric, license, source URL, download/review status hoặc thành công của remote job.
12. Final report nêu action, kết quả, checks đã chạy, phần chưa xác minh và bước tiếp theo.

## 13. Các lựa chọn còn mở và thời điểm cần chốt

| Cần chốt | Khi nào bắt buộc | Scaffold xử lý thế nào |
|---|---|---|
| Backbone/model revision + target modules | Trước model integration thật/pilot | Placeholder hợp lệ ở template; execution gate báo thiếu |
| Dataset source locations/licenses/revisions | Trước fetch/merge/upload tương ứng | Registry giữ unknown + action cần xác minh |
| Nhãn ambiguous/overlap và annotation examples | Trước canonical freeze | Review queue, không tự gán lớp |
| Split ratio và seed | Trước canonical freeze | Protocol ghi chưa khóa; fixture dùng giá trị riêng |
| Local CUDA/GPU; Kaggle account nếu cần fallback | Trước pilot thật | Doctor xác minh CUDA; fallback dùng dry-run, không giả định quyền Kaggle |
| Modal account/GPU và budget mỗi run/tổng | Trước calibration/full thật | Fail closed khi thiếu budget/quyền |
| HF private dataset/model repo IDs | Trước sync thật | URI placeholder, không tự tạo repo/upload |
| Validation review criteria | Trước pilot review | Structural gates có sẵn; không tự bịa ngưỡng Macro-F1 |

Những lựa chọn này không cản việc dựng skeleton và kiểm nghiệm local. Khi execute scaffold hoàn tất, next research action vẫn là xác minh VIMD và WeaponDetection, hoàn thiện canonical dataset, rồi mới chạy baseline thật.

## 14. Checklist bàn giao

- [x] Đọc handoff đầy đủ và đối chiếu flow đã thống nhất.
- [x] Kiểm tra repo hiện tại: có `.gitignore` và handoff; chưa có code training.
- [x] Lập kế hoạch scaffold và điều kiện nghiệm thu.
- [x] Triển khai S1–S7 ở mức scaffold v1.
- [x] Chạy core checks và offline fixture workflow.
- [ ] Kiểm chứng local CUDA pilot thật khi canonical dataset/model sẵn sàng.
- [ ] Kiểm chứng Kaggle fallback thật nếu local 8 GB không đủ.
- [ ] Kiểm chứng integration Modal thật khi có pilot/budget phù hợp.

Remote runners chỉ được đánh dấu live-tested sau lần chạy thật có account, dataset và budget hợp lệ.
