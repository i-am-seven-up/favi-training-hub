# Manifest contract

Input là JSONL, một ảnh mỗi dòng: `sample_id`, `image_uri` (relative to dataset root),
`sha256`, `canonical_label`, `original_label`, `source_id`, `source_revision`,
`group_id`, `subtype`, `review_status`, `split`, `non_sexual`, `non_violent`.
Optional: `duplicate_group`, `dhash` (16 hex chars), `near_duplicate_reviewed`.
Safe requires both boolean flags true. Frame sources require group_id.
`review_status=approved` is sample-level approval, not merely source-level approval.

Assign splits deliberately before freeze; the audit enforces group/duplicate isolation.
Freeze creates a JSON envelope containing records, sources, split counts and fingerprint.
Use paths relative to dataset root; image paths cannot escape that root.
Manifest is metadata only: `data inspect` verifies actual files and creates an inventory.
Actual training rechecks image checksums for train/validation before loading.
