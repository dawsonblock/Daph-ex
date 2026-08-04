# Evidence schema v3.4.0

Every serious model run gets a new immutable folder. Historical metadata is never overwritten.

```text
evidence/e3_run_<id>/
  artifact_metadata.json
  environment.json
  source_model.json
  source_revision.txt
  source_digest.json
  tokenizer_manifest.json
  config.json
  dataset_manifest.json
  train_ids.json
  selection_ids.json
  test_ids.json
  natural_test_ids.json
  hardcase_manifest.json
  profile_digest.json
  training_receipt.json
  checkpoint_digest.json
  per_example_results.jsonl
  per_effort_metrics.json
  quality_bootstrap.json
  utility_bootstrap.json
  lambda_sweep.json
  rescue_regression.json
  effort_frontier.json
  decision.json
  summary.md
```

`artifact_metadata.json` requires `artifact_commit`, `repository_version`, UTC creation time, `test_count_at_creation`, pytest digest, config digest, source-tree digest, and claim strength.

Each paired E2/E3 row requires task ID, quality and compute for both arms, materialized utilities, all three deltas, correctness/rescue/regression, family, template, difficulty, generator/verifier versions, seed metadata, both compute receipts, and the normalization rule. Effort IDs are descriptive only and cannot determine cost.

The postprocessor [`scripts/qualify_e3_results.py`](../scripts/qualify_e3_results.py) creates the paired statistical portion of this folder. Model-training scripts must additionally emit source/tokenizer/data/training/checkpoint receipts because those values cannot be reconstructed honestly afterward.
