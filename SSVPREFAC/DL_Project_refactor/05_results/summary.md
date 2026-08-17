# Captioning Pipeline Summary

## What Changed

- Added spatial caption targets in the pipeline so captions can resolve to deterministic phrases such as `no anomaly` or `anomaly at <location>`.
- Added `_mask_to_spatial_location` and caption target builders in `03_code/scripts/run_full_pipeline.py`.
- Added a BLIP caption fine-tuning path that updates the caption decoder while leaving the segmentation model untouched.
- Expanded `03_code/scripts/evaluate_captions.py` to report BERTScore, anomaly exact-match, spatial exact-match, optional NLI entailment, and image-level detection metrics.
- Added deterministic caption synthesis from ground-truth masks and from predicted anomaly maps.

## Main Results

- Full test predicted-map synthesis reached perfect caption match on the spatial reference set: BERTScore F1 = 1.0000.
- On the 74-image full test run, predicted-map synthesis also produced image-level detection metrics of AUROC 96.83, F1-Max 88.89, and AP 94.04.
- The full-test BLIP fine-tuned run improved BERTScore F1 from 0.8604 to 0.9289, a gain of 0.0685.
- The 15-epoch spatial-only run improved from 0.8571 to 0.9262, a gain of 0.0691.
- Smoke runs were consistent with the larger runs: deterministic synthesis gave the strongest text-match gains, while plain spatial fine-tuning produced smaller but stable improvements.

## Interpretation

- Deterministic synthesis is an upper bound because the references were written to match the same spatial phrasing.
- BLIP fine-tuning is the more realistic caption-quality improvement path because it improves text match without relying on reference-aligned synthesis.
- The predicted anomaly maps are strong enough to support caption synthesis, as shown by the full-test detection metrics.

## Artifacts

- Full structured results: [results.json](results.json)
- Key detailed report: [caption_eval_full_compare/caption_before_after_report.json](caption_eval_full_compare/caption_before_after_report.json)
- Key detailed report: [caption_eval_full_predicted_det/caption_before_after_report.json](caption_eval_full_predicted_det/caption_before_after_report.json)
