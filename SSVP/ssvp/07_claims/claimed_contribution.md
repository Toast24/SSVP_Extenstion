# Claimed Contribution

What we reproduced:
- Reimplemented the SSVP zero-shot industrial anomaly detection pipeline on the MVTec AD cable category, including the semantic-visual fusion backbone with CLIP and DINOv2 features, the visual-conditioned prompt generator, and the visual-text anomaly mapper.
- Reproduced the standard anomaly metrics and evaluation flow for clean test images, using image-level and pixel-level AUROC, F1, AP, and per-region overlap.
- Reproduced the core idea of prompt-driven zero-shot anomaly segmentation without retraining on new categories, using the SSVP-style text and vision prompt interaction.

What we modified:
- Added a captioning branch that generates human-readable defect descriptions alongside pixel anomaly maps, and applied INT8 dynamic quantization to the caption transformer for practical deployment.
- Changed the dataset split to a strict 70/20/10 train/validation/test partition with anti-leakage validation-based threshold calibration, instead of relying on the standard MVTec evaluation split for all decisions.
- Extended the project with systematic heavy synthetic noise robustness evaluation, measuring clean-to-noisy deltas across all metrics.
- Built a gated compression workflow that evaluates depth pruning and differentiated pruning candidates on a 3-epoch preliminary stage, and only promotes models that satisfy explicit performance acceptance criteria.
- Added a distillation pipeline and caption prompt variants to explore trade-offs between caption quality, runtime, and compression.

What did not work:
- Removing consistency regularization caused severe performance collapse: the no_consistency variant dropped clean image AUROC from 97.42% to 78.67%, and LORA without consistency performed even worse.
- Neither depth pruning nor differentiated-head pruning met the acceptance threshold in the gated compression workflow, so those compression paths were not adopted as final deployment models.
- The prompt improvement caption variant produced defect-specific descriptions but was about ten times slower than the baseline.

What we believe is our contribution:
- A practical extended SSVP implementation with explainable caption generation and INT8-compressed LLM-based captioning for zero-shot anomaly detection.
- A rigorously documented robustness evaluation under heavy synthetic noise, showing run21 maintains usable performance (97.42% clean image AUROC, 87.00% noisy image AUROC) and quantifying the robustness deltas.
- A gated compression and distillation process that enforces explicit metric acceptance before model promotion, making compression decisions more reliable.
