# Data Description

Dataset used: MVTec AD (cable category).

Current layout:
- datasets/cable: original cable category layout (train/test/ground_truth)
- datasets/cable_resplit/cable: generated 70/20/10 train/test/val split with masks
- sample_inputs: small representative demo inputs

Preprocessing and split:
- Resplit generated with 03_code/scripts/prepare_cable_split.py
- Split ratio: 70% train, 20% test, 10% val
- Seed: 42
