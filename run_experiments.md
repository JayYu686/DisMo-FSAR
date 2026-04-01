# D2ST-lite Few-Shot Experiment Runs

This document provides directly runnable commands for the minimal E0/E1/E2 ablations.

## Preconditions

- Python 3.10+ is recommended.
- Keep `semantic=false`, `motion=false`, `use_classification=false` for all runs.
- These configs assume the default dataset roots documented in `README.md`:
  - SSV2 Small: `/data/nvme1/yujielun_dataset/SSv2/ssv2_mp4_h264`
  - HMDB51: `/data/nvme1/yujielun_dataset/HMDB51/`
- The few-shot trainer consumes one episodic task per step in the current code path, so all configs use:
  - `TRAIN.BATCH_SIZE: 1`
  - `TEST.BATCH_SIZE: 1`

## Config Summary

| Config | Dataset | Core difference vs previous run |
|---|---|---|
| `E0_baseline_ssv2_small_5w1s.yaml` | SSV2 Small 5-way 1-shot | Legacy DiSMo path, no adapter, no local token matching |
| `E1_adapter_only_ssv2_small_5w1s.yaml` | SSV2 Small 5-way 1-shot | Legacy DiSMo path, backbone adapter on, no local token matching |
| `E2_d2st_lite_local_matching_ssv2_small_5w1s.yaml` | SSV2 Small 5-way 1-shot | D2ST-lite V1 path, backbone adapter on, local token matching on |
| `E2_d2st_lite_local_matching_hmdb51_5w1s.yaml` | HMDB51 5-way 1-shot | Same as E2 SSV2 config, dataset/splits switched to HMDB51 |

## Shared Training Logic

All four configs intentionally share the same trainer-side settings:

- `NUM_GPUS: 1`
- `TRAIN.NUM_TRAIN_TASKS: 10000`
- `TRAIN.NUM_TEST_TASKS: 2000`
- `TRAIN.VAL_FRE_ITER: 1000`
- `SOLVER.MAX_EPOCH: 1`
- `SOLVER.STEPS_ITER: 10000`
- `OPTIMIZER/SOLVER.BASE_LR: 5e-4`
- `OPTIMIZER/SOLVER.OPTIM_METHOD: adamw`
- `TRAIN.SAVE_BEST_ONLY: true`

`SOLVER.MAX_EPOCH` is set to `1` on purpose. The current few-shot trainer uses task-budget iteration rather than a normal outer epoch loop, so effective training length is controlled by `TRAIN.NUM_TRAIN_TASKS` and `SOLVER.STEPS_ITER`.

## Train Commands

### E0: Legacy baseline on SSV2 Small

```bash
python runs/run.py --cfg configs/pool/run/training/E0_baseline_ssv2_small_5w1s.yaml \
  TRAIN.ENABLE true TEST.ENABLE false
```

### E1: Adapter-only ablation on SSV2 Small

```bash
python runs/run.py --cfg configs/pool/run/training/E1_adapter_only_ssv2_small_5w1s.yaml \
  TRAIN.ENABLE true TEST.ENABLE false
```

### E2: D2ST-lite + local matching on SSV2 Small

```bash
python runs/run.py --cfg configs/pool/run/training/E2_d2st_lite_local_matching_ssv2_small_5w1s.yaml \
  TRAIN.ENABLE true TEST.ENABLE false
```

### E2: D2ST-lite + local matching on HMDB51

```bash
python runs/run.py --cfg configs/pool/run/training/E2_d2st_lite_local_matching_hmdb51_5w1s.yaml \
  TRAIN.ENABLE true TEST.ENABLE false
```

## Evaluation Commands

After training finishes, evaluate with the same config by flipping the train/test toggles. The current checkpoint loader will automatically read the latest checkpoint under the experiment `OUTPUT_DIR/checkpoints/` when `TEST.CHECKPOINT_FILE_PATH` is empty.

### E0 eval

```bash
python runs/run.py --cfg configs/pool/run/training/E0_baseline_ssv2_small_5w1s.yaml \
  TRAIN.ENABLE false TEST.ENABLE true
```

### E1 eval

```bash
python runs/run.py --cfg configs/pool/run/training/E1_adapter_only_ssv2_small_5w1s.yaml \
  TRAIN.ENABLE false TEST.ENABLE true
```

### E2 SSV2 eval

```bash
python runs/run.py --cfg configs/pool/run/training/E2_d2st_lite_local_matching_ssv2_small_5w1s.yaml \
  TRAIN.ENABLE false TEST.ENABLE true
```

### E2 HMDB51 eval

```bash
python runs/run.py --cfg configs/pool/run/training/E2_d2st_lite_local_matching_hmdb51_5w1s.yaml \
  TRAIN.ENABLE false TEST.ENABLE true
```

## Optional Multi-GPU Override

The configs default to single-GPU execution because that is the safest directly runnable setup. If you want to launch on 8 GPUs, keep the same config and override `NUM_GPUS` from the command line:

```bash
python runs/run.py --cfg configs/pool/run/training/E2_d2st_lite_local_matching_ssv2_small_5w1s.yaml \
  TRAIN.ENABLE true TEST.ENABLE false NUM_GPUS 8
```

The trainer interprets `TRAIN.NUM_TRAIN_TASKS` as a global task budget and divides it by world size internally.

## Output Directories

- `output/experiments/E0_baseline_ssv2_small_5w1s`
- `output/experiments/E1_adapter_only_ssv2_small_5w1s`
- `output/experiments/E2_d2st_lite_local_matching_ssv2_small_5w1s`
- `output/experiments/E2_d2st_lite_local_matching_hmdb51_5w1s`

Each run stores:

- logs under the experiment output dir
- checkpoints under `OUTPUT_DIR/checkpoints/`
- the saved merged config as `configuration.log`
