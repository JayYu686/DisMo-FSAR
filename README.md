# DiSMo: DINOv2 + Semantic + Motion for Few-Shot Action Recognition

DiSMo is a HyRSM-based FSAR project with three core modules:
- DINOv2 visual backbone
- explicit motion modeling
- semantic supervision from class text descriptions

This repository contains both:
- original HyRSM configs (`configs/projects/hyrsm/`)
- DiSMo configs and tuned variants (`configs/projects/dismo/`, including `v2`/`v3`)

<img src="HyRSM_arch.png" width="800">

## Environment

Use the provided environment file (Python 3.10):

```bash
conda env create -f environment.yaml
conda activate dismo
```

Notes:
- Python 3.10+ is recommended for DINOv2 torch.hub code compatibility.
- If you use a custom env name (e.g. `dismo310`), commands are identical.

## Data Layout

Default dataset roots in configs:

| Dataset | Root dir in config | Split dir |
|---|---|---|
| HMDB51 | `/data/nvme1/yujielun_dataset/HMDB51/` | `splits/hmdb51/` |
| UCF101 | `/data/nvme1/yujielun_dataset/UCF101` | `splits/ucf101/` |
| Kinetics100 subset | `/data/nvme1/yujielun_dataset/Kinetics400` | `splits/kinetics100/` |
| SSV2 Small | `/data/nvme1/yujielun_dataset/SSv2/ssv2_mp4_h264` | `splits/ssv2_small/` |
| SSV2 Full | `/data/nvme1/yujielun_dataset/SSv2/ssv2_mp4_h264` | `splits/ssv2_full/` |

Adjust `DATA.DATA_ROOT_DIR` if your paths differ.

## Run Commands

`runs/run.py` is the main entrypoint. Pass explicit train/test toggles to avoid config-side defaults.

### Train

```bash
python runs/run.py --cfg configs/projects/dismo/kinetics100/DiSMo_K100_5shot_v2.yaml \
  TRAIN.ENABLE true TEST.ENABLE false
```

### Test

```bash
python runs/run.py --cfg configs/projects/dismo/kinetics100/DiSMo_K100_5shot_v2.yaml \
  TRAIN.ENABLE false TEST.ENABLE true \
  TEST.CHECKPOINT_FILE_PATH <path_to_best_checkpoint.pyth>
```

### Common launch issue

If you see `EADDRINUSE` (port already in use), set another init port:

```bash
python runs/run.py --cfg <cfg.yaml> --init_method tcp://127.0.0.1:10001
```

## Recommended Configs

All available DiSMo configs are under `configs/projects/dismo/`.

Suggested starting points:
- K100: `DiSMo_K100_1shot_v2.yaml`, `DiSMo_K100_5shot_v2.yaml`
- HMDB51: `DiSMo_HMDB51_1shot_v2.yaml`, `DiSMo_HMDB51_5shot_v2.yaml`
- UCF101: `DiSMo_UCF101_1shot_v3.yaml`, `DiSMo_UCF101_5shot_v3.yaml`
- SSV2-Small: `DiSMo_SSV2_Small_1shot_v3.yaml`, `DiSMo_SSV2_Small_5shot_v2.yaml`
- SSV2-Full: `DiSMo_SSV2_Full_1shot_v3.yaml`, `DiSMo_SSV2_Full_5shot_v3.yaml`

## LLM Description Generation

Script: `scripts/generate_descriptions.py`

Supported useful options:
- `--class_names_file`
- `--prompt_template_file`
- `--openai_base_url`
- `--proxy_url`
- `--ca_bundle`
- `--insecure_ssl`

### Example (DeepSeek + local proxy, quick workaround)

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="https://api.deepseek.com"
export HTTPS_PROXY="http://127.0.0.1:8888"
export HTTP_PROXY="http://127.0.0.1:8888"

python scripts/generate_descriptions.py \
  --dataset ssv2 \
  --class_names_file data/ssv2_full_classes.txt \
  --method llm \
  --llm_model deepseek-chat \
  --llm_temperature 0.2 \
  --llm_max_tokens 320 \
  --llm_retries 3 \
  --sleep_sec 0.4 \
  --timeout_sec 90 \
  --prompt_template_file scripts/prompts/action_description_ssv2_llm_v1.txt \
  --openai_base_url "$OPENAI_BASE_URL" \
  --proxy_url "http://127.0.0.1:8888" \
  --insecure_ssl \
  --output data/ssv2_full_descriptions_llm.json
```

If you get `CERTIFICATE_VERIFY_FAILED` with a proxy, use either:
- `--insecure_ssl` (quick workaround), or
- `--ca_bundle /path/to/your_proxy_ca.pem` (recommended secure method).

## Checkpoint Policy

Few-shot training supports best-only checkpoint retention:
- `TRAIN.SAVE_BEST_ONLY: true`
- `TRAIN.BEST_METRIC: top1_acc`
- `TRAIN.BEST_MODE: max`

When enabled, the best file is promoted with round + metric naming, e.g.:
- `checkpoint_best_round005_top1_acc85.38.pyth`

## Useful Config Keys

```yaml
VIDEO:
  BACKBONE:
    DINO_MODEL: "dinov2_vitl14"
    FREEZE: true

SEMANTIC:
  ENABLE: true
  TEXT_MODEL: "./data/pretrained/all-MiniLM-L6-v2"
  DESCRIPTIONS_PATH: "./data/<dataset>_descriptions_llm.json"

TRAIN:
  SAVE_BEST_ONLY: true
  BEST_METRIC: top1_acc
  BEST_MODE: max
```

`BACKBONE.FREEZE: true` means DINOv2 parameters are fixed (`requires_grad=False`) and used as a frozen feature extractor.

## Project Structure

```text
configs/projects/dismo/     # DiSMo experiment configs (base + v2/v3)
configs/projects/hyrsm/     # HyRSM baseline configs
models/base/dismo.py        # Main DiSMo few-shot head
models/base/dinov2_backbone.py
models/base/motion_module.py
models/base/semantic_module.py
runs/run.py                 # Unified train/test launcher
scripts/generate_descriptions.py
```

## Acknowledgments

- [HyRSM](https://github.com/alibaba/HyRSM)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [MoLo](https://arxiv.org/abs/2305.05912)
- [CLIP-FSAR](https://arxiv.org/abs/2303.15232)
