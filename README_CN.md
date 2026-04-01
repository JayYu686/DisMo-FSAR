# DiSMo: DINOv2 + Semantic + Motion 少样本动作识别

DiSMo 是基于 HyRSM 的 FSAR 工程版本，包含三个核心模块：
- DINOv2 视觉主干
- 显式运动建模
- 类别文本语义监督

本仓库同时包含：
- 原始 HyRSM 配置（`configs/projects/hyrsm/`）
- DiSMo 配置与调参版本（`configs/projects/dismo/`，含 `v2`/`v3`）

<img src="HyRSM_arch.png" width="800">

## 环境

建议直接使用仓库环境文件（Python 3.10）：

```bash
conda env create -f environment.yaml
conda activate dismo
```

说明：
- 推荐 Python 3.10+，避免 DINOv2 torch.hub 代码兼容性问题。
- 若你使用其他环境名（如 `dismo310`），命令不变。

## 数据目录

当前配置中的默认数据路径：

| 数据集 | 配置默认根目录 | 划分文件目录 |
|---|---|---|
| HMDB51 | `/data/nvme1/yujielun_dataset/HMDB51/` | `splits/hmdb51/` |
| UCF101 | `/data/nvme1/yujielun_dataset/UCF101` | `splits/ucf101/` |
| Kinetics100 子集 | `/data/nvme1/yujielun_dataset/Kinetics400` | `splits/kinetics100/` |
| SSV2 Small | `/data/nvme1/yujielun_dataset/SSv2/ssv2_mp4_h264` | `splits/ssv2_small/` |
| SSV2 Full | `/data/nvme1/yujielun_dataset/SSv2/ssv2_mp4_h264` | `splits/ssv2_full/` |

如目录不同，请修改 `DATA.DATA_ROOT_DIR`。

## 运行方式

统一入口是 `runs/run.py`。建议显式传入 `TRAIN.ENABLE` / `TEST.ENABLE`，避免配置文件里历史开关影响。

### 训练

```bash
python runs/run.py --cfg configs/projects/dismo/kinetics100/DiSMo_K100_5shot_v2.yaml \
  TRAIN.ENABLE true TEST.ENABLE false
```

### 测试

```bash
python runs/run.py --cfg configs/projects/dismo/kinetics100/DiSMo_K100_5shot_v2.yaml \
  TRAIN.ENABLE false TEST.ENABLE true \
  TEST.CHECKPOINT_FILE_PATH <path_to_best_checkpoint.pyth>
```

### 常见启动报错

如果出现 `EADDRINUSE`（端口占用）：

```bash
python runs/run.py --cfg <cfg.yaml> --init_method tcp://127.0.0.1:10001
```

## 推荐配置文件

DiSMo 全部配置在 `configs/projects/dismo/`。

建议起步配置：
- K100: `DiSMo_K100_1shot_v2.yaml`, `DiSMo_K100_5shot_v2.yaml`
- HMDB51: `DiSMo_HMDB51_1shot_v2.yaml`, `DiSMo_HMDB51_5shot_v2.yaml`
- UCF101: `DiSMo_UCF101_1shot_v3.yaml`, `DiSMo_UCF101_5shot_v3.yaml`
- SSV2-Small: `DiSMo_SSV2_Small_1shot_v3.yaml`, `DiSMo_SSV2_Small_5shot_v2.yaml`
- SSV2-Full: `DiSMo_SSV2_Full_1shot_v3.yaml`, `DiSMo_SSV2_Full_5shot_v3.yaml`

## LLM 描述生成

脚本：`scripts/generate_descriptions.py`

常用参数：
- `--class_names_file`
- `--prompt_template_file`
- `--openai_base_url`
- `--proxy_url`
- `--ca_bundle`
- `--insecure_ssl`

### 示例（DeepSeek + 本地代理，快速可跑通）

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

如果遇到 `CERTIFICATE_VERIFY_FAILED`：
- 临时方案：`--insecure_ssl`
- 更安全方案：`--ca_bundle /path/to/proxy_ca.pem`

## Checkpoint 策略

few-shot 训练支持只保留最佳模型：
- `TRAIN.SAVE_BEST_ONLY: true`
- `TRAIN.BEST_METRIC: top1_acc`
- `TRAIN.BEST_MODE: max`

启用后，最佳模型命名会包含轮次和指标，例如：
- `checkpoint_best_round005_top1_acc85.38.pyth`

## 关键配置项

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

`BACKBONE.FREEZE: true` 表示 DINOv2 主干参数固定（`requires_grad=False`），作为冻结特征提取器使用。

## 项目结构

```text
configs/projects/dismo/     # DiSMo 实验配置（base + v2/v3）
configs/projects/hyrsm/     # HyRSM 基线配置
models/base/dismo.py        # DiSMo 主模型头
models/base/dinov2_backbone.py
models/base/motion_module.py
models/base/semantic_module.py
runs/run.py                 # 统一训练/测试入口
scripts/generate_descriptions.py
```

## 致谢

- [HyRSM](https://github.com/alibaba/HyRSM)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [MoLo](https://arxiv.org/abs/2305.05912)
- [CLIP-FSAR](https://arxiv.org/abs/2303.15232)
