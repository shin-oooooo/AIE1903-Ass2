# AIE1903 HW1：LSTM 情感分类与积分梯度分析

## 1. 项目内容

本项目使用单向 LSTM 完成 SST-2 二分类，使用 Integrated Gradients（积分梯度，IG）解释五个测试案例，并通过删词实验比较归因方向与模型输出变化。标签 `0` 表示负面，标签 `1` 表示正面。

## 2. 文件与目录

请将 Notebook、模型脚本和 `data/` 放在同一项目目录。Notebook 文件名以实际提交版本为准。

```text
项目目录/
├── README.md
├── AIE1903_HW1.ipynb
├── lstm.py
├── train_batch.py
├── data/
│   ├── SST2/
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   └── test.tsv
│   ├── embeddings_initial.npz
│   └── vocab.json
└── HW1_experiment_results/
    ├── best_model.pt
    ├── figures/
    ├── data_definitions.json
    ├── training_history.csv
    ├── test_metrics.json
    ├── test_predictions.csv
    ├── ig_attributions.csv
    ├── ig_diagnostics.csv
    └── deletion_results.csv
```

`lstm.py` 是 Notebook 运行所需模型脚本。仅当初始化缓存缺失、需要读取已有 GoogleNews 原文件时，Notebook 才从 `train_batch.py` 提取 `load_bin_vec` 函数；Notebook 不会执行该脚本顶层导入。当前代码不使用 `data_helpers.py`。实验报告另行提交。已解压结果目录保存图表、实验数据与 `best_model.pt`；`run_config.json` 不属于下载结果。

## 3. 运行环境

实验使用大学 Jupyter 环境。原运行记录为 Python 3.12.3、PyTorch `2.12.0a0+5aff3928d8.nv26.05`，PyTorch CUDA 构建版本为 13.2。主要依赖包括 PyTorch、NumPy、pandas、Matplotlib 和 Captum；请按 Notebook 导入语句准备环境。

建议先在兼容 GPU 环境验证。CPU 运行需要将设备设为 `cpu`，并按需调整 GPU 检查；当前说明不代表 Notebook 已通过完整 CPU 验证。不同硬件、依赖版本和随机计算可能导致训练结果存在差异。

## 4. 初始化材料
- `data/embeddings_initial.npz` 保存训练前嵌入矩阵。矩阵形状为 `(13840, 300)`，数据类型为 `float32`；矩阵包含匹配到的 GoogleNews 预训练向量、未匹配 tokens 的随机初始向量，以及零填充向量。
- `data/vocab.json` 保存 tokens 与矩阵行号映射。矩阵必须配合对应词表使用；仅核对词表大小不足以确认对应关系。
- `best_model.pt` 保存验证准确率最高一轮的完整模型参数，包括训练后嵌入矩阵、LSTM 参数和分类层参数。模型文件用于复核测试、IG 与删词结果；模型文件不能替代原实验训练前初始化矩阵。

Notebook 优先加载 `data/` 中成对存在的初始化矩阵与词表，因此重新训练无需下载完整 GoogleNews 文件。缓存不完整时，Notebook 再查找已有 GoogleNews `.bin` 或 `.bin.gz`；已有 `.bin` 可直接读取，`.bin.gz` 需要先解压。程序不自动下载大型词向量文件。

提交包无需包含完整 GoogleNews `.bin` 或 `.bin.gz`。初始化缓存仅覆盖本实验词表，不能替代完整 GoogleNews 文件来初始化任意新词表。

## 5. 重新训练与生成结果

1. 将文件按第 2 节放置，并将 Jupyter 工作目录设为项目目录。
2. 确认 `data/SST2/` 包含三个 TSV，且 `data/` 包含初始化矩阵与对应词表。
3. 在全新内核中按顺序执行 Notebook。第 2 节读取 SST-2；第 5 节加载初始化材料；后续代码完成训练、验证选模与测试。
4. 运行第 8 节计算正面类 IG，再运行第 9 节执行删词检验。
5. 第 10A 节仅保存初始化矩阵与词表，不提供下载按钮。第 10 节生成实验结果 ZIP 并提供下载按钮；本次提交已将 ZIP 解压为 `HW1_experiment_results/`。

初始化矩阵与词表优先保存到项目 `data/`。目录不可写时，程序回退到临时目录并提示路径；请按提示复制文件。图表与模型参数可能写入 `/tmp/aie1903/runs/`，请在会话结束前下载结果，避免临时目录被清理。

重新训练可能产生不同指标，或选出不同案例。若需逐项对照报告，请使用原 `best_model.pt` 并固定报告案例编号：354、1117、188、113、1228。

## 6. 使用已训练模型复核

复核需要相同模型定义、相同预处理、对应词表及 SST-2 数据。请在 `HW1_experiment_results/` 中找到 `best_model.pt`。

当前 Notebook 默认执行重新训练流程，尚未提供独立一键复核入口。手动复核时，应执行必要的数据加载与模型定义，按保存结构读取模型参数，并使用 `map_location=device` 指定设备；加载完成后调用 `model.eval()`。跳过初始化与训练步骤，再执行测试、IG 和删词分析。仅把模型文件放入目录后点击“运行全部”，不会自动跳过训练。

复核已训练模型无需 GoogleNews 文件，也无需训练前初始化矩阵；但复核仍需与模型参数匹配的词表。

## 7. 输出含义

- `p1`：正面类概率，始终表示 `P(y=1|x)`。
- `confidence`：当前预测类别对应概率，即 `max(p1, 1-p1)`。
- `IG`：归因目标固定为正面类对数概率 `log p1`。正值表示沿零嵌入基线到输入路径，token 对目标输出作正向贡献；负值表示负向贡献。
- `输出差`：`input_log_p1 - baseline_log_p1`。
- `delta`：`sum_ig - 输出差`。相对残差为 `abs(delta) / max(abs(输出差), 1e-8)`。较小残差说明归因总和对齐较好，但不能保证每个 token 归因准确。
- `prob_change_pp`：`100 × (p1_after - p1_before)`，单位为百分点。
- `log_drop`（第 9 节记为 `d1`）：`log_p1_before - log_p1_after`。
- `same_sign`：IG 与 `d1` 同号，对应 IG 与正面概率变化异号；`opposite_sign` 表示方向相反；`near_zero` 表示相关数值接近零。
- 测试编号 `index` 从 0 开始；token 位置 `position_1based` 从 1 开始。

删词实验每句删除绝对 IG 最大的纯字母 token。IG 与删词使用不同扰动方式，因此两项结果不要求幅度相等；五例均为 `same_sign` 也不能单独证明所有归因准确。
