# Qwen3-8B 的 RMSNorm 融合，对短输入和长输入的影响一样吗？

这是一个可复现的测量方案和结果分析器。当前仓库已验证分析器的输入检查和统计逻辑，尚未采集昇腾 NPU 上的模型性能数据。

## 怎么测

在适配的昇腾 NPU 环境使用 [CANN Learning Hub 的固定版本教程](https://github.com/hicann/cann-learning-hub/tree/f38f0a09d95660e46f9816d6c6a058d67fb1416f/tutorials/llm_inference/qwen3_8b)。实验基于 BF16、eager、单卡、单请求。模型权重、系统环境和模型运行依赖由该教程说明。

1. 运行第 02 节确认基线可用。用第 03 节为基线单独采集 profile，观察 Add、归约等小算子与相邻任务的时间间隔，记录一张有标注的时间线。
2. 准备两个不含个人信息的 prompt，每组各用一个。在教程所用 recipes 副本的 `dataset/default_prompt.json` 中设置 `text` 并核对实际生成文件中的 prompt。保持相同的输入截断上限，选择分词后长度明显不同且都未超出截断上限的输入。长度以 summary 中的 `input_tokens` 为准。
3. 用第 04 节收集正式 A/B。基线关闭 `enable_npu_rmsnorm` 和 `enable_add_rmsnorm`，融合版同时开启两项。固定模型、设备、其余 YAML、解码设置、预热策略和输出长度。记录输出文本与单算子精度检查。碰到两条路径提前结束长度不同的情况，另行检查原因并保留这次记录。
4. 预热轮次单独记录。对每种输入计划至少三次独立的 A/B，尽可能交替两条路径的执行顺序并记录实际顺序。只将完整可比的正式轮次交给分析器。每次完成立刻将当轮文件另存，教程下一轮可能覆盖同名文件。

## 把真实运行文件交给分析器

以下命令在项目根目录执行。每个编号目录是一对完整 A/B。将 Notebook 04 的以下五个上游文件复制并命名。表中源路径相对于该 Notebook 使用的 `SOURCE_DIR/run_outputs/`。

| 上游相对路径 | 一轮目录中的文件名 |
| --- | --- |
| `benchmark_baseline/metrics/addrmsnorm_baseline_summary.json` | `baseline.json` |
| `benchmark_baseline/outputs/addrmsnorm_baseline_generations.jsonl` | `baseline.jsonl` |
| `benchmark_add_rmsnorm/metrics/addrmsnorm_optimized_summary.json` | `fused.json` |
| `benchmark_add_rmsnorm/outputs/addrmsnorm_optimized_generations.jsonl` | `fused.jsonl` |
| `optimization_comparison.json` | `comparison.json` |

```text
runs/
  environment.json
  short/01/   五个文件，后续独立轮次为 02、03……
  long/01/    同样的文件布局
```

在 `runs/environment.json` 里记录以下九个字符串字段的真实值。程序会把它们带入报告，实际设备和外部采样参数需由采集者核对。

| 字段 | 应记录的内容 |
| --- | --- |
| `device_model`、`driver`、`cann` | 加速卡具体型号、驱动与 CANN 版本 |
| `torch`、`torch_npu` | 运行时的实际包版本 |
| `weights_revision`、`tutorial_commit` | 权重的不可变版本或文件摘要，使用的教程提交 |
| `decoding_policy`、`warmup_policy` | 采样和解码参数、预热次数与正式轮次边界 |

将实际 YAML、原始日志、精度检查、性能时间线及运行顺序保留在实验记录中。分析器只可自动核对 summary 和 generation 所含的条件；应检查两份 YAML 差异，确认除融合配置外其余推理参数相同。对外展示前可选用报告中的汇总与原始样本数值，原始日志可能含本地路径和输入文本。

```console
python inference/analyze.py runs
python -B -m unittest discover -s inference -p "test_*.py"
```

第一条命令向标准输出写 JSON，使用 Python 3 标准库。第二条用临时合成输入验证工具，不需要昇腾设备，测试值不会写入性能结果。

## 怎样读结果

主指标是日志中 **prefill 秒数与 decode 秒数之和**。官方工作流在没有完整阶段计时时可能使用进程墙钟时间。此工具要求两个阶段都有正值，且和 `total_time_s` 一致。该口径描述本次离线运行的记录阶段，不表示在线服务首 token 时延或客户端端到端延迟。生成 token 数取自教程对输出文本的重新分词，不额外将其解释为设备内部计数。

报告包含每一轮的原始阶段秒数、tokens/s、输入输出 token 数、文本健康状态、文件与文本摘要，以及精度与输出近似度检查。每组分别计算融合前后总秒数的中位数和最小到最大范围。所有轮次通过质量检查、实际输出长度相同且有至少三对时，报告才给出 `median_time_change_pct = 100 × (融合中位数 / 基线中位数 − 1)`。负值表示记录的阶段时间缩短。质量未通过或轮次不足时保留原始样本，该百分比为 null。时间范围反映本次观测的波动，不是置信区间。

质量检查采用教程已有的单算子精度状态、文本健康状态和文本相似度。结合原始输出检查语义，才能说明质量是否符合实际任务需求。完成实测后，选一组支持原假设或与预期不符的结果，配上时间线、测量范围和自己的解释。
