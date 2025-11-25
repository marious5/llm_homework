# 机器学习与数据挖掘 - 第三次大作业

## 作业说明

本作业包含两个任务：
1. **任务一**：LLM 数学推理评测 (GSM8K)
2. **任务二**：MLLM 多模态科学问答 (ScienceQA)

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 硬件要求

- **推荐**：NVIDIA GPU (至少 8GB 显存)
- **CPU模式**：可运行但速度较慢
- 使用 4-bit 量化可降低显存需求至约 4-5GB

## 运行方法

### 任务一：GSM8K 数学推理评测

```bash
python task1_gsm8k.py
```

**输出**：
- 控制台显示评测进度和最终准确率
- 生成 `task1_results.json` 包含详细结果

### 任务二：ScienceQA 多模态问答评测

```bash
python task2_scienceqa.py
```

**输出**：
- 控制台显示评测进度和最终准确率
- 生成 `task2_results.json` 包含详细结果

## 代码说明

### task1_gsm8k.py
- 使用 Qwen2.5-7B-Instruct 模型
- 从 GSM8K 测试集随机抽取 50 条数据
- 构建 Chain-of-Thought 推理 Pipeline
- 自动提取答案并与 Ground Truth 对比

### task2_scienceqa.py
- 使用 Qwen2-VL-7B-Instruct 多模态模型
- 筛选包含图片的 ScienceQA 题目
- 从测试集随机抽取 50 条数据
- 处理图片+问题+选项，提取模型选择的答案

## 注意事项

1. 首次运行会自动下载模型，需要较长时间
2. 确保网络连接正常，可访问 HuggingFace
3. 如遇显存不足，可调整 `use_4bit=True` 启用量化
4. 结果文件为 JSON 格式，包含每个样本的详细信息

## 模型说明

### Qwen2.5-7B-Instruct
- 参数量：7B
- 架构：Transformer decoder
- 特点：支持长上下文、中英文双语

### Qwen2-VL-7B-Instruct
- 参数量：7B
- 架构：Vision-Language Transformer
- 特点：支持图文多模态理解
