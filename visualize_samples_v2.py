"""
可视化ScienceQA评测样本 - 优化版本
随机选择4条样本，按2x2格式展示题目、图片、选项、模型输出等信息
"""
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
import textwrap
import warnings

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning)

# 设置中文字体 - Windows系统
import matplotlib.font_manager as fm
import sys

# 尝试设置中文字体
if sys.platform.startswith('win'):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

def wrap_text(text, width=40):
    """文本换行处理"""
    lines = []
    for line in text.split('\n'):
        if line:
            lines.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
        else:
            lines.append('')
    return '\n'.join(lines)

def visualize_samples(json_path, images_dir, num_samples=4, seed=42, output_file="task2_visualization.png"):
    """
    可视化样本数据
    
    Args:
        json_path: JSON结果文件路径
        images_dir: 图片文件夹路径
        num_samples: 要展示的样本数量
        seed: 随机种子
        output_file: 输出图片文件名
    """
    # 读取JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 随机选择样本
    random.seed(seed)
    selected_samples = random.sample(results, min(num_samples, len(results)))
    
    # 创建2x2子图 - 调整布局更紧凑
    fig = plt.figure(figsize=(18, 20))
    
    for idx, sample in enumerate(selected_samples):
        # 获取样本信息
        sample_idx = sample['index']
        question = sample['question']
        choices = sample['choices']
        ground_truth = sample['ground_truth']
        predicted = sample['predicted']
        generated_text = sample['generated_text']
        is_correct = sample['correct']
        image_path = sample.get('image_path')
        
        # 创建子图 - 分为上下两部分，使用更紧凑的布局
        # 上部分：图片 (行高度为1.2)
        ax_img = plt.subplot2grid((10, 2), (idx // 2 * 5, idx % 2), rowspan=2)
        ax_img.axis('off')
        
        # 加载并显示图片
        if image_path and os.path.exists(os.path.join(images_dir, image_path)):
            img = Image.open(os.path.join(images_dir, image_path))
            ax_img.imshow(img)
            ax_img.set_title(f"样本 #{sample_idx}", fontsize=11, fontweight='bold', pad=8)
        
        # 下部分：文本信息 (行高度为2.8)
        ax_text = plt.subplot2grid((10, 2), (idx // 2 * 5 + 2, idx % 2), rowspan=3)
        ax_text.axis('off')
        
        # 格式化选项 - 增加换行宽度
        options_lines = []
        for i, choice in enumerate(choices):
            option_letter = chr(65 + i)  # A, B, C, D...
            # # 标记
            # if option_letter == ground_truth and option_letter == predicted:
            #     marker = " [正确答案+模型预测]"
            # elif option_letter == ground_truth:
            #     marker = " [正确答案]"
            # elif option_letter == predicted:
            #     marker = " [模型预测]"
            # else:
            marker = ""
            
            choice_wrapped = wrap_text(choice, width=1000)
            options_lines.append(f"{option_letter}. {choice_wrapped}{marker}")
        
        options_text = '\n'.join(options_lines)
        
        # 结果状态
        result_status = "正确" if is_correct else "错误"
        result_symbol = "√" if is_correct else "×"
        
        # 构建完整文本 - 增加换行宽度
        info_text = f"""【问题】
{wrap_text(question, width=1000)}

【选项】
{options_text}

【正确答案】{ground_truth}  |  【模型预测】{predicted}  |  【结果】{result_symbol} {result_status}

【模型输出】
{wrap_text(generated_text, width=1000)}
"""
        
        # 设置背景色
        bg_color = '#e8f5e9' if is_correct else '#ffebee'
        
        # 显示文本
        ax_text.text(
            0.02, 0.98,
            info_text,
            transform=ax_text.transAxes,
            fontsize=8.5,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.7', facecolor=bg_color, edgecolor='gray', linewidth=1.5),
            linespacing=1.5
        )
    
    # 添加总标题
    model_name = data["model"].split('/')[-1] if '/' in data["model"] else data["model"]
    fig.suptitle(
        f'ScienceQA 评测样本可视化\n模型: {model_name}  |  准确率: {data["accuracy"]:.2%} ({data["correct"]}/{data["total_samples"]})',
        fontsize=13,
        fontweight='bold',
        y=0.995
    )
    
    # 调整布局 - 减小子图间距
    # plt.subplots_adjust(hspace=0.3, wspace=0.15, top=0.97, bottom=0.02)
    
    # 保存图片
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()
    
    # 打印选中的样本信息
    print(f"\nSelected sample indices: {[s['index'] for s in selected_samples]}")
    print(f"Correct: {sum(1 for s in selected_samples if s['correct'])}/{len(selected_samples)}")
    
    # 打印每个样本的简要信息
    print("\nSample details:")
    for i, s in enumerate(selected_samples):
        status = "[CORRECT]" if s['correct'] else "[WRONG]"
        print(f"  {i+1}. Sample #{s['index']}: {status} (GT: {s['ground_truth']}, Pred: {s['predicted']})")


if __name__ == "__main__":
    json_path = "task2_results.json"
    images_dir = "task2_sample_images"
    
    visualize_samples(
        json_path=json_path,
        images_dir=images_dir,
        num_samples=4,
        seed=20,
        output_file="task2_visualization.png"
    )
