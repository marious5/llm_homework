"""
可视化ScienceQA评测样本
随机选择4条样本，按2x2格式展示题目、图片、选项、模型输出等信息
"""
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import textwrap

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')

# 设置中文字体（尝试多个字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def wrap_text(text, width=50):
    """文本换行处理"""
    return '\n'.join(textwrap.wrap(text, width=width))

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
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 24))
    fig.suptitle('ScienceQA 评测样本可视化', fontsize=20, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, sample in enumerate(selected_samples):
        ax = axes[idx]
        ax.axis('off')
        
        # 获取样本信息
        sample_idx = sample['index']
        question = sample['question']
        choices = sample['choices']
        ground_truth = sample['ground_truth']
        predicted = sample['predicted']
        generated_text = sample['generated_text']
        is_correct = sample['correct']
        image_path = sample.get('image_path')
        
        # 加载图片
        if image_path and os.path.exists(os.path.join(images_dir, image_path)):
            img = Image.open(os.path.join(images_dir, image_path))
            # 在子图上方显示图片
            img_height = 0.25  # 图片占据的高度比例
            img_ax = fig.add_axes([
                ax.get_position().x0,
                ax.get_position().y1 - img_height,
                ax.get_position().width,
                img_height
            ])
            img_ax.imshow(img)
            img_ax.axis('off')
        
        # 格式化选项
        options_text = ""
        for i, choice in enumerate(choices):
            option_letter = chr(65 + i)  # A, B, C, D...
            wrapped_choice = wrap_text(choice, width=45)
            # 标记正确答案和模型预测
            marker = ""
            if option_letter == ground_truth and option_letter == predicted:
                marker = " ✓✓"  # 正确答案且模型预测正确
            elif option_letter == ground_truth:
                marker = " ✓"   # 正确答案
            elif option_letter == predicted:
                marker = " ✗"   # 模型预测但错误
            
            options_text += f"{option_letter}. {wrapped_choice}{marker}\n"
        
        # 构建文本内容
        result_status = "✓ 正确" if is_correct else "✗ 错误"
        result_color = "green" if is_correct else "red"
        
        info_text = f"""样本 #{sample_idx}  [{result_status}]

【问题】
{wrap_text(question, width=60)}

【选项】
{options_text}
【正确答案】 {ground_truth}
【模型预测】 {predicted}

【模型输出】
{wrap_text(generated_text, width=60)}
"""
        
        # 在图片下方显示文本信息
        text_y_position = ax.get_position().y1 - img_height - 0.02 if image_path else ax.get_position().y1
        
        ax.text(
            0.5, text_y_position,
            info_text,
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat' if is_correct else 'lightcoral', alpha=0.3),
            family='monospace'
        )
        
        # 添加结果标签
        ax.text(
            0.5, ax.get_position().y0 + 0.01,
            result_status,
            transform=fig.transFigure,
            fontsize=14,
            fontweight='bold',
            color=result_color,
            verticalalignment='bottom',
            horizontalalignment='center'
        )
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图片
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"可视化图片已保存到: {output_file}")
    plt.close()
    
    # 打印选中的样本索引
    print(f"\n已选择的样本索引: {[s['index'] for s in selected_samples]}")
    print(f"正确数量: {sum(1 for s in selected_samples if s['correct'])}/{len(selected_samples)}")


if __name__ == "__main__":
    json_path = "task2_results.json"
    images_dir = "task2_sample_images"
    
    visualize_samples(
        json_path=json_path,
        images_dir=images_dir,
        num_samples=4,
        seed=42,
        output_file="task2_visualization.png"
    )
