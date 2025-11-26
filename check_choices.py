"""检查ScienceQA数据集中的选项数量"""
import json

# 读取结果文件
with open("task2_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 统计选项数量
choice_counts = {}
max_choices = 0
min_choices = float('inf')

for result in data["results"]:
    choices = result.get("choices", [])
    num_choices = len(choices)
    
    # 更新统计
    if num_choices in choice_counts:
        choice_counts[num_choices] += 1
    else:
        choice_counts[num_choices] = 1
    
    max_choices = max(max_choices, num_choices)
    min_choices = min(min_choices, num_choices)
    
    # 打印超过4个选项的样本
    if num_choices > 4:
        print(f"样本 {result['index']}: {num_choices} 个选项")
        print(f"  问题: {result['question']}")
        print(f"  选项: {choices}")
        print()

print("="*60)
print("选项数量统计:")
print("="*60)
for num, count in sorted(choice_counts.items()):
    print(f"{num} 个选项: {count} 个样本")

print(f"\n最少选项数: {min_choices}")
print(f"最多选项数: {max_choices}")
print(f"\n总样本数: {len(data['results'])}")

# 检查是否所有样本都只有2-4个选项
if max_choices <= 4:
    print("\n[OK] 确认：所有样本的选项数都 <= 4，代码修改正确！")
else:
    print(f"\n[WARNING] 警告：存在 {max_choices} 个选项的样本，需要支持更多选项！")
