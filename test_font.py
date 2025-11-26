"""测试matplotlib中文字体显示"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图
fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, '中文测试：样本 #1\n【问题】这是一个测试\n【选项】A. 选项一', 
        ha='center', va='center', fontsize=14,
        bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title('中文字体测试', fontsize=16, fontweight='bold')
plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
print("Font test image saved to: font_test.png")
print("If you can see Chinese characters in the image, the font is working correctly.")
