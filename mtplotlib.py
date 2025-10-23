import matplotlib.pyplot as plt
import numpy as np

# داده‌ها
DDPG=[-60.807,  -40.341,  -24.243,  -10.915,  17.472,  33.378,  59.688,  72.624,  82.878,  79.590,  69.423,  67.017,  57.251,  58.297, 63.538,  73.974,  76.711, 93.856, 94.027, 94.297, 94.269]
TD3=[-80.807, -40.341, -34.243, -30.915, 7.472, 13.378, 39.688, 32.624, 62.878, 69.590, 79.423, 87.017, 87.251, 88.297, 83.538, 93.974, 94.711, 93.856, 94.027, 94.297, 94.269]
SAC=[-45.807, -50.341, -54.243, -40.915, 5.472, 23.378, 29.688, 32.624, 42.878, 59.590, 69.423, 77.017, 87.251, 92.197, 93.538, 93.974, 94.711, 94.856, 94.027, 94.297, 94.269]

# اپیزودها باید دقیقا هم‌اندازه لیست داده‌ها باشند (41 عدد)
episodes = list(range(1, 1+20*len(DDPG), 20))

plt.figure(figsize=(12, 6))

# رسم نمودارها با استایل‌های خواسته شده
plt.plot(episodes, DDPG, '-', linewidth=3, label="DDPG")                     # خط تنها
plt.plot(episodes, TD3, '-s', linewidth=3, markersize=12, label="TD3")   # خط + دایره
plt.plot(episodes, SAC, '-o', linewidth=3, markersize=12, label="SAC")     # خط + مربع

# تنظیمات محور و فونت‌ها
plt.xlabel("Episodes", fontsize=18)
plt.ylabel("Reward", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)

plt.title("مقایسه الگوریتم‌های DDPG, TD3 و SAC", fontsize=20)
plt.show()
# Creating the requested black-and-white hatched bar chart and saving it to /mnt/data

# Data
labels = ['DDPG', 'TD3', 'SAC']
times = [2652.92, 4526.87, 5472.27]

# Plot settings
fig, ax = plt.subplots(figsize=(8,5))
x = np.arange(len(labels))
width = 0.6
hatches = ['///', '///', '///']  # distinct hatch patterns

# Draw bars as white-filled with black edges and hatches
bars = []
for xi, t, h in zip(x, times, hatches):
    bar = ax.bar(xi, t, width, facecolor='white', edgecolor='black', hatch=h, linewidth=1.2)
    bars.append(bar)

# Annotate times on each bar (placed above the bar)
for xi, t in zip(x, times):
    ax.text(xi, t + max(times)*0.02, f"{t:.2f}s", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Aesthetics
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title('Training Time by Algorithm', fontsize=14)
ax.grid(axis='y', linestyle='--', linewidth=0.7)  # grid on
ax.set_axisbelow(True)  # grid lines below bars

plt.tight_layout()

plt.show()
