import numpy as np
import matplotlib.pyplot as plt

# Travel time range (分钟)
x = np.linspace(0, 500, 2000)

# 调参：根据图 4(a) 的趋势
tau = 25.0     # 控制指数饱和速度（越大越平缓）
k = 0.12       # Sigmoid 陡峭度（越大越陡）
x0 = 60.0      # Sigmoid 拐点（对应70分钟左右）

# 指数饱和函数
y_exp = 1 - np.exp(-x / tau)

# Sigmoid函数
y_sigmoid = 1 / (1 + np.exp(-k * (x - x0)))

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y_exp, label=rf'Exponential: $\tau={tau}$', linewidth=2)
plt.plot(x, y_sigmoid, '--', label=rf'Sigmoid: $k={k}, x_0={x0}$', linewidth=2)

plt.title("Fitted Functions Reflecting Range Anxiety vs. Travel Time")
plt.xlabel("Travel time (minutes)")
plt.ylabel("Normalized anxiety level")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
