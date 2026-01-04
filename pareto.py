import matplotlib.pyplot as plt





def pareto_front(points, eps=1e-9):
    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            # q 支配 p（最小化）
            if (
                q[0] <= p[0] + eps and
                q[1] <= p[1] + eps and
                (q[0] < p[0] - eps or q[1] < p[1] - eps)
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return pareto



points = [
    (1148.9632269957117, 0.7383787323100406),
    (1170.354501846426, 0.7539761171028896),
    (1161.0487566449262, 0.7179713820433813),
    (1160.8391802331314, 0.754683471316061),
    (1163.936323936053, 0.7118316121367172),
    (1144.9093963502364, 0.7055868626908823),
    (1156.0784581872297, 0.7406309886079033),
    (1171.2592005826548, 0.816586541146979),
    (1165.9564081707597, 1.1184799876615086),
    (1182.0349069625345, 0.8599191372848677),
    (1161.072310849985, 0.7242480288706669),
]



pareto_points = pareto_front(points)


# 所有点
x_all = [p[0] for p in points]
y_all = [p[1] for p in points]

# 帕累托点
x_pareto = [p[0] for p in pareto_points]
y_pareto = [p[1] for p in pareto_points]

plt.figure(figsize=(7, 5))

# 所有解
plt.scatter(x_all, y_all, alpha=0.6, label="All solutions")

# 帕累托解
plt.scatter(x_pareto, y_pareto, s=80, label="Pareto front")

# 如果你想连线（可选）
pareto_sorted = sorted(pareto_points, key=lambda x: x[0])
plt.plot(
    [p[0] for p in pareto_sorted],
    [p[1] for p in pareto_sorted],
    linestyle="--",
)

plt.xlabel("Total cost")
plt.ylabel("Anxiety")
plt.title("Pareto Front (Cost vs Anxiety)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
