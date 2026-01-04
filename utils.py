#  coding: UTF-8  #
'''
@Project     : utils
@File        : utils.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/07 21:58
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def to_int_list(arr):
    # 转为纯 int 列表
    return list(map(int, arr))


def plot_clusters(clusters, all_nodes, depot_idx, cs_idx, customer_idx, title="Clustered Customers", save_path=None):
    """
    可视化聚类结果，节点类型分颜色，聚类客户按不同颜色标出
    参数：
        clusters: dict[int → set[int]]，每个聚类对应的客户索引
        all_nodes: list，包含所有节点对象（需包含 location 属性）
        depot_list, cs_list, customer_list: list[int]，不同类型节点的索引
        title: 图标题
        save_path: 如果非空，则保存图像到路径
    """
    coords = np.array([node.location() for node in all_nodes])
    fig, ax = plt.subplots(figsize=(8, 7))
    # # 节点散点
    # depot = ax.scatter(coords[depot_idx,0],  coords[depot_idx,1],  s=40,  marker='^', c='#D02C1B', label='Depot')
    # cs  = ax.scatter(coords[cs_idx,0],   coords[cs_idx,1],   s=40,  marker='s', c='#25A61F', label='CS')
    # customer = ax.scatter(coords[customer_idx,0], coords[customer_idx,1], s=50,  marker='o', c='#3179B5', label='Customer')

    # 不同聚类不同颜色
    colors = cm.get_cmap('tab20', len(clusters))  # 最多支持10种颜色
    for i, customer_set in clusters.items():
        c_idx = list(customer_set)
        cluster_coords = coords[c_idx]
        ax.scatter(cluster_coords[:,0], cluster_coords[:,1], 
                   s=50, marker='o', label=f'Cluster {i}')
    
    # 图例、标题
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # 边界设置
    xmn, xmx = np.nanmin(coords[:,0]), np.nanmax(coords[:,0])
    ymn, ymx = np.nanmin(coords[:,1]), np.nanmax(coords[:,1])
    dx = max((xmx - xmn) * 0.05, 1e-6)
    dy = max((ymx - ymn) * 0.05, 1e-6)
    ax.set_xlim(xmn - dx, xmx + dx)
    ax.set_ylim(ymn - dy, ymx + dy)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()
    return fig, ax


def plot_routes(coordinates, routes, depot_list, cs_list, customer_list, title="EVRP Routes", annotate=True, save_path=None):
    """可视化 EVRP 路线，区分不同节点类型，支持路径标注与保存。
    """
    coords = np.asarray(coordinates, dtype=float)
    dep_idx = np.asarray(depot_list, dtype=int)
    cs_idx  = np.asarray(cs_list, dtype=int)
    cust_idx= np.asarray(customer_list, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 7))
    # 节点散点
    dep = ax.scatter(coords[dep_idx,0],  coords[dep_idx,1],  s=80,  marker='s', c='#D02C1B', label='Depot')
    cs  = ax.scatter(coords[cs_idx,0],   coords[cs_idx,1],   s=70,  marker='^', c='#25A61F', label='CS')
    cus = ax.scatter(coords[cust_idx,0], coords[cust_idx,1], s=35,  marker='o', c='#3179B5', label='Customer')

    # 画路线（不重复打点）
    for k, r in enumerate(routes or []):
        r = list(r)
        if len(r) < 2: 
            continue
        pts = coords[np.asarray(r, dtype=int)]
        ax.plot(pts[:,0], pts[:,1], linewidth=2, alpha=0.9, label=None)

        if annotate:
            for step, nid in enumerate(r):
                x, y = coords[int(nid)]
                ax.text(x + 0.5, y + 0.8, f"{nid}", fontsize=9, 
                        ha='center', va='center', color="black")
                
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)

    # 自适应边界
    xmn, xmx = np.nanmin(coords[:,0]), np.nanmax(coords[:,0])
    ymn, ymx = np.nanmin(coords[:,1]), np.nanmax(coords[:,1])
    dx = max((xmx - xmn) * 0.05, 1e-6)
    dy = max((ymx - ymn) * 0.05, 1e-6)
    ax.set_xlim(xmn - dx, xmx + dx)
    ax.set_ylim(ymn - dy, ymx + dy)

    # 图例（仅节点类别）
    ax.legend(handles=[cus, cs, dep], loc='best')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.show()
    return fig, ax


def plot_history_fitness(fitness_list, title="Best Fitness over Generations", save_path=None):
    """绘制进化过程中的 history最优适应度变化。
    """
    fitness = np.asarray(fitness_list, dtype=float)
    gens = np.arange(1, len(fitness)+1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gens, fitness, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Generation'); ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()
    return fig, ax


# =========================
# local <-> global 之间index转换
# =========================
class NodeIndexMap:
    """局部编号 ↔ 全局编号映射"""
    def __init__(self, node_list):
        self.global_ids = list(node_list)
        self.local2global = {i: nid for i, nid in enumerate(self.global_ids)}
        self.global2local = {nid: i for i, nid in enumerate(self.global_ids)}

    def to_local(self, seq):
        return [self.global2local[x] for x in seq]

    def to_global(self, seq):
        return [self.local2global[x] for x in seq]
    

# =========================
# Node 节点类
# =========================
class Node:
    def __init__(self, idx: int, node_type: str, x: float, y: float,
                 demand: float, ready_time: float, due_time: float, service_time: float):
        self.idx = idx
        self.node_type = node_type  # 'd' = depot, 'c' = customer, 'f' = charging station
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        self.nearest_depot: int = -1
        self.nearest_cs: int = -1

    def is_depot(self):
        return self.node_type == 'd'

    def is_customer(self):
        return self.node_type == 'c'

    def is_charging_station(self):
        return self.node_type == 'f'

    def location(self):
        return (self.x, self.y)

    def __repr__(self):
        return f"<Node {self.idx}: type={self.node_type}, ({self.x},{self.y}), demand={self.demand}>"


# =========================
# 实例载入（加载数据类 + 最近depot/CS写入到 Node 属性）
# =========================
class EVRPInstance:
    def __init__(self, file_path):
        """
        将 file_path 转为具体的实例数据
        """
        self.nodes = []          # List[Node]
        self.customer_ids = []   # List[int]
        self.depot_ids = []      # List[int]
        self.cs_ids = []         # List[int]
        self.distance_matrix: np.ndarray

        self.load_data(file_path)

    def load_data(self, file_path):
        nodes = []
        depot_ids = []
        cs_ids = []
        customer_ids = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if idx == 0:
                continue  # Skip header
            line = line.strip()
            if not line:
                continue
            items = line.split()
            if len(items) < 8:
                continue
            try:
                node_type    = items[1]
                x, y         = float(items[2]), float(items[3])
                demand       = float(items[4])
                ready_time   = float(items[5])
                due_time     = float(items[6])
                service_time = float(items[7])
            except (ValueError, IndexError):
                continue

            node = Node(idx-1, node_type, x, y, demand, ready_time, due_time, service_time)
            nodes.append(node)

            if node_type == 'd':
                depot_ids.append(idx - 1)
            elif node_type == 'f':
                cs_ids.append(idx - 1)
            elif node_type == 'c':
                customer_ids.append(idx - 1)

        node_count = len(nodes)
        distance_matrix = np.zeros((node_count, node_count))
        for i in range(node_count):
            for j in range(node_count):
                distance_matrix[i, j] = np.linalg.norm(
                    np.array(nodes[i].location()) - np.array(nodes[j].location())
                )

        # 设置类属性
        self.nodes = nodes
        self.customer_ids = customer_ids
        self.depot_ids = depot_ids
        self.cs_ids = cs_ids
        self.distance_matrix = distance_matrix









