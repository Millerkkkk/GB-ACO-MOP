#  coding: UTF-8  #
'''
@Project     : GB_FLANN_DBSCAN
@File        : GB_DBSCAN.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/07/12 11:32
'''


import numpy as np
from scipy.spatial import distance
from pyflann import FLANN, set_distance_type    # 注意：pip install pyflann-py3
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import EVRPInstance, plot_clusters
import clusters_result


class GranularBall:
    def __init__(self, gb_points_idx, all_points_data):
        """
        初始化粒球对象。
        :param gb_points_idx: 粒球包含的点的索引列表（在原始数据中的索引）(全局索引 global)
        :param all_points_data: 原始所有点的坐标数据
        """
        # 自动转为 NumPy array，便于后续计算
        if isinstance(all_points_data, list):
            self.data = np.array(all_points_data)
        elif isinstance(all_points_data, np.ndarray):
            self.data = all_points_data
        else:
            raise TypeError("data must be a list of (x, y), a numpy array, or a DataFrame with ['x', 'y']")

        self.points_idx = list(gb_points_idx)
        self.points_data = self.data[self.points_idx]
        self.center = self._calculate_center()
        self.radius = self._calculate_radius()
        self.density = self._calculate_density()

    def _calculate_center(self):
        return np.mean(self.points_data, axis=0)
    
    def _calculate_radius(self):
        return np.max(np.linalg.norm(self.points_data - self.center, axis=1))
    
    def _calculate_density(self):
        return self.radius
    
    def __repr__(self):
        return f"<GB points={len(self.points_idx)}, center={self.center}, radius={self.radius:.3f}, density={self.density:.3f}>"


class GbDbscanCluster:
    def __init__(self, customer_idx, customer_data, split_k=0.3, core_ratio=0.92):
        """
        初始化 GB-DBSCAN 聚类器
        :param customer_idx: customer 节点的索引
        :param customer_data: customer 节点的数据
        :param split_k: granular ball 分裂的邻居数参数系数
        :param core_ratio: 核心粒球占比，用于密度阈值划分
        """

        self.customer_idx = np.array(customer_idx)
        self.customer_data = np.array(customer_data)

        self.split_k = split_k
        self.core_ratio = core_ratio

        self.dist_matrix = distance.cdist(self.customer_data, self.customer_data)
        self.point_labels = np.full(len(self.customer_idx), -1, dtype=int)

        self.gbs = []
        self.core_gbs = []
        self.non_core_gbs = []

        self.local_to_global = {i: self.customer_idx[i] for i in range(len(self.customer_idx))}
        self.global_to_local = {idx: i for i, idx in enumerate(self.customer_idx)}

        
    def generate_granular_balls(self):
        set_distance_type('euclidean')
        flann = FLANN()

        num = self.customer_data.shape[0]
        K = int(np.ceil(np.sqrt(num)) * self.split_k)
        K = min(max(K, 1), num)

        nearest_neighbors, _ = flann.nn(
            self.customer_data, self.customer_data,
            num_neighbors=K, algorithm="kmeans",
            branching=32, iterations=7, checks=-1
        )

        visited = np.full(num, False)
        for i in range(num):
            if not visited[i]:
                local_points_idx = np.unique(nearest_neighbors[i]).astype(int)
                visited[local_points_idx] = True
                gb = GranularBall(local_points_idx, self.customer_data)
                self.gbs.append(gb)

    def partition_core_non_core(self):
        # 先清空，避免累加
        self.core_gbs, self.non_core_gbs = [], []

        if len(self.gbs) == 0:
            self.density_threshold = None
            return

        densities = np.array([gb.density for gb in self.gbs], dtype=float)
        k = int(len(self.gbs) * self.core_ratio)
        k = min(max(k, 1), len(self.gbs))
        threshold = np.sort(densities)[k - 1]

        for gb in self.gbs:
            if gb.density <= threshold:
                self.core_gbs.append(gb)
            else:
                self.non_core_gbs.append(gb)

        self.density_threshold = threshold

    def cluster_core_gbs(self):
        num_core = len(self.core_gbs)
        self.core_labels = []  # 重置

        if num_core == 0:
            return

        core_centers = np.array([gb.center for gb in self.core_gbs])
        core_radii  = np.array([gb.radius for gb in self.core_gbs])

        labels = [-1] * num_core
        unvisited = list(range(num_core))
        cur_label = -1

        while unvisited:
            p = unvisited.pop(0)
            neighbors = []
            for i in range(num_core):
                if i != p:
                    dist = np.linalg.norm(core_centers[i] - core_centers[p])
                    if dist <= (core_radii[i] + core_radii[p]):
                        neighbors.append(i)
            cur_label += 1
            labels[p] = cur_label

            for pi in neighbors:
                if pi in unvisited:
                    unvisited.remove(pi)
                    labels[pi] = cur_label
                    # 扩展 pi 的邻居
                    for j in range(num_core):
                        if j != pi:
                            dist = np.linalg.norm(core_centers[j] - core_centers[pi])
                            if dist <= (core_radii[j] + core_radii[pi]) and (j not in neighbors):
                                neighbors.append(j)

        self.core_labels = labels  # 存标签，核心GB对象仍在 self.core_gbs

        # 给核心GB里的点贴上标签
        for i, gb in enumerate(self.core_gbs):
            for idx in gb.points_idx:     # idx 是局部索引
                self.point_labels[idx] = labels[i]

    def assign_non_core_gbs(self):
        if len(self.core_gbs) == 0:
            # 兜底策略：全部打成一个簇 0
            self.point_labels[:] = 0
            return

        core_centers = np.array([gb.center for gb in self.core_gbs])
        for gb in self.non_core_gbs:
            known_labels, known_points, unknown_points = [], [], []

            for idx in gb.points_idx:
                if self.point_labels[idx] != -1:
                    known_labels.append(self.point_labels[idx])
                    known_points.append(idx)
                else:
                    unknown_points.append(idx)

            if known_points:
                dists = self.dist_matrix[np.ix_(unknown_points, known_points)]
                nearest = np.argmin(dists, axis=1)
                for i, idx in enumerate(unknown_points):
                    self.point_labels[idx] = known_labels[nearest[i]]
            else:
                dists = distance.cdist([gb.center], core_centers)
                nearest_core = np.argmin(dists)
                assign_label = self.core_labels[nearest_core]  # ✅ 用核心标签
                for idx in gb.points_idx:
                    self.point_labels[idx] = assign_label

    def _rebuild_gbs_and_results(self):
        labels = np.unique(self.point_labels)
        labels = labels[labels != -1]
        labels = np.sort(labels)

        new_gbs, local_gbs, global_gbs, gbs_center, gbs_radius = [], [], [], [], []

        for label in labels:
            local_idxs = np.where(self.point_labels == label)[0]
            local_gbs.append(local_idxs.tolist())

            gb = GranularBall(local_idxs.astype(int), self.customer_data)
            new_gbs.append(gb)
            gbs_center.append(gb.center)
            gbs_radius.append(gb.radius)

            global_gb = [int(self.customer_idx[i]) for i in local_idxs]
            global_gbs.append(global_gb)

        self.gbs = new_gbs
        return global_gbs, gbs_center, gbs_radius

    def fit(self):
        # —— 重置状态，避免重复 fit 累加旧数据 ——
        self.point_labels[:] = -1
        self.gbs = []
        self.core_gbs = []
        self.non_core_gbs = []
        self.core_labels = []  # 新增：核心GB的簇标签数组

        self.generate_granular_balls()
        self.partition_core_non_core()
        self.cluster_core_gbs()
        self.assign_non_core_gbs()

        # 用最终标签重建 gbs + 结果，确保顺序一致
        global_gbs, gbs_center, gbs_radius = self._rebuild_gbs_and_results()

        return global_gbs, gbs_center, gbs_radius



def plot_labels(data, labels, title="Clustering Result"):
    """
    可视化聚类标签结果。-1 的点作为噪声点显示为黑色，图例放在图外。
    """
    data = np.array(data)
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for label in unique_labels:
        mask = labels == label
        color = 'k' if label == -1 else None
        plt.scatter(
            data[mask, 0], data[mask, 1],
            c=color,
            cmap='tab10', s=30,
            label=f"Cluster {label}" if label != -1 else "Noise"
        )

    plt.title(title)
    plt.grid(True)

    # 👉 图例移到图像外侧右边
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # 收缩坐标轴宽度
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))          # 图例放右侧中部
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # 读取数据
    file_path = r'D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)\c201_21.txt'


    instance = EVRPInstance(file_path)
    cluster = clusters_result.rc101_21
    
    # customer_idx = np.array(list(cluster[1]))
    # customer_data = np.array([instance.nodes[i].location() for i in cluster[1]])

    customer_idx = np.array(instance.customer_ids)
    customer_data = np.array([instance.nodes[i].location() for i in instance.customer_ids])


    # 聚类
    model = GbDbscanCluster(
        customer_idx, 
        customer_data, 
        split_k=0.3, 
        core_ratio=0.93
    )
    global_gbs, gbs_center, gbs_radius = model.fit()
    print(len(global_gbs))
    print('clusters =', global_gbs)

    # 可视化
    data = customer_data

    # 构造 labels 数组：每个 customer 的 cluster id
    labels = np.full(len(customer_idx), -1)
    for cluster_id, group in enumerate(global_gbs):
        for global_idx in group:
            local_idx = customer_idx.tolist().index(global_idx)
            labels[local_idx] = cluster_id

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=20)
    # plt.title("GB-DBSCAN clustering result")
    plt.grid(False)
    plt.show()





    



