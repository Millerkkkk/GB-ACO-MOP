#  coding: UTF-8  #
'''
@Project     : gbGenerate
@File        : GB_DBSCAN.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/07/12 11:32
'''


import os
import numpy as np
from scipy.spatial import distance
from pyflann import FLANN, set_distance_type    # 注意：pip install pyflann-py3
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import EVRPInstance, plot_clusters
import clusters_result

from sklearn.cluster import KMeans


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



# ******************************  Generate granular balls  *******************************
class GBsGenerator:
    def __init__(self, customer_idx, customer_data, split_k):
        self.ids = np.asarray(customer_idx, dtype=int)                # 全局ID
        self.coords = np.asarray(customer_data, dtype=float)        # (n,2)
        assert self.coords.ndim == 2 and self.coords.shape[1] == 2

        self.n = len(self.ids)
        self.gbs_list = [list(range(self.n))]  # 内部：局部索引球
        self.gbs_center = []
        self.gbs_radius = []

        # threshold for controlling the size of the granular ball.
        # k ∈ [0, 1]
        self.threshold = split_k * np.sqrt(self.n)

    def _gb_center_radius(self, gb_local_idx):
        '''
        Calculate the center and radius of the gbs
        '''
        pts = self.coords[np.array(gb_local_idx, dtype=int)]
        center = pts.mean(axis=0)
        radius = np.max(np.linalg.norm(pts - center, axis=1))
        return (float(center[0]), float(center[1])), float(radius)

    def _split_gb(self, gb_local_idx):
        '''
        Use K-Means to split a ball into two sub-balls.
        '''
        split_k = min(2, len(gb_local_idx))
        pts = self.coords[np.array(gb_local_idx, dtype=int)]
        # kmeans = KMeans(n_clusters=split_k, random_state=0).fit(pts)
        kmeans = KMeans(n_clusters=split_k).fit(pts)
        labels = kmeans.labels_

        sub_balls = []
        for single_label in range(split_k):
            sub_ball = [idx for idx, label in zip(gb_local_idx, labels) if label == single_label]
            sub_balls.append(sub_ball)
        return sub_balls

    def split(self):
        '''
        Split all balls if they have more points than a threshold.
        '''
        gb_list_new = []
        for gb_idx in self.gbs_list:
            if isinstance(gb_idx, int):
                gb_idx = [gb_idx]
            if len(gb_idx) < self.threshold:
                gb_list_new.append(gb_idx)
            else:
                gb_list_new.extend(self._split_gb(gb_idx))
        return gb_list_new

    def generate_gbs(self, plot_each_stage=False, plot_interval=1):
        """
        plot_each_stage=True: 每轮 split 后画
        plot_interval=N: 每 N 轮画一次（避免太多图）
        """
        it = 0
        if plot_each_stage:
            self.plot_stage(title=f"Stage 1-0: Init (1 GB)")

        while True:
            gbs_num = len(self.gbs_list)
            self.gbs_list = self.split()
            gbs_num_new = len(self.gbs_list)

            it += 1
            if plot_each_stage and (it % plot_interval == 0):
                self.plot_stage(title=f"Stage 1: After split iter {it} (GBs={gbs_num_new})")

            if gbs_num == gbs_num_new:
                break

        return self.gbs_list

    def assign_single_point_to_gbs(self):
        '''
        Assigning a ball containing a single point to the nearest multi-point ball
        :return: self.gbs_list
        '''
        # 先分离单点球与多点球
        new_merged_gbs = {}
        single_points = []

        for gb in self.gbs_list:
            if len(gb) == 1:
                single_points.append(gb[0])  # 单点的“局部索引”
            else:
                label = len(new_merged_gbs)
                new_merged_gbs[label] = gb[:]  # copy

        # 如果全是单点，兜底：不合并（或你也可以合成一个球）
        if len(new_merged_gbs) == 0:
            return self.gbs_list

        # 合并每个单点到最近的多点球
        for sp in single_points:
            sp_xy = self.coords[sp]
            min_dist = float("inf")
            best_label = None
            for label, gb in new_merged_gbs.items():
                gb_center = self.coords[np.array(gb, dtype=int)].mean(axis=0)
                d = np.linalg.norm(sp_xy - gb_center)
                if d < min_dist:
                    min_dist = d
                    best_label = label
            new_merged_gbs[best_label].append(sp)

        self.gbs_list = list(new_merged_gbs.values())
        return self.gbs_list

    def merge_overlaped_gbs(self):
        gbs_sum = len(self.gbs_list)

        centers = []
        radii = []
        for gb in self.gbs_list:
            c, r = self._gb_center_radius(gb)
            centers.append(c)
            radii.append(r)

        centers = np.array(centers, dtype=float)   # (m,2)
        radii = np.array(radii, dtype=float)       # (m,)

        unvisited = list(range(gbs_sum))
        cluster = [-1] * gbs_sum
        k = -1

        while unvisited:
            p = unvisited.pop(0)
            neighbors = []
            for i in range(gbs_sum):
                if i == p:
                    continue
                dis = np.linalg.norm(centers[i] - centers[p])
                if dis <= (radii[i] + radii[p]):
                    neighbors.append(i)

            k += 1
            cluster[p] = k

            for pi in neighbors:
                if pi in unvisited:
                    unvisited.remove(pi)
                    neighbors_pi = []
                    for j in range(gbs_sum):
                        if j == pi:
                            continue
                        dis_pi = np.linalg.norm(centers[j] - centers[pi])
                        if dis_pi <= (radii[j] + radii[pi]):
                            neighbors_pi.append(j)
                    for t in neighbors_pi:
                        if t not in neighbors:
                            neighbors.append(t)

                if cluster[pi] == -1:
                    cluster[pi] = k

        merged_gbs = {}
        for idx, label in enumerate(cluster):
            if label not in merged_gbs:
                merged_gbs[label] = self.gbs_list[idx][:]
            else:
                merged_gbs[label].extend(self.gbs_list[idx])

        self.gbs_list = list(merged_gbs.values())
        return self.gbs_list

    def merge_contained_gbs(self, merged_gbs):
        centers_radii = {label: self._gb_center_radius(gb) for label, gb in merged_gbs.items()}

        containment_relations = {}
        for label, (c, r) in centers_radii.items():
            for other_label, (oc, or_) in centers_radii.items():
                if label == other_label:
                    continue
                dist = np.linalg.norm(np.array(c) - np.array(oc))
                if dist + r <= or_:
                    containment_relations.setdefault(other_label, []).append(label)

        new_merged_gbs = {}
        for encloser, contained in containment_relations.items():
            if encloser not in new_merged_gbs:
                new_merged_gbs[encloser] = merged_gbs[encloser][:]
            for lab in contained:
                new_merged_gbs[encloser].extend(merged_gbs[lab])

        non_contained_labels = set(merged_gbs) - set(sum(containment_relations.values(), []))
        for lab in non_contained_labels:
            if lab not in new_merged_gbs:
                new_merged_gbs[lab] = merged_gbs[lab][:]

        return new_merged_gbs

    def run(self, merge_single_gb=True, merge_overlaped=True, merge_contained=True, plot_each_stage=False, plot_interval=1):
        # Stage 1: split 迭代（内部会画）
        self.generate_gbs(plot_each_stage=plot_each_stage, plot_interval=plot_interval)

        # Stage 2: 单点合并
        if merge_single_gb == True:
            self.assign_single_point_to_gbs()
            if plot_each_stage:
                self.plot_stage(title=f"Stage 2: After assign single points (GBs={len(self.gbs_list)})")

        # Stage 3: overlap 合并
        if merge_overlaped == True:
            self.merge_overlaped_gbs()
            if plot_each_stage:
                self.plot_stage(title=f"Stage 3: After merge overlap (GBs={len(self.gbs_list)})")

        # Stage 4: contain 合并
        if merge_contained == True:
            merged_dict = {i: gb[:] for i, gb in enumerate(self.gbs_list)}
            merged_dict = self.merge_contained_gbs(merged_dict)

            self.merge_contained_gbs(merged_dict)
            if plot_each_stage:
                self.plot_stage(title=f"Stage 4: After merge contain (GBs={len(self.gbs_list)})")

        # 输出 centers/radii（按最终球）
        self.gbs_center = []
        self.gbs_radius = []
        for gb in self.gbs_list:
            c, r = self._gb_center_radius(gb)
            self.gbs_center.append(c)
            self.gbs_radius.append(r)

        global_gbs = [[int(self.ids[i]) for i in gb_local] for gb_local in self.gbs_list]
        return global_gbs, self.gbs_center, self.gbs_radius


    def _recalc_centers_radii(self):
        """根据当前 self.gbs_list 重新计算 centers/radii（不改算法，只是方便画图）"""
        centers, radii = [], []
        for gb in self.gbs_list:
            c, r = self._gb_center_radius(gb)
            centers.append(c)
            radii.append(r)
        return centers, radii

    def plot_stage(self, title="", draw_circle=True, show_index=False, s=18):
        """画当前 self.gbs_list 对应的分球结果（使用离散颜色）"""
        centers, radii = self._recalc_centers_radii()

        n_gb = len(self.gbs_list)
        cmap = plt.cm.get_cmap("turbo", max(n_gb, 1))

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # 给每个点标记属于哪个 GB
        tmp_labels = np.full(self.n, -1, dtype=int)
        for gid, gb in enumerate(self.gbs_list):
            tmp_labels[np.array(gb, dtype=int)] = gid

        # 逐 GB 绘制点
        for gid in range(n_gb):
            mask = tmp_labels == gid
            if not np.any(mask):
                continue
            plt.scatter(self.coords[mask, 0], self.coords[mask, 1], color=cmap(gid), s=s)

        # 画圈
        if draw_circle:
            for gid, (c, r) in enumerate(zip(centers, radii)):
                circ = plt.Circle((c[0], c[1]), float(r), fill=False, color=cmap(gid), linewidth=1.0)
                ax.add_patch(circ)
                if show_index:
                    plt.text(c[0], c[1], str(gid), fontsize=10)

        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    




if __name__ == '__main__':
    # 读取数据
    instance_name = 'pr04_evrp'
    base_path = r"D:\02_Research\DataSet\C-mdvrptw-improved"

    # instance_name = 'rc201_21'
    # base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    instance = EVRPInstance(file_path)
    
    # customer_idx = np.array(list(cluster[1]))
    # customer_data = np.array([instance.nodes[i].location() for i in cluster[1]])

    customer_idx = np.array(instance.customer_ids)
    customer_data = np.array([instance.nodes[i].location() for i in instance.customer_ids])

    print(customer_idx)
    print(customer_data)


    # 聚类
    model = GBsGenerator(
        customer_idx, 
        customer_data, 
        split_k=0.8
    )
    global_gbs, gbs_center, gbs_radius = model.run(
        merge_single_gb=True,
        merge_overlaped=False,
        merge_contained=False,

        plot_each_stage=False,
        plot_interval=2  # 每轮 split 都画；如果太多图改成 2/5/10
    )
    print(len(global_gbs))
    print('clusters =', global_gbs)

    # 可视化
    data = customer_data

    # 构造 labels 数组：每个 customer 的 cluster id
    labels = np.full(len(customer_idx), -1, dtype=int)
    gid2local = {int(g): i for i, g in enumerate(customer_idx)}  # 一次性映射

    for cluster_id, group in enumerate(global_gbs):
        for gid in group:
            labels[gid2local[int(gid)]] = cluster_id


    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=20)
    # plt.title("GB-DBSCAN clustering result")
    plt.grid(False)
    plt.show()



    



