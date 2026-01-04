#  coding: UTF-8  #
'''
@Project     : clustering
@File        : clustering.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/23 16:53
'''


import os
import numpy as np
import math
import random
from utils import EVRPInstance, plot_clusters


class ImprovedKMeans:
    def __init__(self, all_nodes, distance_matrix, customer_ids, vehicle_capacity, threshold=10.0, random_state=None):
        self.all_nodes = all_nodes
        self.distance_matrix = distance_matrix
        self.customer_ids = customer_ids
        self.vehicle_capacity = vehicle_capacity
        self.threshold = threshold
        self.random_state = random_state
        self.clusters = {}
        self.cluster_capacities = {}
        self.clusters_center = {}

    def fit(self, max_iter=100):
        self.k = self._get_cluster_num()
        if self.random_state is not None:
            random.seed(self.random_state)
        # # 初始化cluster中心点
        self.clusters_center_idx = random.sample(self.customer_ids, self.k)
        self.clusters_center = {i: np.array(self.all_nodes[idx].location()) for i, idx in enumerate(self.clusters_center_idx)}

        # 迭代过程
        for iter in range(max_iter):
            unallocated = set(self.customer_ids)
            self.clusters = {i: set() for i in range(self.k)}
            self.cluster_capacities = {i: self.vehicle_capacity for i in range(self.k)}

            while unallocated:
                customer_idx = max(unallocated, key=lambda idx: self.all_nodes[idx].demand)
                # 计算customer_idx到各中心距离
                dists = []
                for i in range(self.k):
                    d = np.linalg.norm(self.all_nodes[customer_idx].location() - self.clusters_center[i])
                    dists.append(d)

                # 按距离升序选择cluster
                sorted_indices = np.argsort(dists)
                for i in sorted_indices:
                    if self.cluster_capacities[i] >= self.all_nodes[customer_idx].demand:
                        self.clusters[i].add(customer_idx)
                        self.cluster_capacities[i] -= self.all_nodes[customer_idx].demand
                        unallocated.remove(customer_idx)
                        break
            
            # 重新计算中心              
            new_clusters_center = self._calculate_centroids()
            if self._has_center_shifted(new_clusters_center):
                self.clusters_center = new_clusters_center
                continue
            else:
                return self.clusters
        return self.clusters
    
    def _get_cluster_num(self):
        total_demand = sum([self.all_nodes[c].demand for c in self.customer_ids])
        num = math.ceil(total_demand / self.vehicle_capacity)
        return num

    def _calculate_centroids(self):
        new_centers = {}
        for i in range(self.k):
            customers = list(self.clusters[i])
            if len(customers) == 0:
                center = self.clusters_center[i]
            else:
                coords = np.vstack([self.all_nodes[c].location() for c in customers])
                center = coords.mean(axis=0)
            new_centers[i] = center
        return new_centers

    def _has_center_shifted(self, new_centers):
        for i in range(self.k):
            old_center = self.clusters_center[i]
            new_center = new_centers[i]
            if np.linalg.norm(new_center - old_center) > self.threshold:
                return True
        return False





if __name__ == "__main__":
    # c10_: random_state = 5
    # r10_: random_state = 5
    # file_path = r"D:\02_Research\Project_python\GB-MDHOEVRP\large_instances(100customer21cs_10)\c102_21.txt"

    instance_name = 'pr04_evrp'
    base_path = r"D:\02_Research\DataSet\C-mdvrptw-improved"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    instance = EVRPInstance(file_path)
    clustering = ImprovedKMeans(instance.nodes, instance.distance_matrix, instance.customer_ids,
                                vehicle_capacity=650, threshold=10.0, random_state=None)
    clusters = clustering.fit()

    plot_clusters(clusters, instance.nodes, instance.depot_ids, instance.cs_ids, instance.customer_ids)
    print(clusters)
    





