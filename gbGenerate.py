#  coding: UTF-8  #
'''
@Project     : gbGenerate
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
class GenerateGBs:
    def __init__(self, data, k):
        self.data = data[['x', 'y']]
        self.gbs_list = [self.data.index.tolist()]
        self.gbs_center = []
        self.gbs_radius = []
        # threshold for controlling the size of the granular ball.
        # k ∈ [0, 1]
        self.threshold = k * np.sqrt(len(self.data))

    def _gb_center_radius(self, gb):
        '''
        Calculate the center and radius of the gbs
        '''
        x_center = self.data.loc[gb, 'x'].mean()
        y_center = self.data.loc[gb, 'y'].mean()
        gb_center = (x_center, y_center)
        gb_radius = max(np.sqrt((self.data.loc[gb, 'x'] - x_center) ** 2 + (self.data.loc[gb, 'y'] - y_center) ** 2))
        return gb_center, gb_radius

    def _split_gb(self, gb_idx):
        '''
        Use K-Means to split a ball into two sub-balls.
        '''
        split_k = min(2, len(gb_idx))
        gb = self.data.loc[gb_idx]
        kmeans = KMeans(n_clusters=split_k, random_state=0).fit(gb)
        labels = kmeans.labels_

        sub_balls = []
        for single_label in range(split_k):
            sub_ball = [idx for idx, label in zip(gb_idx, labels) if label == single_label]
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

    def generate_gbs(self):
        '''
        Split the ball iteratively until no new splits are generated
        '''
        while True:
            gbs_num = len(self.gbs_list)
            self.gbs_list = self.split()
            gbs_num_new = len(self.gbs_list)
            if gbs_num == gbs_num_new:
                print('break, all GBs have been generated')
                break
        return self.gbs_list

    def assign_single_point_to_gbs(self):
        '''
        Assigning a ball containing a single point to the nearest multi-point ball
        :return: self.gbs_list
        '''
        gbs_center, gbs_radius = [], []
        for gb in self.gbs_list:
            center, radius = self._gb_center_radius(gb)
            gbs_center.append(center)
            gbs_radius.append(radius)
        gbs_center = np.array(gbs_center)

        new_merged_gbs = {}
        single_point_gbs = []

        # Separate single point GBs and multi-point GBs
        for gb in self.gbs_list:
            if len(gb) == 1:
                single_point_gbs.append(gb[0])
            else:
                label = len(new_merged_gbs)
                new_merged_gbs[label] = gb

        # Merge single point GBs into nearest multi-point GBs
        for sp in single_point_gbs:
            min_distance = float('inf')
            closest_gb_label = None
            sp_center = self.data.loc[sp, ['x', 'y']].values

            for label, gbs in new_merged_gbs.items():
                gb_center = self.data.loc[gbs, ['x', 'y']].mean().values
                distance = np.linalg.norm(sp_center - gb_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_gb_label = label

            if closest_gb_label is not None:
                new_merged_gbs[closest_gb_label].append(sp)

        self.gbs_list = list(new_merged_gbs.values())
        return self.gbs_list

    def plot_merge_gbs(self, merged_gbs, centers_radii, plt_type=0):
        plt.figure()
        plt.axis()
        for i in range(len(merged_gbs)):
            # Plot points in each GB as black dots
            if plt_type == 0:
                plt.plot(self.data.loc[merged_gbs[i], 'x'], self.data.loc[merged_gbs[i], 'y'], '.', c='k')

            # Plot circles and their centers
            if plt_type == 0 or plt_type == 1:
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = centers_radii[i][0][0] + centers_radii[i][1] * np.cos(theta)
                y = centers_radii[i][0][1] + centers_radii[i][1] * np.sin(theta)
                plt.plot(x, y, c='r', linewidth=0.8)
                center_marker = 'x' if plt_type == 0 else '.'
                plt.plot(centers_radii[i][0][0], centers_radii[i][0][1], center_marker, color='r')
                # Add index label near the circle center
                plt.text(centers_radii[i][0][0], centers_radii[i][0][1], str(i), color='blue', fontsize=12, ha='right')

        plt.show()

    def merge_contained_gbs(self, merged_gbs):
        '''
        Merge contained balls, i.e., when one ball is completely contained within another ball, combine them
        '''
        # Compute centers and radii for all GBs
        centers_radii = {label: self._gb_center_radius(gbs) for label, gbs in merged_gbs.items()}

        # Identify containment relationships
        containment_relations = {}
        for label, (center, radius) in centers_radii.items():
            for other_label, (other_center, other_radius) in centers_radii.items():
                if label != other_label:
                    distance = np.linalg.norm(np.array(center) - np.array(other_center))
                    if distance + radius <= other_radius:  # label's circle is inside other_label's circle
                        containment_relations.setdefault(other_label, []).append(label)

        # Merge contained GBs into their enclosers and remove the original contained GBs
        new_merged_gbs = {}
        for encloser, contained in containment_relations.items():
            # Initialize the list for the encloser if not already present
            if encloser not in new_merged_gbs:
                new_merged_gbs[encloser] = merged_gbs[encloser][:]  # Use a copy of the original list
            # Extend the encloser's list with the contents of each contained GB
            for label in contained:
                new_merged_gbs[encloser].extend(merged_gbs[label])

        # Add GBs that are not contained by any other GB
        non_contained_labels = set(merged_gbs) - set(sum(containment_relations.values(), []))
        for label in non_contained_labels:
            if label not in new_merged_gbs:
                new_merged_gbs[label] = merged_gbs[label][:]  # Use a copy of the original list
        return new_merged_gbs

    def merge_gbs(self):
        '''
        Combining overlapping or contained balls
        '''
        gbs_sum = len(self.gbs_list)
        gbs_center, gbs_radius = [], []
        for gb in self.gbs_list:
            center, radius = self._gb_center_radius(gb)
            gbs_center.append(center)
            gbs_radius.append(radius)
        gbs_center = np.array(gbs_center)
        unvisited = [i for i in range(gbs_sum)]
        cluster = [-1 for _ in range(gbs_sum)]
        k = -1

        while len(unvisited) > 0:
            p = unvisited[0]
            unvisited.remove(p)
            neighbors = []
            for i in range(gbs_sum):
                if i != p:
                    dis = np.linalg.norm(gbs_center[i] - gbs_center[p])
                    if dis <= (gbs_radius[i] + gbs_radius[p]):
                        neighbors.append(i)
            k += 1
            cluster[p] = k
            for pi in neighbors:
                if pi in unvisited:
                    unvisited.remove(pi)
                    neighbors_pi = []
                    for j in range(gbs_sum):
                        if j != pi:
                            dis_pi = np.linalg.norm(gbs_center[j] - gbs_center[pi])
                            if dis_pi <= (gbs_radius[j] + gbs_radius[pi]):
                                neighbors_pi.append(j)
                    for t in neighbors_pi:
                        if t not in neighbors:
                            neighbors.append(t)
                if cluster[pi] == -1:
                    cluster[pi] = k

        merged_gbs = {}
        for idx, label in enumerate(cluster):
            if label not in merged_gbs:
                merged_gbs[label] = self.gbs_list[idx]
            else:
                merged_gbs[label].extend(self.gbs_list[idx])

        new_merged_gbs = self.merge_contained_gbs(merged_gbs)
        self.gbs_list = list(new_merged_gbs.values())
        return self.gbs_list

    def run(self, merge_single_gb=1, merge=1):
        '''
        Performs the ball generation and merging process and finally draws the ball
        :param merge_single_gb: 1: combined single point ball; 0: not combined
        :param merge: 1: Combining overlapping or mutually contained balls; 0: not combined
        :return:
        '''
        self.generate_gbs()
        if merge_single_gb == 1:
            self.assign_single_point_to_gbs()

        if merge == 1:
            self.merge_gbs()

        for gb in self.gbs_list:
            center, radius = self._gb_center_radius(gb)
            self.gbs_center.append(center)
            self.gbs_radius.append(radius)
        # self.plot_gbs()
        return self.gbs_list, self.gbs_center, self.gbs_radius

    # Set the blue color using an RGB or hex value
      # This is a commonly used blue color; you can adjust if necessary

    def plot_gbs(self, plt_type=0):
        plt.rcParams['font.family'] = 'Times New Roman'
        blue_color = '#1f77b4'
        plt.figure(figsize=(8, 6))  # Adjust figure size if necessary
        ax = plt.gca()  # Get current axis

        # Plot points in each GB and circles with their centers
        for i in range(len(self.gbs_list)):
            # Plot points in each GB as black dots using the provided color
            if plt_type == 0:
                plt.scatter(self.data.loc[self.gbs_list[i], 'x'], self.data.loc[self.gbs_list[i], 'y'], color='#3179B5',
                            s=45)

            # Plot circles and their centers
            if plt_type == 0 or plt_type == 1:
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = self.gbs_center[i][0] + self.gbs_radius[i] * np.cos(theta)
                y = self.gbs_center[i][1] + self.gbs_radius[i] * np.sin(theta)
                plt.plot(x, y, c='r', linewidth=0.8)  # Circle in red

                # Plot center
                center_marker = 'x' if plt_type == 0 else '.'
                plt.scatter(self.gbs_center[i][0], self.gbs_center[i][1], marker=center_marker, color='r',
                            s=45)  # Center with size 45

                # Add index label near the circle center
                plt.text(self.gbs_center[i][0], self.gbs_center[i][1], str(i), color='blue', fontsize=18, ha='left',
                         va='top')

        # Set axis scaling and tick marks dynamically based on data range
        x_min, x_max = self.data['x'].min(), self.data['x'].max()
        y_min, y_max = self.data['y'].min(), self.data['y'].max()

        ax.set_xlim([x_min - 5, x_max + 5])
        ax.set_ylim([y_min - 5, y_max + 5])

        # Set dynamic ticks based on the data range
        ax.set_xticks(range(int(x_min) - 5, int(x_max) + 10, 10))
        ax.set_yticks(range(int(y_min) - 14, int(y_max) + 10, 10))

        ax.tick_params(axis='both', which='major', labelsize=18, direction='in', top=True, right=True)

        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)  # 隐藏Y轴的第一个刻度标签
        ax.xaxis.get_major_ticks()[0].label1.set_visible(False)  # 隐藏Y轴的第一个刻度标签

        # plt.xlabel('X-label', fontsize=20, labelpad=10)  # Adjust the labelpad to move away from the axis
        # plt.ylabel('Y-label', fontsize=20, labelpad=10)  # Adjust the labelpad to move away from the axis

        # plt.title('Stage 2-1: Generate GBs', fontsize=22, pad=15)

        # Use tight layout and show plot
        plt.tight_layout()
        plt.show()


    def plot_gbs1(self, plt_type=0):
        plt.figure()
        plt.axis()
        for i in range(len(self.gbs_list)):
            # Plot points in each GB as black dots
            if plt_type == 0:
                plt.plot(self.data.loc[self.gbs_list[i], 'x'], self.data.loc[self.gbs_list[i], 'y'], '.', c='k')

            # Plot circles and their centers
            if plt_type == 0 or plt_type == 1:
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = self.gbs_center[i][0] + self.gbs_radius[i] * np.cos(theta)
                y = self.gbs_center[i][1] + self.gbs_radius[i] * np.sin(theta)
                plt.plot(x, y, c='r', linewidth=0.8)
                center_marker = 'x' if plt_type == 0 else '.'
                plt.plot(self.gbs_center[i][0], self.gbs_center[i][1], center_marker, color='r')
                # Add index label near the circle center
                plt.text(self.gbs_center[i][0], self.gbs_center[i][1], str(i), color='blue', fontsize=12, ha='right')

        plt.show()


def generate_gbs(dataframe, cluster_set, k, merge=1, merge_single_gb=1):
    # cluster_set 是一个 set，而不是 dict
    cluster_data = dataframe.loc[list(cluster_set)]
    gbs_generator = GenerateGBs(cluster_data, k)
    gbs, centers, gbs_radius = gbs_generator.run(merge_single_gb, merge)
    return gbs, centers, gbs_radius



    



