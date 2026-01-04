#  coding: UTF-8  #
'''
@Project     : ACCA
@File        : aca.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/07 16:04
'''

import numpy as np
import time
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from utils import to_int_list, plot_routes, NodeIndexMap, EVRPInstance



# =========================
# 参数
# =========================

@dataclass
class ACOParams:
    num_ants: int = 30       # 蚂蚁数量
    num_iter: int = 100      # 迭代次数
    alpha: float = 1.0           # 信息素重要程度
    beta: float = 2.0            # 启发因子重要程度
    rho: float = 0.1           # 信息素挥发率
    epsilon: float = 0.1       # 精英蚂蚁加权


# ===== Ant 蚂蚁类 =====
class Ant:
    def __init__(self, total_nodes_list, pheromone_matrix, heuristic_matrix):
        """
        构建每个蚂蚁的路径（基于 customer节点）
        :param total_nodes_list: 所有客户节点编号
        :param pheromone_matrix: 信息素矩阵（应全局共享）
        :param heuristic_matrix: 启发式矩阵（通常为 1 / distance）
        """
        self.total_nodes_list = list(total_nodes_list)
        self.pheromone_matrix = pheromone_matrix
        self.heuristic_matrix = heuristic_matrix

        start_node = np.random.choice(self.total_nodes_list)
        self.ant_route = [start_node]

        self.taboo_set = set(self.ant_route)
        self.unvisited = list(set(self.total_nodes_list) - self.taboo_set)
        self.current_node = self.ant_route[-1]

        # ===== ant 评估结果类属性 =====
        self.cost: float = float('inf')

    def select_next_node(self, alpha=1, beta=2, greedy_prob=0.3):
        """
        计算转移概率并选择下一个节点
        """
        if not self.unvisited:
            return None

        candidates = self.unvisited
        tau = [self.pheromone_matrix[self.current_node, j] for j in candidates]
        eta = [self.heuristic_matrix[self.current_node, j] for j in candidates]

        # 计算分数
        score = [t**alpha * e**beta for t, e in zip(tau, eta)]
        total = sum(score)
        if total <= 0 or not np.isfinite(total):
            return np.random.choice(candidates)  # 兜底随机/或选距离最近
        probs = [s / total for s in score]

        # 贪心 or 轮盘赌
        if np.random.rand() < greedy_prob:
            next_node = candidates[np.argmax(probs)]
        else:
            next_node = np.random.choice(candidates, p=probs)

        return next_node

    def move_to(self, next_node):
        self.ant_route.append(next_node)
        self.taboo_set.add(next_node)
        self.unvisited.remove(next_node)
        self.current_node = next_node

    def construct_solution(self, alpha=1, beta=2):
        while self.unvisited:
            next_node = self.select_next_node(alpha, beta)
            if next_node is None:
                break
            self.move_to(next_node)

    def __repr__(self):
        return f"<Ant ant_route={to_int_list(self.ant_route)}>"


# ===== ACO 主程序类 =====
class AntColonyOptimizer:
    def __init__(self, gbs_idx, gbs_center, depots_location, 
                 num_ants=10, num_iter=100,
                 alpha=1, beta=2, rho=0.1, epsilon = 0.1):
        
        # 节点编号映射
        self.gbs_idx = gbs_idx
        self.gbs_center = gbs_center
        self.dist_matrix = cdist(gbs_center, gbs_center, metric='euclidean')
        self.depots_location = depots_location

        self.n = len(gbs_idx)
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon

        self.pheromone_matrix = np.ones((self.n, self.n))
        self.heuristic_matrix = 1 / (self.dist_matrix + 1e-6)
        # 对角线设为 0
        np.fill_diagonal(self.heuristic_matrix, 0.0)
        np.fill_diagonal(self.pheromone_matrix, 0.0)

        # 记录全局最优解
        self.best_ant = None
        self.best_cost = float('inf')
    
    def evaluate(self, ant_route):
        total_dist = 0

        depots = np.array(self.depots_location)
        first_customer_coord = np.array(self.gbs_center[ant_route[0]]).reshape(1, -1)
        distances = cdist(first_customer_coord, depots)[0]
        first_depot_dist = distances[np.argmin(distances)]

        last_customer_coord = np.array(self.gbs_center[ant_route[-1]]).reshape(1, -1)
        distances1 = cdist(last_customer_coord, depots)[0]
        last_depot_dist = distances1[np.argmin(distances1)]

        # 路径内部距离
        for i in range(len(ant_route) - 1):
            total_dist += self.dist_matrix[ant_route[i]][ant_route[i+1]]

        # 加上首尾 depot 距离
        total_dist += first_depot_dist
        total_dist += last_depot_dist

        return total_dist

    def update_pheromones(self, ant_population, elite_ant):
        """
        信息素更新：蒸发 + 普通蚂蚁贡献 + 精英蚂蚁贡献强化
        注意：此处为非闭合路径，首尾之间信息素没有更新
        """
        
        # 普通蚂蚁贡献
        delta_tau = np.zeros_like(self.pheromone_matrix)
        for ant in ant_population:
            route = ant.ant_route
            cost = ant.cost
            if cost <= 0:
                continue
            for i in range(len(route) - 1):
                n1, n2 = route[i], route[i+1]
                delta_tau[n1, n2] += 1 / cost

        # 精英蚂蚁贡献强化
        elite_delta_tau = np.zeros_like(self.pheromone_matrix)
        elite_route = elite_ant.ant_route
        elite_cost = elite_ant.cost
        if elite_cost <= 0:
            elite_cost = 1e-6
        for i in range(len(elite_route) - 1):
            n1, n2 = elite_route[i], elite_route[i+1]
            elite_delta_tau[n1, n2] += 1 / elite_cost  # 这里改成 elite_cost

        # 信息素更新
        # 蒸发 + 普通蚂蚁贡献 + 精英蚂蚁贡献强化
        self.pheromone_matrix = (1 - self.rho) * self.pheromone_matrix \
                            + delta_tau \
                            + self.epsilon * elite_delta_tau

    def run(self):
        fitness_list = []

        for gen in range(self.num_iter):
            ant_population = []

            for _ in range(self.num_ants):
                ant = Ant(self.gbs_idx,
                          self.pheromone_matrix,
                          self.heuristic_matrix)
                ant.construct_solution(self.alpha, self.beta)
                ant_population.append(ant)

                # 更新全局最优
                ant.cost = self.evaluate(ant.ant_route)

                if ant.cost < self.best_cost:
                    self.best_ant = ant
                    self.best_cost = ant.cost

            elite_ant = self.best_ant
            self.update_pheromones(ant_population, elite_ant)

            # 记录最优适应度和代价
            fitness_list.append(self.best_cost)

            # print(f"Iteration {gen+1}: Best cost = {self.best_cost}; Best routes = {self.best_ant.full_routes}")

        return self.best_ant.ant_route, self.best_cost, fitness_list


class GbPlanning:
    def __init__(self, gbs_dict, depots_location, aca_params):
        gbs_idx = [i for i in range(len(gbs_dict))]
        gbs_center = np.array([gbs_dict[i]["center"] for i in range(len(gbs_dict))])

        self.aco = AntColonyOptimizer(gbs_idx, 
                                      gbs_center, 
                                      depots_location, 
                                      **aca_params)

    def run(self):
        return self.aco.run()



if __name__ == "__main__":
    file_path = r"D:\02_Research\Project_python\GB-MDHOEVRP\large_instances(100customer21cs_10)\rc101_21.txt"

    aca_params = {
        "num_ants": 30,
        "num_iter": 200,
        "alpha": 1,
        "beta": 2,
        "rho": 0.1,
        "epsilon": 0.1,
    }

    start_time = time.time()

    # # 创建求解器
    # solver = GbPlanning(file_path, aca_params)

    # best_ant, best_cost ,fitness_list, fitness_components_list = solver.run()


    end_time = time.time()
    run_time = end_time - start_time

    print(f"run_time = {run_time:.2f} s")





















