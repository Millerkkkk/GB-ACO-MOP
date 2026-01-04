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
from dataclasses import dataclass
from decoder import EnergyModel, Decoder, calculate_cost
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
        self.costs: list[float] = []
        self.dists: list[float] = []
        self.full_routes: list[list[int]] = []
        self.travel_costs: list[float] = []
        self.dispatch_costs: list[float] = []
        self.service_costs: list[float] = []
        self.charging_costs: list[float] = []

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
    def __init__(self, all_nodes, total_distance_matrix, 
                 customer_ids, depot_ids, 
                 para_depot, prev_gb_last_cus, next_gb_center,
                 num_ants=10, num_iter=100,
                 alpha=1, beta=2, rho=0.1, epsilon = 0.1):
        """
        :param customer_ids: 全局客户节点编号列表 (customer节点, global index)
        :param total_distance_matrix: 全局距离矩阵 (所有节点的距离矩阵 customer, cs, depot)
        :param num_ants: 每轮蚂蚁数量
        :param num_iter: 迭代次数
        :param alpha: 信息素重要程度
        :param beta: 启发因子重要程度
        :param rho: 信息素挥发率
        :param epsilon: 精英强化系数
        """
        # 节点编号映射
        self.mapper = NodeIndexMap(customer_ids)
        self.customer_ids_global = customer_ids  # 全局编号
        self.customer_ids_local = self.mapper.to_local(customer_ids) # 局部编号

        # 提取局部客户距离矩阵
        self.customer_distance_matrix = total_distance_matrix[np.ix_(customer_ids, customer_ids)]

        self.all_nodes = all_nodes
        self.total_distance_matrix = total_distance_matrix
        self.depot_ids = depot_ids
        self.para_depot = para_depot
        self.prev_gb_last_cus = prev_gb_last_cus
        self.next_gb_center = next_gb_center

        self.n = len(customer_ids)
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon

        self.pheromone_matrix = np.ones((self.n, self.n))
        self.heuristic_matrix = 1 / (self.customer_distance_matrix + 1e-6)
        # 对角线设为 0
        np.fill_diagonal(self.heuristic_matrix, 0.0)
        np.fill_diagonal(self.pheromone_matrix, 0.0)

        # 记录全局最优解
        self.best_ant = None
        self.best_cost = float('inf')
    
    def _insert_depot(self, ant_route_global):
        if self.para_depot is None:
            return ant_route_global

        # 起点插入最近的出发depot
        if self.para_depot == "depart":
            first_cus = ant_route_global[0]
            dist_to_depots = self.total_distance_matrix[self.depot_ids, first_cus]
            nearest_depot = self.depot_ids[np.argmin(dist_to_depots)]
            ant_route_global.insert(0, nearest_depot)

        # 终点插入最近的返回depot
        elif self.para_depot == "return":
            last_cus = ant_route_global[-1]
            dist_to_depots = self.total_distance_matrix[self.depot_ids, last_cus]
            nearest_depot = self.depot_ids[np.argmin(dist_to_depots)]
            ant_route_global.append(nearest_depot)

        # 首尾都插
        elif self.para_depot == "depart & return":
            first_cus = ant_route_global[0]
            last_cus = ant_route_global[-1]

            dist_to_depots_start = self.total_distance_matrix[self.depot_ids, first_cus]
            dist_to_depots_end = self.total_distance_matrix[self.depot_ids, last_cus]

            nearest_depart_depot = self.depot_ids[np.argmin(dist_to_depots_start)]
            nearest_return_depot = self.depot_ids[np.argmin(dist_to_depots_end)]

            ant_route_global.insert(0, nearest_depart_depot)
            ant_route_global.append(nearest_return_depot)

        return ant_route_global

    def _insert_prev_gb_last_customer(self, ant_route_global):
        if self.prev_gb_last_cus is None:
            return ant_route_global
        else:
            ant_route_global.insert(0, self.prev_gb_last_cus)
            return ant_route_global

    def evaluate(self, ant_route_local):
        """
        individual: 局部客户编号 [0,1,2,...]
        distance_matrix: 局部客户编号 [0,1,2,...]
        """
        # 首先把局部ids转为 全局 ids
        ant_route_global = self.mapper.to_global(ant_route_local)

        total_dist = 0
        inserted_depot_route = self._insert_depot(ant_route_global)
        inserted_last_cus_route = self._insert_prev_gb_last_customer(inserted_depot_route)
        
        for i in range(len(inserted_last_cus_route) - 1):
            curr_node = inserted_last_cus_route[i]
            next_node = inserted_last_cus_route[i + 1]
            total_dist += self.total_distance_matrix[curr_node, next_node]
        
        if self.next_gb_center is not None:
            last_cus = self.all_nodes[inserted_last_cus_route[-1]].location()
            dist = np.linalg.norm(np.array(last_cus) - np.array(self.next_gb_center))
            total_dist += dist

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
                ant = Ant(self.customer_ids_local,
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
        
        best_route_global = self.mapper.to_global(self.best_ant.ant_route)

        return best_route_global, self.best_cost, fitness_list


class CustomersPlanning:
    def __init__(self, instance, gb_customers, para_depot, prev_gb_last_cus, next_gb_center, aca_params):
        self.instance = instance

        self.aco = AntColonyOptimizer(self.instance.nodes, 
                                      self.instance.distance_matrix, 
                                      gb_customers,
                                      self.instance.depot_ids, 
                                      para_depot, prev_gb_last_cus, next_gb_center,
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

    # start_time = time.time()

    # # 创建求解器
    # solver = EVRPSolver(file_path, aca_params)

    # best_ant, best_cost ,fitness_list, fitness_components_list = solver.run()

    # # 打印
    # if best_ant is not None:
    #     print(f"best_cost = {best_cost}")
    #     print(f"components_cost = {float(sum(best_ant.travel_costs)), 
    #                                float(sum(best_ant.dispatch_costs)), 
    #                                float(sum(best_ant.service_costs)), 
    #                                float(sum(best_ant.charging_costs))}")
    #     print(f"best_dist = {sum(best_ant.dists)}")
    #     print(f"best_routes = {best_ant.full_routes}")
    # else:
    #     print("No feasible solution found.")

    # end_time = time.time()
    # run_time = end_time - start_time

    # print(f"run_time = {run_time:.2f} s")

    # coordinates = [(node.x, node.y) for node in solver.instance.nodes]
    # plot_routes(coordinates, 
    #             best_ant.full_routes, 
    #             solver.instance.depot_ids, 
    #             solver.instance.cs_ids, 
    #             solver.instance.customer_ids)
    
    # plot_fitness(fitness_list)




















