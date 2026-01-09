#  coding: UTF-8  #
'''
@Project     : ACCA
@File        : aca.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/07 16:04
'''

import copy
import numpy as np
import time
from dataclasses import dataclass
from decoder import EnergyModel, Decoder, calculate_cost
from utils import to_int_list, plot_routes, NodeIndexMap, EVRPInstance



# =========================
# 参数
# =========================

@dataclass
class CostParams:
    c1: float = 120.0    # fixed dispatch cost
    c2: float = 0.5    # travel cost per minute
    c3: float = 0.3    # service cost per minute
    c4: float = 0.6    # charging cost per minute


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
    def __init__(self, total_nodes_list, pheromone_matrix, heuristic_matrix, aco=None):
        """
        构建每个蚂蚁的路径（基于 customer节点）
        :param total_nodes_list: 所有客户节点编号
        :param pheromone_matrix: 信息素矩阵（应全局共享）
        :param heuristic_matrix: 启发式矩阵（通常为 1 / distance）
        """
        self.total_nodes_list = list(total_nodes_list)
        self.pheromone_matrix = pheromone_matrix
        self.heuristic_matrix = heuristic_matrix
        self.aco = aco

        start_node = np.random.choice(self.total_nodes_list)
        self.ant_route = [start_node]

        self.taboo_set = set(self.ant_route)
        self.unvisited = list(set(self.total_nodes_list) - self.taboo_set)
        self.current_node = self.ant_route[-1]

        # ===== ant 评估结果类属性 =====
        self.cost: float = float('inf')
        self.ra: float = float('inf')
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
        prev = self.current_node  # 👈 先保存上一点

        self.ant_route.append(next_node)
        self.taboo_set.add(next_node)
        self.unvisited.remove(next_node)
        self.current_node = next_node

        # ✅ 局部信息素更新：每走一步就更新刚走过的边
        if self.aco is not None:
            self.aco.local_update(prev, next_node)

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
    def __init__(self, all_nodes, total_distance_matrix, customer_ids, depot_ids, 
                 para_depot, prev_last, next_gb_center, 
                 state, decoder,
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
        self.prev_gb_last_cus = prev_last
        self.next_gb_center = next_gb_center

        self.state0 = state
        self.decoder = decoder

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
        self.best_cost_ant = None
        self.best_cost = float('inf')

        self.best_ra_ant = None
        self.best_ra = float('inf')



    def local_update(self, i, j):
        # 一个常见的局部更新形式：挥发 + 少量常数补偿
        # 你也可以用 1/d(i,j) 做补偿，下面给两种写法任选其一

        # 写法 A（更像 ACS）：tau = (1-rho)*tau + rho*tau0
        tau0 = 1.0
        self.pheromone_matrix[i, j] = (1 - self.rho) * self.pheromone_matrix[i, j] + self.rho * tau0

        # 写法 B（更贴你现在的风格）：tau = (1-rho)*tau + 1/d
        # self.pheromone_matrix[i, j] = (1 - self.rho) * self.pheromone_matrix[i, j] \
        #                               + 1.0 / (self.customer_distance_matrix[i, j] + 1e-6)


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

        # 在gb_customers 中插入depot, 
        inserted_depot_route = self._insert_depot(ant_route_global)
        inserted_last_cus_route = self._insert_prev_gb_last_customer(inserted_depot_route)

        start_state = copy.deepcopy(self.state0)
        end_state = self.decoder.simulate_within_GB(inserted_last_cus_route, start_state)
    
        if self.next_gb_center is not None:
            last_node = end_state["last_node"]
            total_ra, total_route_cost = self.decoder.simulate_to_gbCenter(
                last_node_from_last_gb=last_node,
                next_gb_center=self.next_gb_center,
                state=end_state,
            )

        return total_ra, total_route_cost, end_state
    
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

    def _deposit_elite(self, elite_ant, mode="cost"):
        route = elite_ant.ant_route
        f = elite_ant.cost
        ra = elite_ant.ra

        # 把 ra 转成 “越大越好”的 u（类似论文的满意度 u）
        # 因为你的 ra 是焦虑（越小越好），所以用 1/(ra+eps) 当作 u
        u = 1.0 / (ra + 1e-6)

        # 论文有类似 u/f 的思想：满意度越高、成本越低，强化越大
        bonus = self.epsilon * (u / (f + 1e-6))

        for i in range(len(route) - 1):
            n1, n2 = route[i], route[i+1]
            self.pheromone_matrix[n1, n2] += bonus

    def update_pheromones_dual_elite(self, ant_population, best_cost_ant, best_ra_ant):
        # 1) 蒸发
        self.pheromone_matrix *= (1 - self.rho)

        # 2) 普通蚂蚁贡献（你原来的逻辑保留）
        for ant in ant_population:
            route = ant.ant_route
            cost = ant.cost
            if cost <= 0 or not np.isfinite(cost):
                continue
            for i in range(len(route) - 1):
                n1, n2 = route[i], route[i+1]
                self.pheromone_matrix[n1, n2] += 1.0 / (cost + 1e-6)

        # 3) cost 精英额外强化
        self._deposit_elite(best_cost_ant, mode="cost")

        # 4) ra 精英额外强化
        self._deposit_elite(best_ra_ant, mode="ra")


    def run(self):
        fitness_ra_list = []
        fitness_cost_list = []

        # ===== 最优解容器：把 route + ra + cost + end_state 绑在一起，避免错配 =====
        best_cost_sol = None  # dict: {'ra':..., 'cost':..., 'end_state':...}
        best_ra_sol = None

        for gen in range(self.num_iter):
            ant_population = []

            for _ in range(self.num_ants):
                ant = Ant(self.customer_ids_local,
                          self.pheromone_matrix,
                          self.heuristic_matrix,
                          aco=self)
                ant.construct_solution(self.alpha, self.beta)
                ant_population.append(ant)

                # 更新全局最优
                ant.ra, ant.cost, end_state = self.evaluate(ant.ant_route)

                sol = {
                    "ra": ant.ra,
                    "cost": ant.cost,
                    "end_state": end_state,   # 下一 GB 要用：last_node、t、Q、load...
                }

                # cost 最优
                if (best_cost_sol is None) or (sol["cost"] < best_cost_sol["cost"]):
                    best_cost_sol = sol

                # ra 最优（ra 越小越好）
                if (best_ra_sol is None) or (sol["ra"] < best_ra_sol["ra"]):
                    best_ra_sol = sol

            # 双目标信息素更新
            best_cost_ant = min(ant_population, key=lambda a: a.cost)
            best_ra_ant   = min(ant_population, key=lambda a: a.ra)   # ra 越小越好（焦虑）
            self.update_pheromones_dual_elite(ant_population, best_cost_ant, best_ra_ant)

            # 单目标信息素更新
            # elite_ant = self.best_ant
            # self.update_pheromones(ant_population, elite_ant)

            # 记录最优适应度和代价
            fitness_cost_list.append(best_cost_sol["cost"])
            fitness_ra_list.append(best_cost_sol["ra"])

        return best_cost_sol, best_ra_sol, fitness_cost_list, fitness_ra_list


class CustomersPlanning:
    def __init__(self, instance, gb_customers, 
                 para_depot, prev_last, next_gb_center, 
                 state, decoder,
                 aca_params):
        self.instance = instance
        self.state = state
        self.decoder = decoder

        self.aco = AntColonyOptimizer(self.instance.nodes, 
                                      self.instance.distance_matrix, 
                                      gb_customers,
                                      self.instance.depot_ids, 
                                      para_depot, prev_last, next_gb_center,
                                      state=self.state, 
                                      decoder=self.decoder,
                                      **aca_params)

    def run(self):
        best_cost_sol, best_ra_sol, fitness_cost_list, fitness_ra_list  = self.aco.run()
        return best_cost_sol, best_ra_sol, fitness_cost_list, fitness_ra_list



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




















