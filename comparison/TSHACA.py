#  coding: UTF-8  #
'''
@Project     : TSHACA: A two-stage hybrid ant colony algorithm for multi-depot half-open time-dependent electric vehicle routing problem
@File        : TSHACA.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/09/05 14:45
'''


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import numpy as np
import time
from dataclasses import dataclass
from decoder import EnergyModel, Decoder, calculate_cost
from utils import to_int_list, plot_routes, NodeIndexMap, EVRPInstance
import clusters_result

from clustering import ImprovedKMeans



# =========================
# 参数
# =========================



@dataclass
class CostParams:
    c1: float = 120.0    # fixed dispatch cost
    c2: float = 0.5    # travel cost per minute
    c3: float = 0.3    # service cost per minute
    c4: float = 0.6    # charging cost per minute



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

    def select_next_node(self, alpha, beta, rand0):
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

        if np.random.rand() < rand0:
            # 贪婪选择
            next_node = candidates[np.argmax(score)]
        else:
            # 概率选择
            total = sum(score)
            if total <= 0 or not np.isfinite(total):
                return np.random.choice(candidates)
            probs = [s / total for s in score]
            next_node = np.random.choice(candidates, p=probs)

        return next_node

    def move_to(self, next_node):
        self.ant_route.append(next_node)
        self.taboo_set.add(next_node)
        self.unvisited.remove(next_node)
        self.current_node = next_node

    def construct_solution(self, alpha, beta, rand0):
        while self.unvisited:
            next_node = self.select_next_node(alpha, beta, rand0)
            if next_node is None:
                break
            self.move_to(next_node)

    def __repr__(self):
        return f"<Ant ant_route={to_int_list(self.ant_route)}>"


# ===== ACO 主程序类 =====
class AntColonyOptimizer:
    def __init__(self, instance, customer_ids, energy_model,
                 num_ants=30, num_iter=200,
                 alpha=1, beta=2, rho=0.2, epsilon=1.0, rand0=0.3,
                 f=10, tau_min=0.05, tau_max=2.0):
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
        # 解包出 set，再转换为 list
        if isinstance(customer_ids, list) and isinstance(customer_ids[0], set):
            customer_ids = list(customer_ids[0])
        elif isinstance(customer_ids, set):
            customer_ids = list(customer_ids)

        # 节点编号映射
        self.mapper = NodeIndexMap(customer_ids)
        self.customer_ids_global = customer_ids  # 全局编号
        self.customer_ids_local = self.mapper.to_local(customer_ids) # 局部编号

       

        self.instance = instance
        self.energy_model = energy_model
        self.all_nodes = instance.nodes
        self.total_distance_matrix = instance.distance_matrix
        self.depot_ids = instance.depot_ids

        # 提取局部客户距离矩阵
        self.customer_distance_matrix = self.total_distance_matrix[np.ix_(customer_ids, customer_ids)]

        self.n = len(customer_ids)
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        self.rand0 = rand0
        self.f = f  # ← 新增：信息素释放常数
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.pheromone_matrix = np.ones((self.n, self.n))
        self.heuristic_matrix = 1 / (self.customer_distance_matrix + 1e-6)
        # 对角线设为 0
        np.fill_diagonal(self.heuristic_matrix, 0.0)
        np.fill_diagonal(self.pheromone_matrix, 0.0)

        # 记录全局最优解
        self.best_ant = None
        self.best_cost = float('inf')

        self.decoder = Decoder(self.instance.nodes, self.instance.distance_matrix, self.energy_model, self.instance.cs_ids, 
                               Q_max=40, margin_energy=0.2, radius_km=3.0, soc_anxiety=0.5)
    
    def _get_nearest_depot(self, customer):
        dist_to_depots = self.instance.distance_matrix[self.instance.depot_ids, customer]
        nearest_depot = self.instance.depot_ids[np.argmin(dist_to_depots)]
        return nearest_depot
    
    def _evaluate(self, flattened_route_local):
        route_global = self.mapper.to_global(flattened_route_local)

        route = route_global.copy()
        route_demand = sum([self.instance.nodes[i].demand for i in route])

        # 插入depot
        start_depot = self._get_nearest_depot([route[0]])
        end_depot = self._get_nearest_depot([route[-1]])
        full_route = [start_depot] + route + [end_depot]

        # 解码ant_route, 插入充电站
        decode_result = self.decoder.decode_backtrack_charging(full_route, route_demand)

        (decode_route, travel_dist, travel_time, charging_time, service_time, num_charge,
        updated_Q, updated_departure_time, updated_load) = decode_result

        dispatch_ev = 1
        # 计算 total cost
        costParams = CostParams()
        route_cost, cost_detail = calculate_cost(dispatch_ev, travel_time, charging_time, service_time, costParams)

        return (route_cost, travel_dist, decode_route, num_charge, 
                cost_detail.travel_cost, cost_detail.dispatch_cost, cost_detail.service_cost, cost_detail.charging_cost)

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
                delta_tau[n1, n2] += self.f / cost

        # 精英蚂蚁贡献强化
        elite_delta_tau = np.zeros_like(self.pheromone_matrix)
        elite_route = elite_ant.ant_route
        elite_cost = elite_ant.cost if elite_ant.cost > 0 else 1e-6
        for i in range(len(elite_route) - 1):
            n1, n2 = elite_route[i], elite_route[i+1]
            elite_delta_tau[n1, n2] += self.f / elite_cost

        # 信息素更新
        # 蒸发 + 普通蚂蚁贡献 + 精英蚂蚁贡献强化
        self.pheromone_matrix = (1 - self.rho) * self.pheromone_matrix \
                            + delta_tau \
                            + self.epsilon * elite_delta_tau
        
        # 加上 τ_min ~ τ_max 限制
        self.pheromone_matrix = np.clip(self.pheromone_matrix, self.tau_min, self.tau_max)

    def run(self):
        fitness_list = []
        best = None

        for gen in range(self.num_iter):
            ant_population = []

            for _ in range(self.num_ants):
                ant = Ant(self.customer_ids_local,
                          self.pheromone_matrix,
                          self.heuristic_matrix)
                
                ant.construct_solution(self.alpha, self.beta, self.rand0)
                ant_population.append(ant)

                # 更新全局最优
                (cost_fwd, dist_fwd, decode_fwd, num_charge_fwd,
                travel_fwd, dispatch_fwd, service_fwd, charge_fwd) = self._evaluate(ant.ant_route)

                ant.cost = cost_fwd

                if ant.cost < self.best_cost:
                    self.best_ant = ant
                    self.best_cost = ant.cost
                    best = (cost_fwd, dist_fwd, decode_fwd, num_charge_fwd,
                            travel_fwd, dispatch_fwd, service_fwd, charge_fwd)

            elite_ant = self.best_ant
            self.update_pheromones(ant_population, elite_ant)

            # 记录最优适应度和代价
            fitness_list.append(self.best_cost)

        return best






aca_params = {
    "num_ants": 30, 
    "num_iter": 200,
    "alpha": 1, 
    "beta": 2, 
    "rho": 0.2,
    "epsilon": 1.0, 
    "rand0": 0.3, 
    "f": 10, 
    "tau_min": 0.05, 
    "tau_max": 2.0
}



def run(instance, clusters, energy_model, aca_params, seed=None):
    rng = np.random.default_rng(seed)

    total_cost = 0.0
    decoded_routes, routes_cost, routes_dist = [], [], []
    total_num_charge = 0
    travel_cost_list, dispatch_cost_list, service_cost_list, charging_cost_list = [], [], [], []

    for cid, cluster in clusters.items():
        assert len(cluster) > 0, f"Cluster {cid} is empty!"

        aco = AntColonyOptimizer(instance, cluster,energy_model, **aca_params)
        best = aco.run()
        
        (cost_fwd, dist_fwd, decode_fwd, num_charge_fwd,
        travel_fwd, dispatch_fwd, service_fwd, charge_fwd) = best

        # 汇总
        decoded_routes.append(decode_fwd)
        routes_cost.append(cost_fwd)
        routes_dist.append(dist_fwd)
        total_num_charge += num_charge_fwd
        travel_cost_list.append(travel_fwd)
        dispatch_cost_list.append(dispatch_fwd)
        service_cost_list.append(service_fwd)
        charging_cost_list.append(charge_fwd)
        total_cost += cost_fwd
        
    # # === 绘图与返回 ===
    # total_coors = [instance.nodes[i].location() for i in range(len(instance.nodes))]
    # plot_routes(total_coors, decoded_routes, instance.depot_ids, instance.cs_ids, instance.customer_ids)

    return {
        "decoded_routes": decoded_routes,
        "routes_cost": routes_cost,
        "routes_dist": routes_dist,
        "total_num_charge": total_num_charge,
        "total_cost": total_cost,
        "travel_cost_list": travel_cost_list,
        "dispatch_cost_list": dispatch_cost_list,
        "service_cost_list": service_cost_list,
        "charging_cost_list": charging_cost_list,
    }


# python -m compareAlgorithm.TSHACA

if __name__ == '__main__':

    # ========== 参数设置 ==========
    instance_name = 'rc203_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    # 设置重复运行次数
    N_RUNS = 20   # 改成 20

    # ========== 存储结果 ==========
    results = {
        "all_routes": [],
        "routes_dist": [],
        "routes_cost": [],
        "dispatch_cost": [],
        "travel_cost": [],
        "service_cost": [],
        "charging_cost": [],
        "routes_num_charge": [],
        "running_time": []
    }

    # ========== 多次运行 ==========
    for run_id in range(N_RUNS):
        start_time = time.time()

        # --- 加载实例 ---
        instance = EVRPInstance(file_path)
        instance_clusters = getattr(clusters_result, instance_name)  # 如果 clusters_result 是 dict，可以改成 clusters_result[instance_name]
        energy_model = EnergyModel()

        solver = run(instance, instance_clusters, energy_model, aca_params)

        end_time = time.time()
        running_time = end_time - start_time

        print(run_id, sum(solver["routes_cost"]))

        # --- 存储结果 ---
        results["all_routes"].append(solver["decoded_routes"])
        results["routes_dist"].append(sum(solver["routes_dist"]))
        results["routes_cost"].append(sum(solver["routes_cost"]))
        results["dispatch_cost"].append(sum(solver["dispatch_cost_list"]))
        results["travel_cost"].append(sum(solver["travel_cost_list"]))
        results["service_cost"].append(sum(solver["service_cost_list"]))
        results["charging_cost"].append(sum(solver["charging_cost_list"]))
        results["routes_num_charge"].append(solver["total_num_charge"])
        results["running_time"].append(running_time)

    # ========== 统计结果 ==========
    print('Instance:', instance_name)
    print(f"\n===== Final Results over {N_RUNS} runs =====")

    # 找到最佳解（以 routes_cost 最小为准）
    best_idx = np.argmin(results["routes_cost"])
    print("Best run index =", best_idx)
    print("Best routes =", results["all_routes"][best_idx])

    # 输出每个指标的统计
    for key, values in results.items():
        if key == "all_routes":
            continue  # 不对路径做统计
        arr = np.array(values)
        print(f"{key:15s} -> mean: {arr.mean():.2f}, best: {arr.min():.2f}, worst: {arr.max():.2f}, std: {arr.std():.2f}")







