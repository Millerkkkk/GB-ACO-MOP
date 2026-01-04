#  coding: UTF-8  #
'''
@Project     : LocalSearch
@File        : localSearch.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/14 15:35
'''


import numpy as np
import random
import copy
from collections import defaultdict


class DestroyOperator:
    def __init__(self, all_nodes, total_distance_matrix, removal_ratio=0.2, pd=6):
        self.all_nodes = all_nodes
        self.distance_matrix = total_distance_matrix
        self.removal_ratio = removal_ratio
        self.pd = pd    # 控制随机性的参数， 通常为 1：均匀分布， >1：越靠前的元素被选中概率越高, ∞：永远选第一个

    def _select_by_pd(self, sorted_customers, q):
        selected = set()
        while len(selected) < q:
            y = random.random()
            idx = int((y ** self.pd) * len(sorted_customers))
            selected.add(sorted_customers[min(idx, len(sorted_customers) - 1)])
        return list(selected)
    
    def _calculate_similarity(self, c1, c2, w_dist=1.0, w_demand=0.5, w_time=1.0, w_route=1.0, w_mode=1.0):
        node1 = self.all_nodes[c1]
        node2 = self.all_nodes[c2]
        dist = self.distance_matrix[c1][c2]
        demand_diff = abs(node1['demand'] - node2['demand'])
        time_diff = abs(node1['tw_start'] - node2['tw_start'])
        same_route = int(node1.get('route_id', -1) == node2.get('route_id', -1))
        same_mode = int(node1.get('mode', '') == node2.get('mode', ''))
        sim = - (w_dist * dist + w_demand * demand_diff + w_time * time_diff
                 - w_route * same_route - w_mode * same_mode)
        return sim
    
    def _shaw_removal(self, route):
        q = max(1, int(len(route) * self.removal_ratio))
        removed = set()
        customers = [c for c in route if c not in removed]

        # 初始随机选一个
        current = random.choice(customers)
        removed.add(current)

        while len(removed) < q:
            similarity_list = []

            for other in customers:
                if other in removed:
                    continue
                sim = self._calculate_similarity(current, other)
                similarity_list.append((other, sim))

            # 按相似度降序排列
            similarity_list.sort(key=lambda x: -x[1])
            L = [c[0] for c in similarity_list]
            y = random.random()
            idx = int((y ** self.pd) * len(L))
            selected = L[min(idx, len(L) - 1)]
            removed.add(selected)
            current = selected  # 下一个起点
        
        new_route = [c for c in route if c not in removed]
        return new_route, list(removed)

    def _worst_cost_removal(self, route, cost_function):
        q = max(1, int(len(route) * self.removal_ratio))
        removal_scores = []
        for customer in route:
            prev_idx = route.index(customer) - 1
            next_idx = route.index(customer) + 1
            if 0 <= prev_idx < len(route) and 0 <= next_idx < len(route):
                prev_c = route[prev_idx]
                next_c = route[next_idx]
                old_cost = self.distance_matrix[prev_c][customer] + self.distance_matrix[customer][next_c]
                new_cost = self.distance_matrix[prev_c][next_c]
                gain = old_cost - new_cost
                removal_scores.append((customer, gain))
            else:
                removal_scores.append((customer, 0))

        removal_scores.sort(key=lambda x: -x[1])
        removed = self._select_by_pd([x[0] for x in removal_scores], q)
        new_route = [c for c in route if c not in removed]
        return new_route, removed

    def _worst_distance_removal(self, route):
        q = max(1, int(len(route) * self.removal_ratio))
        removal_scores = []
        for customer in route:
            prev_idx = route.index(customer) - 1
            next_idx = route.index(customer) + 1
            if 0 <= prev_idx < len(route) and 0 <= next_idx < len(route):
                prev_c = route[prev_idx]
                next_c = route[next_idx]
                old_dist = self.distance_matrix[prev_c][customer] + self.distance_matrix[customer][next_c]
                new_dist = self.distance_matrix[prev_c][next_c]
                gain = old_dist - new_dist
                removal_scores.append((customer, gain))
            else:
                removal_scores.append((customer, 0))

        removal_scores.sort(key=lambda x: -x[1])
        removed = self._select_by_pd([x[0] for x in removal_scores], q)
        new_route = [c for c in route if c not in removed]
        return new_route, removed

    def _history_based_removal(self, route):
        q = max(1, int(len(route) * self.removal_ratio))
        removal_scores = []
        for customer in route:
            current_cost = self._calculate_customer_cost(route, customer)
            best_cost = self.historical_best_cost.get(customer, current_cost)
            self.historical_best_cost[customer] = min(best_cost, current_cost)
            score = current_cost - best_cost
            removal_scores.append((customer, score))

        removal_scores.sort(key=lambda x: -x[1])
        removed = self._select_by_pd([x[0] for x in removal_scores], q)
        new_route = [c for c in route if c not in removed]
        return new_route, removed
    
    def _worst_time_removal(self, route):
        q = max(1, int(len(route) * self.removal_ratio))
        removal_scores = []
        for customer in route:
            node = self.all_nodes[customer]
            tw_length = max(1, node.due_time - node.ready_time)
            relative_diff = (node.service_time - node.ready_time) / tw_length
            removal_scores.append((customer, relative_diff))

        removal_scores.sort(key=lambda x: -x[1])
        removed = self._select_by_pd([x[0] for x in removal_scores], q)
        new_route = [c for c in route if c not in removed]
        return new_route, removed

    def _random_removal(self, route):
        q = max(1, int(len(route) * self.removal_ratio))
        removed = random.sample(route, q)
        new_route = [c for c in route if c not in removed]
        return new_route, removed

    def remove(self, method, route):
        if method == "shaw":
            return self._shaw_removal(route)
        elif method == "worst_cost":
            return self._worst_cost_removal(route, cost_function)
        elif method == "worst_distance":
            return self._worst_distance_removal(route)
        elif method == "history":
            return self._history_based_removal(route)
        elif method == "worst_time":
            return self._worst_time_removal(route)
        elif method == "random":
            return self._random_removal(route)
        else:
            raise ValueError(f"Unknown removal method: {method}")


class RepairOperator:
    def __init__(self, all_nodes, total_distance_matrix, regret_g=2):
        self.all_nodes = all_nodes
        self.distance_matrix = total_distance_matrix
        self.regret_g = regret_g    # 后悔值，计算时考虑的候选插入位置数量；g 表示我们考虑前 g 个最好的插入位置来计算“后悔”有多大。

    def _insertion_cost(self, route, customer, position):
        if not route:
            return 0
        if position == 0:
            next_customer = route[0]
            return self.distance_matrix[customer][next_customer]
        elif position == len(route):
            prev_customer = route[-1]
            return self.distance_matrix[prev_customer][customer]
        else:
            prev_customer = route[position - 1]
            next_customer = route[position]
            removed = self.distance_matrix[prev_customer][next_customer]
            added = self.distance_matrix[prev_customer][customer] + self.distance_matrix[customer][next_customer]
            return added - removed

    def _best_insertion_position(self, route, customer):
        best_position = 0
        best_cost = float("inf")
        for i in range(len(route) + 1):
            cost = self._insertion_cost(route, customer, i)
            if cost < best_cost:
                best_cost = cost
                best_position = i
        return best_position

    def _greedy_insertion(self, route, removed_customers):
        route = route.copy()
        while removed_customers:
            best_customer = None
            best_position = None
            best_cost = float("inf")
            for customer in removed_customers:
                for i in range(len(route) + 1):
                    cost = self._insertion_cost(route, customer, i)
                    if cost < best_cost:
                        best_cost = cost
                        best_customer = customer
                        best_position = i
            route.insert(best_position, best_customer)
            removed_customers.remove(best_customer)
        return route

    def _regret_insertion(self, route, removed_customers):
        route = route.copy()
        while removed_customers:
            regret_list = []
            for customer in removed_customers:
                costs = []
                for i in range(len(route) + 1):
                    cost = self._insertion_cost(route, customer, i)
                    costs.append(cost)
                costs.sort()
                if len(costs) >= self.regret_g:
                    regret = sum(costs[1:self.regret_g]) - costs[0]
                else:
                    regret = 0
                regret_list.append((customer, regret, costs[0]))
            regret_list.sort(key=lambda x: (-x[1], x[2]))  # Maximize regret, break tie with min cost
            best_customer = regret_list[0][0]
            best_position = self._best_insertion_position(route, best_customer)
            route.insert(best_position, best_customer)
            removed_customers.remove(best_customer)
        return route

    def _random_insertion(self, route, removed_customers):
        route = route.copy()
        while removed_customers:
            customer = random.choice(removed_customers)
            position = self._best_insertion_position(route, customer)
            route.insert(position, customer)
            removed_customers.remove(customer)
        return route

    def insert(self, method, route, removed_customers):
        if method == "greedy":
            return self._greedy_insertion(route, removed_customers)
        elif method == "regret":
            return self._regret_insertion(route, removed_customers)
        elif method == "random":
            return self._random_insertion(route, removed_customers)
        else:
            raise ValueError(f"Unknown insertion method: {method}")




class LocalSearch:
    def __init__(self, all_nodes, total_distance_matrix, depot_ids, removal_ratio=0.2, loc_penalty=0.1):
        self.all_nodes = all_nodes
        self.total_distance_matrix = total_distance_matrix
        self.depot_ids = depot_ids
        self.removal_ratio = removal_ratio
        self.loc_penalty = loc_penalty

    def _compute_dist_similarity(self, i, j, flattened_routes):
        d_ij = self.total_distance_matrix[i][j]
        d_max = max(self.total_distance_matrix[i][k] for k in flattened_routes)
        return d_ij / d_max if d_max > 0 else 0

    def _compute_same_route(self, i, j, routes):
        # 检查是否为同一个 loc_cluster
        for route in routes:
            if i in route and j in route:
                return 1
        return 0
    
    def _compute_relateness(self, i, j, routes, flattened_routes):
        rd = self._compute_dist_similarity(i, j, flattened_routes)
        sr = self._compute_same_route(i, j, routes)
        denom = rd + sr
        return 1 / denom if denom > 0 else 0

    def remove_customers(self, routes_cluster):
        routes = copy.deepcopy(routes_cluster)
        order = [node for route in routes_cluster for node in route]

        if not order:
            return routes, [], []

        num_removed = max(1, int(len(order) * self.removal_ratio))
        order_copy = order.copy()
        removed_customers = []

        # 随机移除第一个节点
        first_node = random.choice(order_copy)
        removed_customers.append(first_node)
        order_copy.remove(first_node)

        # 根据相似度选择更多节点
        while len(removed_customers) < num_removed and order_copy:
            cr = random.choice(removed_customers)

            similarity_list = []
            for ci in order_copy:
                rij = self._compute_relateness(cr, ci, routes, order)
                similarity_list.append((ci, rij))
            
            # 降序排列：相关性大的排前面
            similarity_list.sort(key=lambda x: x[1], reverse=True)
            selected_customer = similarity_list[0][0]

            removed_customers.append(selected_customer)
            order_copy.remove(selected_customer)
        
        # 从原始 routes 中移除这些客户
        remove_set = set(removed_customers)
        updated_routes = []
        for route in routes:
            new_route = [n for n in route if n not in remove_set]
            if new_route:  # 路径非空才保留
                updated_routes.append(new_route)

        return updated_routes, removed_customers, order_copy

    def _get_nearest_depot(self, customer):
        dist_to_depots = self.total_distance_matrix[self.depot_ids, customer]
        nearest_depot = self.depot_ids[np.argmin(dist_to_depots)]
        return nearest_depot
    
    def _distance_increase(self, route, node, pos):
        if not route:
            # 空路径：old = 0，new = depot(node)→node→depot(node)
            d0 = self._get_nearest_depot(node)
            new_dist = float(self.total_distance_matrix[d0, node] + self.total_distance_matrix[node, d0])
            return new_dist  # old=0

        node_depot = self._get_nearest_depot(node)

        if pos == 0:
            prev_node = route[0]
            prev_depot = self._get_nearest_depot(prev_node)
            old_dist = self.total_distance_matrix[prev_depot][prev_node]
            new_dist = self.total_distance_matrix[node_depot][node] + self.total_distance_matrix[node][prev_node]

        elif pos == len(route):
            prev_node = route[-1]
            prev_depot = self._get_nearest_depot(prev_node)

            old_dist = self.total_distance_matrix[prev_node][prev_depot]
            new_dist = self.total_distance_matrix[prev_node][node] + self.total_distance_matrix[node][node_depot]

        else:
            prev_node = route[pos - 1]
            next_node = route[pos]

            old_dist = self.total_distance_matrix[prev_node][next_node]
            new_dist = self.total_distance_matrix[prev_node][node] + self.total_distance_matrix[node][next_node]
        
        return new_dist - old_dist

    def reinsert_customers(self, flattened_route, removed_customers):
        """
        将 removed_customers 插入到 flattened_route 中，按需求优先插入靠前位置。
        self.loc_penalty: 控制位置惩罚的权重，值越大越倾向把高需求客户放前面。
        """
        route = flattened_route.copy()
        removed = sorted(removed_customers, key=lambda c: self.all_nodes[c].demand, reverse=True)

        while removed:
            best_score = float('inf')
            best_cust = None
            best_insert_pos = -1

            # 当前考虑的客户（高需求优先）
            c_j = removed[0]

            for pos in range(len(route) + 1):
                DI_jp = self._distance_increase(route, c_j, pos)
                pos_penalty = pos / (len(route) + 1)
                cost_score = DI_jp + self.loc_penalty * pos_penalty

                if cost_score < best_score:
                    best_score = cost_score
                    best_cust = c_j
                    best_insert_pos = pos

            if best_cust is not None and best_insert_pos != -1:
                route.insert(best_insert_pos, best_cust)

            # 从待插入集中移除
            removed.pop(0)

        return route


    def _distance_increase_chain(self, routes, ridx, pos, node):
        """
        适用于“多段仅作为显示，整条路径是一条链”的增量计算。
        - 只在全局起点/终点接仓库；
        - 段与段之间：上一段末 → 下一段首 直接相连；
        - 在 routes[ridx] 的下标 pos 处插入 node，返回距离增量 Δ = new - old
        """
        M = self.total_distance_matrix

        # 找到全局“前驱”和“后继”节点（跨段考虑）
        def _prev_node(routes, ridx, pos):
            if pos > 0 and len(routes[ridx]) > 0:
                return routes[ridx][pos - 1]
            # pos == 0，向前找上一段最后一个非空节点
            k = ridx - 1
            while k >= 0:
                if routes[k]:
                    return routes[k][-1]
                k -= 1
            return None  # 全局最前，无前驱 → 用仓库

        def _next_node(routes, ridx, pos):
            r = routes[ridx]
            if pos < len(r):
                return r[pos]
            # pos 在尾部，向后找下一段第一个非空节点
            k = ridx + 1
            while k < len(routes):
                if routes[k]:
                    return routes[k][0]
                k += 1
            return None  # 全局最后，无后继 → 用仓库

        pred = _prev_node(routes, ridx, pos)
        succ = _next_node(routes, ridx, pos)

        # old 边：pred→succ（全局起点用 depot→succ；全局终点用 pred→depot；两端都无时 old=0）
        if pred is None and succ is None:
            old_dist = 0.0
        elif pred is None:
            depot_succ = self._get_nearest_depot(succ)
            old_dist = float(M[depot_succ, succ])
        elif succ is None:
            depot_pred = self._get_nearest_depot(pred)
            old_dist = float(M[pred, depot_pred])
        else:
            old_dist = float(M[pred, succ])

        # new 边：把 node 插进去（全局起点用 depot(node)→node；全局终点用 node→depot(node)）
        if pred is None and succ is None:
            depot_node = self._get_nearest_depot(node)
            new_dist = float(M[depot_node, node] + M[node, depot_node])
        elif pred is None:
            depot_node = self._get_nearest_depot(node)
            new_dist = float(M[depot_node, node] + M[node, succ])
        elif succ is None:
            depot_node = self._get_nearest_depot(node)
            new_dist = float(M[pred, node] + M[node, depot_node])
        else:
            new_dist = float(M[pred, node] + M[node, succ])

        return new_dist - old_dist

    def reinsert_customers_multi(self, routes, removed_customers):
        """
        在多段显示但“本质是一条链”的路径结构中，逐个将 removed_customers
        按全局最小增量插回到某段的某位置；返回扁平路径与分段路径。
        """
        # 防就地修改
        routes = [r.copy() for r in routes]
        if not routes:
            routes = [[]]

        # 高需求优先
        removed = sorted(removed_customers, key=lambda c: self.all_nodes[c].demand, reverse=True)

        while removed:
            customer = removed[0]
            best_score = float('inf')
            best_ridx, best_pos = -1, -1  # 用 -1 避免 None 的类型告警

            # 穷举所有段与位置
            for ridx, r in enumerate(routes):
                L = len(r)
                for pos in range(L + 1):
                    # 关键改动：使用“链式增量”，而不是单段独立增量
                    DI = self._distance_increase_chain(routes, ridx, pos, customer)
                    # 位置惩罚（可按需）
                    pos_penalty = (pos / (L + 1)) if L > 0 else 0.0
                    score = DI + self.loc_penalty * pos_penalty
                    if score < best_score:
                        best_score = score
                        best_ridx, best_pos = ridx, pos

            if best_ridx == -1:
                # 理论上不会发生：至少有一个候选点
                routes.append([customer])
            else:
                routes[best_ridx].insert(best_pos, customer)

            removed.pop(0)

        flat_route = [n for r in routes for n in r]
        return flat_route, routes

    def fit_cross_gbs(self, routes_cluster):
        updated_routes, removed_customers, order_copy = self.remove_customers(routes_cluster)
        # print(updated_routes, removed_customers)
        flat_route, new_routes = self.reinsert_customers_multi(updated_routes, removed_customers)

        return flat_route, new_routes


    def remove_customers_clusters(self, routes, routes_by_clusters):
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)

        order = [node for route in routes_copy for node in route]
        if not order:
            return routes_copy, [], [], routes_by_clusters_copy

        num_removed = max(1, int(len(order) * self.removal_ratio))
        order_copy = order.copy()
        removed_customers = []

        # 随机移除第一个
        first_node = random.choice(order_copy)
        removed_customers.append(first_node)
        order_copy.remove(first_node)

        # 选择剩余
        while len(removed_customers) < num_removed and order_copy:
            cr = random.choice(removed_customers)
            similarity_list = []
            for ci in order_copy:
                rij = self._compute_relateness(cr, ci, routes_copy, order)
                similarity_list.append((ci, rij))
            similarity_list.sort(key=lambda x: x[1], reverse=True)
            selected_customer = similarity_list[0][0]
            removed_customers.append(selected_customer)
            order_copy.remove(selected_customer)

        # 从 routes 中移除
        remove_set = set(removed_customers)
        updated_routes = []
        for route in routes_copy:
            new_route = [n for n in route if n not in remove_set]
            if new_route:
                updated_routes.append(new_route)

        # 从 routes_by_clusters 中移除
        for cid in routes_by_clusters_copy:
            gb_routes = routes_by_clusters_copy[cid]
            for i in range(len(gb_routes)):
                gb_routes[i] = [n for n in gb_routes[i] if n not in remove_set]
            routes_by_clusters_copy[cid] = [r for r in gb_routes if r]  # 删除空路径

        return updated_routes, removed_customers, order_copy, routes_by_clusters_copy

    def remove_random_customers(self, routes, routes_by_clusters, removal_ratio=0.1):
        """
        从所有路径中随机移除若干客户（不考虑相似性），返回结构同 remove_customers_clusters。

        参数:
            routes: list[list[int]]，当前扁平路径结构（不含充电桩与仓库）
            routes_by_clusters: dict[int, list[list[int]]]，cluster → 每个GB内的路径
            removal_ratio: float，移除比例

        返回:
            updated_routes: list[list[int]]，删除后的路径结构
            removed_customers: list[int]，被移除的客户
            order_copy: list[int]，剩余的客户
            routes_by_clusters_copy: dict[int, list[list[int]]]，更新后的cluster路径结构
        """
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)

        order = [node for route in routes_copy for node in route]
        if not order:
            return routes_copy, [], [], routes_by_clusters_copy

        num_removed = max(1, int(len(order) * removal_ratio))
        removed_customers = random.sample(order, min(num_removed, len(order)))

        order_copy = [n for n in order if n not in removed_customers]
        remove_set = set(removed_customers)

        # 从 routes 中移除
        updated_routes = []
        for route in routes_copy:
            new_route = [n for n in route if n not in remove_set]
            if new_route:
                updated_routes.append(new_route)

        # # 从 routes_by_clusters 中移除
        # for cid in routes_by_clusters_copy:
        #     gb_routes = routes_by_clusters_copy[cid]
        #     for i in range(len(gb_routes)):
        #         gb_routes[i] = [n for n in gb_routes[i] if n not in remove_set]
        #     routes_by_clusters_copy[cid] = [r for r in gb_routes if r]  # 删除空路径

        return updated_routes, removed_customers, order_copy, routes_by_clusters_copy

    def remove_worst_distance(self, routes, routes_by_clusters, removal_ratio=0.1):
        """
        根据客户与其所在路径中心的距离，移除最远的若干客户（用于 LocalSearch 中）。

        参数:
            routes: list[list[int]]，路径结构
            routes_by_clusters: dict[int, list[list[int]]]，cluster → GBS 路径
            removal_ratio: float，移除比例

        返回:
            updated_routes: 删除后的路径
            removed_customers: 被移除的客户
            order_copy: 剩余客户（flat list）
            routes_by_clusters_copy: 更新后的结构
        """
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)

        all_customers = [node for route in routes_copy for node in route]
        if not all_customers:
            return routes_copy, [], [], routes_by_clusters_copy

        num_to_remove = max(1, int(len(all_customers) * removal_ratio))

        distance_list = []

        for route in routes_copy:
            if not route:
                continue
            coords = np.array([self.all_nodes[i].location() for i in route])
            center = np.mean(coords, axis=0)
            for i in route:
                dist = np.linalg.norm(np.array(self.all_nodes[i].location()) - center)
                distance_list.append((i, dist))

        distance_list.sort(key=lambda x: x[1], reverse=True)
        removed_customers = [cid for cid, _ in distance_list[:num_to_remove]]
        remove_set = set(removed_customers)
        order_copy = [i for i in all_customers if i not in remove_set]

        updated_routes = []
        for route in routes_copy:
            new_route = [n for n in route if n not in remove_set]
            if new_route:
                updated_routes.append(new_route)

        for cid in routes_by_clusters_copy:
            gb_routes = routes_by_clusters_copy[cid]
            for i in range(len(gb_routes)):
                gb_routes[i] = [n for n in gb_routes[i] if n not in remove_set]
            routes_by_clusters_copy[cid] = [r for r in gb_routes if r]

        return updated_routes, removed_customers, order_copy, routes_by_clusters_copy

    def reinsert_customers_cluster(self, routes, removed_customers, routes_by_clusters, vehicle_capacity=650):
        """
        将 removed_customers 插入到 routes 中，同时同步插入到 routes_by_clusters。
        插入位置按距离增量 + 惩罚分数最小原则插入，cluster结构同步更新。
        """
        def find_customer_index(gb_route_list, customer):
            # routes 2层嵌套列表,list[list[int]]
            for gb_idx, route in enumerate(gb_route_list):
                for pos_idx, node in enumerate(route):
                    if node == customer:
                        return gb_idx, pos_idx
            return None  # 没找到

        def route_demand(route):
            return sum(self.all_nodes[n].demand for n in route)
    
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)

        # 高需求优先插入
        removed = sorted(removed_customers, key=lambda c: self.all_nodes[c].demand, reverse=True)

        while removed:
            customer = removed[0]
            cust_demand = self.all_nodes[customer].demand
            best_score = float('inf')
            best_ridx, best_pos = -1, -1

            # 找最优插入位置
            for ridx, route in enumerate(routes_copy):
                # 加容量判断
                if route_demand(route) + cust_demand > vehicle_capacity:
                    continue  # 超过容量，跳过

                L = len(route)
                for pos in range(L + 1):
                    DI = self._distance_increase(route, customer, pos)
                    pos_penalty = (pos / (L + 1)) if L > 0 else 0.0
                    score = DI + self.loc_penalty * pos_penalty
                    
                    if score < best_score:
                        best_score = score
                        best_ridx, best_pos = ridx, pos

            if best_ridx == -1:
                routes_copy.append([customer])
                routes_by_clusters_copy.setdefault(0, []).append([customer])
            else:
                routes_copy[best_ridx].insert(best_pos, customer)
                
                if best_pos == 0:
                    routes_by_clusters_copy[best_ridx][0].insert(0, customer)
                else:
                    prev_node = routes_copy[best_ridx][best_pos-1]
                    result = find_customer_index(routes_by_clusters_copy[best_ridx], prev_node)
                    if result is not None:
                        gb_idx, pos_idx = result
                        routes_by_clusters_copy[best_ridx][gb_idx].insert(pos_idx+1, customer)
                    else:
                        # fallback：找不到 prev_node（理论上不该发生），默认插入第一条 GB
                        routes_by_clusters_copy[best_ridx][0].insert(0, customer)
                    
            removed = removed[1:]
        
        flat_route = [n for r in routes_copy for n in r]
        return flat_route, routes_copy, routes_by_clusters_copy


    def remove_customers_by_boundary_destroy_v1(self,
        routes, routes_by_clusters, W=2, quota_per_boundary=1, chain_prob=0.3
    ):
        """
        对 routes_by_clusters 中的每个 cluster 应用边界销毁算子，返回更新后的结果结构。

        返回:
            updated_routes: list[list[int]]
            removed_customers: list[int]
            order_copy: list[int]
            routes_by_clusters_copy: dict[int, list[list[int]]]
        """
        print('routes', routes)
        print('routes_by_clusters', routes_by_clusters)
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)
        order = [node for route in routes_copy for node in route]
        num_removed = max(1, int(len(order) * self.removal_ratio))

        
        updated_routes = []
        removed_customers = []

        for cid, gbs_paths in routes_by_clusters.items():
            print(gbs_paths, len(gbs_paths))
            if len(gbs_paths) < 2:
                updated_paths = gbs_paths  # 无边界，跳过
                removed = []
            else:
                # ========== Step 1: 收集边界候选节点 ==========
                removed = []
                candidates = []
                boundaries = [(i, gbs_paths[i], gbs_paths[i + 1]) for i in range(len(gbs_paths) - 1)]

                for b_idx, P_left, P_right in boundaries:
                    # 左窗口
                    L = P_left[-W:] if len(P_left) >= W else P_left[:]
                    for j, nid in enumerate(reversed(L), start=1):
                        score = (W - j + 1) / W
                        local_index = len(P_left) - j
                        candidates.append((score, nid, b_idx, 'L', j, local_index))
                    # 右窗口
                    R = P_right[:W] if len(P_right) >= W else P_right[:]
                    for j, nid in enumerate(R, start=1):
                        score = (W - j + 1) / W
                        local_index = j - 1
                        candidates.append((score, nid, b_idx, 'R', j, local_index))

                # ========== Step 2: 全局排序并挑选要删除的节点 ==========
                # candidates.sort(key=lambda x: x[0], reverse=True)
                # per_boundary_used = defaultdict(int)
                # picked = []
                # picked_set = set()
                # for score, nid, b_idx, side, d, local_index in candidates:
                #     if len(picked) >= num_removed:
                #         break
                #     if per_boundary_used[b_idx] >= quota_per_boundary:
                #         continue
                #     if nid in picked_set:
                #         continue
                #     picked.append((nid, b_idx, side, d, local_index))
                #     picked_set.add(nid)
                #     per_boundary_used[b_idx] += 1

                def softmax(x):
                    e_x = np.exp(x - np.max(x))
                    return e_x / e_x.sum()
                # 拿出 score 列，转为 numpy 数组
                scores = np.array([score for score, *_ in candidates])
                probs = softmax(scores)
                # 确保不超过候选数量
                sample_size = min(num_removed, len(candidates))
                # 采样索引
                picked_indices = np.random.choice(len(candidates), size=sample_size, p=probs, replace=False)
                # 初步选中节点
                pre_picked = [candidates[i] for i in picked_indices]
                # 额外加一层筛选：防止某条边界 quota 超限 or 重复 nid
                per_boundary_used = defaultdict(int)
                picked = []
                picked_set = set()
                for score, nid, b_idx, side, d, local_index in pre_picked:
                    if len(picked) >= num_removed:
                        break
                    if per_boundary_used[b_idx] >= quota_per_boundary:
                        continue
                    if nid in picked_set:
                        continue
                    picked.append((nid, b_idx, side, d, local_index))
                    picked_set.add(nid)
                    per_boundary_used[b_idx] += 1
                
                # ========== Step 3: 链式删除邻居 ==========
                to_remove = set()
                for nid, b_idx, side, d, local_index in picked:
                    to_remove.add(nid)
                    if random.random() < chain_prob:
                        if side == 'L':
                            P = gbs_paths[b_idx]
                            nei_idx = local_index - 1
                            if 0 <= nei_idx < len(P):
                                to_remove.add(P[nei_idx])
                        else:  # side == 'R'
                            P = gbs_paths[b_idx + 1]
                            nei_idx = local_index + 1
                            if 0 <= nei_idx < len(P):
                                to_remove.add(P[nei_idx])
                
                # ========== Step 4: 删除节点，构造新的路径 ==========
                updated_paths = []
                for path in gbs_paths:
                    to_delete = [n for n in path if n in to_remove]
                    if len(path) - len(to_delete) < 2:
                        to_delete = to_delete[: max(0, len(path) - 2)]
                    new_path = [n for n in path if n not in to_delete]
                    if new_path:
                        updated_paths.append(new_path)
                    removed += to_delete  # 累加删除记录
                print(updated_paths, len(updated_paths))
 
            # ========== Step 5: 保存 cluster 内的更新结果 ==========
            if updated_paths:
                routes_by_clusters_copy[cid] = updated_paths
            else:
                routes_by_clusters_copy[cid] = []  # 保留 key，防止后续 KeyError

            updated_routes.extend(updated_paths)
            removed_customers.extend(removed)

            # ========== Step 6: 合并每个 cluster 的路径 ==========
            cluster_routes_merged = []
            for cid, paths in routes_by_clusters_copy.items():
                merged = [n for path in paths for n in path]  # 扁平化 cluster 内所有路径
                if merged:  # 避免空的
                    cluster_routes_merged.append(merged)

        order_copy = [n for route in updated_routes for n in route]
        print('updated_routes=', updated_routes)
        print('removed_customers', removed_customers)
        print('routes_by_clusters_copy', routes_by_clusters_copy)
        return cluster_routes_merged, removed_customers, order_copy, routes_by_clusters_copy

    def remove_customers_by_boundary_destroy_v2(self,
        routes, routes_by_clusters, W=2, quota_per_boundary=1, chain_prob=0.3
    ):
        """
        对 routes_by_clusters 中的每个 cluster 应用边界销毁算子，返回更新后的结果结构。

        返回:
            updated_routes: list[list[int]]
            removed_customers: list[int]
            order_copy: list[int]
            routes_by_clusters_copy: dict[int, list[list[int]]]
        """
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)
        order = [node for route in routes_copy for node in route]
        num_removed = max(1, int(len(order) * self.removal_ratio))
        print(routes_by_clusters_copy)

        updated_routes = []
        removed_customers = []

        # ==== NEW: 构造每个 cluster 的完整路径（flat list） ====
        full_cluster_paths = {
            cid: [n for path in gbs_paths for n in path]
            for cid, gbs_paths in routes_by_clusters.items()
        }

        # ==== NEW: 定义“越靠近端点得分越高”的权重函数 ====
        # def full_path_endpoint_weight(nid, full_path, weight_boost=2.0):
        #     if nid not in full_path or len(full_path) <= 1:
        #         return 1.0
        #     idx = full_path.index(nid)
        #     norm_pos = idx / (len(full_path) - 1)
        #     # 两端可达 2.0，中间为 1.0
        #     return 1.0 + weight_boost * (1 - abs(0.5 - norm_pos) * 2)
        def full_path_endpoint_weight(nid, full_path, weight_boost=2.0):
            if nid not in full_path or len(full_path) <= 1:
                return 1.0
            idx = full_path.index(nid)
            norm_pos = idx / (len(full_path) - 1)
            # 两端可达 2.0，中间为 1.0
            return 1.0 + weight_boost * abs(0.5 - norm_pos) * 2


        for cid, gbs_paths in routes_by_clusters.items():
            if len(gbs_paths) < 2:
                updated_paths = gbs_paths
                removed = []
            else:
                removed = []
                candidates = []
                boundaries = [(i, gbs_paths[i], gbs_paths[i + 1]) for i in range(len(gbs_paths) - 1)]

                full_path = full_cluster_paths[cid]

                for b_idx, P_left, P_right in boundaries:
                    # 左窗口
                    L = P_left[-W:] if len(P_left) >= W else P_left[:]
                    for j, nid in enumerate(reversed(L), start=1):
                        base_score = (W - j + 1) / W
                        ep_weight = full_path_endpoint_weight(nid, full_path)
                        score = base_score * ep_weight
                        local_index = len(P_left) - j
                        candidates.append((score, nid, b_idx, 'L', j, local_index))

                    # 右窗口
                    R = P_right[:W] if len(P_right) >= W else P_right[:]
                    for j, nid in enumerate(R, start=1):
                        base_score = (W - j + 1) / W
                        ep_weight = full_path_endpoint_weight(nid, full_path)
                        score = base_score * ep_weight
                        local_index = j - 1
                        candidates.append((score, nid, b_idx, 'R', j, local_index))

                # Softmax sampling
                def softmax(x):
                    e_x = np.exp(x - np.max(x))
                    return e_x / e_x.sum()

                scores = np.array([score for score, *_ in candidates])
                probs = softmax(scores)
                sample_size = min(num_removed, len(candidates))
                picked_indices = np.random.choice(len(candidates), size=sample_size, p=probs, replace=False)
                pre_picked = [candidates[i] for i in picked_indices]

                per_boundary_used = defaultdict(int)
                picked = []
                picked_set = set()
                for score, nid, b_idx, side, d, local_index in pre_picked:
                    if len(picked) >= num_removed:
                        break
                    if per_boundary_used[b_idx] >= quota_per_boundary:
                        continue
                    if nid in picked_set:
                        continue
                    picked.append((nid, b_idx, side, d, local_index))
                    picked_set.add(nid)
                    per_boundary_used[b_idx] += 1

                # 链式删除
                to_remove = set()
                for nid, b_idx, side, d, local_index in picked:
                    to_remove.add(nid)
                    if random.random() < chain_prob:
                        if side == 'L':
                            P = gbs_paths[b_idx]
                            nei_idx = local_index - 1
                            if 0 <= nei_idx < len(P):
                                to_remove.add(P[nei_idx])
                        else:
                            P = gbs_paths[b_idx + 1]
                            nei_idx = local_index + 1
                            if 0 <= nei_idx < len(P):
                                to_remove.add(P[nei_idx])

                # 删除并构造新路径
                updated_paths = []
                for path in gbs_paths:
                    to_delete = [n for n in path if n in to_remove]
                    if len(path) - len(to_delete) < 2:
                        to_delete = to_delete[: max(0, len(path) - 2)]
                    new_path = [n for n in path if n not in to_delete]
                    if new_path:
                        updated_paths.append(new_path)
                    removed += to_delete

            if updated_paths:
                routes_by_clusters_copy[cid] = updated_paths
            else:
                routes_by_clusters_copy[cid] = []  # 避免 KeyError

            updated_routes.extend(updated_paths)
            removed_customers.extend(removed)

        # 合并每个 cluster 的路径
        cluster_routes_merged = []
        for cid, paths in routes_by_clusters_copy.items():
            merged = [n for path in paths for n in path]
            if merged:
                cluster_routes_merged.append(merged)

        order_copy = [n for route in updated_routes for n in route]
        print('cluster_routes_merged', cluster_routes_merged)
        print('removed_customers', removed_customers)
        print('order_copy', order_copy)
        print('routes_by_clusters_copy', routes_by_clusters_copy)
        return cluster_routes_merged, removed_customers, order_copy, routes_by_clusters_copy


    def remove_customers_by_boundary_destroy_v3(self,
        routes, routes_by_clusters, alpha=0.5
    ):
        """
        使用 正常方法归一化
        对 routes_by_clusters 中的每个 cluster 应用边界销毁算子（无窗口、无链式删除），
        删除越靠近 GB 边界与簇边界的客户概率越大。

        参数:
            routes: list[list[int]]，全局路径集合
            routes_by_clusters: dict[int, list[list[int]]]，簇 -> 该簇的 GB 内路径集合
            alpha: float in [0,1]，融合权重（越大越偏向 GB 边界，越小越偏向簇边界）

        返回:
            cluster_routes_merged: list[list[int]]，每个簇的路径合并为一条序列
            removed_customers: list[int]，被删除的客户
            order_copy: list[int]，删除后全局路径的线性序列
            routes_by_clusters_copy: dict[int, list[list[int]]]，更新后的簇-路径结构
        """
        # print('routes_by_clusters', routes_by_clusters)
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)

        # 全局删除数量
        order = [node for route in routes_copy for node in route]
        num_removed_total = max(1, int(len(order) * self.removal_ratio))

        # 各簇删除数量
        cluster_sizes = {cid: len(route) for cid, route in enumerate(routes_copy)}
        total_nodes = sum(cluster_sizes.values())
        num_removed_clusters = {
            cid: int(round(cluster_sizes[cid] / total_nodes * num_removed_total))
            for cid in cluster_sizes
        }
        # print(num_removed_clusters)
        # print('routes_copy', routes_copy)
        # print('routes_by_clusters_copy', routes_by_clusters_copy)
        # print('order', order)
        # print('num_removed', num_removed)

        updated_routes = []
        removed_customers = []

        # ===== 定义靠近度函数 =====
        def proximity(idx, length):
            """归一化后，越靠近两端得分越高"""
            if length <= 1:
                return 0.0
            return 1.0 - min(idx, length - 1 - idx) / (length - 1)

        for cid, gbs_paths in routes_by_clusters.items():
            if not gbs_paths:
                routes_by_clusters_copy[cid] = []
                continue

            # 簇完整路径
            full_cluster_path = [n for path in gbs_paths for n in path]
            L_cl = len(full_cluster_path)
            # print(gbs_paths)
            candidates = []
            for p_idx, path in enumerate(gbs_paths):
                L_gb = len(path)
                for i, nid in enumerate(path):
                    # GB 边界靠近度
                    prox_gb = proximity(i, L_gb)
                    # 簇边界靠近度
                    j = full_cluster_path.index(nid)
                    prox_cl = proximity(j, L_cl)
                    # 融合得分
                    score = alpha * prox_gb + (1 - alpha) * prox_cl
                    candidates.append((score, nid, p_idx, i))
                    # print('nid', nid)
                    # print(i, prox_gb)
                    # print(j, prox_cl)
                    # print(score)

                # print('candidates', candidates)

            # 按簇分配 quota
            remove_size = min(num_removed_clusters.get(cid, 0), len(candidates))
            if remove_size == 0:
                # 🔒 保证字典里有 key
                routes_by_clusters_copy[cid] = gbs_paths
                updated_routes.extend(gbs_paths)
                continue

            # 归一化采样
            scores = np.array([s for s, *_ in candidates])
            probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(scores)) / len(scores)
            picked_indices = np.random.choice(len(candidates), size=remove_size, p=probs, replace=False)
            picked = [candidates[i] for i in picked_indices]
            
            # print(scores)
            # print(probs)
            # print('remove_size', remove_size)
            # print('picked', picked)

            to_remove = {nid for _, nid, _, _ in picked}

            # 删除并重构路径
            updated_paths = []
            removed = []
            for path in gbs_paths:
                to_delete = [n for n in path if n in to_remove]
                if len(path) - len(to_delete) < 2:  # 保底约束：至少留2个节点
                    to_delete = to_delete[: max(0, len(path) - 2)]
                new_path = [n for n in path if n not in to_delete]
                if new_path:  # 非空才写入
                    updated_paths.append(new_path)
                removed.extend(to_delete)
            
            # 🔒 不管删没删，都写回 cid
            if updated_paths:
                routes_by_clusters_copy[cid] = updated_paths
            else:
                routes_by_clusters_copy[cid] = []  # 避免 KeyError

            updated_routes.extend(updated_paths)
            removed_customers.extend(removed)

        # 合并每个簇的路径
        cluster_routes_merged = []
        for cid, paths in routes_by_clusters_copy.items():
            merged = [n for path in paths for n in path]
            if merged:
                cluster_routes_merged.append(merged)

        order_copy = [n for route in updated_routes for n in route]

        # print('cluster_routes_merged', cluster_routes_merged)
        # print('removed_customers', removed_customers)
        # print('order_copy', order_copy)
        # print('routes_by_clusters_copy', routes_by_clusters_copy)
        return cluster_routes_merged, removed_customers, order_copy, routes_by_clusters_copy

    def remove_customers_by_boundary_destroy_v31(self,
        routes, routes_by_clusters, alpha=0.5
    ):
        """
        使用softmax归一化
        """
        # 定义 softmax 函数（带数值稳定处理）
        def softmax(x):
            e_x = np.exp(x - np.max(x))   # 防止溢出
            return e_x / e_x.sum()

        # print('routes_by_clusters', routes_by_clusters)
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)

        # 全局删除数量
        order = [node for route in routes_copy for node in route]
        num_removed_total = max(1, int(len(order) * self.removal_ratio))

        # 各簇删除数量
        cluster_sizes = {cid: len(route) for cid, route in enumerate(routes_copy)}
        total_nodes = sum(cluster_sizes.values())
        num_removed_clusters = {
            cid: int(round(cluster_sizes[cid] / total_nodes * num_removed_total))
            for cid in cluster_sizes
        }
        # print(num_removed_clusters)
        # print('routes_copy', routes_copy)
        # print('routes_by_clusters_copy', routes_by_clusters_copy)
        # print('order', order)
        # print('num_removed', num_removed)

        updated_routes = []
        removed_customers = []

        # ===== 定义靠近度函数 =====
        def proximity(idx, length):
            """归一化后，越靠近两端得分越高"""
            if length <= 1:
                return 0.0
            return 1.0 - min(idx, length - 1 - idx) / (length - 1)

        for cid, gbs_paths in routes_by_clusters.items():
            if not gbs_paths:
                routes_by_clusters_copy[cid] = []
                continue

            # 簇完整路径
            full_cluster_path = [n for path in gbs_paths for n in path]
            L_cl = len(full_cluster_path)
            # print(gbs_paths)
            candidates = []
            for p_idx, path in enumerate(gbs_paths):
                L_gb = len(path)
                for i, nid in enumerate(path):
                    # GB 边界靠近度
                    prox_gb = proximity(i, L_gb)
                    # 簇边界靠近度
                    j = full_cluster_path.index(nid)
                    prox_cl = proximity(j, L_cl)
                    # 融合得分
                    score = alpha * prox_gb + (1 - alpha) * prox_cl
                    candidates.append((score, nid, p_idx, i))
                    # print('nid', nid)
                    # print(i, prox_gb)
                    # print(j, prox_cl)
                    # print(score)

                # print('candidates', candidates)

            # 按簇分配 quota
            remove_size = min(num_removed_clusters.get(cid, 0), len(candidates))
            if remove_size == 0:
                # 🔒 保证字典里有 key
                routes_by_clusters_copy[cid] = gbs_paths
                updated_routes.extend(gbs_paths)
                continue

            # 归一化采样
            scores = np.array([s for s, *_ in candidates])
            probs = softmax(scores) if len(scores) > 0 else np.ones(len(scores)) / len(scores)
            picked_indices = np.random.choice(len(candidates), size=remove_size, p=probs, replace=False)
            picked = [candidates[i] for i in picked_indices]
            
            # print(scores)
            # print(probs)
            # print('remove_size', remove_size)
            # print('picked', picked)

            to_remove = {nid for _, nid, _, _ in picked}

            # 删除并重构路径
            updated_paths = []
            removed = []
            for path in gbs_paths:
                to_delete = [n for n in path if n in to_remove]
                if len(path) - len(to_delete) < 2:  # 保底约束：至少留2个节点
                    to_delete = to_delete[: max(0, len(path) - 2)]
                new_path = [n for n in path if n not in to_delete]
                if new_path:  # 非空才写入
                    updated_paths.append(new_path)
                removed.extend(to_delete)
            
            # 🔒 不管删没删，都写回 cid
            if updated_paths:
                routes_by_clusters_copy[cid] = updated_paths
            else:
                routes_by_clusters_copy[cid] = []  # 避免 KeyError

            updated_routes.extend(updated_paths)
            removed_customers.extend(removed)

        # 合并每个簇的路径
        cluster_routes_merged = []
        for cid, paths in routes_by_clusters_copy.items():
            merged = [n for path in paths for n in path]
            if merged:
                cluster_routes_merged.append(merged)

        order_copy = [n for route in updated_routes for n in route]

        # print('cluster_routes_merged', cluster_routes_merged)
        # print('removed_customers', removed_customers)
        # print('order_copy', order_copy)
        # print('routes_by_clusters_copy', routes_by_clusters_copy)
        return cluster_routes_merged, removed_customers, order_copy, routes_by_clusters_copy



    def remove_customers_by_boundary_destroy_v4(self,
        routes, routes_by_clusters, alpha=0.5
    ):
        """
        全局版边界销毁算子：
        删除越靠近 GB 边界与簇边界的客户概率越大（全局统一采样）。

        参数:
            routes: list[list[int]]，全局路径集合
            routes_by_clusters: dict[int, list[list[int]]]，簇 -> 该簇的 GB 内路径集合
            alpha: float in [0,1]，融合权重（越大越偏向 GB 边界，越小越偏向簇边界）

        返回:
            cluster_routes_merged: list[list[int]]，每个簇的路径合并为一条序列
            removed_customers: list[int]，被删除的客户
            order_copy: list[int]，删除后全局路径的线性序列
            routes_by_clusters_copy: dict[int, list[list[int]]]，更新后的簇-路径结构
        """
        routes_copy = copy.deepcopy(routes)
        routes_by_clusters_copy = copy.deepcopy(routes_by_clusters)

        # 全局删除数量
        order = [node for route in routes_copy for node in route]
        num_removed_total = max(1, int(len(order) * self.removal_ratio))

        # ===== 定义靠近度函数 =====
        def proximity(idx, length):
            """归一化后，越靠近两端得分越高"""
            if length <= 1:
                return 0.0
            return 1.0 - min(idx, length - 1 - idx) / (length - 1)

        # ===== 构造全局候选池 =====
        candidates = []
        for cid, gbs_paths in routes_by_clusters.items():
            if not gbs_paths:
                continue
            full_cluster_path = [n for path in gbs_paths for n in path]
            L_cl = len(full_cluster_path)

            for p_idx, path in enumerate(gbs_paths):
                L_gb = len(path)
                for i, nid in enumerate(path):
                    prox_gb = proximity(i, L_gb)
                    j = full_cluster_path.index(nid)
                    prox_cl = proximity(j, L_cl)
                    score = alpha * prox_gb + (1 - alpha) * prox_cl
                    candidates.append((score, nid, cid, p_idx, i))
        # print(candidates)
        if not candidates:
            return routes_copy, [], order, routes_by_clusters_copy

        # ===== 全局采样 =====
        scores = np.array([s for s, *_ in candidates])
        probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(scores)) / len(scores)
        sample_size = min(num_removed_total, len(candidates))
        picked_indices = np.random.choice(len(candidates), size=sample_size, p=probs, replace=False)
        picked = [candidates[i] for i in picked_indices]
        to_remove = {nid for _, nid, _, _, _ in picked}

        # ===== 删除并更新结构 =====
        updated_routes = []
        removed_customers = []

        for cid, gbs_paths in routes_by_clusters.items():
            updated_paths = []
            removed = []
            for path in gbs_paths:
                # 删除选中的节点
                new_path = [n for n in path if n not in to_remove]
                # 保底约束：路径至少保留 2 个节点
                if len(new_path) < 2:
                    new_path = path[:]
                    removed_local = []
                else:
                    removed_local = [n for n in path if n in to_remove]

                updated_paths.append(new_path)
                removed.extend(removed_local)

            # 保证字典里始终有 key
            routes_by_clusters_copy[cid] = updated_paths if updated_paths else []
            updated_routes.extend(updated_paths)
            removed_customers.extend(removed)

        # ===== 合并簇路径 =====
        cluster_routes_merged = []
        for cid, paths in routes_by_clusters_copy.items():
            merged = [n for path in paths for n in path]
            if merged:
                cluster_routes_merged.append(merged)

        order_copy = [n for route in updated_routes for n in route]

        return cluster_routes_merged, removed_customers, order_copy, routes_by_clusters_copy



    def fit_cross_clusters(self, routes_clusters, routes_by_clusters, vehicle_capacity=650):
        new_routes = {cid: route.copy() for cid, route in routes_clusters.items()}
        routes = [route for i, route in new_routes.items()]

        updated_routes, removed, _, updated_gb_routes = self.remove_customers_by_boundary_destroy_v3(routes, routes_by_clusters, alpha=0.8)
        flat_route, new_routes, new_gb_routes = self.reinsert_customers_cluster(updated_routes, removed, updated_gb_routes, vehicle_capacity)

        # print('new_routes=', new_routes)
        # print('new_gb_routes=', new_gb_routes)
        return flat_route, new_routes, removed, new_gb_routes




    def remove_customers_by_clusters_boundary_destroy(self, routes, W=2): 
        routes_copy = copy.deepcopy(routes)   # routes 是 list
        order = [node for path in routes_copy for node in path]  # 去掉 .values()
        num_removed = max(1, int(len(order) * self.removal_ratio))

        updated_routes = []
        removed_customers = []

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        def endpoint_weight(nid, full_path, weight_boost=2.0):
            if nid not in full_path or len(full_path) <= 1:
                return 1.0
            idx = full_path.index(nid)
            norm_pos = idx / (len(full_path) - 1)
            return 1.0 + weight_boost * abs(0.5 - norm_pos) * 2

        for path in routes_copy:   # 遍历 list
            if len(path) <= 2:
                updated_path = path[:]
                removed = []
            else:
                candidates = []
                L = path[:W]
                for j, nid in enumerate(L, start=1):
                    base_score = (W - j + 1) / W
                    score = base_score * endpoint_weight(nid, path)
                    candidates.append((score, nid, "L"))

                R = path[-W:]
                for j, nid in enumerate(reversed(R), start=1):
                    base_score = (W - j + 1) / W
                    score = base_score * endpoint_weight(nid, path)
                    candidates.append((score, nid, "R"))

                scores = np.array([c[0] for c in candidates])
                probs = softmax(scores)
                sample_size = min(num_removed, len(candidates))
                picked_idx = np.random.choice(len(candidates), size=sample_size, p=probs, replace=False)
                picked = [candidates[i][1] for i in picked_idx]

                removed = list(set(picked))
                updated_path = [n for n in path if n not in removed]

            updated_routes.append(updated_path)
            removed_customers.extend(removed)

        cluster_routes_merged = updated_routes
        order_copy = [n for route in updated_routes for n in route]

        return cluster_routes_merged, removed_customers, order_copy, routes_copy

    def reinsert_customers_by_clusters(self, routes, removed_customers, vehicle_capacity=650):
        """
        将 removed_customers 插回 routes 中。
        插入位置按距离增量 + 惩罚分数最小原则。
        
        参数:
            routes: list[list[int]]        所有车辆路径
            removed_customers: list[int]   待插入客户
            vehicle_capacity: int          车辆容量约束
        
        返回:
            flat_route: list[int]          扁平化路径
            routes_copy: list[list[int]]   更新后的路径
        """
        def route_demand(route):
            return sum(self.all_nodes[n].demand for n in route)

        routes_copy = copy.deepcopy(routes)
        removed = sorted(removed_customers, key=lambda c: self.all_nodes[c].demand, reverse=True)

        while removed:
            customer = removed[0]
            cust_demand = self.all_nodes[customer].demand
            best_score = float("inf")
            best_ridx, best_pos = -1, -1

            # 找最优插入位置
            for ridx, route in enumerate(routes_copy):
                if route_demand(route) + cust_demand > vehicle_capacity:
                    continue

                L = len(route)
                for pos in range(L + 1):
                    DI = self._distance_increase(route, customer, pos)
                    pos_penalty = (pos / (L + 1)) if L > 0 else 0.0
                    score = DI + self.loc_penalty * pos_penalty

                    if score < best_score:
                        best_score = score
                        best_ridx, best_pos = ridx, pos

            if best_ridx == -1:
                # 没有合适位置，新建一条路径
                routes_copy.append([customer])
            else:
                routes_copy[best_ridx].insert(best_pos, customer)

            removed = removed[1:]

        flat_route = [n for r in routes_copy for n in r]
        return flat_route, routes_copy

    def fit_only_cross_clusters(self, routes, vehicle_capacity=650):

        updated_routes, removed, _, _ = self.remove_customers_by_clusters_boundary_destroy(routes)
        flat_route, new_routes = self.reinsert_customers_by_clusters(updated_routes, removed, vehicle_capacity)

        # print('new_routes=', new_routes)
        # print('new_gb_routes=', new_gb_routes)
        return flat_route, new_routes, removed
    





        














