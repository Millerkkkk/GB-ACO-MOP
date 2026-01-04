#  coding: UTF-8  #
'''
@Project     : running
@File        : main.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/24 19:34
'''


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from dataclasses import dataclass
import numpy as np
import os
import time
from gbDbscan import GbDbscanCluster
from gbGenerate import generate_gbs
from utils import EVRPInstance, plot_routes, plot_clusters
import clusters_result
from localSearch import LocalSearch
from decoder import EnergyModel, Decoder, calculate_cost

from aca_gbs import GbPlanning 
from aca_customers import CustomersPlanning
from clustering import ImprovedKMeans


aca_gb_params = {
        "num_ants": 10,
        "num_iter": 60,
        "alpha": 1,
        "beta": 2,
        "rho": 0.1,
        "epsilon": 0.1,
        }

aca_customer_params = {
        "num_ants": 20,
        "num_iter": 100,
        "alpha": 1,
        "beta": 2,
        "rho": 0.1,
        "epsilon": 0.1,
        }


@dataclass
class CostParams:
    c1: float = 120.0    # fixed dispatch cost
    c2: float = 0.5    # travel cost per minute
    c3: float = 0.3    # service cost per minute
    c4: float = 0.6    # charging cost per minute


def calculate_total_distance(routes, dist_matrix):
    total_distance = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            total_distance += dist_matrix[u][v]
    return total_distance


def compute_center_and_radius(gb, nodes):
    coords = np.array([nodes[i].location() for i in gb])  # 取 (x,y)
    center = coords.mean(axis=0)  # 均值作为中心
    radius = np.max(np.linalg.norm(coords - center, axis=1))  # 最大欧氏距离
    return center, radius


class GBACOPlanner:
    def __init__(self, instance_name, file_base_path, clusters_result, split_k=0.4, core_ratio=0.93, max_iter=10, removal_ratio=0.3):
        self.instance_name = instance_name
        self.file_path = os.path.join(file_base_path, f"{instance_name}.txt")
        self.instance = EVRPInstance(self.file_path)
        self.instance_clusters = getattr(clusters_result, instance_name)
        # self.instance_clusters = clusters_result
        self.split_k = split_k
        self.core_ratio = core_ratio
        self.max_iter = max_iter
        self.removal_ratio = removal_ratio

        self.energy_model = EnergyModel()
        self.decoder = Decoder(self.instance.nodes, self.instance.distance_matrix, self.energy_model, self.instance.cs_ids, 
                               Q_max=40, margin_energy=0.2, radius_km=3.0, soc_anxiety=0.5)
        self.localSearch = LocalSearch(self.instance.nodes, self.instance.distance_matrix, self.instance.depot_ids, self.removal_ratio, loc_penalty=0.1)

    def _generate_gbs(self, cluster):
        cluster_idx = np.array(list(cluster))
        cluster_coords = np.array([self.instance.nodes[i].location() for i in cluster_idx])

        # # 1. 聚类 Granular Balls
        gb_model = GbDbscanCluster(cluster_idx, cluster_coords, split_k=self.split_k, core_ratio=self.core_ratio)
        gbs, centers, radii = gb_model.fit()
        
        # df = pd.read_csv(file_path, sep=r'\s+')
        # for i in ['x', 'y', 'demand', 'ReadyTime', 'DueDate', 'ServiceTime']:
        #     df[i] = pd.to_numeric(df[i], errors='coerce')
        # gbs, centers, radii = generate_gbs(df, cluster, k=0.8, merge=1, merge_single_gb=1)
        # print(gbs, centers, radii)

        # clusters = {idx: set(group) for idx, group in enumerate(gbs)}
        # plot_clusters(clusters, self.instance.nodes, self.instance.depot_ids, self.instance.cs_ids, self.instance.customer_ids)

        gbs_info_list = [
            {"gbs": gbs[i], "center": centers[i], "radius": radii[i]}
            for i in range(len(gbs))
        ]
        return gbs_info_list

    def _plan_gbs_order(self, gbs_info_list):
        # 2. 规划 GB 顺序
        depot_coords = [self.instance.nodes[i].location() for i in self.instance.depot_ids]
        gb_solver = GbPlanning(gbs_info_list, depot_coords, aca_gb_params)
        gbs_route, _, _ = gb_solver.run()

        # # # Because gbs is directionless, set the 0.5 to change the path direction
        # random_number = np.random.rand()
        # if random_number > 0.5:
        #     gbs_route = gbs_route[::-1]
        return gbs_route
    
    def _plan_internal_gbs_order(self, gbs_info_list, gbs_route):
        # Step 1: 调整 gbs 顺序
        gbs_info_list = [gbs_info_list[i] for i in gbs_route]

        if len(gbs_info_list) == 0:
            return []

        routes = []
        # Step 2: 逐个 GB 内部路径规划
        for i in range(len(gbs_info_list)):
            current_gb = gbs_info_list[i]["gbs"]

            if len(current_gb) == 0:
                continue  # 跳过空 GB（可选）

            if len(gbs_info_list) == 1:
                para_depot = 'depart & return'
                prev_last = None
                next_center = None
            elif i == 0:
                para_depot = 'depart'
                prev_last = None
                next_center = gbs_info_list[i + 1]["center"]
            elif i == len(gbs_info_list) - 1:
                para_depot = 'return'
                prev_last = routes[-1][-1]
                next_center = None
            else:
                para_depot = None
                prev_last = routes[-1][-1]
                next_center = gbs_info_list[i + 1]["center"]

            solver = CustomersPlanning(
                self.instance, current_gb, para_depot, prev_last,
                next_center, aca_customer_params
            )
            best_route, _, _ = solver.run()
            routes.append(best_route)

        # 把每个 route[i] 赋值回 gbs_info_list[i]["gbs"]
        for i in range(len(routes)):
            gbs_info_list[i]["gbs"] = routes[i]

        return routes, gbs_info_list

    def _cross_gb_LS(self, cluster_route):
        flat_route, new_routes = self.localSearch.fit(cluster_route)
        return flat_route, new_routes

    def _get_nearest_depot(self, customer):
        dist_to_depots = self.instance.distance_matrix[self.instance.depot_ids, customer]
        nearest_depot = self.instance.depot_ids[np.argmin(dist_to_depots)]
        return nearest_depot
    
    def _evaluate(self, flattened_route_global):

        route = flattened_route_global.copy()
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

    def run_original(self):
        all_decoded_routes = []
        all_routes_cost = []
        all_routes_dist = []
        total_num_charge = 0

        travel_cost_list, dispatch_cost_list, service_cost_list, charging_cost_list = [], [], [], []

        for cluster_id, cluster in self.instance_clusters.items():
            # 1) 基于 cluster 生成 GBS 并规划 GB 顺序 + 每个 GB 内的分路径
            gbs_dict = self._generate_gbs(cluster)
            gbs_route = self._plan_gbs_order(gbs_dict)
            routes, gbs_dict = self._plan_internal_gbs_order(gbs_dict, gbs_route)

            # 2) 迭代本地搜索 + 解码，选最优
            best = {
                "route_cost": float('inf'),
                "travel_dist": None,
                "decode_route": None,
                "num_charge": None,
                "travel_cost": None,
                "dispatch_cost": None,
                "service_cost": None,
                "charging_cost": None,
                "flat_route": None,
                "local_routes": None,
            }
            flattened_cluster_route = [i for route in routes for i in route]

            for iter in range(self.max_iter):
                flat_route, localSearch_routes = self._cross_gb_LS(routes)

                (route_cost, travel_dist, decode_route, num_charge,
                travel_cost, dispatch_cost, service_cost, charging_cost) = self._evaluate(flat_route)
                
                # 若更优则更新 best
                if route_cost < best["route_cost"]:
                    best.update(
                        route_cost=route_cost,
                        travel_dist=travel_dist,
                        decode_route=decode_route,
                        num_charge=num_charge,
                        travel_cost=travel_cost,
                        dispatch_cost=dispatch_cost,
                        service_cost=service_cost,
                        charging_cost=charging_cost,
                        flat_route=flat_route,
                        local_routes=localSearch_routes,
                    )
                
                routes = localSearch_routes

                # # “用 local_routes 重算 GBS”
                # centers, radii = [], []
                # for gb in routes:
                #     c, r = compute_center_and_radius(gb, self.instance.nodes)
                #     centers.append(c); radii.append(r)
                # gbs_dict = {
                #     i: {"gbs": routes[i], "center": centers[i], "radius": radii[i]}
                #     for i in range(len(routes))
                # }

            total_num_charge += best["num_charge"]
            all_decoded_routes.append(best["decode_route"])
            all_routes_cost.append(best["route_cost"])
            all_routes_dist.append(best["travel_dist"])
            travel_cost_list.append(best["travel_cost"])
            dispatch_cost_list.append(best["dispatch_cost"])
            service_cost_list.append(best["service_cost"])
            charging_cost_list.append(best["charging_cost"])

            # # flattened_route = [[node for route in cluster_route for node in route]]
        total_coors = [self.instance.nodes[i].location() for i in range(len(self.instance.nodes))]
        plot_routes(total_coors, all_decoded_routes, self.instance.depot_ids, self.instance.cs_ids, self.instance.customer_ids)

        return (all_decoded_routes, all_routes_cost, all_routes_dist, total_num_charge,
            travel_cost_list, dispatch_cost_list, service_cost_list, charging_cost_list)


        # --- 小工具 ---
    


    # --- 小工具 ---
    # def _flatten(self, routes_by_cluster):
    #     return {cid: [n for r in routes for n in r] for cid, routes in routes_by_cluster.items()}
    
    def _calculate_dist_total(self, flat_routes):
        total_dist = 0
        for route in flat_routes.values():
            if not route: return 0.0
            dist_matrix = self.instance.distance_matrix
            s, t = route[0], route[-1]
            dep_s = self._get_nearest_depot(s)
            dep_t = self._get_nearest_depot(t)
            dist = float(dist_matrix[dep_s, s]) + float(dist_matrix[t, dep_t])
            for i in range(len(route)-1):
                dist += float(dist_matrix[route[i], route[i+1]])
            total_dist += dist
        return total_dist
    
    def _improve_better(self, new_flat, old_flat, rel_improve_thresh=-0.003):
        """替身代价门：new 比 old 至少提升 0.3% 才放行解码。"""
        old = self._calculate_cost_total(old_flat)
        new = self._calculate_cost_total(new_flat)
        # improvement = (new-old)/old ；负值更好
        return (new - old) / (old + 1e-9) <= rel_improve_thresh
    
    def _evaluate_clusters(self, flat_routes):
        """对每个簇解码+计费，返回聚合结果与分项（包括正向与反向）"""
        total_cost = 0.0
        decoded_routes, routes_cost, routes_dist = [], [], []
        total_num_charge = 0
        travel_cost_list, dispatch_cost_list, service_cost_list, charging_cost_list = [], [], [], []

        for cid, route in flat_routes.items():
            # 正向路径评估
            (cost_fwd, dist_fwd, decode_fwd, num_charge_fwd,
            travel_fwd, dispatch_fwd, service_fwd, charge_fwd) = self._evaluate(route)
            # 反向路径评估
            reversed_route = list(reversed(route))
            (cost_bwd, dist_bwd, decode_bwd, num_charge_bwd,
            travel_bwd, dispatch_bwd, service_bwd, charge_bwd) = self._evaluate(reversed_route)

            # 选择成本更低的路径
            if cost_bwd < cost_fwd:
                route_cost, travel_dist = cost_bwd, dist_bwd
                decode_route = decode_bwd
                num_charge = num_charge_bwd
                travel_cost = travel_bwd
                dispatch_cost = dispatch_bwd
                service_cost = service_bwd
                charging_cost = charge_bwd
            else:
                route_cost, travel_dist = cost_fwd, dist_fwd
                decode_route = decode_fwd
                num_charge = num_charge_fwd
                travel_cost = travel_fwd
                dispatch_cost = dispatch_fwd
                service_cost = service_fwd
                charging_cost = charge_fwd

            # 汇总
            decoded_routes.append(decode_route)
            routes_cost.append(route_cost)
            routes_dist.append(travel_dist)
            total_num_charge += num_charge
            travel_cost_list.append(travel_cost)
            dispatch_cost_list.append(dispatch_cost)
            service_cost_list.append(service_cost)
            charging_cost_list.append(charging_cost)
            total_cost += route_cost

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
    

    def rebuild_gbs_info_list(self, new_gb_routes):
        """
        重建每个 cluster 的 gbs_info_list（gbs + center + radius）

        参数:
            new_gb_routes: dict[int, list[list[int]]]，每个 cluster 下多个 granule，每个 granule 是一个客户编号列表
            customer_data: dict[int, Node]，包含每个客户的位置数据，要求每个 Node 有 .location() 方法

        返回:
            dict[int, list[dict]]，每个 cluster 下的 gbs_info_list
        """
        rebuilt_info = {}
        for cid, gbs in new_gb_routes.items():
            cluster_info_list = []
            for gb in gbs:
                coords = np.array([self.instance.nodes[i].location() for i in gb])
                center = np.mean(coords, axis=0)
                radius = np.max(np.linalg.norm(coords - center, axis=1)) if len(gb) > 1 else 0.0

                cluster_info_list.append({
                    "gbs": gb,
                    "center": center,
                    "radius": radius
                })

            rebuilt_info[cid] = cluster_info_list

        return rebuilt_info

    def _detect_gbs_shift(self, gbs_cache, routes, removed_customers, member_ratio_thresh=0.2):
  
        old_assign = gbs_cache["assign"]  # customer_id → old_cid 映射
        new_assign = {}





        to_rebuild = set()
        for cid in affected_cids:
            prev = gbs_cache.get(cid, {"members": set()})["members"]
            curr = set(flat_routes.get(cid, []))
            if not prev and not curr:
                continue
            change = 1.0 - (len(prev & curr) / max(1, len(prev | curr)))
            if change >= member_ratio_thresh:
                to_rebuild.add(cid)
        return to_rebuild
    
    def _cross_cluster_LS(self, flat_routes, removal_ratio=0.1, alpha=0.1, seed=None):
        """跨簇 LS：各簇内移除一批相关点 → 跨簇全局最小增量插回（含仓库边）"""
        rng = np.random.default_rng(seed)
        M = self.instance.distance_matrix
        depots = self.instance.depot_ids
        nodes = self.instance.nodes

        def nearest_depot(x):
            d = M[depots, x]; return depots[int(np.argmin(d))]

        def delta_insert(route, pos, node):
            # old
            if not route: old = 0.0
            elif pos == 0:
                ds = nearest_depot(route[0]); old = float(M[ds, route[0]])
            elif pos == len(route):
                dt = nearest_depot(route[-1]); old = float(M[route[-1], dt])
            else:
                old = float(M[route[pos-1], route[pos]])
            # new
            if not route:
                d0 = nearest_depot(node); new = float(M[d0, node] + M[node, d0])
            elif pos == 0:
                d0 = nearest_depot(node); new = float(M[d0, node] + M[node, route[0]])
            elif pos == len(route):
                d1 = nearest_depot(node); new = float(M[route[-1], node] + M[node, d1])
            else:
                new = float(M[route[pos-1], node] + M[node, route[pos]])
            L = len(route); pos_pen = (pos/(L+1)) if L>0 else 0.0
            return (new - old) + alpha * pos_pen

        new_routes = {cid: r.copy() for cid, r in flat_routes.items()}
        affected = set()
        to_reinsert = []

        # 全局 d_max 供归一化（仅用于你“相关性”指标）
        all_nodes = [n for r in new_routes.values() for n in r]
        if all_nodes:
            d_max = max(float(M[i, k]) for i in all_nodes for k in all_nodes if i != k)
            d_max = d_max if d_max > 0 else 1.0
        else:
            d_max = 1.0

        def rd(i, j): return float(M[i, j]) / d_max

        # 1) 各簇内移除
        for cid, route in list(new_routes.items()):
            n = len(route)
            if n == 0: continue
            num_removed = max(1, int(np.ceil(n * removal_ratio)))
            order = route.copy(); removed = []

            first_idx = int(rng.integers(low=0, high=len(order)))
            seed_node = order[first_idx]; removed.append(seed_node); order.pop(first_idx)

            while len(removed) < num_removed and order:
                cr = removed[int(rng.integers(low=0, high=len(removed)))]
                # 相关性 = 1/(rd + 1) —— “同段奖”恒为1（簇内）
                sims = [(cand, 1.0/(rd(cr, cand)+1.0)) for cand in order]
                sims.sort(key=lambda x: x[1], reverse=True)
                chosen = sims[0][0]
                removed.append(chosen); order.remove(chosen)

            if removed:
                affected.add(cid)
                S = set(removed)
                new_routes[cid] = [x for x in new_routes[cid] if x not in S]
                for v in removed:
                    to_reinsert.append((cid, v))

        # 2) 跨簇插回（高需求优先）
        to_reinsert.sort(key=lambda t: nodes[t[1]].demand, reverse=True)
        for cid_from, node in to_reinsert:
            best = (float('inf'), None, None)  # gain, cid, pos
            for cid, r in new_routes.items():
                L = len(r)
                for pos in range(L+1):
                    g = delta_insert(r, pos, node)
                    if g < best[0]:
                        best = (g, cid, pos)
            if best[1] is None:
                new_routes[cid_from].append(node); affected.update([cid_from])
            else:
                new_routes[best[1]].insert(best[2], node); affected.update([best[1], cid_from])

        return new_routes, affected

    def _rebuild_gbs_subset(self, flat_routes, cand_cids, run_upper_aco=True, run_lower_aco=True):
        """仅对 cand_cids 重建 GBS，并按需跑 ACO；返回更新后的 flat_routes 与新的 cache 条目"""
        for cid in cand_cids:
            route = flat_routes.get(cid, [])
            if not route:
                continue
            cluster = set(route)
            # ① GBS
            gbs_dict = self._generate_gbs(cluster)                       # GbDbscanCluster.fit():contentReference[oaicite:7]{index=7}
            # ② 上层 ACO（可开关）
            if run_upper_aco:
                gb_seq = self._plan_gbs_order(gbs_dict)                  # GbPlanning.run():contentReference[oaicite:8]{index=8}
            else:
                gb_seq = list(gbs_dict.keys())
            # ③ 下层 ACO（可开关）
            if run_lower_aco:
                routes = self._plan_internal_gbs_order(gbs_dict, gb_seq) # CustomersPlanning.run():contentReference[oaicite:9]{index=9}
            else:
                # 退化到 GBS 内的原始顺序
                routes = [gbs_dict[g]["gbs"] for g in gb_seq]
            flat_routes[cid] = [n for r in routes for n in r]
        return flat_routes

    def run_cross_cluster1(self, max_iter=30, surrogate_gate=True, 
                          gate_thresh=-0.003, patience=5, seed=None):
        """
        - 初始化：clusters→GBS→两层 ACO → 初解；评估作为 best
        - 外层迭代：跨簇 LS（不解码）→ gate → 解码评估 → 判优
        - 若受影响簇发生显著偏移：只对这些簇重建 GBS 并两层 ACO
        """
        rng = np.random.default_rng(seed)

        # === 初始化：每簇生成 GBS + 两层 ACO ===
        gbs_cache = {}
        routes_by_clusters = {}

        for cid, cluster in self.instance_clusters.items():
            gbs_dict = self._generate_gbs(cluster)
            gb_seq = self._plan_gbs_order(gbs_dict)
            routes, gbs_info_list = self._plan_internal_gbs_order(gbs_dict, gb_seq)
            routes_by_clusters[cid] = routes
            gbs_cache[cid] = gbs_info_list

        # === 初始解评估 ===
        flat_routes = {cid: [n for r in routes for n in r] for cid, routes in routes_by_clusters.items()}
        best = self._evaluate_clusters(flat_routes)
        best["flat_routes"] = {cid: r.copy() for cid, r in flat_routes.items()}
        # print(f"[Init] total cost = {best['total_cost']:.2f}")

        no_improve = 0
        for iter in range(1, max_iter + 1):
            # print(f"\n[Iter {iter}] =============")

            # === A) 跨簇局部扰动（未解码） ===
            flat_route, LS_routes, removed_customers, new_gb_routes = self.localSearch.fit_cross_clusters(flat_routes, routes_by_clusters, vehicle_capacity=650)
            LS_routes = {i: route for i, route in enumerate(LS_routes)}


            # # === B) 替身门：提升不足则跳过解码 ===
            # if surrogate_gate:
            #     old_cost = best["total_cost"]
            #     dummy_flat = {cid: r.copy() for cid, r in LS_routes.items()}
            #     dummy_cost = self._calculate_dist_total(dummy_flat)
            #     improvement = (old_cost - dummy_cost) / (old_cost + 1e-9)
            #     print(f"Surrogate improvement: {improvement:.5f}")
            #     if improvement < abs(gate_thresh):
            #         no_improve += 1
            #         print(f"🟡 Skipped decoding: improvement {improvement:.5f} < threshold {gate_thresh}")
            #         if no_improve >= patience:
            #             print("🔴 Early stopping due to no improvement.")
            #             break
            #         continue


            # === C) 真实评估新解 ===
            cand = self._evaluate_clusters(LS_routes)         
            cand["flat_routes"] = {cid: r.copy() for cid, r in LS_routes.items()}
            # print(f"Candidate cost = {cand['total_cost']:.2f}")

            if cand["total_cost"] + 1e-9 < best["total_cost"]:
                best = cand
                flat_routes = cand["flat_routes"]
                # print(f"✅ Updated best! New total cost = {best['total_cost']:.2f}")
                no_improve = 0
            else:
                # print("🔺 No improvement.")
                no_improve += 1
                if no_improve >= patience:
                    # print("🔴 Early stopping due to no improvement.")
                    break


            # === D) 重建 GBS + 两层 ACO（仅重建被影响簇） ===
            gbs_info_dict = self.rebuild_gbs_info_list(new_gb_routes)
            routes_by_clusters = {}
            for cid, gbs_info_list in gbs_info_dict.items():
                gb_seq = self._plan_gbs_order(gbs_info_list)
                routes, _ = self._plan_internal_gbs_order(gbs_info_list, gb_seq)
                routes_by_clusters[cid] = routes
            # print(f"✅ Rebuilt GBS for {len(gbs_info_dict)} clusters.")

        # # === 绘图与返回 ===
        # total_coors = [self.instance.nodes[i].location() for i in range(len(self.instance.nodes))]
        # plot_routes(total_coors, best["decoded_routes"], self.instance.depot_ids, self.instance.cs_ids, self.instance.customer_ids)

        # 最终返回（与你 run_cross_cluster 的返回格式保持一致）:contentReference[oaicite:15]{index=15}
        return (best["decoded_routes"], best["routes_cost"], best["routes_dist"], best["total_num_charge"],
                best["travel_cost_list"], best["dispatch_cost_list"], best["service_cost_list"], best["charging_cost_list"])


    def run_cross_cluster(self, max_iter=1, patience=5, seed=None):
        rng = np.random.default_rng(seed)

        best = None
        no_improve = 0

        for iter in range(1, max_iter + 1):
            # === A) 每轮都重新生成 GBS + 两层 ACO ===
            routes_by_clusters = {}
            for cid, cluster in self.instance_clusters.items():
                gbs_dict = self._generate_gbs(cluster)
                gb_seq = self._plan_gbs_order(gbs_dict)
                routes, _ = self._plan_internal_gbs_order(gbs_dict, gb_seq)
                routes_by_clusters[cid] = routes

            flat_routes = {
                cid: [n for r in routes for n in r]
                for cid, routes in routes_by_clusters.items()
            }

            # # # === B) 局部扰动（跨簇 LS）===
            # flat_route, LS_routes, removed_customers, new_gb_routes = \
            #     self.localSearch.fit_cross_clusters(flat_routes, routes_by_clusters, vehicle_capacity=650)
            # LS_routes = {i: route for i, route in enumerate(LS_routes)}

            # === C) 评估新解 ===
            cand = self._evaluate_clusters(flat_routes)
            cand["flat_routes"] = {cid: r.copy() for cid, r in flat_routes.items()}

            if best is None or cand["total_cost"] + 1e-9 < best["total_cost"]:
                best = cand
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return (best["decoded_routes"], best["routes_cost"], best["routes_dist"],
                best["total_num_charge"], best["travel_cost_list"],
                best["dispatch_cost_list"], best["service_cost_list"],
                best["charging_cost_list"])




 
if __name__ == '__main__':
    # ========== 参数设置 ==========
    instance_name = 'r204_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    # 设置重复运行次数
    N_RUNS = 20   # 可以改成 20

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

        # 运行 clustering 得到 clusters_result 结果
        # clustering = ImprovedKMeans(instance.nodes, instance.distance_matrix, instance.customer_ids,
        #                             vehicle_capacity=650, threshold=10.0, random_state=run_id)  # 用 run_id 变化随机种子
        # clusters = clustering.fit()
        # solver = GBACOPlanner(instance_name, base_path, clusters)

        # 直接使用 clusters_result 结果
        solver = GBACOPlanner(instance_name, base_path, clusters_result)
        
        routes, routes_cost, routes_dist, routes_num_charge, travel_cost_list, dispatch_cost_list, service_cost_list, charging_cost_list = solver.run_cross_cluster()

        end_time = time.time()
        running_time = end_time - start_time

        print(run_id, sum(routes_cost))
        
        # --- 存储结果（取总和/汇总值） ---
        results["all_routes"].append(routes)
        results["routes_dist"].append(sum(routes_dist))
        results["routes_cost"].append(sum(routes_cost))
        results["dispatch_cost"].append(sum(dispatch_cost_list))
        results["travel_cost"].append(sum(travel_cost_list))
        results["service_cost"].append(sum(service_cost_list))
        results["charging_cost"].append(sum(charging_cost_list))
        results["routes_num_charge"].append(sum(routes_num_charge) if isinstance(routes_num_charge, list) else routes_num_charge)
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

        















