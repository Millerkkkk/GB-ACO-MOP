#  coding: UTF-8  #
'''
@Project     : running
@File        : main.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/24 19:34
'''

from matplotlib import pyplot as plt
import pandas as pd
from dataclasses import dataclass
import numpy as np
import os
import time
from RA import RangeAnxietyModel
from gbDbscan import GbDbscanCluster
from gbGenerate import generate_gbs
from gbKmeans import GBsGenerator
from utils import EVRPInstance, plot_routes, plot_clusters
import clusters_result
from localSearch import LocalSearch
from decoder import EnergyModel, Decoder, calculate_cost

from aca_gbs import GbPlanning 
from aca_customers import CustomersPlanning
from clustering import ImprovedKMeans

from NSGAII import update_archive_nondominated, dedup_archive, crowding_distance



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
    def __init__(self, instance_name, file_base_path, clusters_result, 
                 soc_max, vehicle_capacity,
                 ra_safe, ra_risk, 
                 
                 split_k=0.4, core_ratio=0.93, max_iter=10, removal_ratio=0.3):
        
        self.instance_name = instance_name
        self.file_path = os.path.join(file_base_path, f"{instance_name}.txt")
        self.instance = EVRPInstance(self.file_path)
        # self.instance_clusters = getattr(clusters_result, instance_name)
        self.instance_clusters = clusters_result
        self.split_k = split_k
        self.core_ratio = core_ratio
        self.max_iter = max_iter
        self.removal_ratio = removal_ratio

        self.soc_max = soc_max
        self.vehicle_capacity = vehicle_capacity

        self.ra_safe = ra_safe
        self.ra_risk = ra_risk

        self.energy_model = EnergyModel()
        self.ra_model = RangeAnxietyModel(self.soc_max)
        self.decoder = Decoder(self.instance.nodes, self.instance.distance_matrix, self.energy_model, self.instance.cs_ids, 
                               self.ra_model, self.ra_safe, self.ra_risk, self.soc_max, margin_energy=0.2, radius_km=3.0)
        self.localSearch = LocalSearch(self.instance.nodes, self.instance.distance_matrix, self.instance.depot_ids, self.removal_ratio, loc_penalty=0.1)

    def _generate_gbs(self, cluster):
        cluster_idx = np.array(list(cluster))
        cluster_coords = np.array([self.instance.nodes[i].location() for i in cluster_idx])

        # # 1. 聚类 Granular Balls
        gb_model = GbDbscanCluster(cluster_idx, cluster_coords, split_k=self.split_k, core_ratio=self.core_ratio)
        gbs, centers, radii = gb_model.fit()

        # gb_model = GBsGenerator(cluster_idx, cluster_coords, split_k=self.split_k)
        # gbs, centers, radii = gb_model.run(
        #     merge_single_gb=True,
        #     merge_overlaped=True,
        #     merge_contained=True,
        #     plot_each_stage=False,
        #     plot_interval=2  # 每轮 split 都画；如果太多图改成 2/5/10
        # )
        
        # df = pd.read_csv(file_path, sep=r'\s+')
        # for i in ['x', 'y', 'demand', 'ReadyTime', 'DueDate', 'ServiceTime']:
        #     df[i] = pd.to_numeric(df[i], errors='coerce')
        # gbs, centers, radii = generate_gbs(df, cluster, k=0.8, merge=1, merge_single_gb=1)
        # print(gbs, centers, radii)

        clusters = {idx: set(group) for idx, group in enumerate(gbs)}
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
        decode_result = self.decoder.decode_backtrack_charging_final(full_route, route_demand)

        (decode_route, travel_dist, travel_time, charging_time, service_time, num_charge,
        updated_Q, updated_departure_time, updated_load, total_ra) = decode_result

        dispatch_ev = 1
        # 计算 total cost
        costParams = CostParams()
        route_cost, cost_detail = calculate_cost(dispatch_ev, travel_time, charging_time, service_time, costParams)

        return (total_ra, route_cost, travel_dist, decode_route, num_charge, 
                cost_detail.travel_cost, cost_detail.dispatch_cost, cost_detail.service_cost, cost_detail.charging_cost)


    def _evaluate_clusters(self, flat_routes):
        """对每个簇解码+计费，返回聚合结果与分项（包括正向与反向）"""
        total_cost = 0.0
        routes_ra = []
        decoded_routes, routes_cost, routes_dist = [], [], []
        total_num_charge = 0
        travel_cost_list, dispatch_cost_list, service_cost_list, charging_cost_list = [], [], [], []

        for cid, route in flat_routes.items():
            # 正向路径评估
            (ra_fwd, cost_fwd, dist_fwd, decode_fwd, num_charge_fwd,
            travel_fwd, dispatch_fwd, service_fwd, charge_fwd) = self._evaluate(route)
            # 反向路径评估
            reversed_route = list(reversed(route))
            (ra_bwd, cost_bwd, dist_bwd, decode_bwd, num_charge_bwd,
            travel_bwd, dispatch_bwd, service_bwd, charge_bwd) = self._evaluate(reversed_route)

            # 选择成本更低的路径
            if cost_bwd < cost_fwd:
                ra = ra_bwd
                route_cost, travel_dist = cost_bwd, dist_bwd
                decode_route = decode_bwd
                num_charge = num_charge_bwd
                travel_cost = travel_bwd
                dispatch_cost = dispatch_bwd
                service_cost = service_bwd
                charging_cost = charge_bwd
            else:
                ra = ra_fwd
                route_cost, travel_dist = cost_fwd, dist_fwd
                decode_route = decode_fwd
                num_charge = num_charge_fwd
                travel_cost = travel_fwd
                dispatch_cost = dispatch_fwd
                service_cost = service_fwd
                charging_cost = charge_fwd

            # 汇总
            routes_ra.append(ra)
            decoded_routes.append(decode_route)
            routes_cost.append(route_cost)
            routes_dist.append(travel_dist)
            total_num_charge += num_charge
            travel_cost_list.append(travel_cost)
            dispatch_cost_list.append(dispatch_cost)
            service_cost_list.append(service_cost)
            charging_cost_list.append(charging_cost)
            total_cost += route_cost
            
        total_ra = float(np.sum(routes_ra))


        return {
            "routes_ra": routes_ra,
            "decoded_routes": decoded_routes,
            "routes_cost": routes_cost,
            "routes_dist": routes_dist,
            "total_num_charge": total_num_charge,
            "total_cost": total_cost,
            "total_ra": total_ra,

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


    def run_cross_cluster2(self, max_iter=30, patience=5, seed=None, log_every=1):
        rng = np.random.default_rng(seed)

        archive = []
        A_MAX = 80   # archive 最大保留 80 个 Pareto 解（你可以调 50/100/200）


        best = None
        no_improve = 0

        history_points = []   # 记录每一代候选点 (iter, cost, ra)
        archive = []          # Pareto archive（存 cand 的关键信息）

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

            # === B) 局部扰动（跨簇 LS）===
            flat_route, LS_routes, removed_customers, new_gb_routes = \
                self.localSearch.fit_cross_clusters(flat_routes, routes_by_clusters, vehicle_capacity=650)
            LS_routes = {i: route for i, route in enumerate(LS_routes)}

            # === C) 评估新解 ===
            cand = self._evaluate_clusters(LS_routes)
            cand["flat_routes"] = {cid: r.copy() for cid, r in LS_routes.items()}
            print(sum(cand["routes_cost"]), sum(cand["routes_ra"]))

            sol = {
                "cost": float(cand["total_cost"]),
                "ra": float(cand["total_ra"]),
                "flat_routes": {cid: r.copy() for cid, r in LS_routes.items()},
                "decoded_routes": cand["decoded_routes"],  # 可选
            }

            archive = update_archive_nondominated(archive, sol, tol=1e-9)
            archive = dedup_archive(archive)

            # 超过 A_MAX 就简单按 crowding/或随机删；先用最简单的：按 cost 排序取 A_MAX
            if len(archive) > A_MAX:
                archive = sorted(archive, key=lambda s: s["cost"])[:A_MAX]



            if best is None or cand["total_cost"] + 1e-9 < best["total_cost"]:
                best = cand
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return (best["decoded_routes"], best["routes_cost"], best["routes_ra"],
                best["routes_dist"], best["total_num_charge"], 
                best["travel_cost_list"], best["dispatch_cost_list"], best["service_cost_list"], best["charging_cost_list"], 
                history_points, archive)


    def run_cross_cluster(self,
                        max_iter=30,
                        patience=5,
                        seed=None,
                        log_every=1,
                        A_MAX=80,
                        K=20,
                        eps=1e-9,
                        explore_prob=0.2):
        """
        Multi-objective version with strict Pareto archive:
        - objectives: minimize total_cost and total_ra
        - each outer iteration samples K perturbed candidates
        - archive is maintained as strict nondominated set (Pareto)
        - archive truncated by crowding distance when > A_MAX
        - next working solution is chosen from archive (mix exploit + explore)

        Returns:
            final_best: dict  (one selected solution from archive, default: min cost)
            history_points: list[(iter, cost, ra)]
            archive: list[dict]  (Pareto archive, size <= A_MAX)
        """

        rng = np.random.default_rng(seed)

        # --- helper: truncate Pareto archive by crowding distance (keep diversity) ---
        def truncate_by_crowding(arch, max_size):
            if len(arch) <= max_size:
                return arch
            # arch 应该已经是非支配集（同一 front）
            front = list(range(len(arch)))
            cd = crowding_distance(arch, front)
            keep_idx = sorted(front, key=lambda i: cd[i], reverse=True)[:max_size]
            return [arch[i] for i in keep_idx]

        # --- helper: choose next working solution from archive ---
        def pick_work_solution(arch):
            # 混合策略：大多数时候选 cost 最小（收敛），少数时候随机选（探索）
            if len(arch) == 0:
                return None
            if rng.random() < explore_prob:
                return arch[int(rng.integers(0, len(arch)))]
            return min(arch, key=lambda s: s["cost"])

        archive = []
        history_points = []

        # ========== 初始化：用你原始 pipeline 生成一个可行工作解 ==========
        routes_by_clusters = {}
        for cid, cluster in self.instance_clusters.items():
            gbs_dict = self._generate_gbs(cluster)
            gb_seq = self._plan_gbs_order(gbs_dict)
            routes, _ = self._plan_internal_gbs_order(gbs_dict, gb_seq)
            routes_by_clusters[cid] = routes

        flat_routes = {cid: [n for r in routes for n in r] for cid, routes in routes_by_clusters.items()}

        # 初始化也评估一次，放进 archive（否则 archive 可能一开始为空）
        init_cand = self._evaluate_clusters(flat_routes)
        init_sol = {
            "cost": float(init_cand["total_cost"]),
            "ra": float(init_cand["total_ra"]),
            "flat_routes": {cid: r.copy() for cid, r in flat_routes.items()},
            "decoded_routes": init_cand["decoded_routes"],
            # 如需更多字段可在这里保存：
            # "routes_cost": init_cand["routes_cost"],
            # "routes_ra": init_cand["routes_ra"],
            # "routes_dist": init_cand["routes_dist"],
            # "total_num_charge": init_cand["total_num_charge"],
        }
        archive = update_archive_nondominated(archive, init_sol, tol=eps)
        archive = dedup_archive(archive)
        archive = truncate_by_crowding(archive, A_MAX)

        best_cost_seen = init_sol["cost"]
        no_improve = 0

        # ========== 外层迭代 ==========
        for it in range(1, max_iter + 1):
            iter_best_cost = float("inf")

            # 每代采样 K 个候选
            for k in range(K):
                # --- 扰动：跨簇 LS ---
                _, LS_routes, _, _ = self.localSearch.fit_cross_clusters(
                    flat_routes, routes_by_clusters,
                    vehicle_capacity=self.vehicle_capacity
                )

                # 你的 LS_routes 可能是 list 或 dict，这里做统一：
                if isinstance(LS_routes, list):
                    LS_routes = {i: route for i, route in enumerate(LS_routes)}

                # --- 真实评估 ---
                cand = self._evaluate_clusters(LS_routes)
                cost = float(cand["total_cost"])
                ra = float(cand["total_ra"])

                history_points.append((it, cost, ra))

                sol = {
                    "cost": cost,
                    "ra": ra,
                    "flat_routes": {cid: r.copy() for cid, r in LS_routes.items()},
                    "decoded_routes": cand["decoded_routes"],
                }

                # --- 关键：严格 Pareto archive 更新（不会保留被支配点） ---
                archive = update_archive_nondominated(archive, sol, tol=eps)

                if cost < iter_best_cost:
                    iter_best_cost = cost

            # 去重 + 截断（crowding distance）
            archive = dedup_archive(archive)
            archive = truncate_by_crowding(archive, A_MAX)

            # 选一个 archive 解作为下一代工作解
            work = pick_work_solution(archive)
            if work is None:
                # 理论上不应发生（因为 init 已加入），防御一下
                break

            flat_routes = {cid: r.copy() for cid, r in work["flat_routes"].items()}

            # routes_by_clusters 同步（最小一致性：每簇一个“单GB”）
            # 这样 localSearch.fit_cross_clusters 至少不会拿旧结构扰动
            routes_by_clusters = {cid: [route.copy()] for cid, route in flat_routes.items()}

            # early stop based on cost improvement
            if iter_best_cost + eps < best_cost_seen:
                best_cost_seen = iter_best_cost
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            if (it % log_every) == 0:
                # 同时打印当前 archive 的 cost 最小和 ra 最小，方便你观察 trade-off
                min_cost = min(s["cost"] for s in archive) if archive else float("inf")
                min_ra = min(s["ra"] for s in archive) if archive else float("inf")
                print(f"[Iter {it}] archive={len(archive)}  best_cost_seen={best_cost_seen:.3f}  "
                    f"min_cost_in_arch={min_cost:.3f}  min_ra_in_arch={min_ra:.6f}")

        # 默认输出一个“单解”：cost 最小的 Pareto 解（你也可以换 knee）
        final_best = min(archive, key=lambda s: s["cost"]) if archive else init_sol
        return final_best, history_points, archive

 
if __name__ == '__main__':
    # # ========== 参数设置 ==========
    instance_name = 'pr06_evrp'
    base_path = r"D:\02_Research\DataSet\C-mdvrptw-improved"
    

    # instance_name = 'r103_21'
    # base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"

    file_path = os.path.join(base_path, f"{instance_name}.txt")

    # 设置重复运行次数
    N_RUNS = 1
    

    ra_safe=0.1
    ra_risk=0.7

    vehicle_capacity = 650
    soc_max = 40

    split_k = 0.4
    core_ratio = 0.93
    max_iter = 10
    removal_ratio = 0.3


    def dominates(a, b):
        # a,b: dict with keys "cost","ra" (both minimize)
        return (a["cost"] <= b["cost"] and a["ra"] <= b["ra"]) and \
            (a["cost"] <  b["cost"] or  a["ra"] <  b["ra"])

    def check_archive_is_nondominated(archive, tol=0.0):
        # tol 可用于容忍浮点误差，比如 1e-9
        n = len(archive)
        bad = []
        for i in range(n):
            for j in range(n):
                if i == j: 
                    continue
                ai, aj = archive[i], archive[j]
                # 带容差的支配
                if ((ai["cost"] <= aj["cost"] + tol) and (ai["ra"] <= aj["ra"] + tol) and
                    ((ai["cost"] < aj["cost"] - tol) or (ai["ra"] < aj["ra"] - tol))):
                    bad.append((i, j))
                    break
        return bad

    def pareto_front_from_points(points, tol=1e-9):
        # points: list of (cost, ra)
        front = []
        for i, (ci, ri) in enumerate(points):
            dominated = False
            for j, (cj, rj) in enumerate(points):
                if i == j: 
                    continue
                if (cj <= ci + tol and rj <= ri + tol) and (cj < ci - tol or rj < ri - tol):
                    dominated = True
                    break
            if not dominated:
                front.append((ci, ri))
        front.sort(key=lambda x: x[0])
        return front



    # ========== 存储结果 ==========
    results = {
        "all_routes": [],
        "routes_dist": [],
        "routes_cost": [],
        "routes_ra": [],
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
        clustering = ImprovedKMeans(instance.nodes, instance.distance_matrix, instance.customer_ids,
                                    vehicle_capacity=650, threshold=10.0, random_state=run_id)  # 用 run_id 变化随机种子
        clusters = clustering.fit()
        solver = GBACOPlanner(instance_name, base_path, clusters, soc_max, vehicle_capacity, 
                              ra_safe, ra_risk, 
                              split_k, core_ratio, max_iter, removal_ratio)

        # # 直接使用 clusters_result 结果
        # solver = GBACOPlanner(instance_name, base_path, clusters_result, soc_anxiety)
        
        # routes, routes_cost, routes_ra, routes_dist, routes_num_charge, \
        # travel_cost_list, dispatch_cost_list, service_cost_list, charging_cost_list, \
        # history_points, archive = solver.run_cross_cluster()

        final_best, history_points, archive = solver.run_cross_cluster()
        best_routes = final_best["decoded_routes"]
        best_cost   = final_best["cost"]
        best_ra     = final_best["ra"]

        end_time = time.time()
        running_time = end_time - start_time

        print(run_id, best_cost, best_ra)
        
        # # --- 存储结果（取总和/汇总值） ---
        # results["all_routes"].append(routes)
        # results["routes_dist"].append(sum(routes_dist))
        # results["routes_cost"].append(sum(routes_cost))
        # results['routes_ra'].append(sum(routes_ra))
        # results["dispatch_cost"].append(sum(dispatch_cost_list))
        # results["travel_cost"].append(sum(travel_cost_list))
        # results["service_cost"].append(sum(service_cost_list))
        # results["charging_cost"].append(sum(charging_cost_list))
        # results["routes_num_charge"].append(sum(routes_num_charge) if isinstance(routes_num_charge, list) else routes_num_charge)
        results["running_time"].append(running_time)

    # ========== 统计结果 ==========
    print('Instance:', instance_name)
    print(f"\n===== Final Results over {N_RUNS} runs =====")
    print('results["running_time"]', results["running_time"])

    # # 找到最佳解（以 routes_cost 最小为准）
    # best_idx = np.argmin(results["routes_cost"])
    # print("Best run index =", best_idx)
    # print("Best routes =", results["all_routes"][best_idx])

    # # 输出每个指标的统计
    # for key, values in results.items():
    #     if key == "all_routes":
    #         continue  # 不对路径做统计
    #     arr = np.array(values)
    #     print(f"{key:15s} -> mean: {arr.mean():.2f}, best: {arr.min():.2f}, worst: {arr.max():.2f}, std: {arr.std():.2f}")
    


    # ✅ 检查 1：archive 内部不应该互相支配
    bad = check_archive_is_nondominated(archive, tol=1e-9)
    print("archive size:", len(archive))
    print("domination violations inside archive:", len(bad))
    if bad:
        i, j = bad[0]
        print("Example: archive[i] dominates archive[j]")
        print("i:", archive[i]["cost"], archive[i]["ra"])
        print("j:", archive[j]["cost"], archive[j]["ra"])

    
    # ✅ 计算“全部候选”的 Pareto 前沿（推荐你一定做）
    all_points = [(c, a) for _, c, a in history_points]
    true_front = pareto_front_from_points(all_points)

    print("True front size (from ALL candidates):", len(true_front))
    print("First 20 front points:")
    for p in true_front[:20]:
        print(p[0], p[1])

    # 对比 archive：
    archive_points = sorted([(s["cost"], s["ra"]) for s in archive], key=lambda x: x[0])

    print("Archive size:", len(archive_points))
    print("First 20 archive points:")
    for p in archive_points[:20]:
        print(p[0], p[1])



    # ✅ 直接找：有没有蓝点支配橙点？
    # 建立 archive 点集合（便于过滤）
    arch = [{"cost": s["cost"], "ra": s["ra"]} for s in archive]

    # 从 history_points 找到最强“打脸”例子：某个蓝点支配某个橙点
    best_counterexample = None

    for _, c, a in history_points:
        candidate = {"cost": float(c), "ra": float(a)}
        for s in arch:
            if dominates(candidate, s):
                # 找一个差距最大的例子
                gap = (s["cost"] - candidate["cost"]) + (s["ra"] - candidate["ra"])
                if (best_counterexample is None) or (gap > best_counterexample[0]):
                    best_counterexample = (gap, candidate, s)

    if best_counterexample:
        gap, blue, orange = best_counterexample
        print("FOUND: a candidate point dominates an archive point!")
        print("Candidate (blue):", blue["cost"], blue["ra"])
        print("Archive   (orange):", orange["cost"], orange["ra"])
    else:
        print("OK: No candidate point dominates any archive point.")








    # 画图
    # 1) 过程散点（每代一个点）
    xs = [c for _, c, _ in history_points]
    ys = [a for _, _, a in history_points]

    # 2) Pareto archive
    ax = [s["cost"] for s in archive]
    ay = [s["ra"] for s in archive]

    archive_sorted = sorted(zip(ax, ay), key=lambda x: x[0])

    plt.figure(figsize=(7,5))
    plt.scatter(xs, ys, alpha=0.35, label="Candidates over iterations")
    plt.scatter(ax, ay, s=80, label="Pareto archive")
    # plt.plot([p[0] for p in archive_sorted], [p[1] for p in archive_sorted], linestyle="--")
    plt.xlabel("Total cost")
    plt.ylabel("Total anxiety (RA)")
    plt.title("Evolution: candidates and Pareto archive")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    # best_routes = results["all_routes"][best_idx]
    coordinates = [node.location() for node in instance.nodes]

    plot_routes(
        coordinates=coordinates,
        routes=best_routes,
        depot_list=instance.depot_ids,
        cs_list=instance.cs_ids,
        customer_list=instance.customer_ids,
        title=f"Best Routes of {instance_name}"
    )



    # soc_anxiety = [1]

    # for anxiety in soc_anxiety:
    #     print('anxiety', anxiety)
    #     base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)\123"
    #     N_RUNS = 20

    #     summary_results = []

    #     for file_name in os.listdir(base_path):
    #         if not file_name.endswith(".txt"):
    #             continue

    #         instance_name = file_name.replace(".txt", "")
    #         file_path = os.path.join(base_path, file_name)

    #         results = {
    #             "all_routes": [],
    #             "routes_dist": [],
    #             "routes_cost": [],
    #             "dispatch_cost": [],
    #             "travel_cost": [],
    #             "service_cost": [],
    #             "charging_cost": [],
    #             "routes_num_charge": [],
    #             "running_time": []
    #         }

    #         for run_id in range(N_RUNS):
    #             start_time = time.time()

    #             instance = EVRPInstance(file_path)
    #             solver = GBACOPlanner(instance_name, base_path, clusters_result, anxiety)
    #             routes, routes_cost, routes_dist, routes_num_charge, travel_cost_list, dispatch_cost_list, service_cost_list, charging_cost_list = solver.run_cross_cluster()

    #             end_time = time.time()
    #             running_time = end_time - start_time
    #             print('run_id', run_id, sum(routes_cost))

    #             results["all_routes"].append(routes)
    #             results["routes_dist"].append(sum(routes_dist))
    #             results["routes_cost"].append(sum(routes_cost))
    #             results["dispatch_cost"].append(sum(dispatch_cost_list))
    #             results["travel_cost"].append(sum(travel_cost_list))
    #             results["service_cost"].append(sum(service_cost_list))
    #             results["charging_cost"].append(sum(charging_cost_list))
    #             results["routes_num_charge"].append(sum(routes_num_charge) if isinstance(routes_num_charge, list) else routes_num_charge)
    #             results["running_time"].append(running_time)

    #         best_idx = np.argmin(results["routes_cost"])
    #         best_routes = results["all_routes"][best_idx]

    #         row = {"instance": instance_name, "Best routes": str(best_routes)}

    #         for key, values in results.items():
    #             if key == "all_routes":
    #                 continue
    #             arr = np.array(values)
    #             row[key] = f"mean: {arr.mean():.2f}, best: {arr.min():.2f}, worst: {arr.max():.2f}, std: {arr.std():.2f}"

    #         summary_results.append(row)

    #         print('file_name', file_name, '完成')

    #     # 保存到单独文件
    #     df = pd.DataFrame(summary_results)
    #     output_path = os.path.join(r"D:\02_Research\DataSet\evrptw_instances_LijunFan", f"final_results_summary_style_anxiety_{anxiety}.xlsx")
    #     df.to_excel(output_path, index=False)

    #     print(f"保存完成 (soc_anxiety={anxiety}):", output_path)
















