def decode_partial_charging(self, route, load, departure_time=0):
    """
    1 逐边检查可达性：依次判断 prev → curr 是否可行；若电量不足，则在 prev 后插入最近充电站，并采用部分充电策略补能。
    2 更新行驶与充电状态：在插入充电站后，计算到达时间、电量恢复、行驶距离和充电时间等，并更新全局累计指标。
    3 直奔终点模式：若在某次充电时无需充满即可直接支撑到终点（电量未达到电池上限），则锁定为“直奔终点模式”，后续跳过“curr → 最近 CS”的检查，只继续执行路径直至终点。
    4 特殊情况处理：
        若 curr → 最近 CS 也不可达，则回退到 prev 并在 prev 插入充电站。
    5结果输出：最终返回更新后的路径、总距离、总行驶时间、总充电时间、服务时间、充电次数、剩余电量和结束时刻。
    """
    Q = self.Q_max
    total_charging_time = 0.0
    total_travel_time = 0.0
    total_service_time = 0.0
    total_dist = 0.0
    num_charge = 0

    # 新增：是否锁定“直奔终点模式”
    lock_to_depot = False
    EPS = 1e-9  # 浮点容差，避免边界抖动

    # print("Route=", route)
    # print("起始状态:", "Q=", Q, "Load=", load, "Departure_time=", departure_time)

    i = 1
    while i < len(route):
        prev_node = route[i - 1]
        curr_node = route[i]

        # print(f"\nindex {i}: {prev_node} -> {curr_node}, 当前Q={Q}, Load={load}, Departure={departure_time}")

        # ---------- 1. 计算 prev → curr ----------
        prev_to_curr_dist = self.dist_matrix[prev_node, curr_node]
        prev_to_curr_energy, prev_to_curr_time, _ = self.energy_model.calculate_one_node_energy_time(
            prev_to_curr_dist, departure_time, load
        )
        # print(f"  prev→curr 耗能={prev_to_curr_energy}, 时间={prev_to_curr_time}")

        # ---------- A. 判断 Q 是否足够去 curr ----------
        if Q < prev_to_curr_energy:
            
            nearest_cs_prev = self._get_nearest_cs(prev_node)
            # print(f"  ❌ 电量不足! Q={Q}, 需要={prev_to_curr_energy}")
            # print(f"  在 {prev_node} 插入最近充电站 {nearest_cs_prev}")

            dist_to_cs_prev = self.dist_matrix[prev_node, nearest_cs_prev]
            energy_cs_prev, time_cs_prev, _ = self.energy_model.calculate_one_node_energy_time(
                dist_to_cs_prev, departure_time, load
            )
            # print(f"  prev→CS 耗能={energy_cs_prev}, 时间={time_cs_prev}")

            # 插桩：prev → CS
            route.insert(i, nearest_cs_prev)
            arrival_cs_time = departure_time + time_cs_prev
            arrival_cs_Q = Q - energy_cs_prev
            # print(f"  到达CS: Arrival_time={arrival_cs_time}, Arrival_Q={arrival_cs_Q}")

            # 使用 partial charging 策略
            required_charge, charge_time = self._estimate_partial_charge(
                cs_index_in_route=i,
                route=route,
                Q_remain=arrival_cs_Q,
                departure_time=arrival_cs_time,
                load=load
            )
            # print(f"  充电: 需要={required_charge}, 时间={charge_time}")

            # ★ 触发“直奔终点模式”的充分条件（未用满可充余量）
            if required_charge + arrival_cs_Q + EPS < self.Q_max:
                lock_to_depot = True

            # 更新状态
            Q = arrival_cs_Q + required_charge
            departure_time = arrival_cs_time + charge_time
            total_travel_time += time_cs_prev
            total_dist += dist_to_cs_prev
            total_charging_time += charge_time
            num_charge += 1

            # print(f"  ✅ 充电后: Q={Q}, Departure={departure_time}, 总行驶={total_dist}, 总充电={total_charging_time}")

            i += 1
            continue

        # ✅ prev 可以到达 curr：执行前进
        arrival_time = departure_time + prev_to_curr_time
        departure_time = arrival_time + self.all_nodes[curr_node].service_time
        Q -= prev_to_curr_energy
        load -= self.all_nodes[curr_node].demand

        total_travel_time += prev_to_curr_time
        total_service_time += self.all_nodes[curr_node].service_time
        total_dist += prev_to_curr_dist

        # print(f"  ✅ 到达 {curr_node}: Arrival_time={arrival_time}, Departure={departure_time}, Q剩余={Q}, Load={load}")

        # ---------- B. 判断 curr → 最近 CS ----------
        # 若 curr 已经是最后一个节点，跳过
        if i == len(route) - 1:
            # print("  🔚 最后一个节点，无需判断 curr→CS")
            i += 1
            continue
        
        # 若已进入“直奔终点模式”，跳过所有 curr→CS 判断
        if lock_to_depot:
            # print(f"  ⏭️ 跳过 curr→CS 判断（直奔终点模式），i={i}, curr={curr_node}")
            i += 1
            continue
        
        # 常规：判断 curr→最近CS 是否可达
        nearest_cs_curr = self._get_nearest_cs(curr_node)
        dist_to_cs_curr = self.dist_matrix[curr_node, nearest_cs_curr]
        energy_cs_curr, time_cs_curr, _ = self.energy_model.calculate_one_node_energy_time(
            dist_to_cs_curr, departure_time, load
        )
        # print(f"  curr→CS 耗能={energy_cs_curr}, 当前Q={Q}")

        if Q < energy_cs_curr:
            # print(f"  ❌ curr电量不足! 回退插桩 prev→CS")

            # 回退：恢复到 prev 状态
            departure_time -= self.all_nodes[curr_node].service_time + prev_to_curr_time
            Q += prev_to_curr_energy
            load += self.all_nodes[curr_node].demand
            total_travel_time -= prev_to_curr_time
            total_service_time -= self.all_nodes[curr_node].service_time
            total_dist -= prev_to_curr_dist
            # print(f"  回退后状态: Departure={departure_time}, Q={Q}, Load={load}")

            # 重新计算 prev → CS
            nearest_cs_prev = self._get_nearest_cs(prev_node)
            dist_to_cs_prev = self.dist_matrix[prev_node, nearest_cs_prev]
            energy_cs_prev, time_cs_prev, _ = self.energy_model.calculate_one_node_energy_time(
                dist_to_cs_prev, departure_time, load
            )
            # print(f"  prev→CS (回退) 耗能={energy_cs_prev}")

            # 插桩 prev → CS
            route.insert(i, nearest_cs_prev)
            arrival_cs_time = departure_time + time_cs_prev
            arrival_cs_Q = Q - energy_cs_prev

            # 使用 partial charging 策略
            required_charge, charge_time = self._estimate_partial_charge(
                cs_index_in_route=i,
                route=route,
                Q_remain=arrival_cs_Q,
                departure_time=arrival_cs_time,
                load=load
            )
            # print(f"  回退充电: 需要={required_charge}, 时间={charge_time}")
            
            # ★ 触发“直奔终点模式”的充分条件
            if required_charge + arrival_cs_Q + EPS < self.Q_max:
                lock_to_depot = True

            # 更新状态
            Q = arrival_cs_Q + required_charge
            departure_time = arrival_cs_time + charge_time
            total_travel_time += time_cs_prev
            total_dist += dist_to_cs_prev
            total_charging_time += charge_time
            num_charge += 1

            # print(f"  ✅ 回退充电后: Q={Q}, Departure={departure_time}, 总行驶={total_dist}, 总充电={total_charging_time}")

            i += 1
            continue

        i += 1

    # print("\n最终结果:")
    # print("Route:", route)
    # print("总距离=", total_dist, "总行驶时间=", total_travel_time,
    #     "总充电时间=", total_charging_time, "总服务时间=", total_service_time,
    #     "充电次数=", num_charge, "剩余Q=", Q, "最终Departure=", departure_time, "剩余Load=", load)

    return (route, total_dist, total_travel_time, total_charging_time,
            total_service_time, num_charge, Q, departure_time, load)









def decode_fully_charging(self, route, load, departure_time=0):
    """
    改进版 decode v22：先判断 prev→curr 是否可达，然后判断 curr→最近CS 是否可达。
    若任一不满足，则在 prev 后插入最近 CS，充电后再走 curr。
    """
    Q = self.Q_max
    total_charging_time = 0.0
    total_travel_time = 0.0
    total_service_time = 0.0
    total_dist = 0.0
    num_charge = 0

    i = 1
    while i < len(route):
        prev_node = route[i - 1]
        curr_node = route[i]

        # ---------- 1. 计算 prev → curr ----------
        prev_to_curr_dist = self.dist_matrix[prev_node, curr_node]
        prev_to_curr_energy, prev_to_curr_time, _ = self.energy_model.calculate_one_node_energy_time(
            prev_to_curr_dist, departure_time, load
        )

        # ---------- A. 判断 Q 是否足够去 curr ----------
        if Q < prev_to_curr_energy:
            # ❌ prev 无法到达 curr.    prev -> 插 CS -> curr
            nearest_cs_prev = self._get_nearest_cs(prev_node)
            dist_to_cs_prev = self.dist_matrix[prev_node, nearest_cs_prev]
            energy_cs_prev, time_cs_prev, _ = self.energy_model.calculate_one_node_energy_time(
                dist_to_cs_prev, departure_time, load
            )

            # 插桩：prev → CS
            route.insert(i, nearest_cs_prev)
            arrival_cs_time = departure_time + time_cs_prev
            charge_time = self.energy_model.calculate_charging_time(self.Q_max - (Q - energy_cs_prev))

            departure_time = arrival_cs_time + charge_time
            Q = self.Q_max
            total_travel_time += time_cs_prev
            total_dist += dist_to_cs_prev
            total_charging_time += charge_time
            num_charge += 1

            i += 1
            continue

        # ✅ prev 可以到达 curr：执行前进
        arrival_time = departure_time + prev_to_curr_time
        departure_time = arrival_time + self.all_nodes[curr_node].service_time
        Q -= prev_to_curr_energy
        load -= self.all_nodes[curr_node].demand

        total_travel_time += prev_to_curr_time
        total_service_time += self.all_nodes[curr_node].service_time
        total_dist += prev_to_curr_dist

        # ---------- B. 判断 curr → 最近 CS ----------
        if i == len(route) - 1:     # 如果是最后一个节点，直接退出循环，不然会在depot前不断的插入cs 的情况
            i += 1
            continue
        # ⭐ 新增：判断 curr 的下一个是不是 depot
        if i + 1 < len(route):
            next_node = route[i + 1]
            if self.all_nodes[next_node].is_depot():
                # 直接判断 curr → depot 能否到达
                dist_to_depot = self.dist_matrix[curr_node, next_node]
                energy_to_depot, time_to_depot, _ = self.energy_model.calculate_one_node_energy_time(
                    dist_to_depot, departure_time, load
                )

                if Q >= energy_to_depot:
                    # ✅ 可以直接到 depot，跳过插桩判断
                    i += 1
                    continue

        nearest_cs_curr = self._get_nearest_cs(curr_node)
        dist_to_cs_curr = self.dist_matrix[curr_node, nearest_cs_curr]
        energy_cs_curr, time_cs_curr, _ = self.energy_model.calculate_one_node_energy_time(
            dist_to_cs_curr, departure_time, load
        )

        if Q < energy_cs_curr:
            # # ❌ curr → CS 也无法满足，必须回退插桩 prev → CS → curr

            # 回退前状态 prev
            departure_time = departure_time - self.all_nodes[curr_node].service_time - prev_to_curr_time
            Q = Q + prev_to_curr_energy
            load = load + self.all_nodes[curr_node].demand
            total_travel_time = total_travel_time - prev_to_curr_time
            total_service_time = total_service_time - self.all_nodes[curr_node].service_time
            total_dist = total_dist - prev_to_curr_dist

            # 重新计算 prev → CS
            nearest_cs_prev = self._get_nearest_cs(prev_node)
            dist_to_cs_prev = self.dist_matrix[prev_node, nearest_cs_prev]
            energy_cs_prev, time_cs_prev, _ = self.energy_model.calculate_one_node_energy_time(
                dist_to_cs_prev, departure_time, load
            )

            # 插桩 prev → CS
            route.insert(i, nearest_cs_prev)
            arrival_cs_time = departure_time + time_cs_prev
            charge_time = self.energy_model.calculate_charging_time(self.Q_max - (Q - energy_cs_prev))

            departure_time = arrival_cs_time + charge_time
            Q = self.Q_max
            total_travel_time += time_cs_prev
            total_dist += dist_to_cs_prev
            total_charging_time += charge_time
            num_charge += 1

            i += 1
            continue

        i += 1

    return (route, total_dist, total_travel_time, total_charging_time,
            total_service_time, num_charge, Q, departure_time, load)








    def run_cross_cluster(self, max_iter=40, removal_ratio=0.4, alpha=0.1,
                             surrogate_gate=True, gate_thresh=-0.003,
                             member_ratio_thresh=0.3, patience=3, seed=None):
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

        # 基线：评估初解（更稳的做法）
        flat_routes = {cid: [n for r in routes for n in r] for cid, routes in routes_by_clusters.items()}
        best = self._evaluate_clusters(flat_routes)
        best["flat_routes"] = {cid: r.copy() for cid, r in flat_routes.items()}

        no_improve = 0
        for iter in range(1, max_iter + 1):
            # A) 跨簇 LS（不解码）
            flat_route, LS_routes, removed_customers, new_gb_routes = self.localSearch.fit_cross_clusters(flat_routes, routes_by_clusters, vehicle_capacity=650)
            LS_routes = {i: route for i, route in enumerate(LS_routes)}

            # # B) 可选 gate：替身代价未见改善就跳过解码
            # if surrogate_gate and not self._improve_better(LS_routes, best["flat_routes"], gate_thresh):
            #     no_improve += 1
            #     if no_improve >= patience: break
            #     continue

            # C) 真实评估
            cand = self._evaluate_clusters(LS_routes)         
            cand["flat_routes"] = {cid: r.copy() for cid, r in LS_routes.items()}

            if cand["total_cost"] + 1e-9 < best["total_cost"]:
                best = cand
                no_improve = 0
            else:
                no_improve += 1

            # D) 偏移检测（只看受影响簇），按需重建 + 两层 ACO
            routes_by_clusters = {}
            gbs_info_list = self.rebuild_gbs_info_list(new_gb_routes)
            for gbs_list in gbs_info_list:
                gb_seq = self._plan_gbs_order(gbs_list)
                routes, gbs_info_list = self._plan_internal_gbs_order(gbs_dict, gb_seq)
                routes_by_clusters[cid] = routes

            print('gbs_info_list=', gbs_info_list)

            

            break
            #     # 更新 cache
            #     for cid in rebuild_set:
            #         gbs_cache[cid] = self._make_gbs_cache_entry(best["flat_routes"][cid])

            #     # （可选）重建后做一次快速 gate；若很差可回退（此处略）

            # if no_improve >= patience:
            #     break
        
        total_coors = [self.instance.nodes[i].location() for i in range(len(self.instance.nodes))]
        plot_routes(total_coors, best["decoded_routes"], self.instance.depot_ids, self.instance.cs_ids, self.instance.customer_ids)

        # 最终返回（与你 run_cross_cluster 的返回格式保持一致）:contentReference[oaicite:15]{index=15}
        return (best["decoded_routes"], best["routes_cost"], best["routes_dist"], best["total_num_charge"],
                best["travel_cost_list"], best["dispatch_cost_list"], best["service_cost_list"], best["charging_cost_list"])






















def decode_partial_charging_with_3km(self, route, load, departure_time=0):
    """
    1 逐边检查可达性：依次判断 prev → curr 是否可行；若电量不足，则在 prev 后插入最近充电站，并采用部分充电策略补能。
    2 更新行驶与充电状态：在插入充电站后，计算到达时间、电量恢复、行驶距离和充电时间等，并更新全局累计指标。
    3 直奔终点模式：若在某次充电时无需充满即可直接支撑到终点（电量未达到电池上限），则锁定为“直奔终点模式”，后续跳过“curr → 最近 CS”的检查，只继续执行路径直至终点。
    4 特殊情况处理：
        若 curr → 最近 CS 也不可达，则回退到 prev 并在 prev 插入充电站。
    5结果输出：最终返回更新后的路径、总距离、总行驶时间、总充电时间、服务时间、充电次数、剩余电量和结束时刻。
    """
    Q = self.Q_max
    total_charging_time = 0.0
    total_travel_time = 0.0
    total_service_time = 0.0
    total_dist = 0.0
    num_charge = 0
    lock_to_depot = False   # 是否锁定“直奔终点模式”
    low_soc_mode = False    # SOC≤阈值（如 30%）
    self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])    # 查找 cs_dist<3km 的点，返回 {(node,cs).....}

    i = 1
    while i < len(route):
        prev_node = route[i - 1]
        curr_node = route[i]

        # ---------- 1. 计算 prev → curr ----------
        dist_prev_to_curr = self.dist_matrix[prev_node, curr_node]
        energy_prev_to_curr, time_prev_to_curr, _ = self.energy_model.calculate_one_node_energy_time(dist_prev_to_curr, departure_time, load)

        # ---------- A. 判断 Q 是否足够去 curr ----------
        if Q + self.EPS < energy_prev_to_curr:
            inserted = self._insert_nearest_cs(i, prev_node, route, Q, departure_time, load)
            if inserted is None:
                raise RuntimeError(f"从 {prev_node} 出发的电量无法到达任何CS，路径不可行。")
            cs_id, time_arrival_cs, Q_arrival_cs, dist_prev_to_cs, time_prev_to_cs = inserted

            # 使用 partial charging 策略
            required_charge, charge_time = self._estimate_partial_charge(
                cs_index_in_route=i, route=route,
                Q_remain=Q_arrival_cs, departure_time=time_arrival_cs, load=load
            )

            # 更新状态
            Q = min(self.Q_max, Q_arrival_cs + required_charge)
            departure_time = time_arrival_cs + charge_time
            total_travel_time += time_prev_to_cs
            total_dist += dist_prev_to_cs
            total_charging_time += charge_time
            num_charge += 1

            # ★ 触发“直奔终点模式”的充分条件（未用满可充余量）
            # if required_charge + self.EPS < self.Q_max:
            #     lock_to_depot = True
            # 严格校验“直奔终点模式”
            if self._can_reach_final_depot(i, route, Q, departure_time, load):
                lock_to_depot = True

            i += 1
            continue

        # ✅ prev 可以到达 curr：执行前进
        arrival_time = departure_time + time_prev_to_curr
        departure_time = arrival_time + self.all_nodes[curr_node].service_time
        Q -= energy_prev_to_curr
        load -= self.all_nodes[curr_node].demand

        total_travel_time += time_prev_to_curr
        total_service_time += self.all_nodes[curr_node].service_time
        total_dist += dist_prev_to_curr

        # 若已到终点，结束此步
        if i == len(route) - 1:
            i += 1
            continue

        # 低SOC触发：仅在 非 lock_to_depot 下，才可能进入低SOC模式
        if not lock_to_depot and not low_soc_mode and Q <= self.low_soc:
                low_soc_mode = True

        # 若已进入低SOC模式，且 curr 是 anchor 节点，则优先在其锚点桩充电
        if not lock_to_depot and low_soc_mode and (curr_node in self.cs_in_3km_node):
            cs_anchor = self.cs_in_3km_node[curr_node]
            # 保守：确保 curr→锚点桩 可达
            dist_to_anchor = self.dist_matrix[curr_node, cs_anchor]
            energy_to_anchor, time_to_anchor, _ = self.energy_model.calculate_one_node_energy_time(dist_to_anchor, departure_time, load)

            if Q + self.EPS >= energy_to_anchor:
                # 在 curr 后插入锚点桩
                route.insert(i + 1, cs_anchor)
                time_arrival_cs = departure_time + time_to_anchor
                Q_arrival_cs = Q - energy_to_anchor

                # 部分充电估计（你的原函数）
                required_charge, charge_time = self._estimate_partial_charge(
                    cs_index_in_route=i + 1, route=route,
                    Q_remain=Q_arrival_cs, departure_time=time_arrival_cs, load=load
                )

                # 更新状态
                Q = min(self.Q_max, Q_arrival_cs + required_charge)
                departure_time = time_arrival_cs + charge_time
                total_travel_time += time_to_anchor
                total_dist += dist_to_anchor
                total_charging_time += charge_time
                num_charge += 1

                # 可选：“直奔终点模式”
                # if required_charge + arrival_cs_Q + self.EPS < self.Q_max:
                #     lock_to_depot = True
                # 严格校验“直奔终点模式”
                if self._can_reach_final_depot(i + 1, route, Q, departure_time, load):
                    lock_to_depot = True

                # 已在 curr 充过电，跳过 B 判断；越过新插入的 CS
                i += 2
                continue

        # ---------- B. 判断 curr → 最近 CS ----------         
        # 若已进入“直奔终点模式”，跳过所有 curr→CS 判断
        if lock_to_depot:
            i += 1
            continue
        
        # 常规：判断 curr→最近CS 是否可达
        nearest_cs_curr = self._get_nearest_cs(curr_node)
        dist_to_cs_curr = self.dist_matrix[curr_node, nearest_cs_curr]
        energy_cs_curr, time_cs_curr, _ = self.energy_model.calculate_one_node_energy_time(dist_to_cs_curr, departure_time, load)

        if Q + self.EPS < energy_cs_curr:
            # 回退：恢复到 prev 状态
            departure_time -= self.all_nodes[curr_node].service_time + time_prev_to_curr
            Q += energy_prev_to_curr
            load += self.all_nodes[curr_node].demand
            total_travel_time -= time_prev_to_curr
            total_service_time -= self.all_nodes[curr_node].service_time
            total_dist -= dist_prev_to_curr

            inserted = self._insert_nearest_cs(i, prev_node, route, Q, departure_time, load)
            if inserted is None:
                raise RuntimeError(f"回退后从 {prev_node} 出发的电量仍无法到达任何CS，路径不可行。")
            cs_id, time_arrival_cs, Q_arrival_cs, dist_prev_to_cs, time_prev_to_cs = inserted

            # 使用 partial charging 策略
            required_charge, charge_time = self._estimate_partial_charge(
                cs_index_in_route=i, route=route,
                Q_remain=Q_arrival_cs, departure_time=time_arrival_cs, load=load
            )

            # 更新状态
            Q = min(self.Q_max, Q_arrival_cs + required_charge)
            departure_time = time_arrival_cs + charge_time
            total_travel_time += time_prev_to_cs
            total_dist += dist_prev_to_cs
            total_charging_time += charge_time
            num_charge += 1

            # # ★ 触发“直奔终点模式”的充分条件
            # if required_charge + arrival_cs_Q + self.EPS < self.Q_max:
            #     lock_to_depot = True
            # 严格校验“直奔终点模式”
            if self._can_reach_final_depot(i, route, Q, departure_time, load):
                lock_to_depot = True

            i += 1
            continue

        i += 1

    return (route, total_dist, total_travel_time, total_charging_time,
            total_service_time, num_charge, Q, departure_time, load)







gbs_route = [5, 56, 57, 58, 59, 60, 61, 64, 65, 63, 62, 93, 75, 117, 102, 82, 89, 76, 121, 91, 23, 27, 28, 29, 67, 25, 66, 26, 24, 22, 7]


e = 26.59-3.46

t = 175.50

print(e)












import matplotlib.pyplot as plt

thresholds = [10, 30, 50, 70, 90]
TD = [596.55, 590.21, 586.01, 587.62, 585.73]
VC = [358.40, 354.26, 353.00, 353.30, 349.45]
CC = [45.03, 43.91, 42.93, 43.29, 43.35]
CN = [3.05, 3.00, 3.05, 3.00, 4.15]

fig, axs = plt.subplots(2, 2, figsize=(10,8))

axs[0,0].plot(thresholds, TD, marker='o'); axs[0,0].set_title("Total travel distance (km)")
axs[0,1].plot(thresholds, VC, marker='s'); axs[0,1].set_title("Total travel cost (yuan)")
axs[1,0].plot(thresholds, CC, marker='^'); axs[1,0].set_title("Total charging cost (yuan)")
axs[1,1].plot(thresholds, CN, marker='d'); axs[1,1].set_title("Total charging times (times)")

for ax in axs.flat:
    ax.set_xlabel("Range-Anxiety Threshold (%)")
    ax.grid(True)

plt.tight_layout()
plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

data = {
    "Type": ["C"]*5 + ["R"]*5 + ["RC"]*5,
    "RAT": [10, 30, 50, 70, 90]*3,
    "TD":  [551.02,549.21,539.77,539.82,539.22,
            691.86,690.32,689.79,677.35,677.82,
            688.39,688.32,676.98,677.16,676.18],
    "TC":  [1022.42,1020.99,1014.39,1014.30,1013.00,
            1154.26,1152.67,1151.37,1137.28,1137.73,
            1133.07,1133.10,1129.48,1129.19,1130.82],
    "VC":  [326.54,325.44,320.91,320.80,319.42,
            433.00,431.80,430.65,418.93,419.38,
            408.55,408.66,408.67,408.27,410.58],
    "CC":  [35.89,35.55,33.49,33.50,33.59,
            61.26,60.87,60.72,58.35,58.35,
            64.52,64.45,60.81,60.92,60.23],
    "CN":  [3.01,3.00,3.01,3.00,4.04,
            3.11,3.21,3.85,4.91,5.14,
            3.42,3.42,4.39,4.37,5.35],
    "RT":  [15.45,16.68,14.67,14.90,14.73,
            15.03,15.69,14.71,15.09,17.52,
            15.77,14.81,14.46,14.72,17.77]
}

df = pd.DataFrame(data)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7,5))
sns.boxplot(x="RAT", y="CN", hue="Type", data=df,
            palette="Set2", linewidth=1.2)

plt.xlabel("Range-Anxiety Threshold (RAT, %)")
plt.ylabel("Charging Number (CN)")
plt.title("Distribution of Charging Frequency under Different RAT Levels")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 1️⃣ 构建原始数据
# ==========================
data = {
    "Type": ["C"]*5 + ["R"]*5 + ["RC"]*5,
    "RAT": [10, 30, 50, 70, 90]*3,
    "TD":  [551.02,549.21,539.77,539.82,539.22,
            691.86,690.32,689.79,677.35,677.82,
            688.39,688.32,676.98,677.16,676.18]
}

df = pd.DataFrame(data)

# ==========================
# 2️⃣ 绘制箱线图
# ==========================
plt.figure(figsize=(7,5))
sns.boxplot(x="RAT", y="TD", hue="Type", data=df,
            palette="Set2", linewidth=1.2)

# ==========================
# 3️⃣ 美化图形
# ==========================
plt.title("Distribution of Total Distance (TD) under Different Range-Anxiety Thresholds", fontsize=11)
plt.xlabel("Range-Anxiety Threshold (RAT, %)", fontsize=10)
plt.ylabel("Total Distance (TD)", fontsize=10)
plt.grid(True, axis='y', alpha=0.3)
plt.legend(title="Instance Type")
plt.tight_layout()
plt.show()


















