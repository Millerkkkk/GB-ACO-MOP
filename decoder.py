#  coding: UTF-8  #
'''
@Project     : decoder
@File        : evaluate.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/08 15:29
'''



import copy
from dataclasses import dataclass
import numpy as np
from collections import namedtuple

@dataclass
class CostParams:
    c1: float = 120.0    # fixed dispatch cost
    c2: float = 0.5    # travel cost per minute
    c3: float = 0.3    # service cost per minute
    c4: float = 0.6    # charging cost per minute



def calculate_cost(ev_number, travel_time, charging_time, service_time, params):

    CostDetail = namedtuple("CostDetail", [
        "dispatch_cost",
        "travel_cost",
        "service_cost",
        "charging_cost",
        # "wait_cost",
        # "delay_cost"
    ])

    dispatch_cost = params.c1 * ev_number
    travel_cost = params.c2 * travel_time
    service_cost = params.c3 * service_time
    charging_cost = params.c4 * charging_time

    cost_detail = CostDetail(dispatch_cost, travel_cost, service_cost, charging_cost)
    total_cost = sum(cost_detail)
    return total_cost, cost_detail



# =========================
# 能耗模型
# =========================
class EnergyModel:
    def __init__(self, phi_motor=1.184692, phi_battery=1.112434, g=9.8, theta_ij = 0,
                 C_r=0.012, L=3000, R=0.7, A=3.8, rho=1.2041):
        """
        初始化能耗模型参数
        :param phi_motor: 驱动电机输出效率系数 (φ^d)
        :param phi_battery: 电池输出效率系数 (φ^a)
        :param g: 重力加速度 (m/s^2)
        :param theta_ij: 路段坡度 (弧度)
        :param C_r: 滚动阻力系数
        :param L: 车辆自重 (kg)
        :param R: 空气阻力系数 (Cd)
        :param A: 车辆正面迎风面积 (m^2)
        :param rho: 空气密度 (kg/m^3)
        """
        self.phi_motor = phi_motor
        self.phi_battery = phi_battery
        self.g = g
        self.theta_ij = theta_ij
        self.C_r = C_r
        self.L = L
        self.R = R
        self.A = A
        self.rho = rho

    def calculate_energy_consumption(self, u_ijk, v_ijk_R, t_ijk_R):
        """
        计算电动车在路段 (i,j) 上的能量消耗
        Args:
            u_ijk (float): 电动车在路段 (i,j) 上的实时载荷, kg
            v_ijk_R (float): 电动车在路段 (i,j) 上的行驶速度, km/h
            t_ijk_R (float): 电动车在路段 (i,j) 上的行驶时间, h
        Returns:
            float: 消耗的能量
        """
        # The rolling and air resistance components of energy consumption 滚动阻力 + 坡度阻力
        resistance_energy = (self.g * np.sin(self.theta_ij) + self.C_r * self.g * np.cos(self.theta_ij)) * (self.L + u_ijk) / 3600.0
        # The aerodynamic drag component of energy consumption 空气阻力
        aerodynamic_drag_energy = (self.R * self.A * self.rho * (v_ijk_R ** 2)) / 76140.0
        # Total energy consumption for the EV 总能耗
        e_ijk_R = self.phi_motor * self.phi_battery * (resistance_energy + aerodynamic_drag_energy) * v_ijk_R * t_ijk_R
        return e_ijk_R
    
    def calculate_one_node_energy_time(self, distance, departure_time, u_ijk):
        max_minutes = 1440  # 一天 = 1440 分钟
        period_length = 60  # 每段 60 分钟（1 小时）
        total_energy = 0.0
        total_travel_time = 0.0
        remaining_distance = distance
        travel_speed = 0

        while remaining_distance > 0:
            # Converting time to 24-hour format
            current_time = departure_time % max_minutes
            period = int(current_time // period_length)
            period_end = (period + 1) * period_length

            current_hour = (departure_time / 60) % 24
            if 0 <= current_hour <= 2 or 10 <= current_hour <= 12:
                travel_speed = 30
            else:
                travel_speed = 60

            # Calculates the travel time in the current time period
            time_available = period_end - current_time
            max_travel_time = remaining_distance / travel_speed * 60.0  # → 以分钟计算
            t_ijk_R = min(time_available, max_travel_time)  # 本段行驶时间（分钟）

            # Calculate the energy consumption in the current time period
            energy_this_period = self.calculate_energy_consumption(u_ijk, travel_speed, t_ijk_R / 60.0)  # 转为小时

            # Update
            total_energy += energy_this_period
            total_travel_time += t_ijk_R
            remaining_distance -= travel_speed * (t_ijk_R / 60.0)
            departure_time += t_ijk_R

        return total_energy, total_travel_time, travel_speed

    def calculate_charging_time(self, q_ik):
        charging_time = max(q_ik, 0.0)     # 防止负值
        return charging_time * 60 / (0.9 * 60)  # 分钟



# =========================
# Decoder 解码器 (把ant_route 解码为插入cs后的完整route)
# =========================
class Decoder:
    def __init__(self, all_nodes, dist_matrix, energy_model, cs_ids, customer_ids,
                 ra_model, ra_safe, ra_risk, 
                 Q_max=40, margin_energy=0.5, radius_km=3.0):
        
        self.all_nodes = all_nodes
        self.dist_matrix = dist_matrix
        self.energy_model = energy_model
        self.cs_ids = cs_ids
        self.customer_ids = customer_ids
        self.Q_max = Q_max
        self.margin_energy = margin_energy
        self.radius_km = radius_km
        self.EPS = 1e-9  # 浮点容差，避免边界抖动

        self.node_to_nearest_cs = self._build_all_nearest_cs_map()

        # === RA 相关 ===
        self.ra_model = ra_model        # 续航焦虑模型（RangeAnxietyModel）
        self.ra_safe = ra_safe          # R < ra_safe：安全区
        self.ra_risk = ra_risk          # R >= ra_risk：风险区
        self.ra_log = []                # 每次解码记录一条 RA 轨迹
    
    def _build_all_nearest_cs_map(self):
        """
        为每个节点（客户 + depot）找到最近的充电站 CS，返回映射字典：
        {node_id: nearest_cs_id}
        """
        nearest_cs_map = {}
        for node in range(len(self.all_nodes)):
            min_dist = float('inf')
            closest_cs = None
            for cs in self.cs_ids:
                d = self.dist_matrix[node, cs]
                if d < min_dist:
                    min_dist = d
                    closest_cs = cs
            nearest_cs_map[node] = closest_cs
        return nearest_cs_map

    def _get_nearest_cs(self, customer):
        dist_to_css = self.dist_matrix[self.cs_ids, customer]
        nearest_cs = self.cs_ids[np.argmin(dist_to_css)]
        return nearest_cs
    
    def _estimate_partial_charge(self, cs_index_in_route, route, Q_remain, departure_time, load):
        """
        预测从当前充电站 curr_cs 出发，到达最终 depot 的整段路径能耗。
        如果预计能耗 > Q_max，则充满；
        否则按需充电，返回 required_charge 和 charge_time。
        """
        # print('0000000000000000000000000000000000000000000000000000000000000000000000000')
        # print('cs_index_in_route', cs_index_in_route, route)
        Q_max = self.Q_max
        margin = self.margin_energy      # 安全余量（若你的业务允许“刚好到达”即可，可将 margin 设为 0）

        total_energy_needed = 0.0
        temp_departure = departure_time
        temp_load = load

        # 从 curr_cs 出发，到达 route[i+1:] 全部节点，包括终点
        for j in range(cs_index_in_route, len(route)-1):
            prev_node = route[j]
            curr_node = route[j+1]

            dist = self.dist_matrix[prev_node, curr_node]
            energy, travel_time, _ = self.energy_model.calculate_one_node_energy_time(
                dist, temp_departure, temp_load
            )

            total_energy_needed += energy
            temp_departure += travel_time

            # 到客户点服务 + 卸货
            service_time = self.all_nodes[curr_node].service_time
            demand = self.all_nodes[curr_node].demand

            temp_departure += service_time
            temp_load -= demand

        total_energy_needed += margin

        # 计算所需充电量
        theoretical_required = max(0.0, total_energy_needed - Q_remain)
        required_charge = min(Q_max - Q_remain, theoretical_required)

        charge_time = self.energy_model.calculate_charging_time(required_charge)

        return required_charge, charge_time

    def _build_anchor_cs_map(self, route):
        """
        为每个节点找一个 3km 内的 CS（取最近/能量最小的一个），形成 {node: cs} 映射。
        """
        anchor_cs = {}
        for n in route:
            best = None
            best_d = float('inf')
            for cs in self.cs_ids:
                d = self.dist_matrix[n, cs]
                if d <= self.radius_km and d < best_d:
                    best = cs
                    best_d = d
            if best is not None:
                anchor_cs[n] = best
        return anchor_cs

    def _insert_nearest_cs(self, prev_idx, prev_node, route, Q, departure_time, load):
        """
        在路径中某个节点后面插入一个可达的最近充电站（CS），并返回插入后的相关状态（到达时间、剩余电量等）。这是为了避免因为电量不足导致的路径中断或死循环。
        """
        best = float('inf')
        selected_cs = None
        for cs in self.cs_ids:
            to_cs_dist = self.dist_matrix[prev_node, cs]
            to_cs_energy, to_cs_time, _ = self.energy_model.calculate_one_node_energy_time(to_cs_dist, departure_time, load)
            if Q + self.EPS >= to_cs_energy and to_cs_dist < best:
                best = to_cs_dist
                selected_cs = (cs, to_cs_time, to_cs_energy, to_cs_dist)
        if not selected_cs:
            return None
        cs, time_to_cs, energy_to_cs, dist_to_cs = selected_cs
        # 插 cs
        route.insert(prev_idx, cs)
        time_arrival_cs = departure_time + time_to_cs
        Q_arrival_cs = Q - energy_to_cs
        return (cs, time_arrival_cs, Q_arrival_cs, dist_to_cs, time_to_cs)

    def _can_reach_final_depot(self, cs_index_in_route, route, Q_after_charge, departure_time, load):
        "判断从某个充电站出发，在当前路径结构下，是否可以不再进桩直接完成剩余路径直到终点 depot。"
        "用于判断是否可以安全启用 “锁定直奔终点模式”，从而跳过之后 curr → CS 的路径检查"
        Q = Q_after_charge
        t = departure_time
        l = load
        for i in range(cs_index_in_route, len(route) - 1):
            u, v = route[i], route[i + 1]
            dist = self.dist_matrix[u, v]
            e, dt, _ = self.energy_model.calculate_one_node_energy_time(dist, t, l)
            if Q + self.EPS < e:
                return False
            Q -= e
            t += dt
            t += self.all_nodes[v].service_time
            l -= self.all_nodes[v].demand
        return True

    def _to_nearest_cs(self, prev_node, Q_at_prev, t_depart_prev, load_prev):
            nearest_cs = self.node_to_nearest_cs[prev_node]
            d = self.dist_matrix[prev_node, nearest_cs]
            e, dt, _ = self.energy_model.calculate_one_node_energy_time(d, t_depart_prev, load_prev)
            if Q_at_prev + self.EPS >= e:
                return (nearest_cs, d, e, dt)
            return None
    
    def _ra_mode(self, ra):
        """根据当前 RA 值把状态分成三类：safe / opportunity / risk。"""
        if self.ra_model is None:
            return "safe_mode"  # 没有模型就当作不焦虑

        if ra >= self.ra_risk:
            return "risk_mode"
        elif ra < self.ra_safe:
            return "safe_mode"
        else:
            return "opportunity_mode"


    def decode_backtrack_charging_before(self, route, load, departure_time=0): 
        """
        与 decode_backtrack_charging 类似，（前瞻 + 回退插桩策略）+ low_soc_mode 模式调试输出
        区别是：存在两个节点之间距离过于长从而不可达，在检测到不可达并回退插桩后，若依然无法到达下一个目标，
        此时直接返回一个“大惩罚解”，而不是无限尝试或报错。
        """
        # print(route)
        Q = self.Q_max
        EPS = self.EPS
        lock_to_depot = False   # 是否锁定“直奔终点模式”
        low_soc_mode = False

        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])  # {node: nearest_cs}， route[1:-1]：去掉首尾depot
        # print('self.cs_in_3km_node', self.cs_in_3km_node)

        total_charging_time = 0.0
        total_travel_time = 0.0
        total_service_time = 0.0
        total_dist = 0.0
        num_charge = 0

        state_log = []
        i = 1

        BIG_PENALTY = 1e6           # 无解惩罚
        WATCHDOG_LIMIT = 200        # 看门狗阈值，防死循环
        watchdog = 0
        tried_pairs = set()  # 记录 (prev_i, cs_id) 已尝试过的回退插桩，防止反复
        # 统一的惩罚返回
        def penalty_exit():
            return (
                route,          # 当前（可能已修改）路径
                BIG_PENALTY,    # total_dist
                BIG_PENALTY,    # total_travel_time
                BIG_PENALTY,    # total_charging_time
                BIG_PENALTY,    # total_service_time
                999,            # num_charge 标识异常
                0.0,            # Q
                departure_time, # 到达/离开时间，维持当前
                load            # 剩余负载
            )

        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                # print("⚠️ 看门狗触发 → 直接惩罚退出")
                return penalty_exit()

            prev_node = route[i - 1]
            curr_node = route[i]
            # print(f"\n[{i}] 当前处理: {prev_node} → {curr_node}, 当前电量: {Q:.2f}, 负载: {load}, 出发时间: {departure_time:.2f}")
            # print(route)
            # 记录快照（用于回退）
            state_log.append({
                "prev_i": i - 1,
                "route": route[:],
                "Q": Q,
                "departure_time": departure_time,
                "load": load,
                'total_travel_time': total_travel_time, 
                'total_service_time': total_service_time, 
                'total_charging_time': total_charging_time, 
                'total_dist': total_dist
            })

            # 进入 low_soc 模式（仅根据当前 Q 判断，不在焦虑分支中修改 Q）
            # if (not lock_to_depot) and Q <= self.low_soc:
            #     low_soc_mode = True

            new_low_soc = (not lock_to_depot) and (Q <= self.low_soc)
            low_soc_mode = new_low_soc

            # ---------------- 焦虑插桩：只做“位置选择 + 插入”，不做任何状态更新 ----------------
            if (not lock_to_depot) and low_soc_mode and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                # print(f"💡 当前节点 {curr_node} 附近有焦虑插桩可用 CS: {cs_id}")

                # 邻接去重：若已经是 [prev, cs, curr] 或 [prev, curr, cs]，则不重复插
                already_before = (route[i-1] == cs_id) if i-1 >= 0 else False
                already_after  = (i+1 < len(route) and route[i+1] == cs_id)

                inserted_now = False
                if (not already_before) and (not already_after):
                    next_node = route[i+1] if i+1 < len(route) else None
                    # 以距离增量选择前插/后插
                    d_prev_cs   = self.dist_matrix[prev_node, cs_id]
                    d_cs_curr   = self.dist_matrix[cs_id,   curr_node]

                    d_prev_curr = self.dist_matrix[prev_node, curr_node]
                    d_curr_cs   = self.dist_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.dist_matrix[curr_node, next_node]
                        d_cs_next   = self.dist_matrix[cs_id,    next_node]
                        # 比较四段：prev-cs-curr-next vs prev-curr-cs-next
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost  = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        # 没有 next：退化成三段比较
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost  = d_prev_curr + d_curr_cs

                    if insert_before_cost  + self.EPS < insert_after_cost:
                        route.insert(i, cs_id)
                        inserted_now = True
                        # print("前插 CS", cs_id)
                    else:
                        route.insert(i+1, cs_id)
                        inserted_now = True
                        # print("后插 CS", cs_id)

                # 只插不算：回到同一 prev（i 不变），下一轮再统一推进
                if inserted_now:
                    continue

            # ---------- 常规前进 prev → curr ----------
            dist = self.dist_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(dist, departure_time, load)
            # print(f"➡️ 前往 {curr_node}，需要能量: {energy_needed:.2f}, 当前电量: {Q:.2f}")
            # print('   需要时间：', travel_time, "服务时间", self.all_nodes[curr_node].service_time)

            # 不可达则回退
            if Q + EPS < energy_needed:
                # print(f"⛔ 无法前往 {curr_node}，开始回退寻找可达 CS 插桩点")
                inserted = False
                while state_log:
                    snap = state_log.pop()
                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_Q = snap["Q"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]
                    snap_tt = snap["total_travel_time"]
                    snap_st = snap["total_service_time"]
                    snap_ct = snap["total_charging_time"]
                    snap_dist = snap["total_dist"]

                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(prev_node2, snap_Q, snap_t, snap_load)
                    if cs_result is None:
                        # print(f"🔄 回退点 {prev_node2} 无法到达任何 CS，继续回退...")
                        continue

                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result
                    # print(f"🔁 在回退点 {prev_node2} 后插入 CS[{cs_id}]")

                    # 保险：避免将客户节点当成 CS；避免重复相邻插桩
                    if cs_id == prev_node2:
                        # print(f"⚠️ 最近CS {cs_id} 与回退点相同，跳过该回退点")
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        # print(f"⚠️ CS[{cs_id}] 已经在 {prev_node2} 后面，跳过插入")
                        continue
                    # 防回绕：同一 (prev_i, cs_id) 不重复尝试
                    if (prev_i, cs_id) in tried_pairs:
                        # print(f"⚠️ (prev_i={prev_i}, cs={cs_id}) 已尝试过，仍走回到此状态 → 直接惩罚退出")
                        return penalty_exit()
                    tried_pairs.add((prev_i, cs_id))

                    # 执行插入 + 充电估计
                    snap_route.insert(prev_i + 1, cs_id)

                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy
                    required_charge, charge_time = self._estimate_partial_charge(prev_i + 1, snap_route, Q_arrive_cs, t_arrive_cs, snap_load)

                    Q = min(self.Q_max, Q_arrive_cs + required_charge)
                    departure_time = t_arrive_cs + charge_time
                    load = snap_load
                    total_travel_time = snap_tt + to_cs_time
                    total_service_time = snap_st
                    total_charging_time = snap_ct + charge_time
                    total_dist = snap_dist + to_cs_dist
                    num_charge += 1

                    if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                        lock_to_depot = True

                    # print(f"🔁 插入回退充电桩 CS[{cs_id}] 于 {prev_node2} 之后，恢复 Q = {Q:.2f}，继续主路径")

                    route = snap_route
                    i = prev_i + 2
                    state_log = state_log[:prev_i + 1]
                    inserted = True

                    # ⭐ 插完就立刻检查“下一跳”是否可达；不可达则直接惩罚退出
                    if i < len(route):
                        next_node = route[i]
                        dist2 = self.dist_matrix[route[i - 1], next_node]
                        energy2, _, _ = self.energy_model.calculate_one_node_energy_time(
                            dist2, departure_time, load
                        )
                        if Q + EPS < energy2:
                            # print(f"⚠️ 插了 CS[{cs_id}] 后依然到不了 {next_node}，直接返回惩罚")
                            return penalty_exit()
                    
                    break

                # 可能存在 两点之间距离太远，即使充电也不能到达的情况
                if not inserted:
                    # print("⚠️ 回退点用尽仍不可达 → 直接惩罚退出")
                    return penalty_exit()
                
                # 成功插入后，继续主 while
                continue

            # 可达：推进 prev→curr（含服务）
            arrival_time = departure_time + travel_time
            service_time = self.all_nodes[curr_node].service_time
            departure_time = arrival_time + service_time
            Q -= energy_needed
            load -= self.all_nodes[curr_node].demand

            total_travel_time += travel_time
            total_service_time += service_time
            total_dist += dist
            # print(f"✅ 到达 {curr_node}：Q={Q:.2f}, t={departure_time:.2f}")

            # ---------- 新增：如果“当前节点就是 CS”，在此刻执行充电 ----------
            if curr_node in self.cs_ids:
                # 在 CS 处估算后续所需补能（从 i 开始的剩余路径）
                required_charge, charge_time = self._estimate_partial_charge(
                    i, route, Q, departure_time, load
                )

                Q = min(self.Q_max, Q + required_charge)
                departure_time += charge_time
                total_charging_time += charge_time
                num_charge += 1
                # print(f"🔋 在 CS[{curr_node}] 充电：ΔQ={required_charge:.2f}，Q={Q:.2f}，t={departure_time:.2f}")

                # 若一充电即能直奔终点，则锁定直奔终点模式（可选）
                if self._can_reach_final_depot(i, route, Q, departure_time, load):
                    lock_to_depot = True
                    # print("🚀 锁定直奔终点模式")

            i += 1

        # print(f"\n🔚 解码完成，总充电次数: {num_charge}, 总距离: {total_dist:.2f}, 总行驶时间: {total_travel_time:.2f}, 剩余电量: {Q:.2f}")
        return (
            route, total_dist, total_travel_time,
            total_charging_time, total_service_time,
            num_charge, Q, departure_time, load
        )


    def decode_backtrack_charging_final(self, route, load, departure_time=0): 
        """
        与 decode_backtrack_charging 类似，（前瞻 + 回退插桩策略）+ low_soc_mode 模式调试输出
        区别是：存在两个节点之间距离过于长从而不可达，在检测到不可达并回退插桩后，若依然无法到达下一个目标，
        此时直接返回一个“大惩罚解”，而不是无限尝试或报错。
        """
        # print(route)
        Q = self.Q_max
        EPS = self.EPS
        lock_to_depot = False   # 是否锁定“直奔终点模式”

        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])  # {node: nearest_cs}， route[1:-1]：去掉首尾depot
        # print('self.cs_in_3km_node', self.cs_in_3km_node)

        # ---------- RA 状态 ----------
        t_since_charge = 0.0   # 从上次充电起的累计行驶时间（分钟）
        Q_ref = Q              # 上次充电后的电量基线（用于相对消耗Qtr）
        total_ra = 0.0         # 累计焦虑（SUMR 增量累加）

        total_charging_time = 0.0
        total_travel_time = 0.0
        total_service_time = 0.0
        total_dist = 0.0
        num_charge = 0

        state_log = []
        i = 1

        BIG_PENALTY = 1e6           # 无解惩罚
        WATCHDOG_LIMIT = 200        # 看门狗阈值，防死循环
        watchdog = 0
        tried_pairs = set()  # 记录 (prev_i, cs_id) 已尝试过的回退插桩，防止反复

        # 防止“回退插桩分支已充电”后，到达同一CS又重复充电/计数
        skip_charge_once = False
        skip_charge_cs_id = None

        # 统一的惩罚返回
        def penalty_exit():
            return (
                route,
                BIG_PENALTY,    # total_dist
                BIG_PENALTY,    # total_travel_time
                BIG_PENALTY,    # total_charging_time
                BIG_PENALTY,    # total_service_time
                999,            # num_charge
                0.0,            # Q
                departure_time, # departure_time
                load,           # load
                BIG_PENALTY     # total_ra
            )

        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                # print("⚠️ 看门狗触发 → 直接惩罚退出")
                return penalty_exit()

            prev_node = route[i - 1]
            curr_node = route[i]
            # print(f"\n[{i}] 当前处理: {prev_node} → {curr_node}, 当前电量: {Q:.2f}, 负载: {load}, 出发时间: {departure_time:.2f}")
            # print(route)
            # 记录快照（用于回退）
            state_log.append({
                "prev_i": i - 1,
                "route": route[:],
                "Q": Q,
                "Q_ref": Q_ref,
                "t_since_charge": t_since_charge,
                "departure_time": departure_time,
                "load": load,
                "total_travel_time": total_travel_time,
                "total_service_time": total_service_time,
                "total_charging_time": total_charging_time,
                "total_dist": total_dist,
                "total_ra": total_ra,
                "lock_to_depot": lock_to_depot,
            })

            # === 计算 RA & 模式（分钟制）===
            if getattr(self, "ra_model", None) is not None:
                # 用“相对上次充电后的消耗”实现：充电后瞬时 RA≈0
                Qtr = max(0.0, Q_ref - Q)  # 已消耗（相对上次充电）
                Ttr = t_since_charge       # 分钟
                hour_of_day = (departure_time / 60.0) % 24.0  # departure_time也是分钟的话
                ra = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra)
            else:
                ra = 0.0
                mode = "safe_mode"


            # ---------- risk_mode：主动强制去最近可达 CS ----------
            if (not lock_to_depot) and (mode == "risk_mode"):
                cs_result = self._to_nearest_cs(prev_node, Q, departure_time, load)
                if cs_result is not None:
                    cs_id, _, _, _ = cs_result
                    # 若当前目标已经是这个CS就不重复插
                    if curr_node != cs_id:
                        route.insert(i, cs_id)
                        continue
                    # 如果从 prev_node 到不了任何CS，则后面“不可达”时会触发回溯；回溯仍失败则惩罚退出

            # ---------------- opportunity_mode：只做“位置选择 + 插入”，不做任何状态更新 ----------------
            if (not lock_to_depot) and (mode == "opportunity_mode") and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                # print(f"💡 当前节点 {curr_node} 附近有焦虑插桩可用 CS: {cs_id}")

                # 邻接去重：若已经是 [prev, cs, curr] 或 [prev, curr, cs]，则不重复插
                already_before = (route[i-1] == cs_id) if i-1 >= 0 else False
                already_after  = (i+1 < len(route) and route[i+1] == cs_id)

                inserted_now = False
                if (not already_before) and (not already_after):
                    next_node = route[i+1] if i+1 < len(route) else None
                    # 以距离增量选择前插/后插
                    d_prev_cs   = self.dist_matrix[prev_node, cs_id]
                    d_cs_curr   = self.dist_matrix[cs_id,   curr_node]

                    d_prev_curr = self.dist_matrix[prev_node, curr_node]
                    d_curr_cs   = self.dist_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.dist_matrix[curr_node, next_node]
                        d_cs_next   = self.dist_matrix[cs_id,    next_node]
                        # 比较四段：prev-cs-curr-next vs prev-curr-cs-next
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost  = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        # 没有 next：退化成三段比较
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost  = d_prev_curr + d_curr_cs

                    if insert_before_cost  + self.EPS < insert_after_cost:
                        route.insert(i, cs_id)
                        inserted_now = True
                        # print("前插 CS", cs_id)
                    else:
                        route.insert(i+1, cs_id)
                        inserted_now = True
                        # print("后插 CS", cs_id)

                # 只插不算：回到同一 prev（i 不变），下一轮再统一推进
                if inserted_now:
                    continue

            # ---------- 常规前进 prev → curr ----------
            dist = self.dist_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(dist, departure_time, load)
            # print(f"➡️ 前往 {curr_node}，需要能量: {energy_needed:.2f}, 当前电量: {Q:.2f}")
            # print('   需要时间：', travel_time, "服务时间", self.all_nodes[curr_node].service_time)

            # 不可达则回退
            if Q + EPS < energy_needed:
                # print(f"⛔ 无法前往 {curr_node}，开始回退寻找可达 CS 插桩点")
                inserted = False

                while state_log:
                    snap = state_log.pop()
                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_Q = snap["Q"]
                    snap_Q_ref = snap["Q_ref"]
                    snap_t_since = snap["t_since_charge"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]

                    snap_tt = snap["total_travel_time"]
                    snap_st = snap["total_service_time"]
                    snap_ct = snap["total_charging_time"]
                    snap_dist = snap["total_dist"]
                    snap_ra_total = snap["total_ra"]
                    snap_lock = snap["lock_to_depot"]

                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(prev_node2, snap_Q, snap_t, snap_load)
                    if cs_result is None:
                        # print(f"🔄 回退点 {prev_node2} 无法到达任何 CS，继续回退...")
                        continue

                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result
                    # print(f"🔁 在回退点 {prev_node2} 后插入 CS[{cs_id}]")

                    # 保险：避免将客户节点当成 CS；避免重复相邻插桩
                    if cs_id == prev_node2:
                        # print(f"⚠️ 最近CS {cs_id} 与回退点相同，跳过该回退点")
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        # print(f"⚠️ CS[{cs_id}] 已经在 {prev_node2} 后面，跳过插入")
                        continue
                    # 防回绕：同一 (prev_i, cs_id) 不重复尝试
                    if (prev_i, cs_id) in tried_pairs:
                        # print(f"⚠️ (prev_i={prev_i}, cs={cs_id}) 已尝试过，仍走回到此状态 → 直接惩罚退出")
                        return penalty_exit()
                    tried_pairs.add((prev_i, cs_id))

                    # 执行插入
                    snap_route.insert(prev_i + 1, cs_id)
                    # 先到CS
                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy
                    # 估算部分充电
                    required_charge, charge_time = self._estimate_partial_charge(prev_i + 1, snap_route, Q_arrive_cs, t_arrive_cs, snap_load)

                    # 更新主状态（回退点之后继续）
                    Q = min(self.Q_max, Q_arrive_cs + required_charge)
                    departure_time = t_arrive_cs + charge_time
                    load = snap_load
                    total_travel_time = snap_tt + to_cs_time
                    total_service_time = snap_st
                    total_charging_time = snap_ct + charge_time
                    total_dist = snap_dist + to_cs_dist
                    num_charge += 1

                    # ---- 回退插桩这次充电后：重置“从上次充电起行驶时间”和Q_ref ----
                    t_since_charge = 0.0
                    Q_ref = Q

                    # ---- 累计焦虑：把 prev_node2 -> cs_id 这一段的 SUMR 增量加上 ----
                    if getattr(self, "ra_model", None) is not None:
                        # 段开始权重取回退快照的状态（段起点=prev_node2）
                        Ttr0 = snap_t_since
                        hour0 = (snap_t / 60.0) % 24.0

                        Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                        Qtr_after = Qtr_before + to_cs_energy

                        seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                        total_ra = snap_ra_total + seg_ra
                    else:
                        total_ra = snap_ra_total

                    # 锁定直奔终点
                    if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                        lock_to_depot = True

                    # print(f"🔁 插入回退充电桩 CS[{cs_id}] 于 {prev_node2} 之后，恢复 Q = {Q:.2f}，继续主路径")

                    route = snap_route
                    i = prev_i + 2
                    state_log = state_log[:prev_i + 1]
                    inserted = True

                    # 防二次充电：下一次真正“到达这个cs节点”时跳过充电块一次
                    skip_charge_once = True
                    skip_charge_cs_id = cs_id

                    # ⭐ 插完就立刻检查“下一跳”是否可达；不可达则直接惩罚退出
                    if i < len(route):
                        next_node = route[i]
                        dist2 = self.dist_matrix[route[i - 1], next_node]
                        energy2, _, _ = self.energy_model.calculate_one_node_energy_time(
                            dist2, departure_time, load
                        )
                        if Q + EPS < energy2:
                            # print(f"⚠️ 插了 CS[{cs_id}] 后依然到不了 {next_node}，直接返回惩罚")
                            return penalty_exit()
                    
                    break

                # 可能存在 两点之间距离太远，即使充电也不能到达的情况
                if not inserted:
                    # print("⚠️ 回退点用尽仍不可达 → 直接惩罚退出")
                    return penalty_exit()
                
                # 成功插入后，继续主 while
                continue
            
            # ---------- 可达：推进 prev→curr ----------
            if getattr(self, "ra_model", None) is not None:
                # 段起点权重（出发前）
                Ttr0 = t_since_charge
                hour0 = (departure_time / 60.0) % 24.0

                Qtr_before = max(0.0, Q_ref - Q)
                Qtr_after = Qtr_before + energy_needed

                seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                gb_ra += seg_ra

            # 2) 走这段路
            arrival_time = departure_time + travel_time
            service_time = self.all_nodes[curr_node].service_time
            departure_time = arrival_time + service_time
            Q -= energy_needed
            load -= self.all_nodes[curr_node].demand

            total_travel_time += travel_time
            total_service_time += service_time
            total_dist += dist
            # print(f"✅ 到达 {curr_node}：Q={Q:.2f}, t={departure_time:.2f}")

            # “从上次充电起的行驶时间”只加行驶，不加服务
            t_since_charge += travel_time

            # ---------- 到达CS：执行充电 ----------
            if curr_node in self.cs_ids:
                # 若这是“回退插桩分支已算过充电”的那一次，到达时跳过一次
                if skip_charge_once and (curr_node == skip_charge_cs_id):
                    skip_charge_once = False
                    skip_charge_cs_id = None
                else:
                    required_charge, charge_time = self._estimate_partial_charge(
                        i, route, Q, departure_time, load
                    )

                    if required_charge > 0 or charge_time > 0:
                        Q = min(self.Q_max, Q + required_charge)
                        departure_time += charge_time
                        total_charging_time += charge_time
                        num_charge += 1

                    # 充电后：焦虑参考点重置（实现“瞬时RA≈0”）
                    t_since_charge = 0.0
                    Q_ref = Q

                    if self._can_reach_final_depot(i, route, Q, departure_time, load):
                        lock_to_depot = True
            # print('---------------------------------', ra, total_ra)
            i += 1

        # print(f"\n🔚 解码完成，总充电次数: {num_charge}, 总距离: {total_dist:.2f}, 总行驶时间: {total_travel_time:.2f}, 剩余电量: {Q:.2f}")
        return (
            route, total_dist, total_travel_time,
            total_charging_time, total_service_time,
            num_charge, Q, departure_time, load, total_ra
        )




    def simulate_within_GB(self, init_route, state):
        """
        GB 内串行模拟（一步回滚插桩版）
        - 不计算 cost，只推进：Q / t / load / RA / 充电次数 / 各类时间与距离统计
        - 允许插入 CS（充电站）以保证可行性
        - 关键：如果“到达 curr 后才发现不可行”，会回滚到 prev 的快照，再在 prev 后插入 prev 最近 CS，重新开始

        规则（每条边 prev->curr）：
        1) prev 处保存快照 snap_prev（用于回滚）
        2) 若 prev->curr 不可达：直接在 prev 后插 cs_prev（无需回滚，因为未推进）
        3) 若可达：推进到 curr（更新状态）
        4) 推进后做“后验可行性检查”：
            - 若 curr 处不可继续（例如 curr 到不了任何 CS，且后续还需要继续走），则回滚到 prev 快照，
            并在 prev 后插 cs_prev，重新尝试
        5) 若可行：i += 1

        参数：
        route: list[int]  节点序列（可含 depot/cs/customer/prev_last）
        state: dict       串行状态（不会原地修改，会 deepcopy）

        返回：
        end_state: dict   终止状态（累计）
        breakdown: dict   本 GB 增量信息（时间/次数/RA等 + final_route）
        """

        # ------------------- helpers -------------------
        def _is_customer(node_id):
            return node_id in self.customer_ids

        def _penalty_exit(cur_idx):
            # 返回“不可行”state + 大惩罚RA（用于淘汰）
            end_state = copy.deepcopy(st0)

            end_state["last_node"] = route[max(0, cur_idx - 1)] if route else end_state.get("last_node")

            end_state["total_ra"] = BIG_PENALTY
            end_state["total_route_cost"] = BIG_PENALTY
            end_state["total_distance"] = BIG_PENALTY
            
            # 把 GB 增量字段也塞进去（统一放 state）
            end_state["total_dispatch_cost"] = BIG_PENALTY
            end_state["total_travel_cost"] = BIG_PENALTY
            end_state["total_service_cost"] = BIG_PENALTY
            end_state["total_charging_cost"] = BIG_PENALTY
            end_state["num_charges"] = BIG_PENALTY
            end_state["energy_charged"] = BIG_PENALTY  
            return end_state

        # ------------------- init -------------------
        st0 = copy.deepcopy(state)
        route = copy.deepcopy(init_route)

        EPS = getattr(self, "EPS", 1e-9)
        BIG_PENALTY = 1e6
        WATCHDOG_LIMIT = 5000  # 防止插桩来回卡死（一步回退一般不会，但仍建议保留）

        # 关于 输入状态（上个GB）中的信息
        Q = st0["Q"]
        load = st0["load"]
        t = st0["t"]

        st0.setdefault("full_route", [])
        st0.setdefault("decoded_full_route", [])

        total_ra = st0["total_ra"]
        total_route_cost = st0["total_route_cost"]
        total_distance = st0["total_distance"]
        
        t_since_charge = st0["t_since_charge"]
        Q_after_last_charge = st0["Q_after_last_charge"]

        total_dispatch_cost = st0["total_dispatch_cost"]
        total_travel_cost = st0["total_travel_cost"]
        total_service_cost = st0["total_service_cost"]
        total_charging_cost = st0["total_charging_cost"]

        num_charges = st0["num_charges"]
        energy_charged = st0["energy_charged"]


        # 关于本GB中新增的 Cost/RA
        gb_ra = 0.0
        gb_dist = 0.0
        gb_travel_time = 0.0
        gb_service_time = 0.0
        gb_charging_time = 0.0
        
        # 关于其他
        gb_num_charge = 0
        gb_energy_charged = 0.0

        # 其他
        watchdog = 0
        # 防止“插了 prev 的 cs 但下一轮又插同一个”无限循环
        tried_at_prev = set()  # (prev_node, cs_id, next_node) 防止重复同一处插同一cs

        self.cs_in_3km_node = self._build_anchor_cs_map(route)  # {node: nearest_cs}

        # ------------------- main loop -------------------
        i = 1
        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                return _penalty_exit(i)

            prev_node = route[i - 1]
            curr_node = route[i]


            # ---- 1) prev 处快照（用于回滚）----
            cs_prev = self._to_nearest_cs(prev_node, Q, t, load)  # (cs_id, dist, energy, time) or None
            snap_prev = {
                "i": i-1,
                "node": prev_node,
                "Q": Q,
                "t": t,
                "load": load,
                "t_since_charge": t_since_charge,
                "Q_after_last_charge": Q_after_last_charge,
                "gb_ra": gb_ra,
                "gb_dist": gb_dist,

                "gb_travel_time": gb_travel_time,
                "gb_service_time": gb_service_time,
                "gb_charging_time": gb_charging_time,

                "gb_num_charge": gb_num_charge,
                "gb_energy_charged": gb_energy_charged,

                "cs_prev": cs_prev,
            }


            # ---- 2) risk/opportunity ----
            if getattr(self, "ra_model", None) is not None:
                Qtr = max(0.0, Q_after_last_charge - Q)
                Ttr = t_since_charge
                hour_of_day = (t / 60.0) % 24.0
                ra_inst = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra_inst)
            else:
                mode = "safe_mode"

            # risk_mode：主动在 prev 后插最近 cs（不推进）
            if mode == "risk_mode" and cs_prev is not None:
                cs_id, _, _, _ = cs_prev
                # 如果当前目标不是这个CS，则在 prev 后插入
                if curr_node != cs_id:
                    key = (prev_node, cs_id, curr_node)
                    if key in tried_at_prev:
                        return _penalty_exit(i)
                    tried_at_prev.add(key)
                    route.insert(i, cs_id)
                    continue
                # 若 prev 到不了任何CS，则继续走后面的可达性判断，不行会惩罚退出

            # opportunity_mode：按 anchor 插桩（不推进）
            if (mode == "opportunity_mode") and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                already_before = (route[i - 1] == cs_id)
                already_after = (i + 1 < len(route) and route[i + 1] == cs_id)

                if (not already_before) and (not already_after):
                    next_node = route[i + 1] if i + 1 < len(route) else None

                    d_prev_cs = self.dist_matrix[prev_node, cs_id]
                    d_cs_curr = self.dist_matrix[cs_id, curr_node]
                    d_prev_curr = self.dist_matrix[prev_node, curr_node]
                    d_curr_cs = self.dist_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.dist_matrix[curr_node, next_node]
                        d_cs_next = self.dist_matrix[cs_id, next_node]
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost = d_prev_curr + d_curr_cs

                    if insert_before_cost + EPS < insert_after_cost:
                        route.insert(i, cs_id)
                    else:
                        route.insert(i + 1, cs_id)

                    # 只插不推进，让下一轮统一推进
                    continue


            # ---------- 3) 可达性检查, 计算 prev->curr 的能耗/时间 ----------
            # （未推进，不需要回滚）
            dist = self.dist_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(
                dist, t, load
            )

            if Q + EPS < energy_needed:
                if cs_prev is None:
                    return _penalty_exit(i)

                cs_id, _, _, _ = cs_prev 

                # 避免重复插同一个cs导致死循环
                key = (prev_node, cs_id, curr_node)
                if key in tried_at_prev:
                    return _penalty_exit(i)
                tried_at_prev.add(key)

                if curr_node == cs_id:
                    # 目标就是这个CS但仍不可达 => 无解
                    return _penalty_exit(i)

                route.insert(i, cs_id)
                continue  # 不推进，下一轮处理 prev->cs
            

            # ---- 4) 推进 prev->curr（开始污染状态）----
            # 4.1 RA 段增量
            if getattr(self, "ra_model", None) is not None:
                Ttr0 = t_since_charge
                hour0 = (t / 60.0) % 24.0
                Qtr_before = max(0.0, Q_after_last_charge - Q)
                Qtr_after = Qtr_before + energy_needed
                seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                gb_ra += seg_ra

            # 4.2 推进物理状态
            arrival_time = t + travel_time
            service_time = self.all_nodes[curr_node].service_time
            t = arrival_time + service_time
            if _is_customer(curr_node):
                load -= self.all_nodes[curr_node].demand

            Q = Q - energy_needed

            gb_travel_time += travel_time
            gb_service_time += service_time
            gb_dist += dist
            t_since_charge += travel_time  # 只加行驶时间

            # 4.3 若 curr 是 CS：充电并重置参考点
            if curr_node in self.cs_ids:
                required_charge = max(0.0, self.Q_max - Q)
                if required_charge > 0:
                    charge_time = self.energy_model.calculate_charging_time(required_charge)
                    Q = self.Q_max
                    t += charge_time
                    gb_charging_time += charge_time
                    gb_num_charge += 1
                    gb_energy_charged += required_charge

                t_since_charge = 0.0
                Q_after_last_charge = Q


            # ---- 5) 推进后判断当前节点是否可以到达最近的cs ----
            # 判断：在 curr 的状态下，是否还能到达任意一个 CS
            cs_from_curr = self._to_nearest_cs(curr_node, Q, t, load)
            if cs_from_curr is None:
                # 5.1 回滚到 prev 快照
                Q = snap_prev["Q"]
                t = snap_prev["t"]
                load = snap_prev["load"]

                t_since_charge = snap_prev["t_since_charge"]
                Q_after_last_charge = snap_prev["Q_after_last_charge"]
                gb_ra = snap_prev["gb_ra"]
                gb_dist = snap_prev["gb_dist"]

                gb_travel_time = snap_prev["gb_travel_time"]
                gb_service_time = snap_prev["gb_service_time"]
                gb_charging_time = snap_prev["gb_charging_time"]
 
                gb_num_charge = snap_prev["gb_num_charge"]
                gb_energy_charged = snap_prev["gb_energy_charged"]

                # 5.2 在 prev 后插 prev 最近 CS
                if snap_prev["cs_prev"] is None:
                    return _penalty_exit(i)
                
                cs_id, _, _, _ = snap_prev["cs_prev"]

                key = (prev_node, cs_id, curr_node)
                if key in tried_at_prev:
                    return _penalty_exit(i)
                tried_at_prev.add(key)

                if curr_node == cs_id:
                    return _penalty_exit(i)

                route.insert(i, cs_id)
                continue


            # ---- 6) 成功：确认这一跳，进入下一节点 ----
            i += 1


        costParams = CostParams()
        gb_route_cost, cost_detail = calculate_cost(0, gb_travel_time, gb_charging_time, 
                                                 gb_service_time, costParams)
        
        # ------------------- finalize -------------------
        end_state = copy.deepcopy(st0)
        end_state["last_node"] = route[-1] if route else end_state.get("last_node")
        end_state["Q"] = Q
        end_state["load"] = load
        end_state["t"] = t
        end_state["full_route"] = init_route
        end_state["decoded_full_route"] = route

        end_state["total_ra"] = total_ra + gb_ra
        end_state["total_route_cost"] = total_route_cost + gb_route_cost
        end_state["total_distance"] = total_distance + gb_dist

        end_state["t_since_charge"] = t_since_charge
        end_state["Q_after_last_charge"] = Q_after_last_charge
        
        end_state["total_dispatch_cost"] = total_dispatch_cost + cost_detail.dispatch_cost
        end_state["total_travel_cost"] = total_travel_cost + cost_detail.travel_cost
        end_state["total_service_cost"] = total_service_cost + cost_detail.service_cost
        end_state["total_charging_cost"] = total_charging_cost + cost_detail.charging_cost

        end_state["num_charges"] = num_charges + gb_num_charge
        end_state["energy_charged"] = energy_charged + gb_energy_charged

        return end_state


    def simulate_to_gbCenter(self, last_node_from_last_gb, next_gb_center, state):
        """
        模拟一段“虚拟 leg”：from_node(真实节点) -> to_xy(坐标点，比如 next_gb_center)
        - 不服务、不扣 demand
        - 不插桩（默认），只做一次段推进
        - 返回：end_state（拷贝后的新state） + breakdown(这段的增量)

        allow_cs_insert: 这里建议 False。因为 to_xy 不是节点，插桩会很怪。
        """
        st0 = copy.deepcopy(state)
        EPS = getattr(self, "EPS", 1e-9)
        BIG_PENALTY = 1e6
        costParams = CostParams()

        Q = st0["Q"]
        load = st0["load"]
        t = st0["t"]

        t_since_charge = st0["t_since_charge"]
        Q_after_last_charge = st0["Q_after_last_charge"]

        total_ra = st0["total_ra"]
        total_route_cost = st0["total_route_cost"]


        # ---------- 内部小工具：RA 增量（与 simulate_within_GB 同口径） ----------
        def _ra_cal(energy_used, t_since, Qref, Qnow, t_now):
            if getattr(self, "ra_model", None) is None:
                return 0.0
            hour0 = (t_now / 60.0) % 24.0
            Qtr_before = max(0.0, Qref - Qnow)
            Qtr_after = Qtr_before + energy_used
            return self.ra_model.SUMR(Qtr_after, t_since, hour0) - self.ra_model.SUMR(Qtr_before, t_since, hour0)


        # 计算欧氏距离
        # --- from node -> center (euclid) ---
        loc_last = np.array(self.all_nodes[last_node_from_last_gb].location(), dtype=float)
        loc_center = np.array(next_gb_center, dtype=float)
        dist = float(np.linalg.norm(loc_last - loc_center))

        # 真实能耗/时间
        energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(dist, t, load)

        # ========== A) 直达可行：直接走 ==========
        if Q + EPS >= energy_needed:
            ra = _ra_cal(energy_needed, t_since_charge, Q_after_last_charge, Q, t)

            route_cost, cost_detail = calculate_cost(0, travel_time, 0, 0, costParams)

            total_ra += ra
            total_route_cost += route_cost

            return total_ra, total_route_cost
        

        # ========== B) 直达不可行：去最近可达 CS 充电，再走 ==========
        cs_prev = self._to_nearest_cs(last_node_from_last_gb, Q, t, load)  # (cs_id, dist, energy, time) or None
        if cs_prev is None:
            # from_node 本身到不了任何 CS，那就无解
            total_ra += BIG_PENALTY
            total_route_cost += BIG_PENALTY
            return total_ra, total_route_cost

        cs_id, dist_to_cs, e_to_cs, tt_to_cs = cs_prev

        # from -> cs 必须可达（_to_nearest_cs 按理已保证，但再保险）
        if Q + EPS < e_to_cs:
            total_ra += BIG_PENALTY
            total_route_cost += BIG_PENALTY
            return total_ra, total_route_cost
        
        # 1) 走到 CS
        ra_to_cs = _ra_cal(e_to_cs, t_since_charge, Q_after_last_charge, Q, t)
        Q_cs = Q - e_to_cs
        t_cs_arrive = t + tt_to_cs

        # 充满电
        required = max(0.0, self.Q_max - Q_cs)
        charge_time = self.energy_model.calculate_charging_time(required)
        Q_after_charge = self.Q_max
        t_after_charge = t_cs_arrive + charge_time


        # 充电后：RA 参考点重置
        t_since_after = 0.0
        Q_ref_after = Q_after_charge

        # 2) 从 CS 走到 center
        cs_xy = np.array(self.all_nodes[cs_id].location(), dtype=float)
        dist_cs_to_xy = float(np.linalg.norm(cs_xy - loc_center))
        e2, tt2, _ = self.energy_model.calculate_one_node_energy_time(dist_cs_to_xy, t_after_charge, load)

        if Q_after_charge + EPS < e2:
            # 即使充满也到不了 to_xy（极端：太远）
            total_ra += BIG_PENALTY
            total_route_cost += BIG_PENALTY
            return total_ra, total_route_cost
        
        ra_to_center = _ra_cal(e2, t_since_after, Q_ref_after, Q_after_charge, t_after_charge)
        

        route_cost, _ = calculate_cost(0, tt_to_cs + tt2, charge_time, 0.0, costParams)



        return total_ra + ra_to_cs + ra_to_center, total_route_cost + route_cost













