

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



# --- 加载实例 ---
instance_name = 'rc203_21'
base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
file_path = os.path.join(base_path, f"{instance_name}.txt")

instance = EVRPInstance(file_path)
energy_model = EnergyModel()
decoder = Decoder(instance.nodes, instance.distance_matrix, energy_model, instance.cs_ids, 
                               Q_max=40, margin_energy=0.2, radius_km=3.0, soc_anxiety=0.5)


# --- 手动测试一条路径 ---
test_route = [0, 102, 75, 117, 115, 114, 92, 93, 96, 30, 35, 32, 31, 33, 
              68, 38, 37, 36, 34, 108, 80, 118, 79, 45, 43, 70, 40, 41, 
              120, 73, 87, 86, 90, 119, 109, 76, 121, 111, 0]

load = sum(instance.nodes[c].demand for c in test_route if c not in instance.depot_ids)

result = decoder.decode_backtrack_charging_final(test_route, load)

print("解码结果：")
print("路径:", result[0])
print("总距离:", result[1])
print("行驶时间:", result[2])
print("充电时间:", result[3])
print("服务时间:", result[4])
print("充电次数:", result[5])
print("剩余电量:", result[6])
print("到达时间:", result[7])
print("剩余负载:", result[8])
