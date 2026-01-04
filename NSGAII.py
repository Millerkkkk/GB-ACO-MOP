
def dominates(a, b, tol=1e-9):
    # minimize both: cost, ra
    return (a["cost"] <= b["cost"] + tol and a["ra"] <= b["ra"] + tol) and \
           (a["cost"] <  b["cost"] - tol or  a["ra"] <  b["ra"] - tol)


def fast_nondominated_sort(pop):
    """返回 fronts: list[list[idx]]，fronts[0] 是第一前沿"""
    S = [set() for _ in range(len(pop))]
    n = [0 for _ in range(len(pop))]
    fronts = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].add(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # 最后一个空的去掉
    return fronts

def crowding_distance(pop, front):
    """对某个 front(索引列表)计算 crowding distance，返回 dict idx->dist"""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist

    # 两个目标：cost, ra
    for key in ["cost", "ra"]:
        front_sorted = sorted(front, key=lambda i: pop[i][key])
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")

        min_v = pop[front_sorted[0]][key]
        max_v = pop[front_sorted[-1]][key]
        denom = (max_v - min_v) if (max_v - min_v) > 1e-12 else 1e-12

        for k in range(1, len(front_sorted) - 1):
            prev_v = pop[front_sorted[k - 1]][key]
            next_v = pop[front_sorted[k + 1]][key]
            dist[front_sorted[k]] += (next_v - prev_v) / denom

    return dist

def nsga2_truncate(pop, max_size):
    """对 pop 执行 NSGA-II 选择，返回截断后的新pop"""
    if len(pop) <= max_size:
        return pop

    fronts = fast_nondominated_sort(pop)
    new_pop = []
    for front in fronts:
        if len(new_pop) + len(front) <= max_size:
            new_pop.extend([pop[i] for i in front])
        else:
            # 需要在这个 front 内按 crowding distance 选
            cd = crowding_distance(pop, front)
            front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
            need = max_size - len(new_pop)
            new_pop.extend([pop[i] for i in front_sorted[:need]])
            break
    return new_pop





def update_archive_nondominated(archive, cand, tol=1e-9):
    """
    维护一个严格的 Pareto archive（只保留非支配解）
    """
    # 1) 如果 cand 被 archive 中任何点支配，则丢弃
    for s in archive:
        if dominates(s, cand, tol=tol):
            return archive

    # 2) 否则 cand 进入 archive，同时删掉被 cand 支配的旧解
    new_archive = []
    for s in archive:
        if not dominates(cand, s, tol=tol):
            new_archive.append(s)
    new_archive.append(cand)
    return new_archive

def dedup_archive(archive, tol_cost=1e-9, tol_ra=1e-9):
    """
    去掉几乎重复的点（避免你现在 True front 里重复很多次）
    """
    out = []
    for s in archive:
        ok = True
        for t in out:
            if abs(s["cost"] - t["cost"]) <= tol_cost and abs(s["ra"] - t["ra"]) <= tol_ra:
                ok = False
                break
        if ok:
            out.append(s)
    return out




