
import json, math, time, itertools, copy, random
from random import choice
from itertools import combinations, product
from mip import *
import numpy as np

def Euclidean_fun(A, B):
    return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))

class data_process(object):

    def task_date_fun(self, T, min_skill, max_skill, budget_range):
        ## task的技能要求数随机生成
        K = []
        while T != len(K):
            x = random.randint(min_skill, max_skill) ## task所要求的技能数
            K.append(x)
        # print(K, len(K))   # 每个task的技能
        Lt = []
        task_dict = {}
        for i in range(T):
            ## 位置
            l1 = random.uniform(0, 1)
            l2 = random.uniform(0, 1)
            LL = [l1, l2]
            ## 成本
            Cb = random.randint(budget_range[0], budget_range[1])
            ## 技能
            while len(Lt) != K[i]:
                x = random.randint(1, 10)
                if x in Lt:
                    continue
                else:
                    Lt.append(x)
                    Lt.sort()
            task_dict[i + 1] = {"Lt": LL, "Kt": Lt, "budget": Cb}
            Lt = []
        with open('task_noreal.json', 'w', encoding='utf-8') as fp_task:
            json.dump(task_dict, fp_task)
        print('task dict:', len(task_dict.keys()))
        return task_dict

    def worker_date_fun(self, W):

        # 每个worker的技能
        Lw = []
        worker_dict = {}
        for i in range(1, W + 1):

            worker_dict[i] = {}
            worker_dict[i]['Lw'] = [random.uniform(0, 1), random.uniform(0, 1)]
            skill_quantity = 1
            Kt = []
            while len(Kt) != skill_quantity:
                x = random.randint(1, 10)
                if x in Kt:
                    continue
                else:
                    Kt.append(x)
                    Kt.sort()
            worker_dict[i]['Kw'] = Kt

        with open('worker_noreal.json', 'w', encoding='utf-8') as fp_worker:
            json.dump(worker_dict, fp_worker)
        print('worker', len(worker_dict.keys()))
        return worker_dict

    def cooperation_date_fun(self):

        history_task_range = random.randint(100, 120)
        task_history_number_range = [10, 20]

        task_history_quantity = []  ## 历史服务 task的个数
        while len(worker_dict) != len(task_history_quantity):
            x = random.randint(task_history_number_range[0], task_history_number_range[1])
            task_history_quantity.append(x)
        # 合作 task数
        task_history_served = []
        served_history_dict = {}
        for i in range(len(worker_dict)):
            while len(task_history_served) != task_history_quantity[i]:
                x = random.randint(1, history_task_range)
                if x in task_history_served:
                    continue
                else:
                    task_history_served.append(x)
                    task_history_served.sort()
            served_history_dict[i + 1] = {"task_history_served": [task_history_quantity[i], task_history_served]}
            task_history_served = []

        cooperation_score_dict = {}
        for i in range(1, len(worker_dict.keys()) + 1):
            # print("task_history:", i, served_history.get(i)["task_history_served"][1])
            cap_served_list = []
            cup_served_list = []
            cooperation_score = []
            for j in range(1, len(worker_dict.keys()) + 1):
                if i == j:
                    cooperation_score.append(0)
                else:
                    worker_i_served = served_history_dict.get(i)["task_history_served"][1]
                    worker_j_served = served_history_dict.get(j)["task_history_served"][1]
                    cap_pair = list(set(worker_i_served).intersection(set(worker_j_served)))
                    cup_pair = list(set(worker_i_served).union(set(worker_j_served)))
                    cap_pair.sort()
                    cup_pair.sort()
                    cap_served_list.append(len(cap_pair))
                    cup_served_list.append(len(cup_pair))
                    cooperation_score.append(0.5 * 0.5 + 0.5 * float(len(cap_pair) / len(cup_pair)))
            cooperation_score_dict[i] = cooperation_score

        with open('cooperation_noreal.json', 'w', encoding='utf-8') as fp_cooperation:
            json.dump(cooperation_score_dict, fp_cooperation)

        return cooperation_score_dict

    def real_task_fun(self):
        with open('Real-data//task_500.txt', 'r', encoding='utf-8') as f1:
            task_dict = {}
            counttask = 0
            for line in f1.readlines():
                counttask += 1  # 重新编码POI的序号
                line = line.strip('\n')  # 去掉换行符\n
                b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
                b = list(map(float, b))
                # for i in range(len(b)):
                #     b[i] = int((b[i] * 100) + 0.5) / 100.0
                k = counttask
                task_dict[k] = {}
                task_dict[k]['Lt'] = b
                skill_quantity = random.randint(3, 5)
                Kt = []
                # T = len(task_dict.keys())
                while len(Kt) != skill_quantity:
                    # x = random.randint(1, 1)
                    # Kt.append(x)
                    x = random.randint(1, 10)
                    if x in Kt:
                        continue
                    else:
                        Kt.append(x)
                        Kt.sort()
                task_dict[k]['Kt'] = Kt
                task_dict[k]['budget'] = random.randint(5, 10)
        print('task dict:', len(task_dict), task_dict[1])
        with open('task.json', 'w', encoding='utf-8') as fp_task:
            json.dump(task_dict, fp_task)

        return task_dict

    def real_worker_fun(self, cooperation_dict):
        with open('Real-data//worker_1000.txt', 'r', encoding='utf-8') as f1:
            new_cooperation_dict = {}
            worker_dict = {}
            counttask = 0
            for line in f1.readlines():
                counttask += 1  # 重新编码POI的序号
                line = line.strip('\n')  # 去掉换行符\n
                b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
                b = list(map(float, b))
                b[0] = int(b[0])
                k = counttask
                worker_dict[k] = {}
                worker_dict[k]['Lw'] = [b[1], b[2]]
                skill_quantity = 1
                Kt = []
                while len(Kt) != skill_quantity:
                    x = random.randint(1, 10)
                    if x in Kt:
                        continue
                    else:
                        Kt.append(x)
                        Kt.sort()
                worker_dict[k]['Kw'] = Kt
                if b[0] in cooperation_dict.keys():
                    new_cooperation_dict[k] = cooperation_dict[b[0]]
        print('worker_dict:', len(worker_dict), worker_dict[1])
        # print('new cooperation dict:', len(new_cooperation_dict), new_cooperation_dict)
        cooperation_score_dict = {}
        for i in new_cooperation_dict.keys():
            cooperation_score_dict_low = []
            for j in new_cooperation_dict.keys():
                if i == j:
                    cooperation_score_dict_low.append(0)
                else:
                    cap_ij = set(new_cooperation_dict[i]).intersection(set(new_cooperation_dict[j]))
                    cup_ij = set(new_cooperation_dict[i]).union(set(new_cooperation_dict[j]))
                    cooperation_score_dict_low.append(0.5 * 0.5 + 0.5 * float(len(cap_ij) / len(cup_ij)))
            cooperation_score_dict[i] = cooperation_score_dict_low

        with open('worker.json', 'w', encoding='utf-8') as fp_worker:
            json.dump(worker_dict, fp_worker)
        with open('cooperation.json', 'w', encoding='utf-8') as fp_cooperation:
            json.dump(cooperation_score_dict, fp_cooperation)

        return worker_dict, cooperation_score_dict

class cartesian(object):
    def __init__(self):
        self._data_list = []

    def add_data(self, data=[]):  # 添加生成笛卡尔积的数据列表
        self._data_list.append(data)

    def build(self):  # 计算笛卡尔积
        combin = []
        for item in itertools.product(*self._data_list):
            combin.append(item)
            # print(item)
        return combin

def max_price(t, v, worker_list):
    dis = []
    for j in worker_list:
        dis.append(Euclidean_fun(task_dict[t]['Lt'], worker_dict[j]['Lw']))

    return task_dict[t]['budget'] - len(task_dict[t]['Kt']) * min(dis) * v

def cooperation_score(worker_list):
    if worker_list == None or len(worker_list) <= 1:
        return 0
    else:
        total = 0
        for wi in worker_list:
            for wj in worker_list:
                total += cooperation_score_dict[wi][wj - 1]

        return total / (len(worker_list) - 1)

def price_score(t, v, worker_list):
    if len(worker_list) == 0:
        return 0
    else:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw'])

        return task_dict[t]['budget'] - dis * v

def satisfaction_score(cooperation_s, price_s):
    return alpha * (cooperation_s / max_c) + (1 - alpha) * (price_s / max_p)

def increment(ti, tj, conflict_worker):

    # cooperation 的增量
    with_wi_cooperation = cooperation_score(best_assign[ti])
    without_wi_list = copy.deepcopy(best_assign[ti])
    without_wi_list.remove(conflict_worker)
    without_wi_cooperation = cooperation_score(without_wi_list)
    cooperation_increment_wi = (with_wi_cooperation - without_wi_cooperation) / max_c

    with_wj_cooperation = cooperation_score(best_assign[tj])
    without_wj_list = copy.deepcopy(best_assign[tj])
    without_wj_list.remove(conflict_worker)
    without_wj_cooperation = cooperation_score(without_wj_list)
    cooperation_increment_wj = (with_wj_cooperation - without_wj_cooperation) / max_c

    # price 的增量
    price_increment_wi = (task_dict[ti]['budget'] - Euclidean_fun(task_dict[ti]['Lt'],
                          worker_dict[conflict_worker]['Lw']) * v) / max_p

    price_increment_wj = (task_dict[tj]['budget'] - Euclidean_fun(task_dict[tj]['Lt'],
                          worker_dict[conflict_worker]['Lw']) * v) / max_p

    return cooperation_increment_wi + price_increment_wi, cooperation_increment_wj + price_increment_wj

def add_increment(t, worker):

    without_w_cooperation = cooperation_score(best_assign[t])
    without_w_list = copy.deepcopy(best_assign[t])
    without_w_list.append(worker)
    with_w_cooperation = cooperation_score(without_w_list)
    cooperation_increment_w = (with_w_cooperation - without_w_cooperation) / max_c
    price_increment_w = (task_dict[t]['budget'] - Euclidean_fun(task_dict[t]['Lt'], worker_dict[worker]['Lw']) * v) / max_p

    return alpha * cooperation_increment_w + (1 - alpha) * price_increment_w

def check(task, worker_list):
    skill_workers = []
    for i in worker_list:
        skill_workers.append(worker_dict[i]['Kw'][0])

    if len(task_dict[task]['Kt']) == len(skill_workers):
        flag = True
    else:
        flag = False

    return flag

def conflict_check(best_assign):

    flag = True
    for i in best_assign.keys():
        for j in best_assign.keys():
            if i == j:
                continue
            else:
                if len(set(best_assign[i]).intersection(set(best_assign[j]))) != 0:
                    flag = False
                    break

    return flag

def count_fun(best_assign):

    game_task_assigned = 0
    game_worker_assigned = []
    game_potential_satisfaction = 0
    game_optimal_satisfaction = 0
    perfect_assignment = 0

    for i in best_assign.keys():
        if len(best_assign[i]) != 0:
            game_task_assigned += 1
            for k in best_assign[i]:
                game_worker_assigned.append(k)
            game_potential_satisfaction += (
                                               (1 - alpha) * price_score(i, v, best_assign[i]) / max_p + alpha * cooperation_score(
                    best_assign[i]) / max_c)
        if len(best_assign[i]) == len(task_dict[i]['Kt']):
            game_optimal_satisfaction += ((1 - alpha) * price_score(i, v, best_assign[i]) / max_p + alpha * cooperation_score(
                best_assign[i]) / max_c)
            perfect_assignment += 1
    # print('assignment:', game_task_assigned)
    # print('perfect assignment:', perfect_assignment)
    # print('assigned workers:', len(game_worker_assigned), len(set(game_worker_assigned)))
    # print('potential satisfaction:', game_potential_satisfaction)
    # print('optimal satisfaction:', game_optimal_satisfaction)

    return game_worker_assigned, game_potential_satisfaction

def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2

def GT_algorithm(candidate_worker):
    initial_workers = copy.deepcopy(candidate_worker)
    states = {}
    assigned_workers_game = []
    for p in best_assign.keys():
        for s in best_assign[p]:
            assigned_workers_game.append(s)
    players = list(set(initial_workers).difference(set(assigned_workers_game)))
    players.sort()
    print('remaining workers:', len(set(players)))

    # 初始化 player的信息
    for w in players:
        states[w] = {}
        states[w]['strategy'] = 0
        states[w]['space'] = []
        states[w]['utility'] = 0
        states[w]['list'] = []

    # 统计 player的决策空间
    for j in players:
        for i in task_dict.keys():
            if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
                    list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                states[j]['space'].append(i)

        for k in states[j]['space']:

            if len(best_assign[k]) < 3:
                skills_workers = []
                for s in best_assign[k]:
                    skills_workers.append(worker_dict[s]['Kw'][0])

                # print(j, k, task_dict[k]['Kt'], skills_workers, worker_dict[j]['Kw'], skill_group[k])

    # 开始博弈
    be_strategy_list = []
    for i in states.keys():
        be_strategy_list.append(states[i]['strategy'])
    # print('before strategy:', len(be_strategy_list), be_strategy_list)
    af_strategy_list = [1]

    game_worker_assigned, _ = count_fun(best_assign)

    while be_strategy_list != af_strategy_list:

        increase_utility = 0
        # 初始化 player
        states = {}
        assigned_workers_game = []
        for p in best_assign.keys():
            for s in best_assign[p]:
                assigned_workers_game.append(s)
        players = list(set(initial_workers).difference(set(assigned_workers_game)))
        players.sort()
        # print('remaining workers:', len(set(players)))

        # 初始化 player的信息
        for w in players:
            states[w] = {}
            states[w]['strategy'] = 0
            states[w]['space'] = []
            states[w]['utility'] = 0
            states[w]['list'] = []

        # 统计 player的决策空间
        for j in players:
            for i in task_dict.keys():
                if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
                        list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                    states[j]['space'].append(i)

        be_strategy_list = copy.deepcopy(af_strategy_list)
        af_strategy_list = []

        for i in players:
            if i in game_worker_assigned:
                continue

            best_list = []
            update_task = 0
            best_utility = 0

            for t in states[i]['space']:

                # 当只能进行 worker的替换，展开博弈
                if len(best_assign[t]) == len(task_dict[t]['Kt']):
                    # print('worker:', i, 'worker skill:', worker_dict[i]['Kw'], 'space task:', t, 'assigned workers:', best_assign[t])

                    replace_worker = 0

                    for p in best_assign[t]:

                        if p in skill_group[t][worker_dict[i]['Kw'][0]] and p != t:
                            replace_worker = p
                            # print('placed worker:', replace_worker)
                            break

                    if replace_worker != 0:
                        before_satisfaction = alpha * (cooperation_score(best_assign[t]) / max_c) + (1 - alpha) * (
                                price_score(t, v, best_assign[t]) / max_p)
                        # print(t, 'before satisfaction:', before_satisfaction)
                        new_worker_list = copy.deepcopy(best_assign[t])
                        new_worker_list.remove(replace_worker)
                        new_worker_list.append(i)
                        after_satisfaction = alpha * (cooperation_score(new_worker_list) / max_c) + (1 - alpha) * (
                                price_score(t, v, new_worker_list) / max_p)
                        # print('new worker list:', new_worker_list)
                        # print(t, 'after satisfaction:', after_satisfaction)

                        if after_satisfaction - before_satisfaction > best_utility:
                            best_utility = after_satisfaction - before_satisfaction
                            best_list = new_worker_list
                            update_task = t
                            # print('after:', best_assign[t], states[i]['utility'])

                if len(best_assign[t]) < len(task_dict[t]['Kt']):
                    # 判断是替换还是补 worker
                    assigned_worker_skill = []
                    for j in best_assign[t]:
                        assigned_worker_skill.append(worker_dict[j]['Kw'][0])
                    # print(i, t, worker_dict[i]['Kw'], task_dict[t]['Kt'], assigned_worker_skill)
                    new_worker = []

                    if len(set(worker_dict[i]['Kw']).intersection(set(assigned_worker_skill))) != 0:

                        # 替换已分配的 workers
                        replace_worker = 0
                        for p in best_assign[t]:
                            if p in skill_group[t][worker_dict[i]['Kw'][0]] and p != t:
                                replace_worker = p
                                # print('placed worker:', replace_worker)
                                break
                        if replace_worker != 0:
                            before_satisfaction = alpha * (cooperation_score(best_assign[t]) / max_c) + (1 - alpha) * (
                                    price_score(t, v, best_assign[t]) / max_p)
                            # print(t, 'before satisfaction:', before_satisfaction)
                            new_worker_list = copy.deepcopy(best_assign[t])
                            new_worker_list.remove(replace_worker)
                            new_worker_list.append(i)
                            after_satisfaction = alpha * (cooperation_score(new_worker_list) / max_c) + (1 - alpha) * (
                                    price_score(t, v, new_worker_list) / max_p)
                            # print('new worker list:', new_worker_list)
                            # print(t, 'after satisfaction:', after_satisfaction)

                            if after_satisfaction - before_satisfaction > best_utility:
                                best_utility = after_satisfaction - before_satisfaction
                                best_list = new_worker_list
                                update_task = t
                                # print('after:', best_assign[t], states[i]['utility'])

                    if len(set(worker_dict[i]['Kw']).intersection(set(assigned_worker_skill))) == 0:
                        # 补入 worker-list中

                        # if len(best_assign[t])

                        before_satisfaction = alpha * (cooperation_score(best_assign[t]) / max_c) + (1 - alpha) * (
                                price_score(t, v, best_assign[t]) / max_p)
                        # print(t, 'before satisfaction:', before_satisfaction)
                        new_list = copy.deepcopy(best_assign[t])
                        new_list.append(i)
                        after_satisfaction = alpha * (cooperation_score(new_list) / max_c) + (1 - alpha) * (
                                price_score(t, v, new_list) / max_p)
                        # print(t, 'after satisfaction:', after_satisfaction)
                        # print('add worker:', i, t, best_assign[t], worker_dict[i]['Kw'], task_dict[t]['Kt'], assigned_worker_skill)
                        # print(cooperation_score(best_assign[t]) / max_c, price_score(t, v, best_assign[t]) / max_p, before_satisfaction, after_satisfaction, after_satisfaction - before_satisfaction)

                        if after_satisfaction - before_satisfaction > best_utility:
                            best_utility = after_satisfaction - before_satisfaction
                            best_list = new_list
                            update_task = t

            if best_utility != 0:
                # print(i, update_task, 'best utility:', best_utility, best_list, best_assign[update_task])

                states[i]['list'] = best_list
                states[i]['utility'] = best_utility
                states[i]['strategy'] = update_task
                best_assign[update_task] = best_list
                increase_utility += best_utility
                # print('best utility:', update_task, best_utility, states[i]['list'])

        for i in states.keys():
            af_strategy_list.append(states[i]['strategy'])

        # 由 task留下 utility最高的 worker
        for i in states.keys():
            for ii in states.keys():
                if i != ii:
                    if states[i]['strategy'] == states[ii]['strategy'] and states[i]['utility'] != 0 and states[ii]['utility'] != 0:
                        if states[i]['utility'] > states[ii]['utility']:
                            # print('wi:', i, 'utility:', states[i]['utility'], 'wj:', ii, 'utility:',
                            #       states[ii]['utility'], 'task:', states[i]['strategy'], states[i]['list'])
                            best_assign[states[i]['strategy']] = states[i]['list']
                            states[ii]['strategy'] = 0
                            states[ii]['utility'] = 0
                            states[ii]['list'] = []
                        else:
                            # print('wi:', i, 'utility:', states[i]['utility'], 'wj:', ii, 'utility:',
                            #       states[ii]['utility'], 'task:', states[ii]['strategy'], states[ii]['list'])
                            best_assign[states[ii]['strategy']] = states[ii]['list']
                            states[i]['strategy'] = 0
                            states[i]['utility'] = 0
                            states[i]['list'] = []

        game_worker_assigned, _ = count_fun(best_assign)
        # print(len(count_fun(best_assign)[0]), count_fun(best_assign)[1])

        # print('increase utility:', increase_utility)

    # 统计博弈之后的满意度和分配结果
    print(len(count_fun(best_assign)[0]), count_fun(best_assign)[1])
    flag = conflict_check(best_assign)
    # print('game conflict:', flag)
    # print('time:', time.time() - start)
    count_full = 0
    count_partial = 0
    for i in best_assign.keys():
        if len(best_assign[i]) != 0:
            count_partial += 1
        if len(best_assign[i]) == len(task_dict[i]['Kt']):
            count_full += 1
    print('full:', count_full, 'partial:', count_partial)

def GT_ITS_algorithm(beta, candidate_worker, best_assign):

    initial_workers = copy.deepcopy(candidate_worker)
    states = {}
    assigned_workers_game = []
    for p in best_assign.keys():
        for s in best_assign[p]:
            assigned_workers_game.append(s)
    players = list(set(initial_workers).difference(set(assigned_workers_game)))
    players.sort()
    print('remaining workers:', len(set(players)))

    # 初始化 player的信息
    for w in players:
        states[w] = {}
        states[w]['strategy'] = 0
        states[w]['space'] = []
        states[w]['utility'] = 0
        states[w]['list'] = []

    # 统计 player的决策空间
    for j in players:
        for i in task_dict.keys():
            if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
                    list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                states[j]['space'].append(i)

        for k in states[j]['space']:

            if len(best_assign[k]) < 3:
                skills_workers = []
                for s in best_assign[k]:
                    skills_workers.append(worker_dict[s]['Kw'][0])

    # 开始博弈
    be_strategy_list = []
    for i in states.keys():
        be_strategy_list.append(states[i]['strategy'])
    # print('before strategy:', len(be_strategy_list), be_strategy_list)
    af_strategy_list = [1]

    game_worker_assigned, later_satisfaction = count_fun(best_assign)
    initial_satisfaction = 0

    while later_satisfaction - initial_satisfaction > beta * initial_satisfaction:
        _, initial_satisfaction = count_fun(best_assign)

        increase_utility = 0
        # 初始化 player
        states = {}
        assigned_workers_game = []
        for p in best_assign.keys():
            for s in best_assign[p]:
                assigned_workers_game.append(s)
        players = list(set(initial_workers).difference(set(assigned_workers_game)))
        players.sort()
        print('remaining workers:', len(set(players)))

        # 初始化 player的信息
        for w in players:
            states[w] = {}
            states[w]['strategy'] = 0
            states[w]['space'] = []
            states[w]['utility'] = 0
            states[w]['list'] = []

        # 统计 player的决策空间
        for j in players:
            for i in task_dict.keys():
                if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
                        list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                    states[j]['space'].append(i)

        be_strategy_list = copy.deepcopy(af_strategy_list)
        af_strategy_list = []

        for i in players:
            if i in game_worker_assigned:
                continue

            best_list = []
            update_task = 0
            best_utility = 0

            for t in states[i]['space']:

                # 当只能进行 worker的替换，展开博弈
                if len(best_assign[t]) == len(task_dict[t]['Kt']):
                    # print('worker:', i, 'worker skill:', worker_dict[i]['Kw'], 'space task:', t, 'assigned workers:', best_assign[t])

                    replace_worker = 0

                    for p in best_assign[t]:

                        if p in skill_group[t][worker_dict[i]['Kw'][0]] and p != t:
                            replace_worker = p
                            # print('placed worker:', replace_worker)
                            break

                    if replace_worker != 0:
                        before_satisfaction = alpha * (cooperation_score(best_assign[t]) / max_c) + (1 - alpha) * (
                                price_score(t, v, best_assign[t]) / max_p)
                        # print(t, 'before satisfaction:', before_satisfaction)
                        new_worker_list = copy.deepcopy(best_assign[t])
                        new_worker_list.remove(replace_worker)
                        new_worker_list.append(i)
                        after_satisfaction = alpha * (cooperation_score(new_worker_list) / max_c) + (1 - alpha) * (
                                price_score(t, v, new_worker_list) / max_p)
                        # print('new worker list:', new_worker_list)
                        # print(t, 'after satisfaction:', after_satisfaction)

                        if after_satisfaction - before_satisfaction > best_utility:
                            best_utility = after_satisfaction - before_satisfaction
                            best_list = new_worker_list
                            update_task = t
                            # print('after:', best_assign[t], states[i]['utility'])

                if len(best_assign[t]) < len(task_dict[t]['Kt']):
                    # 判断是替换还是补 worker
                    assigned_worker_skill = []
                    for j in best_assign[t]:
                        assigned_worker_skill.append(worker_dict[j]['Kw'][0])
                    # print(i, t, worker_dict[i]['Kw'], task_dict[t]['Kt'], assigned_worker_skill)
                    new_worker = []

                    if len(set(worker_dict[i]['Kw']).intersection(set(assigned_worker_skill))) != 0:

                        # 替换已分配的 workers
                        replace_worker = 0
                        for p in best_assign[t]:
                            if p in skill_group[t][worker_dict[i]['Kw'][0]] and p != t:
                                replace_worker = p
                                # print('placed worker:', replace_worker)
                                break
                        if replace_worker != 0:
                            before_satisfaction = alpha * (cooperation_score(best_assign[t]) / max_c) + (1 - alpha) * (
                                    price_score(t, v, best_assign[t]) / max_p)
                            # print(t, 'before satisfaction:', before_satisfaction)
                            new_worker_list = copy.deepcopy(best_assign[t])
                            new_worker_list.remove(replace_worker)
                            new_worker_list.append(i)
                            after_satisfaction = alpha * (cooperation_score(new_worker_list) / max_c) + (1 - alpha) * (
                                    price_score(t, v, new_worker_list) / max_p)
                            # print('new worker list:', new_worker_list)
                            # print(t, 'after satisfaction:', after_satisfaction)

                            if after_satisfaction - before_satisfaction > best_utility:
                                best_utility = after_satisfaction - before_satisfaction
                                best_list = new_worker_list
                                update_task = t
                                # print('after:', best_assign[t], states[i]['utility'])

                    if len(set(worker_dict[i]['Kw']).intersection(set(assigned_worker_skill))) == 0:
                        # 补入 worker-list中

                        # if len(best_assign[t])

                        before_satisfaction = alpha * (cooperation_score(best_assign[t]) / max_c) + (1 - alpha) * (
                                price_score(t, v, best_assign[t]) / max_p)
                        # print(t, 'before satisfaction:', before_satisfaction)
                        new_list = copy.deepcopy(best_assign[t])
                        new_list.append(i)
                        after_satisfaction = alpha * (cooperation_score(new_list) / max_c) + (1 - alpha) * (
                                price_score(t, v, new_list) / max_p)
                        # print(t, 'after satisfaction:', after_satisfaction)
                        # print('add worker:', i, t, best_assign[t], worker_dict[i]['Kw'], task_dict[t]['Kt'], assigned_worker_skill)
                        # print(cooperation_score(best_assign[t]) / max_c, price_score(t, v, best_assign[t]) / max_p, before_satisfaction, after_satisfaction, after_satisfaction - before_satisfaction)

                        if after_satisfaction - before_satisfaction > best_utility:
                            best_utility = after_satisfaction - before_satisfaction
                            best_list = new_list
                            update_task = t

            if best_utility != 0:
                # print(i, update_task, 'best utility:', best_utility, best_list, best_assign[update_task])

                states[i]['list'] = best_list
                states[i]['utility'] = best_utility
                states[i]['strategy'] = update_task
                best_assign[update_task] = best_list
                increase_utility += best_utility
                # print('best utility:', update_task, best_utility, states[i]['list'])

        for i in states.keys():
            af_strategy_list.append(states[i]['strategy'])

        # 由 task留下 utility最高的 worker
        for i in states.keys():
            for ii in states.keys():
                if i != ii:
                    if states[i]['strategy'] == states[ii]['strategy'] and states[i]['utility'] != 0 and states[ii]['utility'] != 0:
                        if states[i]['utility'] > states[ii]['utility']:
                            # print('wi:', i, 'utility:', states[i]['utility'], 'wj:', ii, 'utility:',
                            #       states[ii]['utility'], 'task:', states[i]['strategy'], states[i]['list'])
                            best_assign[states[i]['strategy']] = states[i]['list']
                            states[ii]['strategy'] = 0
                            states[ii]['utility'] = 0
                            states[ii]['list'] = []
                        else:
                            # print('wi:', i, 'utility:', states[i]['utility'], 'wj:', ii, 'utility:',
                            #       states[ii]['utility'], 'task:', states[ii]['strategy'], states[ii]['list'])
                            best_assign[states[ii]['strategy']] = states[ii]['list']
                            states[i]['strategy'] = 0
                            states[i]['utility'] = 0
                            states[i]['list'] = []

        game_worker_assigned, later_satisfaction = count_fun(best_assign)
        print(len(count_fun(best_assign)[0]), count_fun(best_assign)[1])

        # print('increase utility:', increase_utility)

    # 统计博弈之后的满意度和分配结果
    # print(len(count_fun(best_assign)[0]), count_fun(best_assign)[1])
    flag = conflict_check(best_assign)
    print('game conflict:', flag)
    # print('time:', time.time() - start)
    count_full = 0
    count_partial = 0
    for i in best_assign.keys():
        if len(best_assign[i]) != 0:
            count_partial += 1
        if len(best_assign[i]) == len(task_dict[i]['Kt']):
            count_full += 1
    print('full:', count_full, 'partial:', count_partial)

def random_algorithm():
    assigned_workers = []
    random_best_assign = {}
    total_satisfaction = 0
    total_cooperation = 0
    total_price = 0
    for i in task_dict.keys():

        skill_group[i] = {}
        candidate_list = []
        for j in worker_dict.keys():
            if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
                    list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                candidate_list.append(j)
                candidate_worker.append(j)

        candidate[i] = candidate_list

        if len(candidate[i]) > 0:

            random_best_assign[i] = []

            d = [[] for i in range(len(task_dict[i]['Kt']))]
            for k in range(0, len(task_dict[i]['Kt'])):
                for j in candidate[i]:
                    if worker_dict[j]['Kw'][0] == task_dict[i]['Kt'][k]:
                        d[k].append(j)
                skill_group[i][task_dict[i]['Kt'][k]] = d[k]
            # print(i, skill_group[i])

            worker_list = []
            for r in skill_group[i].keys():
                skill_list = list(set(skill_group[i][r]).difference(set(assigned_workers)))
                if len(skill_list) != 0:
                    # print(choice(skill_list))
                    worker_w = choice(skill_list)
                    worker_list.append(worker_w)
                    assigned_workers.append(worker_w)
            if len(worker_list) != 0:
                profit_w = price_score(i, v, worker_list)
                random_best_assign[i] = worker_list
                score_w = cooperation_score(worker_list)
                total_satisfaction += alpha * (score_w / max_c) + (1 - alpha) * (profit_w / max_p)
                total_cooperation += score_w / max_c
                total_price += profit_w / max_p

    # print('cooperation:', total_cooperation, 'price:', total_price)
    print('candidate workers:', len(candidate[1]), 'total satisfaction:', total_satisfaction)
    # print(random_best_assign)
    # print('assignment result:', len(count_fun(random_best_assign)[0]), count_fun(random_best_assign)[1])
    # print('random time:', time.time() - random_time)
    count_full = 0
    count_partial = 0
    for i in random_best_assign.keys():
        if len(random_best_assign[i]) != 0:
            count_partial += 1
        if len(random_best_assign[i]) == len(task_dict[i]['Kt']):
            count_full += 1
    print('full:', count_full, 'partial:', count_partial)

    return random_best_assign

def baseline_greedy():
    assigned_workers = []
    greedy_best_assign = {}
    total_greedy_satisfaction = 0

    for i in task_dict.keys():

        skill_group[i] = {}
        candidate_list = []
        for j in worker_dict.keys():
            if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
                    list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                candidate_list.append(j)
                candidate_worker.append(j)

        candidate[i] = candidate_list

        if len(candidate_list) > 0:

            greedy_best_assign[i] = []
            candidate_list = list(set(candidate_list).difference(set(assigned_workers)))
            # print(i, candidate_list)
            if len(candidate_list) == 0:
                continue

            d = [[] for i in range(len(task_dict[i]['Kt']))]
            for k in range(0, len(task_dict[i]['Kt'])):
                for j in candidate_list:
                    if worker_dict[j]['Kw'][0] == task_dict[i]['Kt'][k]:
                        d[k].append(j)
                skill_group[i][task_dict[i]['Kt'][k]] = d[k]

            worker_start = []
            record = []
            while len(worker_start) == 0:

                x = choice(candidate_list)
                record.append(x)
                if x in assigned_workers:
                    continue
                else:
                    worker_start.append(x)

                if len(set(candidate_list).difference(set(record))) == 0:
                    break

            if len(worker_start) > 0:

                worker_start_index = 0
                for r in skill_group[i].keys():
                    if worker_start in skill_group[i][r]:
                        worker_start_index = r
                        break
                # print('start worker:', worker_start)

                best_s = 0
                best_list = [worker_start[0]]
                best_satisfaction = 0
                for r in skill_group[i].keys():
                    if r == worker_start_index:
                        continue
                    else:
                        best_step = []
                        for s in skill_group[i][r]:
                            if s in assigned_workers:
                                continue

                            best_list_copy = copy.deepcopy(best_list)
                            best_list_copy.append(s)
                            price_s = price_score(i, v, best_list_copy)
                            if price_s < 0:
                                continue
                            else:
                                cooperation_s = cooperation_score(best_list_copy)

                            if alpha * (cooperation_s / max_c) + (1 - alpha) * (
                                    price_s / max_p) > best_satisfaction:
                                best_satisfaction = alpha * (cooperation_s / max_c) + (1 - alpha) * (price_s / max_p)
                                best_step = best_list_copy

                        best_list = best_step
                # print(i, best_list, best_satisfaction)

                if len(best_list) != 0:
                    greedy_best_assign[i] = best_list
                    for k in best_list:
                        assigned_workers.append(k)
                    total_greedy_satisfaction += best_satisfaction
                    greedy_best_assign[i] = best_list

        # print(i, greedy_best_assign[i])

    flag = conflict_check(greedy_best_assign)
    # print('conflict greedy:', flag)

    # for i in task_dict.keys():
    #
    #     skill_group[i] = {}
    #     candidate_list = []
    #     for j in worker_dict.keys():
    #         if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
    #                 list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
    #                 candidate_list.append(j)
    #                 candidate_worker.append(j)
    #
    #     candidate[i] = candidate_list
    #
    #     if len(candidate_list) > 0:
    #
    #         greedy_best_assign[i] = []
    #
    #         d = [[] for i in range(len(task_dict[i]['Kt']))]
    #         for k in range(0, len(task_dict[i]['Kt'])):
    #             for j in candidate_list:
    #                 if worker_dict[j]['Kw'][0] == task_dict[i]['Kt'][k]:
    #                     d[k].append(j)
    #             skill_group[i][task_dict[i]['Kt'][k]] = d[k]
    #         # print(i, skill_group[i])
    #
    #         car = cartesian()
    #         for p in range(len(d)):
    #             d[p] = list(set(d[p]).difference(set(assigned_workers)))
    #             car.add_data(d[p])
    #         worker_combin = car.build()
    #
    #         part_time = time.time()
    #
    #         # 遍历所有笛卡尔积组合，找到组合中满意度最高的组合
    #         best_group = []
    #         best_group_score = 0
    #         best_satisfaction = 0
    #         for c in worker_combin:
    #
    #             profit_c = price_score(i, v, c)
    #
    #             if profit_c <= 0:
    #                 continue
    #             else:
    #                 score_c = cooperation_score(c)
    #                 if profit_c > 0 and alpha * (score_c / max_c) + (1 - alpha) * (profit_c / max_p) > best_satisfaction:
    #                     best_satisfaction = alpha * (score_c / max_c) + (1 - alpha) * (profit_c / max_p)
    #                     best_group = c
    #
    #         if len(best_group) != 0:
    #             greedy_best_assign[i] = best_group
    #             for k in best_group:
    #                 assigned_workers.append(k)
    #             total_greedy_satisfaction += best_satisfaction

    print('basic greedy satisfaction:', total_greedy_satisfaction)
    print(greedy_best_assign)
    count_full = 0
    count_partial = 0
    for i in greedy_best_assign.keys():
        if len(greedy_best_assign[i]) != 0:
            count_partial += 1
        if len(greedy_best_assign[i]) >= len(task_dict[i]['Kt']):
            count_full += 1
    print('full:', count_full, 'partial:', count_partial)
    # print('greedy time:', time.time() - greedy_time)

def all_combin_worker(workerset):
    combin_list = []
    for k in range(len(workerset), 0, -1):
        for linelist in list(combinations(workerset, k)):
            linelist = list(linelist)
            for i in product(*linelist):
                i = list(i)
                combin_list.append(i)
    return combin_list

def satisfaction(task, v, worker_set):
    return 10 * (alpha * (cooperation_score(worker_set) / max_c) + (1 - alpha) * (
                              price_score(task, v, worker_set) / max_p))

def sum_satisfaction(assignment):
    total_score = 0
    for i in assignment.keys():
        total_score += satisfaction(i, v, assignment[i])
    return total_score

def baseline_sa():

    sa_start = time.time()
    sa_assign = {}

    # 找到skill group
    for i in task_dict.keys():
        skill_group[i] = {}
        candidate_list = []
        for j in worker_dict.keys():
            if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
                    list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                candidate_list.append(j)
                candidate_worker.append(j)
        candidate[i] = candidate_list

        if len(candidate[i]) > 0:
            sa_assign[i] = []
            d = [[] for i in range(len(task_dict[i]['Kt']))]
            for k in range(0, len(task_dict[i]['Kt'])):
                for j in candidate[i]:
                    if worker_dict[j]['Kw'][0] == task_dict[i]['Kt'][k]:
                        d[k].append(j)
                skill_group[i][task_dict[i]['Kt'][k]] = d[k]

    # task 的所有 worker set组合
    worker_combine_dict = {}
    for t in task_dict.keys():
        skillset = []
        for k in skill_group[t].keys():
            if len(skill_group[t][k]) != 0:
                skillset.append(skill_group[t][k])
        worker_combine_dict[t] = all_combin_worker(skillset)

    initial_score = 0
    assigned_worker_sa = []
    for t in task_dict.keys():

        for wlist in worker_combine_dict[t]:

            flag = True
            for s in wlist:
                if s in assigned_worker_sa:
                    flag = False

            if flag == True:
                t_set = wlist
                for k in t_set:
                    assigned_worker_sa.append(k)
                worker_combine_dict[t].remove(t_set)
                initial_score += alpha * (cooperation_score(t_set) / max_c) + (1 - alpha) * (price_score(t, v, t_set) / max_p)
                sa_assign[t] = t_set
                # print(t, t_set, initial_score)

                break
            else:
                continue
    with open('sa.json', 'w', encoding='utf-8') as fp_sa:
        json.dump(sa_assign, fp_sa)

    with open('sa.json', 'r') as f_sa:
        sa_1 = json.load(f_sa)
    sa_assign = {}
    for k in sa_1.keys():
        sa_assign[int(k)] = sa_1[k]
    print(sa_assign)

    # 初始分配
    assigned_worker_sa = []
    sa_assign = random_algorithm()
    for t in sa_assign.keys():
        for k in sa_assign[t]:
            assigned_worker_sa.append(k)

    initial_sa = 0
    for i in sa_assign.keys():
        initial_sa += satisfaction(i, v, sa_assign[i])
    print('initial satisfaction:', initial_sa)
    print('initial flag:', conflict_check(sa_assign), len(assigned_worker_sa), len(set(assigned_worker_sa)))
    print('initial time:', time.time() - sa_start)

    round_d = 1000
    t = round_d + 1
    p = 0.1
    new_sa = 0

    # 后续随机分配
    task_list = list(sa_assign.keys())
    for i in range(round_d):
        # random choose task
        random_t = choice(task_list)
        temp_set = []

        # find candidate workers for the task except assigned workers
        skill_group[random_t] = {}
        candidate_list = []
        for j in worker_dict.keys():
            if Euclidean_fun(worker_dict[j]['Lw'], task_dict[random_t]['Lt']) <= R and len(
                    list(set(worker_dict[j]['Kw']).intersection(set(task_dict[random_t]['Kt'])))) != 0:
                if j not in assigned_worker_sa:
                    candidate_list.append(j)

        # if candidate is not empty, obtain the skill_group dict
        if len(candidate_list) > 0:
            # sa_assign[random_t] = []
            d = [[] for b in range(len(task_dict[random_t]['Kt']))]
            for k in range(0, len(task_dict[random_t]['Kt'])):
                for j in candidate_list:
                    if worker_dict[j]['Kw'][0] == task_dict[random_t]['Kt'][k]:
                        d[k].append(j)
                skill_group[random_t][task_dict[random_t]['Kt'][k]] = d[k]

        # find all combination worker set
        worker_combine_dict[random_t]= {}
        skillset = []
        for k in skill_group[random_t].keys():
            if len(skill_group[random_t][k]) != 0:
                skillset.append(skill_group[random_t][k])
        worker_combine_dict[random_t] = all_combin_worker(skillset)

        # find valid candidate combination
        for temp in worker_combine_dict[random_t]:
            if len(list(set(temp).intersection(set(assigned_worker_sa)))) == 0:
                temp_set.append(temp)
        # print(i, random_t, len(assigned_worker_sa), len(set(assigned_worker_sa)), len(temp_set), temp_set)

        if len(temp_set) != 0:
            old_set = sa_assign[random_t]

            random_set = choice(temp_set)
            # print(random_t, old_set, random_set, satisfaction(random_t, v, old_set), satisfaction(random_t, v, random_set))
            new_s = satisfaction(random_t, v, random_set)
            old_s = satisfaction(random_t, v, old_set)

            if new_s > old_s:
                for s in old_set:
                    if s in assigned_worker_sa:
                        assigned_worker_sa.remove(s)

                sa_assign[random_t] = random_set
                for s in random_set:
                    assigned_worker_sa.append(s)
                # print('overall:', sum_satisfaction(sa_assign))
                new_sa = sum_satisfaction(sa_assign)
                if new_sa > initial_sa:
                    initial_sa = new_sa

            elif math.exp((new_s - old_s) / t * initial_sa) > p:
                for s in old_set:
                    if s in assigned_worker_sa:
                        assigned_worker_sa.remove(s)

                # print('True......', math.exp((new_s - old_s)/t*initial_sa))
                sa_assign[random_t] = random_set
                for s in random_set:
                    assigned_worker_sa.append(s)
                new_sa = sum_satisfaction(sa_assign)
                if new_sa > initial_sa:
                    initial_sa = new_sa

        t -= 1
    print('SA satisfaction:', initial_sa)
    print(sum_satisfaction(sa_assign))
    print(conflict_check(sa_assign), len(assigned_worker_sa), len(set(assigned_worker_sa)))
    assigned_worker_sa = []
    for i in sa_assign.keys():
        for k in sa_assign[i]:
            assigned_worker_sa.append(k)
    print('new:', len(assigned_worker_sa), len(set(assigned_worker_sa)))

    count_full = 0
    count_partial = 0
    for i in sa_assign.keys():
        if len(sa_assign[i]) != 0:
            count_partial += 1
        if len(sa_assign[i]) == len(task_dict[i]['Kt']):
            count_full += 1

def dist(t, w):
    return Euclidean_fun(task_dict[t]['Lt'], worker_dict[w]['Lw'])


# from methods import random_algorithm, baseline_greedy, baseline_sa, GT_algorithm, GT_ITS_algorithm

# from utils import Euclidean_fun, price_score, cooperation_score, conflict_check, increment, count_fun, check, add_increment

if __name__ == '__main__':

    print('=============== Real data =====================')

    # 读取基本的数据
    with open('cooperation_group.json', 'r') as f_cooperation:
        cooperation_group_1 = json.load(f_cooperation)
    cooperation_group = {}
    for k in cooperation_group_1.keys():
        cooperation_group[int(k)] = cooperation_group_1[k]
    print('whole cooperation arr:', len(cooperation_group.keys()))

    # basic_data = data_process()
    # task_dict = basic_data.real_task_fun()
    # worker_dict, cooperation_score_dict = basic_data.real_worker_fun(cooperation_group)

    count = 0
    with open('task.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        if count <= 19:
            count += 1
            task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    count = 0
    with open('worker.json', 'r') as f_worker:
        worker_dict_1 = json.load(f_worker)
    worker_dict = {}
    for k in worker_dict_1.keys():
        if count <= 99:
            count += 1
            worker_dict[int(k)] = worker_dict_1[k]
    print('worker dict:', len(worker_dict))

    # # 更新预算
    # for i in task_dict.keys():
    #     budget = random.randint(10, 15)
    #     task_dict[i]['budget'] = budget
    # print('task dict:', task_dict[1])
    
    # 更新技能
    for i in task_dict.keys():
        skill_quantity = random.randint(4, 4)
        Kt = []
        # T = len(task_dict.keys())
        while len(Kt) != skill_quantity:
            # x = random.randint(1, 1)  
            # Kt.append(x)
            x = random.randint(1, 10)
            if x in Kt:
                continue
            else:
                Kt.append(x)
                Kt.sort()
        task_dict[i]['Kt'] = Kt
    print('task dict:', task_dict[1])

    with open('cooperation.json', 'r') as f_cooperation:
        cooperation_dict_1 = json.load(f_cooperation)
    cooperation_score_dict = {}
    for k in cooperation_dict_1.keys():
        cooperation_score_dict[int(k)] = cooperation_dict_1[k]
    print('cooperation dict:', len(cooperation_score_dict))

    print('=============== Synthetic data =====================')

    # T = 500
    # W = 1500
    # # #
    # basic_data = data_process()
    # # task_dict = basic_data.task_date_fun(T, min_skill=3, max_skill=3, budget_range=[5, 10])
    # worker_dict = basic_data.worker_date_fun(W)
    # cooperation_score_dict = basic_data.cooperation_date_fun()
    #
    # with open('task_noreal.json', 'r') as f_task:
    #     task_dict_1 = json.load(f_task)
    # task_dict = {}
    # for k in task_dict_1.keys():
    #     task_dict[int(k)] = task_dict_1[k]
    # print('task dict:', len(task_dict))

    # with open('worker_noreal.json', 'r') as f_worker:
    #     worker_dict_1 = json.load(f_worker)
    # worker_dict = {}
    # for k in worker_dict_1.keys():
    #     worker_dict[int(k)] = worker_dict_1[k]
    # print('worker dict:', len(worker_dict))
    #
    # with open('cooperation_noreal.json', 'r') as f_cooperation:
    #     cooperation_dict_1 = json.load(f_cooperation)
    # cooperation_score_dict = {}
    # for k in cooperation_dict_1.keys():
    #     cooperation_score_dict[int(k)] = cooperation_dict_1[k]
    # print('cooperation dict:', len(cooperation_score_dict))

    candidate = {}
    best_assign = {}
    R = 0.1
    v = 20
    alpha = 0.5
    beta = 0.05
    candidate_worker = []
    skill_group = {}

    max_cooperation = -99
    for i in cooperation_score_dict.keys():
        max_row_i = max(cooperation_score_dict[i])
        if max_row_i > max_cooperation:
            max_cooperation = max_row_i
    max_c = max_cooperation * 2 * len(task_dict[1]['Kt']) / (len(task_dict[1]['Kt']) - 1)
    print('max cooperation:', max_c)

    max_p = -99
    for i in task_dict.keys():
        for j in worker_dict.keys():
            min_p = task_dict[i]['budget'] - Euclidean_fun(task_dict[i]['Lt'], worker_dict[j]['Lw']) * v
            if min_p > max_p:
                max_p = min_p
    print('max price:', max_p)

    print('task dict:', task_dict[1])
    print('=========== basic random algorithm ==============')
    random_time = time.time()
    random_algorithm()
    print('random time:', time.time() - random_time)

    print('=========== benchmark Simulated Annealing ================')
    start_time = time.time()
    baseline_sa()
    print('sa time:', time.time() - start_time)

    print('=========== basic greedy algorithm ============')
    greedy_time = time.time()
    baseline_greedy()
    print('basic greedy:', time.time() - greedy_time)

    print('=========== Greedy algorithm =============')

    start = time.time()
    part_time_end = 0

    assigned_workers = []
    best_assign = {}
    total_satisfaction = 0
    for i in task_dict.keys():

        skill_group[i] = {}
        candidate_list = []
        for j in worker_dict.keys():
            if Euclidean_fun(worker_dict[j]['Lw'], task_dict[i]['Lt']) <= R and len(
                    list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                candidate_list.append(j)
                candidate_worker.append(j)

        candidate[i] = candidate_list

        if len(candidate_list) > 0:

            best_assign[i] = []

            d = [[] for i in range(len(task_dict[i]['Kt']))]
            for k in range(0, len(task_dict[i]['Kt'])):
                for j in candidate_list:
                    if worker_dict[j]['Kw'][0] == task_dict[i]['Kt'][k]:
                        d[k].append(j)
                skill_group[i][task_dict[i]['Kt'][k]] = d[k]

            worker_start = choice(candidate[i])
            worker_start_index = 0
            for r in skill_group[i].keys():
                if worker_start in skill_group[i][r]:
                    worker_start_index = r
                    break
            # print('start worker:', worker_start)

            best_s = 0
            best_list = [worker_start]
            best_satisfaction = 0
            for r in skill_group[i].keys():
                if r == worker_start_index:
                    continue
                else:
                    best_step = []
                    for s in skill_group[i][r]:
                        best_list_copy = copy.deepcopy(best_list)
                        best_list_copy.append(s)
                        price_s = price_score(i, v, best_list_copy)
                        if price_s < 0:
                            continue
                        else:
                            cooperation_s = cooperation_score(best_list_copy)

                        if alpha * (cooperation_s / max_c) + (1 - alpha) * (
                                price_s / max_p) > best_satisfaction:
                            best_satisfaction = alpha * (cooperation_s / max_c) + (1 - alpha) * (price_s / max_p)
                            best_step = best_list_copy

                    best_list = best_step
            # print(i, best_list, best_satisfaction)
            best_assign[i] = best_list

    flag = conflict_check(best_assign)
    print('conflict greedy:', flag)

    # 解决冲突的 worker group
    conflict_time = time.time()
    P = list(best_assign.keys())
    for s in P:
        for ss in P:
            if ss <= s:
                continue
            elif len(list(set(best_assign[s]).intersection(set(best_assign[ss])))) != 0:
                conflict_worker = list(set(best_assign[s]).intersection(set(best_assign[ss])))

                for c in conflict_worker:

                    increment_wi, increment_wj = increment(s, ss, c)

                    s_new = copy.deepcopy(best_assign[s])
                    ss_new = copy.deepcopy(best_assign[ss])

                    if increment_wi >= increment_wj:
                        ss_new.remove(c)
                    else:
                        s_new.remove(c)
                    best_assign[ss] = ss_new
                    best_assign[s] = s_new

    flag = conflict_check(best_assign)
    print('after conflict greedy:', flag)
    print('conflict time:', time.time() - conflict_time)

    # 统计剩下的 worker和 task
    candidate_worker = list(set(candidate_worker))
    candidate_task = list(best_assign.keys())
    for i in best_assign.keys():
        # print('best assign:', i, best_assign[i])
        for j in best_assign[i]:
            candidate_worker.remove(j)
        if len(best_assign[i]) == len(task_dict[i]['Kt']):
            candidate_task.remove(i)

    # 记录解决冲突以后的分配结果，作为博弈的初始结果

    conflit_time = time.time() - start
    best_assign_setp = copy.deepcopy(best_assign)
    candidate_worker_step = copy.deepcopy(candidate_worker)

    print('conflict result:', len(count_fun(best_assign)[0]), count_fun(best_assign)[1])

    # 采用 best pair策略再次分配剩下的 task和 worker
    reassign_time = time.time()
    pairs = []
    for t in candidate_task:
        skills = []
        for p in best_assign[t]:
            skills.append(worker_dict[p]['Kw'][0])
        skills = list(set(task_dict[t]['Kt']).difference(set(skills)))

        for k in skills:
            count = 0
            for s in skill_group[t][k]:
                if s in candidate_worker:
                    count += 1
                    pairs.append([t, s, k, add_increment(t, s)])

    # 对 pairs进行降序排序
    pairs.sort(key=lambda k: k[3], reverse=True)

    assigned_task = []
    assigned_worker = []
    for pa in pairs:
        if pa[0] in assigned_task or pa[1] in assigned_worker:
            continue
        if pa[0] in candidate_task and pa[1] in candidate_worker:
            without_w = best_assign[pa[0]]
            without_w.append(pa[1])
            best_assign[pa[0]] = without_w

            # 将分配的 worker删除
            candidate_worker.remove(pa[1])
            # 如果任务能被分配的 pair完成，将任务删除
            if check(pa[0], best_assign[pa[0]]) == True:
                candidate_task.remove(pa[0])

            assigned_task.append(pa[0])
            assigned_worker.append(pa[1])

    # 统计分配结果
    print(len(count_fun(best_assign)[0]), count_fun(best_assign)[1])
    print(conflict_check(best_assign))
    print('remaining workers:', len(candidate_worker))
    print('reassigment time:', time.time() - reassign_time)
    print('time:', time.time() - start)

    count_full = 0
    count_partial = 0
    for i in best_assign.keys():
        if len(best_assign[i]) != 0:
            count_partial += 1
        if len(best_assign[i]) == len(task_dict[i]['Kt']):
            count_full += 1
    print('full:', count_full, 'partial:', count_partial)

    print('============ Game algorithm =============')

    start = time.time()
    candidate_worker = copy.deepcopy(candidate_worker_step)
    best_assign = copy.deepcopy(best_assign_setp)
    print('initial:', len(count_fun(best_assign)[0]), count_fun(best_assign)[1])
    GT_algorithm(candidate_worker)
    print('game time:', conflit_time + time.time() - start)

    print('============ Game ITS algorithm =============')

    start = time.time()
    candidate_worker_1 = copy.deepcopy(candidate_worker_step)
    best_assign_1 = copy.deepcopy(best_assign_setp)
    print('initial:', len(count_fun(best_assign_setp)[0]), count_fun(best_assign_setp)[1])
    GT_ITS_algorithm(0.01, candidate_worker_1, best_assign_1)
    print('game time:', conflit_time + time.time() - start)
    print()

    start = time.time()
    candidate_worker_3 = copy.deepcopy(candidate_worker_step)
    best_assign_3 = copy.deepcopy(best_assign_setp)
    print('initial:', len(count_fun(best_assign_setp)[0]), count_fun(best_assign_setp)[1])
    GT_ITS_algorithm(0.03, candidate_worker_3, best_assign_3)
    print('game time:', conflit_time + time.time() - start)
    print()

    start = time.time()
    candidate_worker_5 = copy.deepcopy(candidate_worker_step)
    best_assign_5 = copy.deepcopy(best_assign_setp)
    print('initial:', len(count_fun(best_assign_setp)[0]), count_fun(best_assign_setp)[1])
    GT_ITS_algorithm(0.05, candidate_worker_5, best_assign_5)
    print('game time:', conflit_time + time.time() - start)
    print()

    start = time.time()
    candidate_worker_8 = copy.deepcopy(candidate_worker_step)
    best_assign_8 = copy.deepcopy(best_assign_setp)
    print('initial:', len(count_fun(best_assign_setp)[0]), count_fun(best_assign_setp)[1])
    GT_ITS_algorithm(0.08, candidate_worker_8, best_assign_8)
    print('game time:', conflit_time + time.time() - start)



