import json, math, random
import numpy as np
from itertools import combinations
from mip import *

def Euclidean_fun(A, B):
    return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))

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

with open('cooperation_group.json', 'r') as f_cooperation:
    cooperation_group_1 = json.load(f_cooperation)
cooperation_group = {}
for k in cooperation_group_1.keys():
    cooperation_group[int(k)] = cooperation_group_1[k]
print('whole cooperation arr:', len(cooperation_group.keys()))

with open('cooperation.json', 'r') as f_cooperation:
    cooperation_dict_1 = json.load(f_cooperation)
cooperation_score_dict = {}
for k in cooperation_dict_1.keys():
    cooperation_score_dict[int(k)] = cooperation_dict_1[k]
print('cooperation dict:', len(cooperation_score_dict))

count = 0
with open('task.json', 'r') as f_task:
    task_dict_1 = json.load(f_task)
task_dict = {}
for k in task_dict_1.keys():
    if count <= 19:
        task_dict[int(k)] = task_dict_1[k]
        count += 1
print('task dict:', len(task_dict), task_dict[1])

count = 0
with open('worker.json', 'r') as f_worker:
    worker_dict_1 = json.load(f_worker)
worker_dict = {}
for k in worker_dict_1.keys():
    if count <= 99:
        worker_dict[int(k)] = worker_dict_1[k]
        count += 1
print('worker dict:', len(worker_dict))

# data updates

# # budget updates
# for i in task_dict.keys():
#     budget = random.randint(20, 25)
#     task_dict[i]['budget'] = budget
# print('task dict:', task_dict[1])
    
# skill updates
for i in task_dict.keys():
    skill_quantity = random.randint(5, 5)
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

def dist(t, w):
    return Euclidean_fun(task_dict[t]['Lt'], worker_dict[w]['Lw'])


# dist_tw = np.zeros((len(task_dict.keys()), len(worker_dict.keys())))
# for t in range(len(task_dict.keys())):
#     for w in range(len(worker_dict.keys())):
#         dist_tw[t, w] = dist(t+1, w+1)
# print('dist:', dist_tw)


set_list = []
for i in range(1, len(task_dict[1]['Kt']) + 1):
    print(len(list(combinations(worker_dict.keys(), i))))
    for s in list(combinations(worker_dict.keys(), i)):
        set_list.append(s)
print(len(set_list))

score_arr = np.zeros((len(task_dict.keys()), len(set_list)))
for t in task_dict.keys():
    for s in range(len(set_list)):
        flag = True

        skill_set = []

        budget = 0
        for w in set_list[s]:

            # skill
            if set(worker_dict[w]['Kw']).intersection(set(task_dict[t]['Kt'])) == None:
                flag = False

            skill_set.append(worker_dict[w]['Kw'][0])

            # range
            if dist(t, w) > R:
                flag = False

            # budget
            budget += dist(t, w)

        if budget * v > task_dict[t]['budget']:
            flag = False

        if len(set(skill_set)) != len(set_list[s]):
            flag = False

        if flag == True:
            score_arr[t - 1, s] = satisfaction_score(cooperation_score(set_list[s]), price_score(t, v, set_list[s]))
        else:
            score_arr[t - 1, s] = 0

print('score array:', score_arr.shape, score_arr)

T = len(task_dict.keys())  # maximum number of bars
L = 250  # bar length
W = len(worker_dict.keys())  # number of requests
w = np.ones((W))  # size of each item
b = [1, 2, 2, 1]  # demand for each item

print('tasks:', T, 'workers:', W)

# creating the model
model = Model()

x = {(t, w): model.add_var(obj=0, var_type=INTEGER, name="x[%d,%d]" % (t, w))
     for t in range(T) for w in range(len(set_list))}

model.objective = maximize(xsum(score_arr[t, s] * x[t, s] for t in range(T) for s in range(len(set_list))))

# constraints
for t in range(T):
    model.add_constr(xsum(x[t, s] for s in range(len(set_list))) <= 1)

for s in range(len(set_list)):
    model.add_constr(xsum(x[t, s] for t in range(T)) <= 1)

# optimizing the model
model.optimize()

# printing the solution
print('')
print('obj:', model.objective_value)
print('Objective value: {model.objective_value:.3}'.format(**locals()))
print('Solution: ', end=" ")
for v in model.vars:
    if v.x > 1e-5:
        print('{v.name} = {v.x}'.format(**locals()))
        print('          ', end='')

