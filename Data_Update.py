import json
import random

# import data
with open('parameter.json', 'r') as f_parameter:
    parameter_dict = json.load(f_parameter)
v = 10
max_Pi = parameter_dict['max_Pi']
max_Ci = 0.47

with open('task_Nov269.json', 'r') as f_task:
    task_dict_1 = json.load(f_task)
task_dict = {}
for k in task_dict_1.keys():
    task_dict[int(k)] = task_dict_1[k]
print('task dict:', len(task_dict), task_dict)

# # task budget
# max_budget = 50
# min_budget = 40
# for i in task_dict.keys():
#     task_dict.get(i)["Cbt"] = random.randint(min_budget, max_budget)
# print('New task dict:', task_dict)
# with open('task_Nov269.json', 'w', encoding='utf-8') as fp_task:
#     json.dump(task_dict, fp_task)

# task skill
Kt = []
T = len(task_dict.keys())
max_task = 6
min_task = 5
while T != len(Kt):
    x = random.randint(min_task, max_task)
    Kt.append(x)
for i in task_dict.keys():
    task_dict.get(i)["Kt"][0] = Kt[i - 1]
    task_skill = []
    while len(task_skill) != Kt[i - 1]:
        x = random.randint(1, 10)
        if x in task_skill:
            continue
        else:
            task_skill.append(x)
            task_skill.sort()
    task_dict.get(i)["Kt"][1] = task_skill
print('New task dict:', task_dict)
with open('task_Nov269.json', 'w', encoding='utf-8') as fp_task:
    json.dump(task_dict, fp_task)

# # worker skill
# with open('worker.json', 'r') as f_worker:
#     worker_dict_1 = json.load(f_worker)
# worker_dict = {}
# for k in worker_dict_1.keys():
#     worker_dict[int(k)] = worker_dict_1[k]
# print('worker dict:', len(worker_dict), worker_dict)
# W = len(worker_dict.keys())
# K = []
# max_worker = 2
# while W != len(K):
#     x = random.randint(1, max_worker)
#     K.append(x)
# for i in worker_dict.keys():
#     worker_dict.get(i)["Kw"][0] = K[i - 1]
#     worker_skill = []
#     while len(worker_skill) != K[i - 1]:
#         x = random.randint(1, 10)
#         if x in worker_skill:
#             continue
#         else:
#             worker_skill.append(x)
#             worker_skill.sort()
#     worker_dict.get(i)["Kw"][1] = worker_skill
# print('New worker dict:', worker_dict)
# with open('worker.json', 'w', encoding='utf-8') as fp_worker:
#     json.dump(worker_dict, fp_worker)


