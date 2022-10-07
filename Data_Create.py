import random
import json
from random import choice

class data_process(object):

    def task_date_fun(self, T, min_task, max_task, budget_range):
        ## task的技能要求数随机生成
        K = []
        while T != len(K):
            x = random.randint(min_task, max_task) ## task所要求的技能数
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
            task_dict[i + 1] = {"Lt": LL, "Kt": Lt, "Cbt": Cb}
            Lt = []
        return task_dict

    def worker_date_fun(self, W, min_worker, max_worker):
        # worker随机生成技能数总和[1，5]，最多有5个技能，这[1，5]之间的技能范围在[1，10]
        K = []
        while W != len(K):
            x = random.randint(min_worker, max_worker)
            K.append(x)
        # print(K, len(K))
        # 每个worker的技能
        Lw = []
        worker_dict = {}
        for i in range(W):
            ## 位置
            l1 = random.uniform(0, 1)
            l2 = random.uniform(0, 1)
            LL = [l1, l2]
            ## 成本
            Cb = random.randint(5, 10)
            ## 技能
            while len(Lw) != K[i]:
                x = random.randint(1, 10)
                if x in Lw:
                    continue
                else:
                    Lw.append(x)
                    Lw.sort()
            worker_dict[i + 1] = {"Lw": LL, "Kw": Lw, "Cbw": Cb}
            Lw = []
        return worker_dict

    def real_task_fun(self):
        with open('Real-data//task_800.txt', 'r', encoding='utf-8') as f1:
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
                skill_quantity = random.randint(3, 3)
                Kt = []
                # T = len(task_dict.keys())
                while len(Kt) != skill_quantity:
                    x = random.randint(1, 10)
                    if x in Kt:
                        continue
                    else:
                        Kt.append(x)
                        Kt.sort()
                task_dict[k]['Kt'] = Kt
                task_dict[k]['budget'] = random.randint(5, 10)
        print('task dict:', len(task_dict), task_dict[1])
        return task_dict

    def real_worker_fun(self):
        with open('Real-data//worker_2400.txt', 'r', encoding='utf-8') as f1:
            worker_dict = {}
            counttask = 0
            for line in f1.readlines():
                counttask += 1  # 重新编码POI的序号
                line = line.strip('\n')  # 去掉换行符\n
                b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
                b = list(map(float, b))
                k = counttask
                worker_dict[k] = {}
                worker_dict[k]['Lw'] = b
                skill_quantity = 1
                Kt = []
                # T = len(task_dict.keys())
                while len(Kt) != skill_quantity:
                    x = random.randint(1, 10)
                    if x in Kt:
                        continue
                    else:
                        Kt.append(x)
                        Kt.sort()
                worker_dict[k]['Kw'] = Kt
        print('worker_dict:', len(worker_dict), worker_dict[1])
        return worker_dict

    def social_date_fun(self, worker_dict, history_task_range, task_history_number_range):
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
        return served_history_dict

T = 500   # task number is 50
W = 1000   # worker number is 100
K = [1, 10]  # total skill range
R = 0.3  # worker range
# v = random.randint(20, 25)  # 单位成本
history_task_range = random.randint(100, 120)
task_history_number_range = [10, 20]
budget_range = [10, 20]
skill_range = [1, 10]
min_task = 3
max_task = 4
min_worker = 1
max_worker = 1

basic_data = data_process()

def real_cooperation_fun():
    with open('Real-data//cooperation_group.txt', 'r', encoding='utf-8') as f1:
        cooperation_dict = {}
        for line in f1.readlines():
            line = line.strip('\n')  # 去掉换行符\n
            b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
            b = list(map(int, b))
            b[0] = int(b[0])
            # print(b)
            if b[0] not in cooperation_dict.keys():
                cooperation_dict[b[0]] = []
                cooperation_dict[b[0]].append(b[1])
            else:
                cooperation_dict[b[0]].append(b[1])
    with open('cooperation_group.json', 'w', encoding='utf-8') as fp_cooperation:
        json.dump(cooperation_dict, fp_cooperation)
    return cooperation_dict

# cooperation_dict = real_cooperation_fun()


def real_worker_2700_fun(cooperation_dict):
    with open('Real-data//worker_2700.txt', 'r', encoding='utf-8') as f1:
        new_cooperation_dict= {}
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
            # T = len(task_dict.keys())
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
    print('new cooperation dict:', new_cooperation_dict)
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
    print(len(cooperation_score_dict[1]), cooperation_score_dict[1])
    return worker_dict, cooperation_score_dict

# real_worker_2700_fun(cooperation_dict)


def real_worker_fun():

    # worker 总体 skill
    with open('Real-data//topic_id.txt', 'r', encoding='utf-8') as f1:
        topic_id = []
        for line in f1.readlines():
            line = line.strip('\n')  # 去掉换行符\n
            b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
            b = list(map(int, b))
            topic_id.append(b)
        final_topic_id = list(set(topic_id[0]).intersection(set(topic_id[1])))
        final_topic_id.sort()
        print(len(final_topic_id), final_topic_id)

    with open('Real-data//worker-topic.txt', 'r', encoding='utf-8') as f1:
        worker_skill_dict = {}
        for line in f1.readlines():
            line = line.strip('\n')  # 去掉换行符\n
            b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
            b = list(map(int, b))
            b[0] = int(b[0])
            # print(b)
            if b[0] not in worker_skill_dict.keys() and b[1] in final_topic_id:
                worker_skill_dict[b[0]] = []
                worker_skill_dict[b[0]].append(b[1])
            elif b[0] in worker_skill_dict.keys() and b[1] in final_topic_id:
                worker_skill_dict[b[0]].append(b[1])


    # worker location
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
            # k = counttask

            if b[0] in worker_skill_dict.keys():
                worker_dict[b[0]] = {}
                worker_dict[b[0]]['Lw'] = [b[1], b[2]]
                worker_dict[b[0]]['Kw'] = choice(topic_id[1])
    print('worker_dict:', len(worker_dict), worker_dict)
    print(len(worker_skill_dict.keys()))
    print(len(set(worker_dict.keys()).intersection(set(worker_skill_dict.keys()))))

    return worker_dict

def real_task_fun():
    with open('Real-data//topic_id.txt', 'r', encoding='utf-8') as f1:
        topic_id = []
        for line in f1.readlines():
            line = line.strip('\n')  # 去掉换行符\n
            b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
            b = list(map(int, b))
            topic_id.append(b)
        final_topic_id = list(set(topic_id[0]).intersection(set(topic_id[1])))
        final_topic_id.sort()
        print(len(final_topic_id), final_topic_id)

    with open('Real-data//task-topic.txt', 'r', encoding='utf-8') as f1:
        task_skill_dict = {}
        for line in f1.readlines():
            line = line.strip('\n')  # 去掉换行符\n
            b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
            b = list(map(int, b))
            b[0] = int(b[0])
            # print(b)
            if b[0] not in task_skill_dict.keys():
                task_skill_dict[b[0]] = []
                task_skill_dict[b[0]].append(b[1])
            elif b[0] in task_skill_dict.keys():
                task_skill_dict[b[0]].append(b[1])

    with open('Real-data//task_1000.txt', 'r', encoding='utf-8') as f1:
        new_cooperation_dict = {}
        task_dict = {}
        counttask = 0
        for line in f1.readlines():
            counttask += 1  # 重新编码POI的序号
            line = line.strip('\n')  # 去掉换行符\n
            b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
            b = list(map(float, b))
            b[0] = int(b[0])
            # k = counttask

            if b[0] in task_skill_dict.keys():
                task_dict[b[0]] = {}
                task_dict[b[0]]['Lt'] = [b[1], b[2]]
                task_dict[b[0]]['Kt'] = task_skill_dict[b[0]]
    print('task dict:', len(task_dict.keys()), task_dict)
    skills = []
    count = 0
    for i in task_dict.keys():

        if len(task_dict[i]['Kt']) > 2:
            print(i, task_dict[i]['Kt'], len(task_dict[i]['Kt']))
            count += 1
        for j in task_dict[i]['Kt']:
            skills.append(j)
    print(len(set(skills)))
    print('count:', count)
    return task_dict

worker_dict = real_worker_fun()
task_dict = real_task_fun()





