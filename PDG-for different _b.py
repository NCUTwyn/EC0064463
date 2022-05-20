import random
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import math
SW = np.load('SW.npy')
sw= nx.DiGraph(SW)
theNET=sw
alpha =0
k=0.1
#定义节点是动作
#def action():
#    if False:
#        return 1
#    else:
#        return 0

#初始NStates
def init_QT_NS(theNET):
    NodeStates = {}
    for n in theNET.nodes():
        NodeStates[str(n)]=random_init_Q()
    return NodeStates


# #随机生成0 or 1，用于初始化节点状态
def random_init_Q():
    if random.random()>0.5:
        return 1
    else:
        return 0

#获得节点的动作对应的向量
def get_action(NodeStates,n):
    if NodeStates[str(n)]== 0:  # 节点i的动作
        z = np.mat(([1], [0]))
    else:
        z = np.mat(([0], [1]))
    return z

#计算reward(i)
def get_reward(n,Amat,NodeStates,nbrs):
    zx = get_action(NodeStates,n)
    reward = 0
    for nbr, eattr in nbrs.items():  # 遍历每一个邻居
        zy = get_action(NodeStates,nbr)
        reward = reward + zx.T * Amat * zy
    return reward[0]


#计算邻居数量
def get_node_n_nbrs(n):
    for m in theNET.nodes():
        n_nbs = []
        nbrs = theNET.neighbors(n)
        for j in nbrs:
            n_nbs.append(j)
        # 获取邻居个数
        n_nbnum = len(n_nbs)
    return n_nbnum

#随机选择的邻居编号
def get_id_nbr(n):
    for m in theNET.nodes():
        n_nbs = []
        nbrs = theNET.neighbors(n)
        for j in nbrs:
            n_nbs.append(j)
        id_nbr = random.choice(n_nbs)
    return id_nbr

def get_theta(n,theNET):
    a=nx.betweenness_centrality(theNET)
    b=sum(a.values())
    theta=(a[n]/b)**alpha
    return theta


#计算策略改变概率
def get_q(n,Amat,NodeStates,nbrs):
    theta=get_theta(n,theNET)
    id_nbr=get_id_nbr(n)
    get_reward(n,Amat,NodeStates,nbrs)
    q=theta*1/(1+math.exp(get_reward(n,Amat,NodeStates,nbrs)-get_reward(id_nbr,Amat,NodeStates,nbrs,))/k)
    return q



# epsilon greedy进行选择动作
def epsilon_greedy( n, NodeStates,q,id_nbr):
    if random.random()<= q:  # greedy exploitation action
         action= NodeStates[str(id_nbr)]
    else:  # exploration action
         action = NodeStates[str(n)]
    return action




def main_proc():
    #Initialize 初始化参
    NodeStates = init_QT_NS(theNET)
    a = np.array([list(NodeStates.values())]).sum()
    l = len(list(NodeStates.values()))
    oldc_ratio = [1 - (a / l)]

    for b in np.arange(1, 2.1, 0.1):
        Amat = np.mat(([1, 0], [b, 0]))
        print(b)
        # 初始状态的矩阵
        oldNodeStates_list = np.array([list(NodeStates.values())])
        # 执行2000次;
        for i in range(500):
            # one loop
            newNStates = copy.deepcopy(NodeStates)  # 新状态，临时变量
            for n, nbrs in theNET.adj.items():  # 遍历每一个节点
                id_nbr = get_id_nbr(n)
                q = get_q(n, Amat, newNStates, nbrs)
                action = epsilon_greedy(n, newNStates, q, id_nbr)  # Exploration or Exploitation
                NodeStates[str(n)] = action

            # 计算节点i的状态
            a = np.array(list(NodeStates.values())).sum()  # 遍历一次所有节点之后节点的状态
            l = len(list(NodeStates.values()))  # 节点个数
        newc_ratio = [1 - (a / l)]  # 对应一个b:计算循环20次之后最终第20次稳定的合作者比例
        oldc_ratio = np.append(oldc_ratio, newc_ratio)  # 对于所有的b:对应的每一个最终的合作者比例
        print(oldc_ratio)
    return oldc_ratio



if __name__ == "__main__":
    (oldc_ratio) = main_proc()
print (oldc_ratio)

#name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#
d= pd.DataFrame(columns=None, data=oldc_ratio)
print(d)
d.to_csv('D:/111111111研三论文/上个论文实验数据100个节点数据/图2-SW-b-100次.csv')