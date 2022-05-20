import random
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import math
BA = np.load('BA.npy')
ba = nx.DiGraph(BA)
theNET=ba
b=1.5
theta=1
alpha = 0.5
gamma = 0.7
#定义节点是动作
def action():
    if random.random()>0.5:
        return 1
    else:
        return 0

#初始化QTable和NStates
def init_QT_NS(theNET):
    QTable = {}
    NodeStates = {}
    for n in theNET.nodes():
        QTable[str(n)]=np.mat(np.zeros((2,2)))
        NodeStates[str(n)]=random_init_Q()
    return QTable,NodeStates
    print(QTable)

# #随机生成0 or 1，用于初始化节点状态
def random_init_Q():
    if random.random()>0.5:
        return 1
    else:
        return 0


# Boltzmann_action进行选择
def Boltzmann_action(lanmuta, state, qtable):
    pc=math.exp(qtable[state,0]/lanmuta)/(math.exp(qtable[state,0]/lanmuta)+math.exp(qtable[state,1]/lanmuta))
    if random.random() <= pc:  # greedy exploitation action
        action=0
    else:
        action=1
    return action


#获得节点i的动作对应的向量
def get_action(lanmuta, n, state, qtable):
    action= Boltzmann_action(lanmuta, state, qtable)
    if action== 0:  # 节点i的动作
        z = np.mat(([1], [0]))
    else:
        z = np.mat(([0], [1]))
    return z


#计算reward(i)
def get_reward(lanmuta,n,nbrs,Amat,NodeStates,QTable):
    zx = get_action(lanmuta, n, NodeStates[str(n)], QTable[str(n)])
    reward = 0
    for nbr, eattr in nbrs.items():  # 遍历每一个邻居
        zy = get_action(lanmuta, nbr,  NodeStates[str(nbr)], QTable[str(nbr)])
        reward = reward + zx.T * Amat * zy
    return reward[0]

#计算邻编号及数量
def get_node_n_nbrs(n):
    for m in theNET.nodes():
        n_nbs = []
        nbrs = theNET.neighbors(n)
        for j in nbrs:
            n_nbs.append(j)
        # 获取邻居个数
        n_nbnum = len(n_nbs)
    return n_nbnum




def main_proc(theNET,alpha,gamma):
    #Initialize 初始化参数
    (QTable, NodeStates) = init_QT_NS(theNET)
    a = np.array([list(NodeStates.values())]).sum()
    l = len(list(NodeStates.values()))
    oldc_ratio = [1 - (a / l)]
    Amat = np.mat(([1,0],[b,0]))
    # 初始状态的矩阵
    oldNodeStates_list=np.array([list(NodeStates.values())])
    d1 = 0

    # 执行2000次;
    for i in range(1000):
        t=i
        lanmuta=5*0.9999**t
        # one loop
        newNStates = copy.deepcopy(NodeStates)  # 新状态，临时变量
        newQtable = copy.deepcopy(QTable)
        rewardi = {}  # 奖励
        d = 0
        for n, nbrs in theNET.adj.items():  # 遍历每一个节点
            action = Boltzmann_action(lanmuta, newNStates[str(n)], newQtable[str(n)])  # Exploration or Exploitation
            NodeStates[str(n)] = action
            newstate = action
            #for nbr, eattr in nbrs.items():  # 遍历每一个邻居
            #    action = Boltzmann_action( newNStates[str(nbr)], newQtable[str(nbr)],lanmuta)  # Exploitation
                # NodeStates[str(nbr)] = action  # 修改邻居的状态
            rewardi[str(n)] = get_reward(lanmuta,n,nbrs,Amat,newNStates ,newQtable)/ get_node_n_nbrs(n)  # 计算reward(i)

            # 计算节点i的Q table 和 状态
            oldstate = NodeStates[str(n)]
            qtable = QTable[str(n)]
            maxcol = qtable.argmax(1)  # 每行最大值的列号（索引位置）
            newmaxvalIndex = maxcol.item(newstate)  # 指定行-新的状态对应的最大值的索引位置
            oldmaxvalIndex = maxcol.item(oldstate)
            newQMax = qtable[newstate, newmaxvalIndex]
            oldQMax = qtable[oldstate, oldmaxvalIndex]
            r = rewardi[str(n)].item(0) ** theta
            Qupdate = oldQMax * (1 - alpha) + alpha * (r + gamma * newQMax)  #
            # 更新
            QTable[str(n)][oldstate, oldmaxvalIndex] = Qupdate

            d=d+rewardi[str(n)].item(0)/100  #所有节点的奖励之和
        d1 = np.row_stack((d1,d))


        newQtable[str(n)] = QTable[str(n)] #遍历所有节点之后更新所有节点的Q-table
        a = np.array(list(NodeStates.values())).sum()
        l = len(list(NodeStates.values()))
        newc_ratio = [1 - (a / l)]  # 计算合作者比例
        oldc_ratio = np.append(oldc_ratio, newc_ratio)
        newNodeStates_list = np.array(list(NodeStates.values()))  # 新状态矩阵
        oldNodeStates_list = np.row_stack((oldNodeStates_list, newNodeStates_list))  # 每一次增加一个新状态矩阵


    return QTable,  oldNodeStates_list, oldc_ratio,d1





if __name__ == "__main__":
    (QTable,  oldNodeStates_list, oldc_ratio,d1 ) =main_proc(theNET,alpha,gamma)

print(QTable,  oldNodeStates_list, oldc_ratio,d1)

#name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#
oldNodeStates_list=np.column_stack((oldNodeStates_list, oldc_ratio))
test = pd.DataFrame(columns=None, data=oldNodeStates_list)
print(test)
test.to_csv('D:/111111111研三论文/更改epsilon之后的数据/Boltzman/Boltzman-0-4-BA-图1-1000次-100个节点.csv')

reward=pd.DataFrame(columns=None, data=d1)
print(reward)
reward.to_csv('D:/111111111研三论文/更改epsilon之后的数据/Boltzman/Boltzman-0-4-reward-BA-图1-1000次-100个节点.csv')

