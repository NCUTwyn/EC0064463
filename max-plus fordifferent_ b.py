import random
from pandas import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

BA = np.load('BA.npy')
ba = nx.DiGraph(BA)
theNET=ba
#b=1.5
eps = 0.4
theta=1
alpha = 0.5
gamma = 0.7
c=(0,1)


#初始化QTable和NStates和UTable
def init_QT_NS(theNET):
    QTable = {}
    NodeStates = {}
    UTable = {}
    newUTable = {}
    for n in theNET.nodes():
        QTable[str(n)]=np.mat(np.zeros((2,2)))
        NodeStates[str(n)]=random_init_Q()
        n_nbnum = get_node_n_nbrs(n)[0]
        a=get_node_n_nbrs(n)[1]
        b=[0,1]
        UTable[str(n)] = pd.DataFrame(np.mat(np.zeros((2, n_nbnum))),index=list(b),columns=list(a))
        newUTable[str(n)]=pd.DataFrame(np.mat(np.zeros((2, n_nbnum))),index=list(b),columns=list(a))

    return QTable,NodeStates,UTable,newUTable

#计算邻编号及数量
def get_node_n_nbrs(n):
    for m in theNET.nodes():
        n_nbs = []
        nbrs = theNET.neighbors(n)
        for j in nbrs:
            n_nbs.append(j)
        # 获取邻居个数
        n_nbnum = len(n_nbs)
    return n_nbnum,n_nbs

# #随机生成0 or 1，用于初始化节点状态
def random_init_Q():
    if random.random()>0.5:
        return 1
    else:
        return 0


# epsilon greedy进行选择
def epsilon_greedy(eps, n,A):
    if random.random() > eps:  # greedy exploitation action
        return A[n]
    else:  # exploration action
        return exploration_action()
# 随机选择
def exploration_action():
    return random_init_Q() #随机生成一个0 或 1， 即C或D


# 执行已有最佳
def exploitation_action(theNET, NodeStates, QTable,UTable):
    newUTable = get_newUTable(theNET)
    A=[] #所有节点的动作
    for i in theNET.nodes():
        nbrs = theNET.neighbors(i) #节点i的所有邻居
        for j in nbrs:
            #cij = 0
            for aj in c: #对于邻居j 的两个动作0，1
                ui = 0
                #nbrsi = UTable[str(i)].columns.values.tolist()  错误原因：应该是theNET.neighbors(i) 应该是i的邻居，而不是j
                for k in nbrs:
                    ai=get_QMax(i, NodeStates[str(i)],QTable[str(i)])[1] # 表示uki(ai) 里面的ai
                    ui = ui + UTable[str(i)].loc[ai,k]    # 求sum-uki(ai)
                cij = (newUTable[str(j)].loc[0, i]+newUTable[str(j)].loc[1, i]) / 2  #(  UTable[str(j)].loc[aj, i]) / 2 #计算指定的j,i, aj=0/1 ,求平均值，写函数，
                UTable[str(j)].loc[aj, i] = get_QMax(i, NodeStates[str(i)], QTable[str(i)])[0] +QTable[str(j)][NodeStates[str(j)], aj] + ui - UTable[str(i)].loc[ai, j]-cij  #UTable[str(j)].loc[aj, i]表示第j个UTtable  的动作为aj，邻居i ,时的值

        gi = g_ai(i, UTable)  #
        if gi[0,0]==gi[0,1]:
           action = random_init_Q()
        else:
           action = gi.argmax(0)[0]  #action =argmax uji(ai)
        A.append(action)
    return A

def get_newUTable(theNET):
    (QTable, NodeStates, UTable, newUTable) = init_QT_NS(theNET)
    for i in theNET.nodes():
        nbrs = theNET.neighbors(i) #节点i的所有邻居
        for j in nbrs:

            for aj in c: #对于邻居j 的两个动作0，1
                ui = 0
                #nbrsi = UTable[str(i)].columns.values.tolist()  错误原因：应该是theNET.neighbors(i) 应该是i的邻居，而不是j
                for k in nbrs:
                    ai=get_QMax(i, NodeStates[str(i)],QTable[str(i)])[1] # 表示uki(ai) 里面的ai
                    ui = ui + UTable[str(i)].loc[ai,k]    # 求sum-uki(ai)，
                newUTable[str(j)].loc[aj, i] = get_QMax(i, NodeStates[str(i)], QTable[str(i)])[0] +QTable[str(j)][NodeStates[str(j)], aj] + ui - UTable[str(i)].loc[ai, j]  #UTable[str(j)].loc[aj, i]表示第j个UTtable  的动作为aj，邻居i ,时的

    return newUTable

#定义Q(si,ai)
def get_QMax(n,state,qtable,):
    maxcol = qtable.argmax(1) # 每行最大值的列号（索引位置）
    maxvalueIndex = maxcol.item(state) # 指定行-新的状态对应的最大值的索引位置（指定状态对应动作ai）
    QMax = qtable[state, maxvalueIndex]   #指定状态对应动作对应的最大q值
    return QMax,maxvalueIndex


#定义g(ai)
def g_ai(n,UTable):
    g=np.zeros((1,2))
    i = n
    nbrs = theNET.neighbors(i)
    for ai in c:
        for k in nbrs:
            g[0,ai]=g[0,ai]+UTable[str(i)].loc[ai,k]

    return g


#获得节点的动作对应的向量
def get_action(n,A):
    action=A[n]  # 节点n的动作
    if action== 0:
        z = np.mat(([1], [0]))
    else:
        z = np.mat(([0], [1]))
    return z

#计算reward(i)
def get_reward(n,nbrs,Amat,A):
    zx = get_action(n,A)
    reward = 0
    for nbr, eattr in nbrs.items():  # 遍历每一个邻居
        zy = get_action(nbr,A)
        reward = reward + zx.T * Amat * zy
    return reward[0]



def main_proc(theNET,eps,alpha,gamma):
    #Initialize 初始化参数
    # Initialize 初始化参数
    (QTable, NodeStates,UTable,newUTable) = init_QT_NS(theNET)
    a = np.array([list(NodeStates.values())]).sum()
    l = len(list(NodeStates.values()))
    oldc_ratio = [1 - (a / l)]
    for b in np.arange(1, 2.1, 0.1):
        Amat = np.mat(([1, 0], [b, 0]))
        print(b)

        # 初始状态的矩阵
        oldNodeStates_list = np.array([list(NodeStates.values())])
        # 执行2000次;
        for i in range(10):
            # one loop
            newNStates = copy.deepcopy(NodeStates)  # 新状态，临时变量
            newQtable = copy.deepcopy(QTable)
            # newUTable= copy.deepcopy(UTable)
            rewardi = {}  # 奖励
            A = exploitation_action(theNET, newNStates, newQtable, UTable)  # 计算所有节点的最佳action,uij,并更新uij
            for n, nbrs in theNET.adj.items():  # 遍历每一个节点
                action = epsilon_greedy(eps, n, A)  # Exploration or Exploitation
                NodeStates[str(n)] = action
                newstate = action
                # for nbr, eattr in nbrs.items():  # 遍历每一个邻居
                # action = epsilon_greedy(eps, nbr,A)  # Exploitation
                # NodeStates[str(nbr)] = action  # 修改邻居的状态
                rewardi[str(n)] = get_reward(n, nbrs, Amat, A) / get_node_n_nbrs(n)[0]  # 计算reward(i)

                # 计算节点n的Q table 和 状态
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

            newQtable[str(n)] = QTable[str(n)]  # 遍历所有节点之后更新所有节点的Q-table
            a = np.array(list(NodeStates.values())).sum()  # 遍历一次所有节点之后节点的状态
            l = len(list(NodeStates.values()))  # 节点个数
        newc_ratio = [1 - (a / l)]  # 对应一个b:计算循环1000次之后最终第1000次稳定的合作者比例
        oldc_ratio = np.append(oldc_ratio, newc_ratio)  # 每次增加一个b:对应的每一个最终的合作者比例 ,循环完之后得到的矩阵[11*1]
        print(oldc_ratio)  # ,对于一个theta:输出所有的b对应的合作者比例
    return oldc_ratio






if __name__ == "__main__":
    (oldc_ratio) = main_proc(theNET, eps, alpha, gamma)

print(oldc_ratio)

# name = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11','12']
c = pd.DataFrame(columns=None, data=oldc_ratio)
print(c)
c.to_csv('D:/111111111研三论文/更改epsilon之后的数据/Maxplus/Maxplus-0-4-图2-BA-b-200次.csv')