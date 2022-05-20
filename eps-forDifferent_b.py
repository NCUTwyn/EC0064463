import random
import pandas as pd
import numpy as np
import networkx as nx
import copy
BA = np.load('BA.npy')
ba = nx.DiGraph(BA)
theNET=ba
theta=1
eps = 0.02
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
# epsilon greedy进行选择
def epsilon_greedy(eps, n, state, qtable):
    if random.random() > eps:  # greedy exploitation action
        return exploitation_action(n, state, qtable)
    else:  # exploration action
        return exploration_action()


# 探索新的
def exploration_action():
    return random_init_Q() #随机生成一个0 或 1， 即C或D


# 执行已有最佳
def exploitation_action(n, state, qtable):
    if qtable[state,0]==qtable[state,1]:
        action=random_init_Q()
    else:
        maxcol = qtable.argmax(0)#每行最大值的列号（索引位置）
        action = maxcol.item(state)
    return action



#获得节点i的动作对应的向量
def get_action(eps, n, state, qtable):
    action=epsilon_greedy(eps, n, state, qtable)
    if action== 0:  # 节点i的动作
        z = np.mat(([1], [0]))
    else:
        z = np.mat(([0], [1]))
    return z

#获得节点i的邻居的动作对应的向量
def n_nbr_action( n, state, qtable):
    action=exploitation_action(n, state, qtable)
    if action== 0:  # 节点i的动作
        z = np.mat(([1], [0]))
    else:
        z = np.mat(([0], [1]))
    return z


#计算reward(i)
def get_reward(n,nbrs,Amat,NodeStates,QTable):
    zx = get_action(eps, n, NodeStates[str(n)], QTable[str(n)])
    reward = 0
    for nbr, eattr in nbrs.items():  # 遍历每一个邻居
        zy = n_nbr_action(nbr, NodeStates[str(nbr)], QTable[str(nbr)])
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




def main_proc(theNET,eps,alpha,gamma):
    # Initialize 初始化参数
    (QTable, NodeStates) = init_QT_NS(theNET)
    a = np.array([list(NodeStates.values())]).sum()
    l = len(list(NodeStates.values()))
    oldc_ratio = [1 - (a / l)]
    for b in np.arange(1, 2.1, 0.1):
        Amat = np.mat(([1, 0], [b, 0]))
        print(b)

        # 初始状态的矩阵
        oldNodeStates_list = np.array([list(NodeStates.values())])
        # 执行2000次;
        for i in range(100):
            # one loop
            newNStates = copy.deepcopy(NodeStates)  # 新状态，临时变量
            newQtable = copy.deepcopy(QTable)
            rewardi = {}  # 奖励
            for n, nbrs in theNET.adj.items():  # 遍历每一个节点
                action = epsilon_greedy(eps, n, newNStates[str(n)], newQtable[str(n)])  # Exploration or Exploitation
                NodeStates[str(n)] = action
                newstate = action
                #for nbr, eattr in nbrs.items():  # 遍历每一个邻居
                #    action = exploitation_action(nbr, newNStates[str(nbr)], newQtable[str(nbr)])  # Exploitation
                    # NodeStates[str(nbr)] = action  # 修改邻居的状态
                rewardi[str(n)] = get_reward(n,nbrs,Amat,newNStates,newQtable) / get_node_n_nbrs(n)  # 计算reward(i)

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

            newQtable[str(n)] = QTable[str(n)]  # 遍历所有节点之后更新所有节点的Q-table
            a = np.array(list(NodeStates.values())).sum()  #遍历一次所有节点之后节点的状态
            l = len(list(NodeStates.values()))             #节点个数
        newc_ratio = [1 - (a / l)]   # 对应一个b:计算循环20次之后最终第20次稳定的合作者比例
        oldc_ratio = np.append(oldc_ratio, newc_ratio) #对于所有的b:对应的每一个最终的合作者比例
        print(oldc_ratio)
    return oldc_ratio





if __name__ == "__main__":
    ( oldc_ratio ) =main_proc(theNET,eps,alpha,gamma)

print(oldc_ratio)

#name = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11','12']
c= pd.DataFrame(columns=None, data=oldc_ratio)
print(c)
c.to_csv('D:/111111111研三论文/更改之后的数据/eps/eps-图2-BA-b-epsilon2.csv')


