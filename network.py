import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
rg = nx.random_graphs.random_regular_graph(3,100)  #生成包含20个节点、每个节点有3个邻居的规则图RG
#pos = nx.spectral_layout(rg)          #定义一个布局，此处采用了spectral布局方式，后变还会介绍其它布局方式，注意图形上的区别
nx.draw(rg,with_labels=False,node_size = 30)  #绘制规则图的图形，with_labels决定节点是非带标签（编号），node_size是节点的直径
plt.show()  #显示图形

#ER网络
er = nx.random_graphs.erdos_renyi_graph(100,0.3)  #生成包含20个节点、以概率0.2连接的随机图
#pos = nx.shell_layout(er)          #定义一个布局，此处采用了shell布局方式
nx.draw(er,with_labels=False,node_size = 30)
plt.show()
#WS网络
ws = nx.random_graphs.watts_strogatz_graph(100,3,0.3)  #生成包含20个节点、每个节点3个近邻、随机化重连概率为0.3的小世界网络
#pos = nx.circular_layout(ws)          #定义一个布局，此处采用了circular布局方式
nx.draw(ws,with_labels=False,node_size = 30)  #绘制图形
plt.show()
#BA网络
ba= nx.random_graphs.barabasi_albert_graph(100,3)  #生成n=20、m=3的BA无标度网络
#pos = nx.spring_layout(ba)          #定义一个布局，此处采用了spring布局方式
nx.draw(ba,with_labels=False,node_size = 30)  #绘制图形
plt.show()


RG= nx.to_numpy_array(rg)
ER= nx.to_numpy_array(er)
WS= nx.to_numpy_array(ws)
BA= nx.to_numpy_array(ba)

np.save('D:/111111111研三论文/最新代码/RG',RG)
np.save('D:/111111111研三论文/最新代码/ER',ER)
np.save('D:/111111111研三论文/最新代码/WS',WS)
np.save('D:/111111111研三论文/最新代码/BA',BA)







