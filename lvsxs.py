import sys
import os
import traci
import traci.constants as tc
import pandas as pd
import numpy as np

#仿真模型
#sumoCmd = ["sumo", "-c", "C:/Users/10539/Documents/My_SUMO/4RL-test/simple6nodes.sumocfg"]
sumoCmd = ["sumo-gui", "-c", "D:/sumopy/4RL-test/simple6nodes.sumocfg", '--start']
##--------------------------初始化仿真 traci -----------------------------
traci.start(sumoCmd)
step = 0 #设置步长计数
a=0
b=[]
## ------------------------------  开始进行仿真  ------------------------------------------
#Run a simulation until all vehicles have arrived
while traci.simulation.getMinExpectedNumber() > 0:
   #print("step", step)
   step+=1
   traci.simulationStep()
  # print('step', step)



def queue_1(Junction):
    Junction = ['edge1', 'edge2', 'edeg3', 'edge4']
    queuecountEdge = 0

    for x in Junction:
        queuecountEdge = queuecountEdge + traci.edge.getLastStepHaltingNumber('x')
    return queuecountEdge

J6= ['eJ3J', 'eJ5J6', 'eDJ6J6', 'eDJ10J6']
a=queue_1(J6)
print(a)
traci.close()