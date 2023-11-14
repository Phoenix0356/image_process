import numpy as npy
import matplotlib.pyplot as pyt
from scipy.special import expit as exp


hidden_node_num = []
hidden_node_efficiency = []
hidden_node_num.append(1)
hidden_node_num.append(2)
hidden_node_num.append(3)
hidden_node_num.append(4)
hidden_node_num.append(5)
hidden_node_num.append(6)
hidden_node_num.append(7)
hidden_node_num.append(8)
hidden_node_num.append(9)
hidden_node_num.append(10)
hidden_node_efficiency.append(88.79)
hidden_node_efficiency.append(93.84)
hidden_node_efficiency.append(92.59)
hidden_node_efficiency.append(94.26)
hidden_node_efficiency.append(94.14)
hidden_node_efficiency.append(94.68)
hidden_node_efficiency.append(94.63)
hidden_node_efficiency.append(94.54)
hidden_node_efficiency.append(93.22)
hidden_node_efficiency.append(94.60)
pyt.plot(hidden_node_num, hidden_node_efficiency)
pyt.show()
