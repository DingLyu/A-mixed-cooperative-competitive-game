import random
import numpy as np
import networkx as nx
from multiprocessing import Process
import matplotlib.pyplot as plt

rounds = 11000
def cooperation_level_within_an_isolated_group(b):
    N = 1000
    M = 4
    c = 1
    epochs = 100
    cooperation_level = np.zeros((epochs, rounds))
    for epoch in range(epochs):
        G = nx.barabasi_albert_graph(N, M)
        Neighbor = {}
        for edge in G.edges():
            if edge[0] not in Neighbor:
                Neighbor[edge[0]] = [edge[1]]
            else:
                Neighbor[edge[0]].append(edge[1])
            if edge[1] not in Neighbor:
                Neighbor[edge[1]] = [edge[0]]
            else:
                Neighbor[edge[1]].append(edge[0])
        actions = {time: {} for time in range(rounds)}
        rewards = {time: {} for time in range(rounds)}
        cooperation_level[epoch][0] += 0.5
        for n in G.nodes():
            rand_v = random.random()
            if rand_v < 0.5:
                actions[0][n] = 'C'
            else:
                actions[0][n] = 'D'
        for t in range(rounds - 1):
            for n in G.nodes():
                rewards[t][n] = 0
                for neighbor in Neighbor[n]:
                    if actions[t][n] == 'C' and actions[t][neighbor] == 'C':
                        rewards[t][n] += b - c
                    if actions[t][n] == 'C' and actions[t][neighbor] == 'D':
                        rewards[t][n] += -c
                    if actions[t][n] == 'D' and actions[t][neighbor] == 'C':
                        rewards[t][n] += b
            for n in G.nodes():
                neighbor = random.choice(Neighbor[n])
                if rewards[t][neighbor] > rewards[t][n]:
                    if random.random() < 1 / (1 + np.exp((rewards[t][n] - rewards[t][neighbor]) / 0.1)):
                        actions[t + 1][n] = actions[t][neighbor]
                    else:
                        actions[t + 1][n] = actions[t][n]
                else:
                    actions[t + 1][n] = actions[t][n]
                if actions[t + 1][n][0] == 'C':
                    cooperation_level[epoch][t + 1] += 1 / N
        np.save('res/b{}.npy'.format(b), cooperation_level)


if __name__ == '__main__':
    process_list = []
    b_list = [3, 4, 5, 6, 7, 8, 9, 10]
    for b in b_list:
        p = Process(target=cooperation_level_within_an_isolated_group, args=(b,))
        p.start()
        process_list.append(p)

