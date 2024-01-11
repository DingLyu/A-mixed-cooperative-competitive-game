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
    epochs = 1000
    cooperation_level = np.zeros((epochs, rounds))
    for epoch in range(epochs):
        if epoch // 100 == 0:
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
                    if random.random() < (rewards[t][neighbor] - rewards[t][n]) / ((b+c) * max(G.degree(n), G.degree(neighbor))):
                        actions[t + 1][n] = actions[t][neighbor]
                    else:
                        actions[t + 1][n] = actions[t][n]
                else:
                    actions[t + 1][n] = actions[t][n]
                if actions[t + 1][n][0] == 'C':
                    cooperation_level[epoch][t + 1] += 1 / N
        np.save('res/robustness/b{}.npy'.format(b), cooperation_level)


def cooperation_level_within_g1(b, r, p):
    N = 1000
    M = 4
    E = int(p * N * N)

    c = 1
    epochs = 1000
    cooperation_level = np.zeros((epochs, rounds))
    for epoch in range(epochs):
        if epoch // 100 == 0:
            g1 = nx.barabasi_albert_graph(N, M)
            g2 = nx.barabasi_albert_graph(N, M)
            G1 = nx.Graph()
            for edge in g1.edges():
                G1.add_edge('r' + str(edge[0]), 'r' + str(edge[1]))
            G2 = nx.Graph()
            for edge in g2.edges():
                G2.add_edge('b' + str(edge[0]), 'b' + str(edge[1]))
            between_network = {}
            between_network_reverse = {}
            num_between_network = 0
            while num_between_network < E:
                edge = (random.choice(list(G1.nodes())), random.choice(list(G2.nodes())))
                if edge[0] in between_network:
                    if edge[1] not in between_network[edge[0]]:
                        between_network[edge[0]].append(edge[1])
                        num_between_network += 1
                else:
                    between_network[edge[0]] = [edge[1]]

                if edge[1] in between_network_reverse:
                    if edge[0] not in between_network_reverse[edge[1]]:
                        between_network_reverse[edge[1]].append(edge[0])
                else:
                    between_network_reverse[edge[1]] = [edge[0]]
        actions = {time: {} for time in range(rounds)}
        rewards = {time: {} for time in range(rounds)}
        cooperation_level[epoch][0] += 0.5
        for n1 in G1.nodes():
            if random.random() < 0.5:
                actions[0][n1] = 'C'
            else:
                actions[0][n1] = 'D'
        for n2 in G2.nodes():
            if random.random() < 0.5:
                actions[0][n2] = 'C'
            else:
                actions[0][n2] = 'D'
        for t in range(rounds-1):
            for n1 in G1.nodes():
                rewards[t][n1] = 0
                for n1_neighbor in nx.neighbors(G1, n1):
                    if actions[t][n1] == 'C':
                        if actions[t][n1_neighbor] == 'C':
                            rewards[t][n1] += b - c
                        if actions[t][n1_neighbor] == 'D':
                            rewards[t][n1] += - c
                    if actions[t][n1][0] == 'D':
                        if actions[t][n1_neighbor] == 'C':
                            rewards[t][n1] += b
            for n2 in G2.nodes():
                rewards[t][n2] = 0
                for n2_neighbor in nx.neighbors(G2, n2):
                    if actions[t][n2] == 'C':
                        if actions[t][n2_neighbor] == 'C':
                            rewards[t][n2] += b - c
                        if actions[t][n2_neighbor] == 'D':
                            rewards[t][n2] += - c
                    if actions[t][n2] == 'D':
                        if actions[t][n2_neighbor] == 'C':
                            rewards[t][n2] += b
            for n1 in between_network:
                for n2 in between_network[n1]:
                    if actions[t][n1] == 'C':
                        if actions[t][n2] == 'C':
                            rewards[t][n1] += r / 2 - c
                            rewards[t][n2] += r / 2 - c
                        if actions[t][n2] == 'D':
                            rewards[t][n1] += r - c
                    if actions[t][n1] == 'D':
                        if actions[t][n2] == 'C':
                            rewards[t][n2] += r - c
            actions[t + 1] = {}
            for n1 in G1.nodes():
                n1_neighbor = random.choice(list(nx.neighbors(G1, n1)))
                if n1 in between_network:
                    k1 = G1.degree(n1) + len(between_network[n1])
                else:
                    k1 = G1.degree(n1)
                if n1_neighbor in between_network:
                    k2 = G1.degree(n1_neighbor) + len(between_network[n1_neighbor])
                else:
                    k2 = G1.degree(n1_neighbor)
                if rewards[t][n1_neighbor] > rewards[t][n1]:
                    if random.random() < (rewards[t][n1_neighbor] - rewards[t][n1]) / ((b + c) * max(k1, k2)):
                        actions[t + 1][n1] = actions[t][n1_neighbor]
                    else:
                        actions[t + 1][n1] = actions[t][n1]
                else:
                    actions[t + 1][n1] = actions[t][n1]
                if actions[t + 1][n1] == 'C':
                    cooperation_level[epoch][t + 1] += 1 / N
            for n2 in G2.nodes():
                n2_neighbor = random.choice(list(nx.neighbors(G2, n2)))
                if n2 in between_network_reverse:
                    k1 = G2.degree(n2) + len(between_network_reverse[n2])
                else:
                    k1 = G2.degree(n2)
                if n2_neighbor in between_network_reverse:
                    k2 = G2.degree(n2_neighbor) + len(between_network_reverse[n2_neighbor])
                else:
                    k2 = G2.degree(n2_neighbor)
                if rewards[t][n2_neighbor] > rewards[t][n2]:
                    if random.random() < (rewards[t][n2_neighbor] - rewards[t][n2]) / ((b + c) * max(k1, k2)):
                        actions[t + 1][n2] = actions[t][n2_neighbor]
                    else:
                        actions[t + 1][n2] = actions[t][n2]
                else:
                    actions[t + 1][n2] = actions[t][n2]
        np.save('res/p{}/b{}r{}.npy'.format(int(p*1000), b, r), cooperation_level)


def cooperation_level_within_g1_degree_heterogenity(b, r, opt):
    N = 1000
    M = 4
    p = 0.001
    E = int(p * N * N)

    c = 1
    epochs = 1000
    cooperation_level1 = np.zeros((epochs, rounds))
    cooperation_level2 = np.zeros((epochs, rounds))
    for epoch in range(epochs):
        if epoch // 100 == 0:
            g1 = nx.barabasi_albert_graph(N, M)
            g2 = nx.barabasi_albert_graph(N, M)
            G1 = nx.Graph()
            for edge in g1.edges():
                G1.add_edge('r' + str(edge[0]), 'r' + str(edge[1]))
            G2 = nx.Graph()
            for edge in g2.edges():
                G2.add_edge('b' + str(edge[0]), 'b' + str(edge[1]))
            G1_deg, G2_deg, G1_deg_inv, G2_deg_inv = [], [], [], []
            for node1 in G1.nodes():
                G1_deg.append(G1.degree(node1))
                G1_deg_inv.append(1 / G1.degree(node1))
            for node2 in G2.nodes():
                G2_deg.append(G2.degree(node2))
                G2_deg_inv.append(1 / G2.degree(node2))
            G1_deg = np.array(G1_deg) / sum(G1_deg)
            G1_deg_inv = np.array(G1_deg_inv) / sum(G1_deg_inv)
            G2_deg = np.array(G2_deg) / sum(G2_deg)
            G2_deg_inv = np.array(G2_deg_inv) / sum(G2_deg_inv)
            between_network = {}
            between_network_reverse = {}
            num_between_network = 0
            while num_between_network < E:
                if opt == 'high2high':
                    edge = (np.random.choice(G1.nodes(), replace=True, p=G1_deg),
                            np.random.choice(G2.nodes(), replace=True, p=G2_deg))
                elif opt == 'high2low':
                    edge = (np.random.choice(G1.nodes(), replace=True, p=G1_deg),
                            np.random.choice(G2.nodes(), replace=True, p=G2_deg_inv))
                elif opt == 'low2low':
                    edge = (np.random.choice(G1.nodes(), replace=True, p=G1_deg_inv),
                            np.random.choice(G2.nodes(), replace=True, p=G2_deg_inv))
                if edge[0] in between_network:
                    if edge[1] not in between_network[edge[0]]:
                        between_network[edge[0]].append(edge[1])
                        num_between_network += 1
                else:
                    between_network[edge[0]] = [edge[1]]

                if edge[1] in between_network_reverse:
                    if edge[0] not in between_network_reverse[edge[1]]:
                        between_network_reverse[edge[1]].append(edge[0])
                else:
                    between_network_reverse[edge[1]] = [edge[0]]
        actions = {time: {} for time in range(rounds)}
        rewards = {time: {} for time in range(rounds)}
        cooperation_level1[epoch][0] += 0.5
        cooperation_level2[epoch][0] += 0.5
        for n1 in G1.nodes():
            if random.random() < 0.5:
                actions[0][n1] = 'C'
            else:
                actions[0][n1] = 'D'
        for n2 in G2.nodes():
            if random.random() < 0.5:
                actions[0][n2] = 'C'
            else:
                actions[0][n2] = 'D'

        for t in range(rounds-1):
            for n1 in G1.nodes():
                rewards[t][n1] = 0
                for n1_neighbor in nx.neighbors(G1, n1):
                    if actions[t][n1] == 'C':
                        if actions[t][n1_neighbor] == 'C':
                            rewards[t][n1] += b - c
                        if actions[t][n1_neighbor] == 'D':
                            rewards[t][n1] += - c
                    if actions[t][n1][0] == 'D':
                        if actions[t][n1_neighbor] == 'C':
                            rewards[t][n1] += b
            for n2 in G2.nodes():
                rewards[t][n2] = 0
                for n2_neighbor in nx.neighbors(G2, n2):
                    if actions[t][n2] == 'C':
                        if actions[t][n2_neighbor] == 'C':
                            rewards[t][n2] += b - c
                        if actions[t][n2_neighbor] == 'D':
                            rewards[t][n2] += - c
                    if actions[t][n2] == 'D':
                        if actions[t][n2_neighbor] == 'C':
                            rewards[t][n2] += b
            for n1 in between_network:
                for n2 in between_network[n1]:
                    if actions[t][n1] == 'C':
                        if actions[t][n2] == 'C':
                            rewards[t][n1] += r / 2 - c
                            rewards[t][n2] += r / 2 - c
                        if actions[t][n2] == 'D':
                            rewards[t][n1] += r - c
                    if actions[t][n1] == 'D':
                        if actions[t][n2] == 'C':
                            rewards[t][n2] += r - c
            actions[t + 1] = {}
            for n1 in G1.nodes():
                n1_neighbor = random.choice(list(nx.neighbors(G1, n1)))
                if n1 in between_network:
                    k1 = G1.degree(n1) + len(between_network[n1])
                else:
                    k1 = G1.degree(n1)
                if n1_neighbor in between_network:
                    k2 = G1.degree(n1_neighbor) + len(between_network[n1_neighbor])
                else:
                    k2 = G1.degree(n1_neighbor)
                if rewards[t][n1_neighbor] > rewards[t][n1]:
                    if random.random() < (rewards[t][n1_neighbor] - rewards[t][n1]) / ((b + c) * max(k1, k2)):
                        actions[t + 1][n1] = actions[t][n1_neighbor]
                    else:
                        actions[t + 1][n1] = actions[t][n1]
                else:
                    actions[t + 1][n1] = actions[t][n1]
                if actions[t + 1][n1] == 'C':
                    cooperation_level1[epoch][t + 1] += 1 / N
            for n2 in G2.nodes():
                n2_neighbor = random.choice(list(nx.neighbors(G2, n2)))
                if n2 in between_network_reverse:
                    k1 = G2.degree(n2) + len(between_network_reverse[n2])
                else:
                    k1 = G2.degree(n2)
                if n2_neighbor in between_network_reverse:
                    k2 = G2.degree(n2_neighbor) + len(between_network_reverse[n2_neighbor])
                else:
                    k2 = G2.degree(n2_neighbor)
                if rewards[t][n2_neighbor] > rewards[t][n2]:
                    if random.random() < (rewards[t][n2_neighbor] - rewards[t][n2]) / ((b + c) * max(k1, k2)):
                        actions[t + 1][n2] = actions[t][n2_neighbor]
                    else:
                        actions[t + 1][n2] = actions[t][n2]
                else:
                    actions[t + 1][n2] = actions[t][n2]
                if actions[t + 1][n2] == 'C':
                    cooperation_level2[epoch][t + 1] += 1 / N
        if opt == 'high2high':
            np.save('res/highhigh/b{}r{}.npy'.format(b, r), cooperation_level1)
        elif opt == 'high2low':
            np.save('res/highlowg1/b{}r{}.npy'.format(b, r), cooperation_level1)
            np.save('res/highlowg2/b{}r{}.npy'.format(b, r), cooperation_level2)
        elif opt == 'low2low':
            np.save('res/lowlow/b{}r{}.npy'.format(b, r), cooperation_level1)