#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:01:55 2018

@author: Degentleman
"""
import pandas as pd
import networkx as nx
import random
import numpy as np
from itertools import combinations


def player_dict(G,player):
    x = dict(G[player])
    return x

# Common path functions:

# Return the path between two agents, resulting in a list of neighbor nodes

path = nx.dijkstra_path

#Returns the distance between two agents in a graph, resulting in an int of E
path_d = nx.dijkstra_path_length

#Returns the shortest path between two agents in a graph, resulting in a list
s_path = nx.shortest_path

username = input('What is your username? ')
filename = '/Users/{username}/Downloads/GSW at HOU WCF - NBA PBP.csv'.format(username=username)
csv_df = pd.read_csv(filename, delimiter=',')

test_list = []
for i in range(len(csv_df)):
    row = csv_df.iloc[i]
    col = list(csv_df)
    lineup = []
    for x in col:
        key = x
        data = row[key]
        if data == "In":
            lineup.append(str(key))
        if key == "Lineup Score":
            lineup.append(int(data))
            test_list.append(lineup)


team_performance = []
for x in test_list:
    matchup = x
    team_performance.append(matchup)
    

lineup_df = pd.DataFrame(team_performance, columns = ['Team(i1)',
                                               'Team(i2)',
                                               'Team(i3)',
                                               'Team(i4)',
                                               'Team(i5)',
                                               'Team(j1)',
                                               'Team(j2)',
                                               'Team(j3)',
                                               'Team(j4)',
                                               'Team(j5)',
                                               'Synergy'])

df_headers = list(lineup_df)    

# Create a list of agents without duplicates
    
lineup_list = []
for i in range(len(lineup_df)):
    row = lineup_df.iloc[i]
    players = [x for x in lineup_df.iloc[i] if type(x) == str]
    for x in players:
        if x not in lineup_list:
            lineup_list.append(x)
lineup_list = sorted(lineup_list)



#Learn Team Capabilities from Training Data

training_data = lineup_df


# Separate players into teams

team_Ai = []
team_Aj = []

for i in range(len(training_data)):
    row = training_data.iloc[i]
    lineup_i = row[0:5]
    for a_i in lineup_i[:]:
        if a_i not in team_Ai:
            team_Ai.append(a_i)
    lineup_j = row[5:10]
    for a_j in lineup_j[:]:
        if a_j not in team_Aj:
            team_Aj.append(a_j)
            
            
# Assign capability value to player from mean of lineup performance

value = []
for player_i in team_Ai:
    player_value = []
    for i in range(len(training_data)):
        row = training_data.iloc[i]
        for x in row:
            if player_i == x:
                C_Ai = row['Synergy']
                player_value.append(int(C_Ai))
    value.append([player_i,round(np.sum(player_value),3),len(player_value),round(np.mean(player_value),3),round(np.std(player_value),3)])

for player_j in team_Aj:
    player_value = []
    for i in range(len(training_data)):
        row = training_data.iloc[i]
        for x in row:
            if player_j == x:
                C_Aj = row['Synergy']
                player_value.append(int(C_Aj))
    value.append([player_j,round(np.sum(player_value),3),len(player_value),round(np.mean(player_value),3),round(np.std(player_value),3)])

k = 1/(len(value)/2)


# Create a Synergy Graph using Training Data from df

D_Train = nx.Graph()

for i in range(len(lineup_df)):
    row = lineup_df.iloc[i]
    player_list = [x for x in lineup_df.iloc[i] if type(x) == str]
    for player in row[0:5]:
        for x in value[:]:
            if x[0] == player:
                C_Ai = round(x[3],4)
        for pair in row[0:5]:
            if player!=pair:
                for x in value[:]:
                    if x[0] == pair:
                        C_Bi = round(x[3],4)
                D_Train.add_edge(player,pair,weight=1,sprd=row[10],Cap_AiBi=round((C_Ai+C_Bi),4))
            if player==pair:
                pass
        for adversary in row[5:10]:
            for x in value[:]:
                if x[0] == adversary:
                    C_Aj = round(x[3],4)
                    D_Train.add_edge(player,adversary,weight=1,sprd=row[10],Cap_AiAj=round((C_Ai-C_Aj),4))
        
    for opp in row[5:9]:
        for x in value[:]:
            if x[0] == opp:
                C_Aj = round(x[3],4)
        for opp_pair in row[5:10]:
            if opp!=opp_pair:
                for x in value[:]:
                    if x[0] == opp_pair:
                        C_Bj = round(x[3],4)
                        D_Train.add_edge(opp,opp_pair,weight=1,sprd=row[10],Cap_AjBj=round((C_Aj+C_Bj),4))
            if opp==opp_pair:
                pass
        for adversary in row[0:5]:
            for x in value[:]:
                if x[0] == adversary:
                    C_Ai = round(x[3],4)
                    D_Train.add_edge(opp,adversary,weight=1,sprd=row[10],Cap_AjAi=round((C_Aj-C_Ai),4))


weighted_lineup = []
for i in range(len(lineup_df)):
    row = lineup_df.iloc[i]
    entry = []
    for data in row:
        if type(data) == str:
            for player in value:
                if data == player[0]:
                    C_A=player[3]
                    entry.append(C_A)
    team_i_lineup = list(row[0:5])
    i_lineup_value = []
    for x,y in list(combinations(team_i_lineup,r=2)):
        if D_Train.has_edge(x,y) == True:
            combo_value = D_Train[x][y]['Cap_AiBi']
            i_lineup_value.append(combo_value)
    i_line_value = np.sum(i_lineup_value)*k
    entry.append(i_line_value)
    team_j_lineup = list(row[5:10])
    j_lineup_value = []
    for w,v in list(combinations(team_j_lineup,r=2)):
        if D_Train.has_edge(w,v) == True:
            combo_value = D_Train[w][v]['Cap_AjBj']
            j_lineup_value.append(combo_value)
    j_line_value = np.sum(j_lineup_value)*k
    entry.append(j_line_value)
    weighted_lineup.append(entry)

w_df = pd.DataFrame(weighted_lineup, columns=['Team(Ci1)',
                                               'Team(Ci2)',
                                               'Team(Ci3)',
                                               'Team(Ci4)',
                                               'Team(Ci5)',
                                               'Team(Cj1)',
                                               'Team(Cj2)',
                                               'Team(Cj3)',
                                               'Team(Cj4)',
                                               'Team(Cj5)',
                                               'Cap_Ai',
                                               'Cap_Aj'])

weighted_lineup_df = pd.concat([lineup_df,w_df],axis=1)

# Create a random unweighted graph

G = nx.Graph()
teams = team_Ai+team_Aj
N = len(teams)

for x in value[:]:
    player = x[0]
    C_A = x[3]
    G.add_node(player,C_A=C_A)

explored = []
while G.size() < int(N-1):
    queue = value[:]
    random.shuffle(queue)
    agent_a = queue[0][0]
    print(agent_a)
    capability_a = queue[0][3]
    queue.remove(queue[0])
    if agent_a not in explored:
        G.add_node(agent_a,CofA=capability_a)
        explored.append(agent_a)
    i = random.randint(0,len(queue)-1)
    agent_b = queue[i][0]
    print(agent_b)
    capability_b = queue[i][3]
    if agent_a != agent_b:
        if G.has_edge(agent_a,agent_b) == False:
            G.add_edge(agent_a,agent_b,weight=1,CofA=capability_a,CofB=capability_b)
    neighbors = list(G.neighbors(agent_a))
    print(neighbors)
print(len(explored))
        
# Evaluate compatibility between agents in random graph G

columns=['Agent A','Agent B','Same Team','Dist','C_A','C_B','Cap_AB']

pairwise_syn = pd.DataFrame(columns=columns)

entry = []
for i in range(len(lineup_df)):
    row = lineup_df.iloc[i]
    player_list = [x for x in lineup_df.iloc[i] if type(x) == str]
    for agent_a in row[0:5]:
        C_Ai = [round(x[1]['C_A'],3) for x in list(G.nodes(data=True)) if x[0] == agent_a][0]
        for agent_b in row[0:5]:
            if agent_b != agent_a:
                if G.has_edge(agent_a,agent_b):
                    dist = path_d(G,agent_a,agent_b)
                else:
                    dist = 1
                C_Bi = [round(x[1]['C_A'],3) for x in list(G.nodes(data=True)) if x[0] == agent_b][0]
                cap_AiBi = round(C_Ai+C_Bi,3)
                same_team = 'Yes'
                entry.append([agent_a,agent_b,same_team,dist,C_Ai,C_Bi,cap_AiBi])
        for agent_b in row[5:10]:
            if G.has_edge(agent_a,agent_b):
                dist = path_d(G,agent_a,agent_b)
            else:
                dist = 1
            C_Aj = [round(x[1]['C_A'],3) for x in list(G.nodes(data=True)) if x[0] == agent_b][0]
            cap_AiAj = round(C_Ai-C_Aj,3)
            same_team = 'No'
            entry.append([agent_a,agent_b,same_team,dist,C_Ai,C_Aj,cap_AiAj])
    for agent_a in row[5:10]:
        C_Aj = [round(x[1]['C_A'],3) for x in list(G.nodes(data=True)) if x[0] == agent_a][0]
        for agent_b in row[5:10]:
            if agent_b != agent_a:
                if G.has_edge(agent_a,agent_b):
                    dist = path_d(G,agent_a,agent_b)
                else:
                    dist = 1
                C_Bj = [round(x[1]['C_A'],3) for x in list(G.nodes(data=True)) if x[0] == agent_b][0]
                cap_AjBj = round(C_Aj+C_Bj,3)
                same_team = 'Yes'
                entry.append([agent_a,agent_b,same_team,dist,C_Aj,C_Bj,cap_AjBj])
        for agent_b in row[0:5]:
            if G.has_edge(agent_a,agent_b):
                dist = path_d(G,agent_a,agent_b)
            else:
                dist = 1
            C_Ai = [round(x[1]['C_A'],3) for x in list(G.nodes(data=True)) if x[0] == agent_b][0]
            cap_AjAi = round(C_Aj-C_Ai,3)
            same_team = 'No'
            entry.append([agent_a,agent_b,same_team,dist,C_Aj,C_Ai,cap_AjAi])

pairwise_syn = pd.DataFrame(entry,columns=columns)
    
#Export the Pairwise Synergy of All Agents

pairwise_syn.to_csv('PairwiseSynergy.csv')
