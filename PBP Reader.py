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

training_size = .9
training_data = lineup_df.iloc[0:round(int(len(lineup_df)) * training_size)]


team_Ai = []
team_Aj = []

for i in range(len(training_data)):
    row = training_data.iloc[i]
    lineup_i = list(row[0:5])
    for a_i in lineup_i:
        if a_i not in team_Ai:
            team_Ai.append(a_i)
    lineup_j = list(row[5:9])
    for a_j in lineup_j:
        if a_j not in team_Aj:
            team_Aj.append(a_j)

value = []
for player in team_Ai:
    player_value = []
    for i in range(len(training_data)):
            row = training_data.iloc[i]
            for x in row:
                if player == x:
                    Ci = row['Synergy']
                    player_value.append(int(Ci))
    value.append([player,round(np.mean(player_value),3)])

for player in team_Aj:
    player_value = []
    for i in range(len(training_data)):
            row = training_data.iloc[i]
            for x in row:
                if player == x:
                    Ci = row['Synergy']
                    player_value.append(int(Ci))
    value.append([player,round(np.mean(player_value),3)])

k = 1/(len(value)/2)


weighted_lineup = []
for i in range(len(lineup_df)):
    row = lineup_df.iloc[i]
    entry = []
    for data in row:
        if type(data) == str:
            for player in value:
                if data == player[0]:
                    Ci=player[1]
                    entry.append(Ci)
    weighted_lineup.append(entry)
w_df = pd.Series(weighted_lineup, name='Weighted Lineup')

weighted_lineup_df = pd.concat([lineup_df,w_df],axis=1)

# Create a random unweighted graph

G = nx.Graph()

queue = lineup_list[:]
while G.size() < int(len(lineup_list)-1):
    x = queue[random.randint(0,len(queue)-1)]
    queue.remove(x)
    y = queue[random.randint(0,len(queue)-1)]
    if len(queue) <= 1:
        queue = lineup_list[:]
    if x != y:    
        G.add_edge(x,y,weight=1)
        
# Evaluate compatibility between agents in random graph G

columns=['i','j','Same Team','C_i','C_j','Cap_i_j']

pairwise_syn = pd.DataFrame(columns=columns)

for i in range(len(value)):
    player_i,c_i = value[i]
    entry = []
    queue = [x for x in value if x[0] != player_i]
    for path in queue:
        if path[0] in team_Ai:
            if player_i in team_Ai:
                cap_i_j = round(c_i+path[1],3)
                same_team = 'Yes'
            if player_i not in team_Ai:
                cap_i_j = round(c_i-path[1],3)
                same_team = 'No'
        if path[0] in team_Aj:
            if player_i in team_Aj:
                cap_i_j = round(c_i+path[1],3)
                same_team = 'Yes'
            if player_i not in team_Aj:
                cap_i_j = round(c_i-path[1],3)
                same_team = 'No'
        entry.append([player_i,path[0],same_team,c_i,path[1],cap_i_j])
        entry_df = pd.DataFrame(entry,columns=columns)
    pairwise_syn = pd.concat([pairwise_syn,entry_df])

pairwise_syn.to_csv('PairwiseSynergy.csv')

# Create a Synergy Graph using Training Data from df

D_Train = nx.Graph()

for i in range(len(lineup_df)):
    row = lineup_df.iloc[i]
    player_list = [x for x in lineup_df.iloc[i] if type(x) == str]
    for player in row[0:5]:
        for pair in row[0:5]:
            if player!=pair:
                D_Train.add_edge(player,pair,weight=1,sprd=row[10])
            if player==pair:
                pass
        for adversary in row[5:10]:
            D_Train.add_edge(player,adversary,weight=1,sprd=row[10])
        
    for opp in row[5:9]:
        for opp_pair in row[5:10]:
            if opp!=opp_pair:
                D_Train.add_edge(player,pair,weight=1,sprd=row[10])
            if opp==opp_pair:
                pass
        for adversary in row[0:5]:
            D_Train.add_edge(player,adversary,weight=1,sprd=row[10])