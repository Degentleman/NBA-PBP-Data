#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 00:06:24 2018

@author: Degentleman
"""
import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
import statsmodels.api as sm

def player_dict(G,player):
    x = dict(G[player])
    return x

def Phi(d):
    return(1/float(d))

def bfs_cc(G):
    # Visited Nodes
    explored = []
    # Nodes to be checked
    queue = list(G.nodes)
    G_prime = nx.Graph(G)
    np.random.shuffle(queue)
    # Loop until there are no remaining nodes to be checked
    while G_prime.size() < len(list(G.nodes))-1:
        node = queue.pop(0)
        if node not in explored:
            explored.append(node)
            if len(queue)>0:
                neighbor = queue[0]
                if G_prime.has_edge(node,neighbor) == False:
                    weight = 1
                    G_prime.add_edge(node,neighbor,weight=weight)
                elif G_prime.has_edge(node,neighbor) == True:
                    continue
    return (G_prime)

def Synergy(A,team_Ai,team_Aj):
    if list(A.nodes(data='mu'))[0][1] is None:
        return(print('mu data missing')) 
    CofAi = [x for x in combinations(team_Ai,r=2)]
    CofAj = [x for x in combinations(team_Aj,r=2)]
    agent_pairs = np.array(['Team','Player','Teammate','mu','var']).reshape(1,5)
    for team in CofAi:
        team_name = 'Ai'
        player = team[0]
        teammate = team[1]
        dist = Phi(path_d(A,player,teammate))
        compat_mu = A.nodes[player]['mu']+A.nodes[teammate]['mu']
        compat_var = A.nodes[player]['var']+A.nodes[teammate]['var']
        pair_mu = round(dist*compat_mu,3)
        pair_var = round(((dist**2)*compat_var),3)
        entry = np.array([team_name,player,teammate,pair_mu,pair_var]).reshape(1,5)
        agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    for team in CofAj:
        team_name = 'Aj'
        player = team[0]
        teammate = team[1]
        dist = Phi(path_d(A,player,teammate))
        compat_mu = A.nodes[player]['mu']+A.nodes[teammate]['mu']
        compat_var = A.nodes[player]['var']+A.nodes[teammate]['var']
        pair_mu = round(dist*compat_mu,3)
        pair_var = round(((dist**2)*compat_var),3)
        entry = np.array([team_name,player,teammate,pair_mu,pair_var]).reshape(1,5)
        agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    syn_df = pd.DataFrame(agent_pairs[1:],columns=['Team','Player One','Player Two','Cap','Var'])
    return(syn_df)
    
    
# Common path functions:

path = nx.dijkstra_path

#Returns the distance between two agents in a graph, resulting in an int of E
path_d = nx.dijkstra_path_length

# Returns the shortest path between two agents in a graph, resulting in a list
# Make Sure that the CSV you're scraping has the field "Lineup Score" and all
# the player names have an in or out specification.

s_path = nx.shortest_path

username = input('What is your username? ')
filename = '/Users/{username}/Downloads/PBP DEN at LAC 101718.csv'.format(username=username)
csv_df = pd.read_csv(filename, delimiter=',')


# Created a DataFrame of the Lineups for each team using sorted lists
csv_lineups = []

for i in range(len(csv_df)):
    row = csv_df.iloc[i]
    col = list(csv_df)
    lineup = []
    entry = []
    for x in col:
        key = x
        data = row[key]
        if data == "In":
            entry.append(str(key))
        if key == "Lineup Score":
            team_Ai_lineup = entry[0:5]
            team_Aj_lineup = entry[5:]
            if 'Tobias Harris' in team_Aj_lineup:
                print(i)
            entry = team_Ai_lineup+team_Aj_lineup
            entry.append(int(data))
    lineup = entry
    csv_lineups.append(lineup)

lineup_df = pd.DataFrame(csv_lineups, columns = ['Team(i1)',
                                               'Team(i2)',
                                               'Team(i3)',
                                               'Team(i4)',
                                               'Team(i5)',
                                               'Team(j1)',
                                               'Team(j2)',
                                               'Team(j3)',
                                               'Team(j4)',
                                               'Team(j5)',
                                               'Performance'])

# Create a list of agents without duplicates


# Separate players into teams

team_Ai = []

team_Aj = []

for i in range(len(lineup_df)):
    row = lineup_df.iloc[i]
    lineup_i = row[0:5]
    for a_i in lineup_i[:]:
        if a_i not in team_Ai:
            team_Ai.append(a_i)
    lineup_j = row[5:10]
    for a_j in lineup_j[:]:
        if a_j not in team_Aj:
            team_Aj.append(a_j)

team_Ai = sorted(team_Ai)

team_Aj = sorted(team_Aj)

lineup_list = team_Ai+team_Aj

# Assign capability value to player from mean of lineup performance

value = []

for player_i in team_Ai:
    player_value = []
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        for x in row:
            if player_i == x:
                C_Ai = row['Performance']
                player_value.append(int(C_Ai))
    value.append([player_i,round(np.mean(player_value),4),round(np.var(player_value),4)])

for player_j in team_Aj:
    player_value = []
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        for x in row:
            if player_j == x:
                C_Aj = row['Performance']
                player_value.append(int(C_Aj))
    value.append([player_j,round(np.mean(player_value),4),round(np.var(player_value),4)])

k = 1/(len(value)/2)

NBA = nx.Graph()

for x in team_Ai[:]:
    for player in value:
        if player[0] == x:
            name = player[0]
            player_mu = player[1]
            player_var = player[2]
            NBA.add_node(name,mu=player_mu,var=player_var)
            
for x in team_Aj[:]:
    for player in value:
        if player[0] == x:
            name = player[0]
            player_mu = player[1]
            player_var = player[2]
            NBA.add_node(name,mu=player_mu,var=player_var)

G_Train = nx.Graph(NBA)

for i in range(len(lineup_df)):
    row = lineup_df.iloc[i]
    player_list = [x for x in lineup_df.iloc[i] if type(x) == str]      
    for player in row[0:5]:
        for x in value[:]:
            if x[0] == player:
                C_Ai = round(x[1],3)
                V_Ai = round(x[2],3)
        for pair in row[0:5]:
            if player!=pair:
                for x in value[:]:
                    if x[0] == pair:
                        C_Bi = round(x[1],3)
                        V_Bi = round(x[2],3)
                        G_Train.add_edge(player,pair,weight=1,S_Cap=round((C_Ai+C_Bi),3),S_Var=round((V_Ai+V_Bi),3))
            if player==pair:
                pass
        for adversary in row[5:10]:
            for x in value[:]:
                if x[0] == adversary:
                    C_Aj = round(x[1],3)
                    V_Aj = round(x[2],3)
                    G_Train.add_edge(player,adversary,weight=1,A_Cap=round((C_Ai-C_Aj),3),A_Var=round((V_Ai-V_Aj),3))
    for opp in row[5:9]:
        for x in value[:]:
            if x[0] == opp:
                C_Aj = round(x[1],3)
                V_Aj = round(x[2],3)
        for opp_pair in row[5:10]:
            if opp!=opp_pair:
                for x in value[:]:
                    if x[0] == opp_pair:
                        C_Bj = round(x[1],3)
                        V_Bj = round(x[2],3)
                        G_Train.add_edge(opp,opp_pair,weight=1,S_Cap=round((C_Aj+C_Bj),3),S_Var=round((V_Aj+V_Bj),3))
            if opp==opp_pair:
                pass
        for adversary in row[0:5]:
            for x in value[:]:
                if x[0] == adversary:
                    C_Ai = round(x[1],3)
                    V_Ai = round(x[2],3)
                    G_Train.add_edge(opp,adversary,weight=1,A_Cap=round((C_Aj-C_Ai),3),A_Var=round((V_Aj-V_Ai),3))

Training_lineups = list(G_Train.edges(data=True))

# Created Weighted Lineups Using Known Data

weighted_lineup = []


# Evaluate Synergy between teams usings lineups

G_Alpha = bfs_cc(NBA)

variation_one = list(G_Alpha.edges(data=True))

#Run Model Alpha

for i in range(len(lineup_df)):
    entry = []
    row = lineup_df.iloc[i]
    team_i_lineup = sorted(list(row[0:5]))
    entry.append(team_i_lineup)
    team_j_lineup = sorted(list(row[5:10]))
    entry.append(team_j_lineup)
    player_list = team_i_lineup+team_j_lineup
    Ai_synergy = []
    Aj_synergy = []
    Adversarial = []
    for x,y in list(combinations(player_list,r=2)):
        dist = path_d(G_Alpha,x,y)
        if x in team_Ai:
            team_name = 'Team Ai'
            if y in team_Ai:
                same_team = True
                x_y_Cap = G_Train[x][y]['S_Cap']*Phi(dist)
                x_y_Var = G_Train[x][y]['S_Var']*Phi(dist)**2
                Ai_synergy.append(x_y_Cap)
            if y in team_Aj:
                same_team = False
                x_y_Cap = G_Train[x][y]['A_Cap']*Phi(dist)
                x_y_Var = G_Train[x][y]['A_Var']*Phi(dist)**2
                Adversarial.append(x_y_Cap)
        if x in team_Aj:
            team_name = 'Team Aj'
            if y in team_Aj:
                same_team = True
                x_y_Cap = G_Train[x][y]['S_Cap']*Phi(dist)
                x_y_Var = G_Train[x][y]['S_Var']*Phi(dist)**2
                Aj_synergy.append(x_y_Cap)
            if y in team_Ai:
                same_team = False  
    matchup_score = [round(np.sum(Ai_synergy),3),
                     round(np.sum(Aj_synergy),3),
                     round(np.sum(Adversarial),3),
                     round((np.sum(Ai_synergy)-np.sum(Aj_synergy)+np.sum(Adversarial)),3)]
    entry.append(matchup_score)
    weighted_lineup.append(entry[0]+entry[1]+entry[2])
    Alpha_w_df = pd.DataFrame(weighted_lineup, columns=['Ai1',
                                                   'Ai2',
                                                   'Ai3',
                                                   'Ai4',
                                                   'Ai5',
                                                   'Aj1',
                                                   'Aj2',
                                                   'Aj3',
                                                   'Aj4',
                                                   'Aj5',
                                                   'Ai_Syn',
                                                   'Aj_Syn',
                                                   'Ai_Aj_Adv',
                                                   'Ai_Aj_Synergy'])
# Join DF of Synergy and Performance
Alpha_weighted_df = pd.concat([Alpha_w_df,lineup_df['Performance']],axis=1)
model_error = []
for i in range(len(Alpha_weighted_df)):
    row = Alpha_weighted_df.iloc[i]
    prediction = float(row['Ai_Aj_Synergy'])
    result = float(row['Performance'])
    sq_error = round((prediction-result)**2,3)
    model_error.append(sq_error)
model_error = pd.Series(data=model_error,name='Sq Error')
Alpha_df = pd.concat([Alpha_weighted_df,model_error],axis=1)

# Performance of Model Alpha
Alpha_performance = np.mean(Alpha_df['Sq Error'])**.5

trials = 0

iterations = 100

#Iterate to find better random model

while trials < iterations:
    if Alpha_performance > 0:
        G_Beta = bfs_cc(NBA)
        weighted_lineup = []
        for i in range(len(lineup_df)):
            entry = []
            row = lineup_df.iloc[i]
            team_i_lineup = sorted(list(row[0:5]))
            entry.append(team_i_lineup)
            team_j_lineup = sorted(list(row[5:10]))
            entry.append(team_j_lineup)
            player_list = team_i_lineup+team_j_lineup
            Ai_synergy = []
            Aj_synergy = []
            Adversarial = []
            for x,y in list(combinations(player_list,r=2)):
                if x in team_Ai:
                    team_name = 'Team Ai'
                    if y in team_Ai:
                        same_team = True
                        dist = path_d(G_Beta,x,y)
                        x_y_Cap = G_Train[x][y]['S_Cap']*Phi(dist)
                        x_y_Var = G_Train[x][y]['S_Var']*Phi(dist)**2
                        Ai_synergy.append(x_y_Cap)
                    if y in team_Aj:
                        same_team = False
                        dist = path_d(G_Beta,x,y)
                        x_y_Cap = G_Train[x][y]['A_Cap']*Phi(dist)
                        x_y_Var = G_Train[x][y]['A_Var']*Phi(dist)**2
                        Adversarial.append(x_y_Cap)
                if x in team_Aj:
                    team_name = 'Team Aj'
                    if y in team_Aj:
                        same_team = True
                        dist = path_d(G_Beta,x,y)
                        x_y_Cap = G_Train[x][y]['S_Cap']*Phi(dist)
                        x_y_Var = G_Train[x][y]['S_Var']*Phi(dist)**2
                        Aj_synergy.append(x_y_Cap)
                    if y in team_Ai:
                        same_team = False  
            matchup_score = [round(np.sum(Ai_synergy),3),
                             round(np.sum(Aj_synergy),3),
                             round(np.sum(Adversarial),3),
                             round((np.sum(Ai_synergy)-np.sum(Aj_synergy)+np.sum(Adversarial)),3)]
            entry.append(matchup_score)
            weighted_lineup.append(entry[0]+entry[1]+entry[2])
            
        # Create a Data Frame from the test Graph
        Beta_w_df = pd.DataFrame(weighted_lineup, columns=['Ai1',
                                                           'Ai2',
                                                           'Ai3',
                                                           'Ai4',
                                                           'Ai5',
                                                           'Aj1',
                                                           'Aj2',
                                                           'Aj3',
                                                           'Aj4',
                                                           'Aj5',
                                                           'Ai_Syn',
                                                           'Aj_Syn',
                                                           'Ai_Aj_Adv',
                                                           'Ai_Aj_Synergy'])
        # Join DF of Synergy and Performance
        Beta_weighted_df = pd.concat([Beta_w_df,lineup_df['Performance']],axis=1)
        model_error = []
        for i in range(len(Beta_weighted_df)):
            row = Beta_weighted_df.iloc[i]
            prediction = float(row['Ai_Aj_Synergy'])
            result = float(row['Performance'])
            sq_error = round((prediction-result)**2,3)
            model_error.append(sq_error)
        model_error = pd.Series(data=model_error,name='Sq Error')
        Beta_df = pd.concat([Beta_weighted_df,model_error],axis=1)
        
        # Performance of Beta Model
        Beta_performance = np.mean(Beta_df['Sq Error'])**.5
        trials +=1
        if Beta_performance < Alpha_performance:
            
            #Copy Beta information to Alpha model
            G_Alpha = G_Beta
            Alpha_weighted_df = Beta_weighted_df
            Alpha_df = Beta_df
            Alpha_performance = Beta_performance
            print('Alpha Performance: ' + str(Alpha_performance)+' Beta Performance: ' + str(Beta_performance))
            print('Beta Graph changed to Alpha Graph on trial #' + str(trials))

print(str(iterations)+' trials complete')

final_variation = list(G_Alpha.edges(data=True))

queue = team_Ai[:]+team_Aj[:]

paths_df = [] 
for x in team_Ai:
    queue.remove(x)
for x in queue:
    for y in team_Ai:
        x_y_d = path_d(G_Alpha,x,y)
        x_mu = G_Train.nodes()[x]['mu']
        x_var = G_Train.nodes()[x]['var']
        y_mu = G_Train.nodes()[y]['mu']
        y_var = G_Train.nodes()[y]['var']
        entry = [x,y,x_y_d,x_mu,x_var,y_mu,y_var]
        paths_df.append(entry)
        
paths_df = pd.DataFrame(data=paths_df,columns=['Ax','Ay','P(Ax_Ay)','x mu','x var','y mu','y var'])

sample_N = int(len(Alpha_df)*.9)
X = Alpha_df.iloc[0:sample_N][['Ai_Syn','Aj_Syn','Ai_Aj_Adv']]

Y = Alpha_df.iloc[0:sample_N][['Performance']]

model = sm.OLS(Y,X).fit()

input_data = Alpha_df[['Ai_Syn','Aj_Syn','Ai_Aj_Adv']]

predictions = pd.Series(data=model.predict(input_data ),name='Predictions')

predictions_df = pd.concat([Alpha_df,predictions],axis=1)

print(np.sum(predictions_df['Sq Error']))

duplicates = []
duplicates_df = []
for i in range(len(Alpha_df[list(Alpha_df)[0:10]])):
    row = Alpha_df[list(Alpha_df)[0:10]].iloc[i]
    performance = Alpha_df.iloc[i]['Performance']
    index_name = [str(i)]
    init_count = 0
    entry = [int(performance)]
    for z in range(len(Alpha_df[list(Alpha_df)[0:10]])):
        row_z = Alpha_df[list(Alpha_df)[0:10]].iloc[z]
        if i != z:
            if list(row) == list(row_z):
                init_count +=1
                duplicates.append([i,z])
                perf_z = Alpha_df.iloc[z]['Performance']
                entry.append(int(perf_z))
    entry = [np.sum(entry)]
    entry = index_name+entry+[init_count]
    duplicates_df.append(entry)

duplicates_df = pd.DataFrame(data=duplicates_df,columns=['Lineup Index','Total Score','Count'])
    
average_score = pd.concat([lineup_df.iloc[:,:-1],duplicates_df[['Total Score','Count']]],axis=1)
    
average_score = average_score.drop_duplicates(subset=list(average_score)[0:10],keep='first').sort_values(by=['Count'],ascending=False)

paths_list = []
    
for x in list(s_path(G_Alpha)):
    for player in list(s_path(G_Alpha)):
        if x != player:
            y = player
            entry = x,y,path_d(G_Alpha,x,y)
            paths_list.append(entry)

paths_list_df = pd.DataFrame(paths_list,columns = ['Player_A','Player_B','Distance'])

synergy_df = Synergy(G_Alpha,team_Ai,team_Aj)

print('----')
for x in team_Ai:
    player_name = x
    filter_by = [player_name]
    filter_df = Alpha_df[(Alpha_df.Ai1.isin(filter_by)) 
            | (Alpha_df.Ai2.isin(filter_by))
            | (Alpha_df.Ai3.isin(filter_by))
             | (Alpha_df.Ai4.isin(filter_by))
            | (Alpha_df.Ai5.isin(filter_by))]
    print(player_name)
    print(round(np.mean(filter_df['Ai_Aj_Synergy']),3))
print('----')
print('Team Ai Mean Synergy:')
print(np.mean(Alpha_df['Ai_Syn']))
print('Team Ai Mean of Cap Pair:')
print(np.mean(np.array(synergy_df[synergy_df.Team == 'Ai']['Cap'],dtype=float)))
print('----')
print('----')
for y in team_Aj:
    player_name = y
    filter_by = [player_name]
    filter_df = Alpha_df[(Alpha_df.Aj1.isin(filter_by)) 
            | (Alpha_df.Aj2.isin(filter_by))
            | (Alpha_df.Aj3.isin(filter_by))
            | (Alpha_df.Aj4.isin(filter_by))
            | (Alpha_df.Aj5.isin(filter_by))]
    print(player_name)
    print(round(np.mean(filter_df['Ai_Aj_Synergy']),3))
print('----')
print('Team Aj Mean Synergy:')
print(np.mean(Alpha_df['Aj_Syn']))
print('Team Aj Mean of Cap Pair:')
print(np.mean(np.array(synergy_df[synergy_df.Team == 'Aj']['Cap'],dtype=float)))
print('----')
print('----')

model_results = [np.sum(Alpha_df.Performance),np.sum(Alpha_df.Ai_Syn), 
                 np.sum(Alpha_df.Aj_Syn),np.sum(Alpha_df.Ai_Aj_Adv)]

proj_spread = ((model_results[1]*.7281)+(model_results[2]*-.1575)+(model_results[3]*.4969))/2
