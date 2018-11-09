#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:20:00 2018
@author: Degentleman
"""
import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
import statsmodels.api as sm

# Create a DataFrame of lineups for each team using the PBP_Perf file in the Play-by-Play Reader
def lineups(DataFrame):
    
    csv_lineups = []
    
    for i in range(len(DataFrame)):
        entry = []
        lu_start = list(DataFrame).index('Player Two')+1
        col_keys = list(DataFrame)[lu_start:]
        row = DataFrame.iloc[i][col_keys]
        performance = row[col_keys[-1]]
        for key in col_keys[:-1]:
            if row[key] == "In":
                entry.append(key)
        if len(entry) != 10:
            print('There are not 10 Players In @ Loc #' + str(i))
        else:
            entry.append(performance)
            csv_lineups.append(entry)
    
    lineup_df = pd.DataFrame(csv_lineups,columns = ['Team(Ai1)',
                                       'Team(Ai2)',
                                       'Team(Ai3)',
                                       'Team(Ai4)',
                                       'Team(Ai5)',
                                       'Team(Aj1)',
                                       'Team(Aj2)',
                                       'Team(Aj3)',
                                       'Team(Aj4)',
                                       'Team(Aj5)',
                                       'Performance'])
    return (lineup_df)

def team_sort(lineup_df):
    #Seperates players into teams.
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
    return(team_Ai,team_Aj)

# Here are some common functions for the Adversarial Synergy Graph.

# Find paths between agents
path = nx.dijkstra_path

#Find the shortest path between two agents in the graph
path_d = nx.dijkstra_path_length

# Returns a list of shortest paths between agents in the graph.
s_path = nx.shortest_path

# Compatibility function
def Phi(d):
    return(1/float(d))

#Search through nodes and create unweighted edges.
def bfs_cc(NBA):
    
    # Visited Nodes
    explored = []
    # Nodes to be checked
    queue = list(NBA.nodes)
    G_prime = nx.Graph(NBA)
    np.random.shuffle(queue)
    
    # Loop until there are no remaining nodes to be checked
    while G_prime.size() < len(list(NBA.nodes))-1:
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

# Create a Blank Unweighted Graph
def LearnCap(lineup_df):
    
    # Learn the mean value of each player from the lineup DataFrame
    value_list = []
    
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        team_Ai_lineup = list(row)[0:5]
        team_Aj_lineup = list(row)[5:10]
        for player_i in team_Ai_lineup:
            i_performance = float(str(list(row)[10]))
            value_list.append([player_i,i_performance])
        for player_j in team_Aj_lineup:
            j_performance = float(str(-list(row)[10]))
            value_list.append([player_j,j_performance])
    value_df = pd.DataFrame(data=value_list, columns=['Player','Performance'])
    
    Agents = list(np.unique(value_df['Player']))
    
    final_value = []
    
    for x in Agents:
        player_name = str(x)
        player_in = value_df[(value_df['Player'] == player_name)]
        mean_performance = round(np.mean(player_in['Performance']),3)
        var_performance = round(np.var(player_in['Performance']),3)
        final_entry = [player_name,mean_performance,var_performance]
        final_value.append(final_entry)
    
    final_value_df = pd.DataFrame(data=final_value,columns=['Player','mu','var' ])
    
    value = final_value_df
    
    # Add training data as nodes to an Unweighted Graph
    D_Train = nx.Graph()
    
    # Add the training data to the NBA graph as player nodes
    for i in range(len(value)):
        row = value.iloc[i]
        name = row['Player']
        player_mu = row['mu']
        player_var = row['var']
        D_Train.add_node(name,mu=player_mu,var=player_var)
    
    return(D_Train,value)


# Regress the compatability of each agent to the performance outcome in the Lineup DataFrame to solve for agent capability.
def Cap_Matrix(G_Alpha,lineup_df):  
    
    Agents = list(G_Alpha)
    A_matrix = []
    for player in Agents:
        for x in list(lineup_df)[0:10]:
            if player in list(lineup_df[x]):
                row_index = list(lineup_df[x]).index(player)
                row = lineup_df.iloc[row_index]
                team_Ai_lineup = sorted(list(row[0:5]))
                team_Aj_lineup = sorted(list(row[5:10]))
                if player in team_Ai_lineup:
                    Ai_combos = list(combinations(team_Ai_lineup,r=2))
                    for Ai1,Ai2 in Ai_combos:
                        if Ai1 == player:
                            dist = path_d(G_Alpha,Ai1,Ai2)
                            Ai1_compat = Phi(dist)
                            Ai2_compat = Phi(dist)
                            pair_cap = int(row['Performance'])
                            Ai_entry = [Ai1,Ai2,Ai1_compat,Ai2_compat,pair_cap]
                            A_matrix.append(Ai_entry)
                        if Ai2 == player:
                            dist = path_d(G_Alpha,Ai2,Ai1)
                            Ai2_compat = Phi(dist)
                            Ai1_compat = Phi(dist)
                            pair_cap = int(row['Performance'])
                            Ai_entry = [Ai2,Ai1,Ai2_compat,Ai1_compat,pair_cap]
                            A_matrix.append(Ai_entry)
                    for Aj1 in team_Aj_lineup:
                        dist = path_d(G_Alpha,player,Aj1)
                        Ai1_compat = Phi(dist)
                        Aj1_compat = Phi(dist)
                        adv_cap = int(row['Performance'])
                        Ai_entry = [player,Aj1,Ai1_compat,Aj1_compat,adv_cap]
                        A_matrix.append(Ai_entry)
                if player in team_Aj_lineup:
                    Aj_combos = list(combinations(team_Aj_lineup,r=2))
                    for Aj1,Aj2 in Aj_combos:
                        if Aj1 == player:
                            dist = path_d(G_Alpha,Aj1,Aj2)
                            Aj1_compat = Phi(dist)
                            Aj2_compat = Phi(dist)
                            pair_cap = int(row['Performance'])
                            Aj_entry = [Aj1,Aj2,Aj1_compat,Aj2_compat,pair_cap]
                            A_matrix.append(Aj_entry)
                        if Aj2 == player:
                            dist = path_d(G_Alpha,Aj2,Aj1)
                            Aj2_compat = Phi(dist)
                            Aj1_compat = Phi(dist)
                            pair_cap = int(row['Performance'])
                            Aj_entry = [Aj2,Aj1,Aj2_compat,Aj1_compat,pair_cap]
                            A_matrix.append(Aj_entry)
                    for Ai1 in team_Ai_lineup:
                        dist = path_d(G_Alpha,player,Ai1)
                        Aj1_compat = Phi(dist)
                        Ai1_compat = Phi(dist)
                        adv_cap = int(row['Performance'])
                        Aj_entry = [player,Ai1,Aj1_compat,Ai1_compat,adv_cap]
                        A_matrix.append(Aj_entry)
    A_M1 = pd.DataFrame(data=A_matrix,columns=['PlayerOne','PlayerTwo','i Dist','j Dist','Performance'])
    
    return(A_M1)

# Create a graph from the Lineup DataFrame to use as training data.
def SimGraph(D_Train,lineup_df):
    
    Agents = list(D_Train)

    # Create a random set of edges between nodes using BFS and form a new Graph

    G_Alpha = bfs_cc(D_Train)
    
    #Learn Agent Capabilites from Graph Distances
    
    A_M1 = Cap_Matrix(G_Alpha,lineup_df)
    
    Agent_Capabilities = []
    
    for player in Agents:
        player_name = str(player)
        X = A_M1[(A_M1.PlayerOne == player_name)][['i Dist','j Dist']]
        Y = A_M1[(A_M1.PlayerOne == player_name)][['Performance']]
        model = sm.OLS(Y,X).fit()
        i_cap,j_cap = model.params
        player_cap_entry = [player_name,round(i_cap,3)]
        Agent_Capabilities.append(player_cap_entry)
        G_Alpha.add_node(player_name,G_Cap=round(i_cap,3))
    
    Agent_Cap_DF = pd.DataFrame(data= Agent_Capabilities, columns=['Player One','G_Cap'])
    
    return(G_Alpha, Agent_Cap_DF)

# Find the synergy between agents who are on the same team.
def Synergy(G_Alpha,team_Ai_lineup,team_Aj_lineup):
    if list(G_Alpha.nodes(data='G_Cap'))[0][1] is None:
        return(print('G_Cap data missing')) 
    CofAi = [x for x in combinations(team_Ai_lineup,r=2)]
    CofAj = [x for x in combinations(team_Aj_lineup,r=2)]
    agent_pairs = np.array(['Team','Player','Teammate','Dist','Syn of Pair']).reshape(1,5)
    for team in CofAi:
        team_name = 'Ai'
        player = team[0]
        teammate = team[1]
        dist = path_d(G_Alpha,player,teammate)
        compat_mu = G_Alpha.nodes[player]['G_Cap']+G_Alpha.nodes[teammate]['G_Cap']
        pair_mu = Phi(dist)*compat_mu
        entry = np.array([team_name, player, teammate, dist, round(pair_mu,3)]).reshape(1,5)
        agent_pairs = np.concatenate([agent_pairs,entry], axis=0)
    for team in CofAj:
        team_name = 'Aj'
        player = team[0]
        teammate = team[1]
        dist = path_d(G_Alpha,player,teammate)
        compat_mu = G_Alpha.nodes[player]['G_Cap']+G_Alpha.nodes[teammate]['G_Cap']
        pair_mu = Phi(dist)*compat_mu
        entry = np.array([team_name,player,teammate,dist,round(pair_mu,4)]).reshape(1,5)
        agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    syn_df = pd.DataFrame(agent_pairs[1:],columns=['Team','Player One','Player Two','Dist','Syn'])
    return(syn_df)
    
# Find the Adversarial Synergy between agents on opposing teams.
def A_Synergy(G_Alpha,team_Ai_lineup,team_Aj_lineup):
    if list(G_Alpha.nodes(data='G_Cap'))[0][1] is None:
        return(print('G_Cap data missing')) 
    agent_pairs = np.array(['Team','i','j','Dist','Ai_Aj_Adv']).reshape(1,5)
    for agent in team_Ai_lineup:
        team_name = 'Ai'
        agent_mu = G_Alpha.nodes[agent]['G_Cap']
        for adversary in team_Aj_lineup:
            adversary_mu = G_Alpha.nodes[adversary]['G_Cap']
            dist = path_d(G_Alpha,agent,adversary)
            Ai_Aj_Adv = Phi(dist)*(agent_mu-adversary_mu)
            entry = np.array([team_name,agent,adversary,dist,round(Ai_Aj_Adv,4)]).reshape(1,5)
            agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    for adversary in team_Aj_lineup:
        team_name = 'Aj'
        adversary_mu = G_Alpha.nodes[adversary]['G_Cap']
        for agent in team_Ai_lineup:
            agent_mu = G_Alpha.nodes[agent]['G_Cap']
            dist = path_d(G_Alpha,agent,adversary)
            Ai_Aj_Adv = Phi(dist)*(agent_mu-adversary_mu)
            entry = np.array([team_name,adversary,agent,dist,round(Ai_Aj_Adv,4)]).reshape(1,5)
            agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    adv_syn_df = pd.DataFrame(agent_pairs[1:],columns=['Team','i','j','Dist','Ai_Aj_Adv'])
    return(adv_syn_df)

# Measure the error of the predictions from the model.
def MeasureError(G_Alpha,lineup_df):
    model_error = []
    predictions = []
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        team_Ai_lineup = sorted(list(row)[0:5])
        team_Aj_lineup = sorted(list(row)[5:10])
        K = 1/(len(team_Ai_lineup+team_Aj_lineup)/2)
        syn_df = Synergy(G_Alpha,team_Ai_lineup,team_Aj_lineup)
        adv_syn_df = A_Synergy(G_Alpha,team_Ai_lineup,team_Aj_lineup)
        Ai_syn = np.sum(np.array(syn_df[(syn_df['Team'] == 'Ai')]['Syn'],dtype=float))
        Aj_syn = np.sum(np.array(syn_df[(syn_df['Team'] == 'Aj')]['Syn'],dtype=float))
        Ai_Aj_Adv = np.sum(np.array(adv_syn_df[(adv_syn_df['Team'] == 'Ai')]['Ai_Aj_Adv'],dtype=float))
        prediction = (Ai_syn*K)-(Aj_syn*K)+(Ai_Aj_Adv*K)
        sq_error = (prediction-row['Performance'])**2
        predictions.append(prediction)
        model_error.append(sq_error)
    spread_sq_error = np.mean(model_error)
    Alpha_performance = spread_sq_error**.5
    Alpha_model_predictions = pd.Series(data=predictions,name='Prediction')
    Alpha_model_error = pd.Series(data=model_error,name='Sq Error')
    predictions_df = pd.concat([lineup_df,Alpha_model_predictions,Alpha_model_error],axis=1)
    return(round(Alpha_performance,3),predictions_df)

# Project the Performance, or in the case of NBA the "spread"

def ProjSpread(G_Alpha,team_Ai_lineup,team_Aj_lineup):
    # Using the A_Synergy and Synergy module in the main script
    
    K = 1/(len(team_Ai_lineup+team_Aj_lineup)/2)
    
    adv_syn_df = A_Synergy(G_Alpha,team_Ai_lineup,team_Aj_lineup)
    
    syn_df = Synergy(G_Alpha,team_Ai_lineup,team_Aj_lineup)
    
    Ai_Syn = np.sum(np.array(syn_df[(syn_df.Team == 'Ai' ) ]['Syn'],dtype=float))
    Aj_Syn = np.sum(np.array(syn_df[(syn_df.Team == 'Aj' ) ]['Syn'],dtype=float))
    Ai_Aj_Adv = np.sum(np.array(adv_syn_df[(adv_syn_df.Team == 'Ai' ) ]['Ai_Aj_Adv'],dtype=float))
    
    Ai_Aj_Syn = round(((Ai_Syn) - (Aj_Syn) + (Ai_Aj_Adv))*K,2)
    
    Adv_Syn_Spread = Ai_Aj_Syn
    
    return Adv_Syn_Spread


# Regress the performance of each lineup to the pairwise distance between agents.
def LineupSyn(G_Alpha,lineup_df):  
    
    Ai_lineups = []
    Aj_lineups = []
    
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        team_Ai_lineup = sorted(list(row[0:5]))
        team_Aj_lineup = sorted(list(row[5:10]))
        if team_Ai_lineup not in Ai_lineups:    
            Ai_lineups.append(team_Ai_lineup)
        if team_Aj_lineup not in Aj_lineups:
            Aj_lineups.append(team_Aj_lineup)
    
    team_Ai = list(np.unique(Ai_lineups))
    team_Aj = list(np.unique(Aj_lineups))
    K = 1/(len(team_Ai+team_Aj)/2)
    Ai_matrix = []
    
    Ai_lineup_id = 0
    for Ai_lineup in Ai_lineups:
        Ai_lineup_index = str(Ai_lineup_id)
        Ai_combos = list(combinations(Ai_lineup,r=2))
        Ai_dist = []
        for Ai1,Ai2 in Ai_combos:
            dist = path_d(G_Alpha,Ai1,Ai2)
            Pair_compat = Phi(dist)
            Ai_dist.append(Pair_compat)
        Ai_dist_array = np.array(Ai_dist,dtype=float)
        Ai_mean_dist= round(np.sum(Ai_dist_array)*K,3)
        Ai_entry = ['Ai',Ai_lineup_index,Ai_mean_dist]
        Ai_matrix.append(Ai_entry)
        Ai_lineup_id +=1
        
    Aj_matrix = []
    
    Aj_lineup_id = 0
    for Aj_lineup in Aj_lineups:
        Aj_lineup_index = str(Aj_lineup_id)
        Aj_combos = list(combinations(Aj_lineup,r=2))
        Aj_dist = []
        for Aj1,Aj2 in Aj_combos:
            dist = path_d(G_Alpha,Aj1,Aj2)
            Pair_compat = Phi(dist)
            Aj_dist.append(Pair_compat)
        Aj_dist_array = np.array(Aj_dist,dtype=float)
        Aj_mean_dist= round(np.sum(Aj_dist_array)*K,3)
        Aj_entry = ['Aj',Aj_lineup_index,Aj_mean_dist]
        Aj_matrix.append(Aj_entry)
        Aj_lineup_id +=1
        
    Ai_scoring_matrix = []
    for row in Ai_matrix:
        Ai_index = int(row[1])
        compat = row[2]
        players = Ai_lineups[Ai_index]
        for i in range(len(lineup_df)):
            df_row = lineup_df.iloc[i]
            ai_check = sorted(list(df_row[0:5]))
            aj_check = sorted(list(df_row[5:10]))
            if players == ai_check:
                Aj_index = Aj_lineups.index(aj_check)
                j_compat = Aj_matrix[Aj_index][2]
                ai_aj_adv = []
                for ai_agent in ai_check:
                    for aj_agent in aj_check:
                        dist = path_d(G_Alpha,ai_agent,aj_agent)
                        ai_aj_compat = Phi(dist)
                        ai_aj_adv.append(ai_aj_compat)
                adv_syn = round(np.sum(ai_aj_adv)*K,3)
                entry = players+aj_check+[compat]+[j_compat]+[adv_syn]+[df_row['Performance']]
                Ai_scoring_matrix.append(entry)
    
    column_headers = ['Ai1','Ai2','Ai3','Ai4','Ai5','Aj1','Aj2','Aj3','Aj4','Aj5','Ai Compat','Aj Compat','Ai Aj Compat','Performance']
    Ai_matrix_df = pd.DataFrame(data=Ai_scoring_matrix,columns=column_headers)
    X = Ai_matrix_df[['Ai Compat','Aj Compat','Ai Aj Compat']]
    Y = Ai_matrix_df['Performance']
    model = sm.OLS(Y,X).fit()
    Ai_Syn_coef, Aj_Syn_coef, Ai_Aj_Adv_Syn_coef = model.params
    Ai_Syn = np.sum(Ai_matrix_df['Ai Compat'])
    Aj_Syn = np.sum(Ai_matrix_df['Aj Compat'])
    Ai_Aj_Adv = np.sum(Ai_matrix_df['Ai Aj Compat'])
    R_Spread = round(((Ai_Syn*Ai_Syn_coef)+(Aj_Syn*Aj_Syn_coef)+(Ai_Aj_Adv*Ai_Aj_Adv_Syn_coef)),2)
    return(Ai_matrix_df, Ai_Syn_coef, Aj_Syn_coef, Ai_Aj_Adv_Syn_coef, R_Spread)

# Evaluate the mean synergy for each agent.
def AlphaBPM(syn_df,Player):
    player_name = str(Player)
    player_df = syn_df[(syn_df['Player One'] == player_name) | (syn_df['Player Two'] == player_name)]
    player_syn = np.array(player_df['Syn'], dtype=float)
    player_mu = np.mean(player_syn)
    return(player_mu)
