#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:01:55 2018
@author: Degentleman
"""

import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
import statsmodels.api as sm

# Common path functions:
path = nx.dijkstra_path

#Returns the distance between two agents in a graph, resulting in an int of E
path_d = nx.dijkstra_path_length

# Returns the shortest path between two agents in a graph, resulting in a list
s_path = nx.shortest_path

def Phi(d):
    return(1/float(d))
    
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
            j_performance = -float(str(list(row)[10]))
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

def Cap_Matrix(G_Alpha, lineup_df):
    Agents = list(G_Alpha)
    A_matrix = []
    for player in Agents:
        team = G_Alpha.node[player]['Team']
        for i in range(len(lineup_df)):
            row = lineup_df.iloc[i]
            if player in list(row)[0:10]:
                team_Ai_lineup = sorted(list(row[0:5]))
                team_Aj_lineup = sorted(list(row[5:10]))
                if player in team_Ai_lineup:
                    for Ai2 in team_Ai_lineup:
                        if Ai2 != player:
                            dist = path_d(G_Alpha,player,Ai2)
                            Ai1_compat = Phi(dist)
                            Ai2_compat = Phi(dist)
                            pair_cap = int(row.Performance)
                            Ai_entry = [team, team, player,Ai2,Ai1_compat,Ai2_compat,pair_cap]
                            A_matrix.append(Ai_entry)
                    for Aj1 in team_Aj_lineup:
                        opp = G_Alpha.node[Aj1]['Team']
                        dist = path_d(G_Alpha,player,Aj1)
                        Ai1_compat = Phi(dist)
                        Aj1_compat = Phi(dist)
                        adv_cap = int(row.Performance)
                        Ai_entry = [team, opp, player, Aj1, Ai1_compat, Aj1_compat, adv_cap]
                        A_matrix.append(Ai_entry)
                if player in team_Aj_lineup:
                    for Aj2 in team_Aj_lineup:
                        if Aj2 != player:
                            dist = path_d(G_Alpha,player,Aj2)
                            Aj1_compat = Phi(dist)
                            Aj2_compat = Phi(dist)
                            pair_cap = -int(row.Performance)
                            Aj_entry = [team, team, player, Aj2, Aj1_compat, Aj2_compat, pair_cap]
                            A_matrix.append(Aj_entry)
                    for Ai1 in team_Ai_lineup:
                        opp = G_Alpha.node[Ai1]['Team']
                        dist = path_d(G_Alpha,player,Ai1)
                        Ai1_compat = Phi(dist)
                        Aj1_compat = Phi(dist)
                        adv_cap = -int(row.Performance)
                        Aj_entry = [team, opp, player, Ai1, Ai1_compat, Aj1_compat, adv_cap]
                        A_matrix.append(Aj_entry)
    A_M1 = pd.DataFrame(data= A_matrix, columns=['Team_One', 'Team_Two', 'Player_One','Player_Two','i Dist','j Dist','Performance'])
    return(A_M1)

def SimGraph(D_Train,lineup_df):
    
    # Identify players included in training data
    Agents = list(D_Train)

    # Create a random set of edges between nodes using BFS and form a new Graph
    G_Alpha = bfs_cc(D_Train)
    
    #Learn Agent Capabilites from Graph Distances
    A_M1 = Cap_Matrix(G_Alpha,lineup_df)
    
    Agent_Capabilities = []
    
    for player in Agents:
        player_name = str(player)
        player_X = A_M1[(A_M1.Player_One == player_name)][['i Dist','j Dist']]
        player_Y = A_M1[(A_M1.Player_One == player_name)][['Performance']]
        model = sm.OLS(player_Y,player_X).fit()
        i_cap,j_cap = model.params
        player_cap_entry = [player_name,round(i_cap,4)]
        Agent_Capabilities.append(player_cap_entry)
        G_Alpha.add_node(player_name, P_Cap=round(i_cap,4))
    
    Agent_Cap_DF = pd.DataFrame(data= Agent_Capabilities, columns=['Player_One','P_Cap'])
    
    return(G_Alpha, Agent_Cap_DF, model)
    
def Synergy(G_Alpha, home_lineup, away_lineup, home_team, away_team):
    if list(G_Alpha.nodes(data='Mu'))[0][1] is None:
        return(print('Mu data missing'))
    CofAi = [x for x in combinations(home_lineup,r=2)]
    CofAj = [x for x in combinations(away_lineup,r=2)]  
    agent_pairs = np.array(['Team','Player','Teammate','Dist','Mu_Syn','P_Cap_Syn']).reshape(1,6)
    
    for team in CofAi:
        team_name = home_team
        player = team[0]
        teammate = team[1]
        dist = path_d(G_Alpha,player,teammate)
        compat_mu = G_Alpha.nodes[player]['Mu']+G_Alpha.nodes[teammate]['Mu']
        compat_p_cap = G_Alpha.node[player]['P_Cap']+G_Alpha.node[teammate]['P_Cap']
        pair_mu = Phi(dist)*compat_mu
        pair_p_cap = Phi(dist)*compat_p_cap
        entry = np.array([team_name, player, teammate, dist, round(pair_mu,3),round(pair_p_cap,3)]).reshape(1,6)
        agent_pairs = np.concatenate([agent_pairs,entry], axis=0)
    for team in CofAj:
        team_name = away_team
        player = team[0]
        teammate = team[1]
        dist = path_d(G_Alpha,player,teammate)
        compat_mu = G_Alpha.nodes[player]['Mu']+G_Alpha.nodes[teammate]['Mu']
        compat_p_cap = G_Alpha.nodes[player]['P_Cap']+G_Alpha.nodes[teammate]['P_Cap']
        pair_mu = Phi(dist)*compat_mu
        pair_p_cap = Phi(dist)*compat_p_cap
        entry = np.array([team_name, player, teammate, dist, round(pair_mu,3),round(pair_p_cap,3)]).reshape(1,6)
        agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    syn_df = pd.DataFrame(agent_pairs[1:],columns=['Team','Player One','Player Two','Dist','Mu_Syn','Syn'])
    return(syn_df)
    
def A_Synergy(G_Alpha, home_lineup, away_lineup, home_team, away_team):
    if list(G_Alpha.nodes(data='Mu'))[0][1] is None:
        return(print('Mu data missing')) 
    agent_pairs = np.array(['Team','i','j','Dist','Mu_Adv','Adv']).reshape(1,6)
    
    for agent in home_lineup:
        team_name = home_team
        agent_mu = G_Alpha.nodes[agent]['Mu']
        agent_p_cap = G_Alpha.nodes[agent]['P_Cap']
        for adversary in away_lineup:
            adversary_mu = G_Alpha.nodes[adversary]['Mu']
            adversary_p_cap = G_Alpha.nodes[adversary]['P_Cap']
            dist = path_d(G_Alpha,agent,adversary)
            Ai_Aj_Adv = Phi(dist)*(agent_p_cap-adversary_p_cap)
            Ai_Aj_Adv_mu = Phi(dist)*(agent_mu-adversary_mu)
            entry = np.array([team_name,agent,adversary,dist,round(Ai_Aj_Adv_mu,4),round(Ai_Aj_Adv,4)]).reshape(1,6)
            agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    for adversary in away_lineup:
        team_name = away_team
        adversary_mu = G_Alpha.nodes[adversary]['Mu']
        adversary_p_cap = G_Alpha.nodes[adversary]['P_Cap']
        for agent in home_lineup:
            agent_mu = G_Alpha.nodes[agent]['Mu']
            agent_p_cap = G_Alpha.nodes[agent]['P_Cap']
            dist = path_d(G_Alpha,adversary,agent)
            Aj_Ai_Adv = Phi(dist)*(adversary_p_cap-agent_p_cap)
            Aj_Ai_Adv_mu = Phi(dist)*(adversary_mu-agent_mu)
            entry = np.array([team_name, adversary, agent, dist,round(Aj_Ai_Adv_mu,4),round(Aj_Ai_Adv,4)]).reshape(1,6)
            agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    adv_syn_df = pd.DataFrame(agent_pairs[1:],columns=['Team','i','j','Dist','Mu_Adv','Adv'])
    return(adv_syn_df)

def MeasureError(G_Alpha, lineup_df):
    
    model_error = []
    mu_error = []
    p_cap_predictions = []
    mu_predictions = []
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        home_lineup = sorted(list(row[0:5]))
        away_lineup = sorted(list(row[5:10]))
        home_team, away_team = teamNSA.team_vote(home_lineup, away_lineup)
        syn_df, adv_syn_df, p_cap_proj, mu_proj = ProjSpread(G_Alpha, home_lineup, away_lineup, home_team, away_team)
        sq_error = (p_cap_proj-row['Performance'])**2
        mu_sq_error = (mu_proj-row['Performance'])**2
        p_cap_predictions.append(p_cap_proj)
        mu_predictions.append(mu_proj)
        model_error.append(sq_error)
        mu_error.append(mu_sq_error)
    spread_sq_error = np.sum(model_error)
    mu_sq_error = np.sum(mu_error)
    Alpha_performance = spread_sq_error**.5
    mu_performance = mu_sq_error**.5
    p_cap_model_predictions = pd.Series(data=p_cap_predictions, name='Prediction')
    mu_model_predictions = pd.Series(data=mu_predictions, name='Mu_Prediction')
    p_cap_model_error = pd.Series(data=model_error, name='Sq_Error')
    mu_model_error = pd.Series(data=mu_error, name='Mu_Sq_Error')
    predictions_df = pd.concat([lineup_df, mu_model_predictions, mu_model_error, p_cap_model_predictions, p_cap_model_error],axis=1)
    return(round(Alpha_performance,4), round(mu_performance,4), predictions_df)

def ProjSpread(G_Alpha, home_lineup, away_lineup, home_team, away_team):
    K = 1/(len(home_lineup+away_lineup)/2)
    # Using the A_Synergy and Synergy module in the main script
    syn_df = Synergy(G_Alpha, home_lineup, away_lineup, home_team, away_team)
    adv_syn_df = A_Synergy(G_Alpha, home_lineup, away_lineup, home_team, away_team)

    #Using linear regression to find capability coefficient for agent
    Ai_Syn = round(np.sum(np.array(syn_df[(syn_df.Team == home_team ) ]['Syn'],dtype=float)),3)
    Aj_Syn = round(np.sum(np.array(syn_df[(syn_df.Team == away_team ) ]['Syn'],dtype=float)),3)
    Ai_Aj_Adv = np.sum(np.array(adv_syn_df[(adv_syn_df.Team == home_team ) ]['Adv'],dtype=float))
    Ai_Aj_Syn = Ai_Syn - Aj_Syn + Ai_Aj_Adv
    Adv_Syn_Spread = round(Ai_Aj_Syn*K,3)
    #Using mean of known lineup
    Ai_Syn_mu = round(np.sum(np.array(syn_df[(syn_df.Team == home_team ) ]['Mu_Syn'],dtype=float)),3)
    Aj_Syn_mu = round(np.sum(np.array(syn_df[(syn_df.Team == away_team ) ]['Mu_Syn'],dtype=float)),3)
    Ai_Aj_Adv_mu = np.sum(np.array(adv_syn_df[(adv_syn_df.Team == home_team) ]['Mu_Adv'],dtype=float))
    Ai_Aj_Syn_mu = (Ai_Syn_mu) - (Aj_Syn_mu) + (Ai_Aj_Adv_mu)
    Adv_Syn_Spread_mu = round(Ai_Aj_Syn_mu*K,3)
    
    return (syn_df, adv_syn_df, Adv_Syn_Spread, Adv_Syn_Spread_mu)

def LineupSyn(G_Alpha, lineup_df, K):  
    
    Ai_entries = []
    Aj_entries = [] 
    Ai_lineups = []
    Aj_lineups = []
    
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        team_Ai_lineup = sorted(list(row[0:5]))
        team_Aj_lineup = sorted(list(row[5:10]))
        
        if team_Ai_lineup not in Ai_lineups:
            home_team, away_team = teamNSA.team_vote(team_Ai_lineup, team_Aj_lineup)
            Ai_lineups.append(team_Ai_lineup)
            Ai_entries.append([home_team]+team_Ai_lineup)
            
        if team_Aj_lineup not in Aj_lineups:
            home_team, away_team = teamNSA.team_vote(team_Ai_lineup, team_Aj_lineup)
            Aj_lineups.append(team_Aj_lineup)
            Aj_entries.append([away_team]+team_Aj_lineup)
    
    Ai_matrix = []
    
    Ai_lineup_id = 0
    for Ai_lineup in Ai_entries:
        home_team = Ai_lineup[0]
        Ai_lineup_index = str(Ai_lineup_id)
        Ai_combos = list(combinations(Ai_lineup[1:],r=2))
        Ai_dist = []
        for Ai1,Ai2 in Ai_combos:
            dist = path_d(G_Alpha,Ai1,Ai2)
            Pair_compat = Phi(dist)
            Ai_dist.append(Pair_compat)
        Ai_dist_array = np.array(Ai_dist,dtype=float)
        Ai_mean_dist= round(np.sum(Ai_dist_array),4)
        Ai_entry = [home_team,Ai_lineup_index,Ai_mean_dist]
        Ai_matrix.append(Ai_entry)
        Ai_lineup_id +=1
        
    Aj_matrix = []
    
    Aj_lineup_id = 0
    for Aj_lineup in Aj_entries:
        away_team = Aj_lineup[0]
        Aj_lineup_index = str(Aj_lineup_id)
        Aj_combos = list(combinations(Aj_lineup[1:],r=2))
        Aj_dist = []
        for Aj1,Aj2 in Aj_combos:
            dist = path_d(G_Alpha,Aj1,Aj2)
            Pair_compat = Phi(dist)
            Aj_dist.append(Pair_compat)
        Aj_dist_array = np.array(Aj_dist,dtype=float)
        Aj_mean_dist= round(np.sum(Aj_dist_array),4)
        Aj_entry = [away_team,Aj_lineup_index,Aj_mean_dist]
        Aj_matrix.append(Aj_entry)
        Aj_lineup_id +=1
        
    Ai_scoring_matrix = []
    for row in Ai_matrix:
        Ai_index = int(row[1])
        compat = row[2]
        players = Ai_lineups[Ai_index]
        team = str(row[0])
        for i in range(len(lineup_df)):
            df_row = lineup_df.iloc[i]
            ai_check = sorted(list(df_row[0:5]))
            aj_check = sorted(list(df_row[5:10]))
            if players == ai_check:
                Aj_index = Aj_lineups.index(aj_check)
                j_compat = Aj_matrix[Aj_index][2]
                opp = Aj_matrix[Aj_index][0]
                ai_aj_adv = []
                for ai_agent in ai_check:
                    for aj_agent in aj_check:
                        dist = path_d(G_Alpha,ai_agent,aj_agent)
                        ai_aj_compat = Phi(dist)
                        ai_aj_adv.append(ai_aj_compat)
                adv_syn = round(np.sum(ai_aj_adv),4)
                entry = [team,opp]+players+aj_check+[compat]+[j_compat]+[adv_syn]+[df_row['Performance']]
                Ai_scoring_matrix.append(entry)
    
    column_headers = ['Team_Ai','Team_Aj','Team_Ai1','Team_Ai2','Team_Ai3','Team_Ai4','Team_Ai5','Team_Aj1','Team_Aj2','Team_Aj3','Team_Aj4','Team_Aj5','Ai_Compat','Aj_Compat','Ai_Aj_Compat','Performance']
    Ai_matrix_df = pd.DataFrame(data=Ai_scoring_matrix,columns=column_headers)
    return(Ai_matrix_df)

def AlphaBPM(syn_df,Player):
    player_name = str(Player)
    player_df = syn_df[(syn_df['Player One'] == player_name) | (syn_df['Player Two'] == player_name)]
    player_syn = np.array(player_df['Syn'], dtype=float)
    player_mu = np.mean(player_syn)
    return(player_mu)
