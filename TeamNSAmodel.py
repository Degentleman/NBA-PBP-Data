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
from NBA_Player_DB import GetDB
Team_Legend = pd.read_csv('NBA PBP - Team Legend.csv', delimiter = ',')   

NBA_Legend = pd.read_csv('NBA Player DF - 2019.csv', delimiter = ',')   

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
        node = queue.pop(-1)
        
        if node not in explored:
            explored.append(node)
            if len(queue)>0:
                neighbor = queue[-1]
                if G_prime.has_edge(node,neighbor) == False:
                    weight = 1
                    G_prime.add_edge(node,neighbor,weight=weight)
                    if len(queue) == 0:
                        break
    return (G_prime)

def lineups(DataFrame):
    
    csv_lineups = []
    
    for i in range(len(DataFrame)):
        entry = []
        lu_start = list(DataFrame).index('Player Two')+1
        col_keys = list(DataFrame)[lu_start:]
        row = DataFrame.iloc[i][col_keys]
        performance = row.Performance
        for key in col_keys[:-1]:
            if row[key] == "In":
                entry.append(key)
        if len(entry) != 10:
            print('There are not 10 Players In @ Loc #' + str(i))
            break
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


def team_vote(team_Ai, team_Aj):
   
    ai_votes = []
    for player in team_Ai:
        if player in list(NBA_Legend.Player):
            row = NBA_Legend[(NBA_Legend.Player == player)]
            ai_vote = row['Team'].values[0]
            ai_votes.append(ai_vote)
    ai_tms, ai_vts = np.unique(ai_votes, return_counts=True)
    ai_leader = 0
    ai_decision = []
    for i in range(len(list(ai_tms))):
        ai_tm_count = int(ai_vts[i])
        if ai_tm_count > ai_leader:
            ai_decision = ai_tms[i]
            ai_leader = ai_tm_count
    home_team = ai_decision
    aj_votes = []
    for player in team_Aj:
        if player in list(NBA_Legend.Player):
            row = NBA_Legend[(NBA_Legend.Player == player)]
            aj_vote = row['Team'].values[0]
            aj_votes.append(aj_vote)
    aj_tms, aj_vts = np.unique(aj_votes, return_counts=True)
    aj_leader = 0
    aj_decision = []
    for i in range(len(list(aj_tms))):
        aj_tm_count = int(aj_vts[i])
        if aj_tm_count > aj_leader:
            aj_decision = aj_tms[i]
            aj_leader = aj_tm_count
    away_team = aj_decision
    return(home_team, away_team)

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

def LearnCap(lineup_df):
    
    # Learn the mean value of each player from the lineup DataFrame
    value_list = []
    NBA = nx.Graph()
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        team_Ai_lineup = list(row)[0:5]
        team_Aj_lineup = list(row)[5:10]
        home_team, away_team = team_vote(team_Ai_lineup, team_Aj_lineup)
        if NBA.has_edge(home_team, away_team) == False:
            NBA.add_edge(home_team, away_team, weight=1)
        for player_i in team_Ai_lineup:
            i_performance = row[10]
            value_list.append([home_team, player_i,i_performance])
        for player_j in team_Aj_lineup:
            j_performance = row[10]
            value_list.append([away_team, player_j,j_performance])
    value_df = pd.DataFrame(data=value_list, columns=['Team','Player','Performance'])
    
    Agents = list(np.unique(value_df['Player']))
    
    final_value = []
    
    for x in Agents:
        player_name = str(x)
        player_in = value_df[(value_df['Player'] == player_name)]
        mean_performance = np.mean(player_in['Performance'])
        var_performance = np.var(player_in['Performance'],ddof=1)
        team = player_in.Team.iloc[0]
        final_entry = [team, player_name, mean_performance, var_performance]
        final_value.append(final_entry)
    
    final_value_df = pd.DataFrame(data=final_value,columns=['Team', 'Player', 'Mean_Perf', 'Var_Perf' ]).sort_values(by=['Team','Player'])
    
    
    
    # Add training data as nodes to an Unweighted Graph
    
    D_Train = nx.Graph()
    
    # Add the training data to the NBA graph as player nodes
    
    for i in range(len(final_value_df)):
        row = final_value_df.iloc[i]
        name = row['Player']
        player_mu = row['Mean_Perf']
        player_var = row['Var_Perf']
        team = row['Team']
        D_Train.add_node(name, Team=team, Mu=player_mu, Var=player_var)
    
    return(D_Train, NBA, final_value_df)
    
def SimGraph(D_Train, NBA, lineup_df):
    # Create a random set of edges between nodes using BFS and form a new Graph
    G_Alpha = bfs_cc(D_Train)
    
    #Learn Agent Capabilites from Graph Distances    
    A_M1 = Cap_Matrix(G_Alpha,lineup_df)
    
    Team_Capabilities = []
    
    Teams = list(NBA)
    
    for team in Teams:
        team_name = str(team)
        opps = list(np.unique(A_M1[(A_M1.Team_One == team_name) & (A_M1.Team_Two != team_name)].Team_Two))
        X = A_M1[(A_M1.Team_One == team_name) & (A_M1.Team_Two == team_name)][['i Dist','j Dist']]
        Y = A_M1[(A_M1.Team_One == team_name) & (A_M1.Team_Two == team_name)][['Performance']]
        model = sm.OLS(Y,X).fit()
        i_cap,j_cap = model.params
        NBA.add_node(team_name, Team_Syn=round(i_cap,4))
    
        for opp in opps:
            X_b = A_M1[(A_M1.Team_One == team_name) & (A_M1.Team_Two == opp)][['i Dist','j Dist']]
            Y_b = A_M1[(A_M1.Team_One == team_name) & (A_M1.Team_Two == opp)][['Performance']]        
            model_b = sm.OLS(Y_b,X_b).fit()
            i_adv_cap,j_adv_cap = model_b.params
            NBA.add_edge(team_name, opp, Adv_Syn=round(i_cap,4))
            team_cap_entry = [team_name, opp, round(i_cap,4), round(i_adv_cap,4)]
            Team_Capabilities.append(team_cap_entry)
            
    NBA_JAM = NBA
    
    Team_Cap_DF = pd.DataFrame(data= Team_Capabilities, columns=['Team','Opp', 'Team_Syn','Adv_Syn'])
    
    Agents = list(G_Alpha)
    
    Agent_Capabilities = []
    
    for player in Agents:
        player_name = str(player)
        X = A_M1[(A_M1.Player_One == player_name)][['i Dist','j Dist']]
        Y = A_M1[(A_M1.Player_One == player_name)][['Performance']]
        model = sm.OLS(Y,X).fit()
        i_cap,j_cap = model.params
        player_cap_entry = [player_name,round(i_cap,4)]
        Agent_Capabilities.append(player_cap_entry)
        G_Alpha.add_node(player_name, P_Cap=round(i_cap,4))
    Agent_Cap_DF = pd.DataFrame(data= Agent_Capabilities, columns=['Player One','P_Cap'])
    
    return(G_Alpha, NBA_JAM, A_M1, Team_Cap_DF, Agent_Cap_DF, model)

def Cap_Matrix(G_Alpha, lineup_df):  
    #Performance only adjusted for Team_Ai
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
                        Ai_entry = [team, opp, player,Aj1,Ai1_compat,Aj1_compat,adv_cap]
                        A_matrix.append(Ai_entry)
                if player in team_Aj_lineup:
                    for Aj2 in team_Aj_lineup:
                        if Aj2 != player:
                            dist = path_d(G_Alpha,player,Aj2)
                            Aj1_compat = Phi(dist)
                            Aj2_compat = Phi(dist)
                            pair_cap = -int(row.Performance)
                            Aj_entry = [team, team, player,Aj2,Aj1_compat,Aj2_compat,pair_cap]
                            A_matrix.append(Aj_entry)
                    for Ai1 in team_Ai_lineup:
                        opp = G_Alpha.node[Ai1]['Team']
                        dist = path_d(G_Alpha,player,Ai1)
                        Ai1_compat = Phi(dist)
                        Aj1_compat = Phi(dist)
                        adv_cap = -int(row.Performance)
                        Aj_entry = [team, opp, player,Ai1,Ai1_compat,Aj1_compat,adv_cap]
                        A_matrix.append(Aj_entry)
    A_M1 = pd.DataFrame(data=A_matrix,columns=['Team_One','Team_Two', 'Player_One','Player_Two','i Dist','j Dist','Performance'])
    
    return(A_M1)
    
def Synergy(G_Alpha, home_lineup, away_lineup, home_team, away_team):
    if list(G_Alpha.nodes(data='Mu'))[0][1] is None:
        return(print('Mu data missing'))
    CofAi = [x for x in combinations(home_lineup,r=2)]
    CofAj = [x for x in combinations(away_lineup,r=2)]  
    agent_pairs = np.array(['Team','Player','Teammate','Dist','mu Syn','P_Cap Syn']).reshape(1,6)
    
    for team in CofAi:
        team_name = home_team
        player = team[0]
        teammate = team[1]
        dist = path_d(G_Alpha,player,teammate)
        compat_mu = G_Alpha.nodes[player]['Mu']+G_Alpha.nodes[teammate]['Mu']
        compat_p_cap = G_Alpha.nodes[player]['P_Cap']+G_Alpha.nodes[teammate]['P_Cap']
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
    agent_pairs = np.array(['Team','i','j','Dist','mu_Adv','Adv']).reshape(1,6)
    
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
            dist = path_d(G_Alpha,agent,adversary)
            Aj_Ai_Adv = Phi(dist)*(adversary_p_cap-agent_p_cap)
            Aj_Ai_Adv_mu = Phi(dist)*(adversary_mu-agent_mu)
            entry = np.array([team_name, adversary, agent, dist,round(Aj_Ai_Adv_mu,4),round(Aj_Ai_Adv,4)]).reshape(1,6)
            agent_pairs = np.concatenate([agent_pairs,entry],axis=0)
    adv_syn_df = pd.DataFrame(agent_pairs[1:],columns=['Team','i','j','Dist','Mu_Adv','Adv'])
    return(adv_syn_df)
    
def ProjSpread(G_Alpha, home_lineup, away_lineup, home_team, away_team):
    syn_df = Synergy(G_Alpha, home_lineup, away_lineup, home_team, away_team)
    adv_syn_df = A_Synergy(G_Alpha, home_lineup, away_lineup, home_team, away_team)
    K = 1/(len(home_lineup+away_lineup)/2)
    #Using linear regression to find capability coefficient for agent
    Ai_Syn = np.sum(np.array(syn_df[(syn_df.Team == home_team ) ]['Syn'],dtype=float))*K
    Aj_Syn = np.sum(np.array(syn_df[(syn_df.Team == away_team ) ]['Syn'],dtype=float))*K
    Ai_Aj_Adv = np.sum(np.array(adv_syn_df[(adv_syn_df.Team == home_team ) ]['Adv'],dtype=float))*K
    Ai_Aj_Syn = Ai_Syn - Aj_Syn + Ai_Aj_Adv
    Adv_Syn_Spread = round(Ai_Aj_Syn,2)
    #Using mean of known lineup
    Ai_Syn_mu = np.sum(np.array(syn_df[(syn_df.Team == home_team ) ]['Mu_Syn'],dtype=float))*K
    Aj_Syn_mu = np.sum(np.array(syn_df[(syn_df.Team == away_team ) ]['Mu_Syn'],dtype=float))*K
    Ai_Aj_Adv_mu = np.sum(np.array(adv_syn_df[(adv_syn_df.Team == home_team) ]['Mu_Adv'],dtype=float))*K
    Ai_Aj_Syn_mu = (Ai_Syn_mu) - (Aj_Syn_mu) + (Ai_Aj_Adv_mu)
    Adv_Syn_Spread_mu = round(Ai_Aj_Syn_mu,2)
    return (syn_df, adv_syn_df, Adv_Syn_Spread, Adv_Syn_Spread_mu)

def MeasureError(G_Alpha, lineup_df):
    matchups = []
    model_error = []
    projections = []
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        home_lineup = sorted(list(row)[0:5])
        away_lineup = sorted(list(row)[5:10])
        home_team, away_team = team_vote(home_lineup, away_lineup)
        syn_df, adv_syn_df, player_model_spread, Adv_Syn_Spread_mu = ProjSpread(G_Alpha, home_lineup, away_lineup, home_team, away_team)
        projections.append(player_model_spread)
        spread = row.Performance
        error = (player_model_spread-spread)**2
        entry = [home_team, away_team, row.Performance]
        matchups.append(entry)
        model_error.append(error)
    matchups_df = pd.DataFrame(data=matchups, columns = ['Team','Opp','Performance'])
    square_error = round(np.sum(model_error)**.5,4)
    projs = pd.Series(projections, name='PL_Proj')
    errors = pd.Series(model_error, name = 'Error')
    model_results = pd.concat([matchups_df, projs, errors], axis=1)
    return(square_error, model_results)
    
def LineupSyn(G_Alpha, lineup_df, Team):
    team_id = Team_Legend[(Team_Legend.Code == Team)].TeamID.iloc[0]
    team_name, team_abbv, roster_df, coach_df, team_df = GetDB(team_id)
    # Sort through training data to filter through players included in data
    loc_list = []
    for i in range(len(lineup_df)):
        row = lineup_df.iloc[i]
        Ai_lineup = list(row)[0:5]
        Aj_lineup = list(row)[5:10]
        for player in Ai_lineup:
            if player in list(team_df.Player):
                loc_list.append(i)
        for player in Aj_lineup:
            if player in list(team_df.Player):
                loc_list.append(i)
    ul = list(np.unique(loc_list))
    DF = lineup_df.iloc[ul].reset_index(drop=True)
    Ai_entries = []
    Aj_entries = [] 
    Ai_lineups = []
    Aj_lineups = []
    for i in range(len(DF)):
        row = DF.iloc[i]
        team_Ai_lineup = sorted(list(row[0:5]))
        team_Aj_lineup = sorted(list(row[5:10]))
        home_team, away_team = team_vote(team_Ai_lineup, team_Aj_lineup)
        if team_Ai_lineup not in Ai_lineups:
            Ai_lineups.append(team_Ai_lineup)
            Ai_entries.append([home_team]+team_Ai_lineup) 
        if team_Aj_lineup not in Aj_lineups:
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
        Ai_dist_array = np.array(Ai_dist, dtype=float)
        Ai_mean_dist= np.sum(Ai_dist_array)
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
        Aj_mean_dist= np.sum(Aj_dist_array)
        Aj_entry = [away_team,Aj_lineup_index,Aj_mean_dist]
        Aj_matrix.append(Aj_entry)
        Aj_lineup_id +=1
        
    Ai_Scoring_Matrix = []
    for row in Ai_matrix:
        if str(row[0]) == Team:
            Ai_index = int(row[1])
            compat = row[2]
            players = Ai_lineups[Ai_index]
            team = str(row[0])
            Ai_matches = np.array([i for i in range(len(DF)) if (list(DF.iloc[i])[0:5] == players)], dtype=int)    
            if len(Ai_matches) > 0:
                ind_mtch = list(np.unique(Ai_matches))
                newdf = DF.iloc[ind_mtch].reset_index(drop=True)
                for i in range(len(newdf)):
                    df_row = newdf.iloc[i]
                    aj_check = sorted(list(df_row[5:10]))
                    Aj_index = Aj_lineups.index(aj_check)
                    j_compat = Aj_matrix[Aj_index][2]
                    opp = Aj_matrix[Aj_index][0]
                    ai_aj_adv = []
                    for ai_agent in players:
                        for aj_agent in aj_check:
                            dist = path_d(G_Alpha,ai_agent,aj_agent)
                            ai_aj_compat = Phi(dist)
                            ai_aj_adv.append(ai_aj_compat)
                    adv_syn = np.sum(ai_aj_adv)
                    entry = [team,opp]+players+aj_check+[compat]+[j_compat]+[adv_syn]+[df_row['Performance']]
                    Ai_Scoring_Matrix.append(entry)
    Aj_Scoring_Matrix = []
    for row in Aj_matrix:
        if str(row[0]) == Team:
            Aj_index = int(row[1])
            compat = row[2]
            players = Aj_lineups[Aj_index]
            team = str(row[0])
            Aj_matches = np.array([i for i in range(len(DF)) if (list(DF.iloc[i])[5:10] == players)], dtype=int)    
            if len(Aj_matches) > 0:
                ind_mtch = list(np.unique(Aj_matches))
                newdf = DF.iloc[ind_mtch].reset_index(drop=True)
                for i in range(len(newdf)):
                    df_row = newdf.iloc[i]
                    ai_check = sorted(list(df_row[0:5]))
                    Ai_index = Ai_lineups.index(ai_check)
                    i_compat = Ai_matrix[Ai_index][2]
                    opp = Ai_matrix[Ai_index][0]
                    ai_aj_adv = []
                    for aj_agent in players:
                        for ai_agent in ai_check:
                            dist = path_d(G_Alpha,aj_agent,ai_agent)
                            ai_aj_compat = Phi(dist)
                            ai_aj_adv.append(ai_aj_compat)
                    adv_syn = np.sum(ai_aj_adv)
                    entry = [team,opp]+players+ai_check+[compat]+[i_compat]+[adv_syn]+[-df_row['Performance']]
                    Aj_Scoring_Matrix.append(entry)
    
    column_headers = ['Team','Opp','T1','T2','T3','T4','T5','O1','O2','O3','O4','O5','Team_Compat','Opp_Compat','Adv_Compat','Performance']
    Ai_matrix_df = pd.DataFrame(data= Ai_Scoring_Matrix, columns= column_headers)
    Aj_matrix_df = pd.DataFrame(data= Aj_Scoring_Matrix, columns= column_headers)
    matrix_df = pd.concat([Ai_matrix_df[(Ai_matrix_df.Team == Team)], Aj_matrix_df[(Aj_matrix_df.Team == Team)]], axis=0).reset_index(drop=True)
    team_data = matrix_df[['Team','Opp','Team_Compat','Opp_Compat','Adv_Compat', 'Performance']]
    train_X = team_data[['Team_Compat','Opp_Compat','Adv_Compat']].astype(float)
    train_Y = team_data[['Performance']].astype(float)
    model = sm.OLS(train_Y,train_X).fit()
    t_c = np.sum(train_X.Team_Compat)
    o_c = np.sum(train_X.Opp_Compat)
    adv_c = np.sum(train_X.Adv_Compat)
    T_Syn_coef, O_Syn_coef, Adv_Syn_coef = model.params
    R_Spread = ((T_Syn_coef*t_c)-(-O_Syn_coef*o_c)+(Adv_Syn_coef*adv_c))
    print('Regressed Spread:' + str(round(R_Spread,2)))
    proj_spread = round(R_Spread,2)
    return(matrix_df, T_Syn_coef, O_Syn_coef, Adv_Syn_coef, proj_spread, model)
