#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:34:43 2018

@author: Degentleman
"""
import requests
import json
import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx

NBA_Legend = pd.read_csv('NBA_Player_DB.csv', delimiter = ',')
    
keys = list(NBA_Legend.PlayerID)

def PBP_Read(game_id,season):

    # Specify which Game ID you need the Play-by-Play (PBP) Data For Below in url_id
    
    url_id = game_id
    season_id = season
    url = 'https://data.nba.com/data/v2015/json/mobile_teams/nba/{season}/scores/pbp/{urlid}_full_pbp.json'.format(urlid=url_id,season=season_id)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    response = requests.get(url, headers = headers)
    data = json.loads(response.content.decode())
    access_key = list(data)[0]
    
    # Use JSON data to pull Matchup Details i.e. day,month, year
    
    day = list(data[access_key].items())[2][1][6:8]
    month = list(data[access_key].items())[2][1][4:6]
    year = list(data[access_key].items())[2][1][0:4]
    away_team = list(data[access_key].items())[2][1][-6:-3]
    home_team = list(data[access_key].items())[2][1][-3:]
    
    pbp_json = data['g']['pd']
    
    # Convert JSON data to DataFrame using for loop
    
    pbp = pd.DataFrame()
    
    for i in range(len(pbp_json)):
        quarter_data = pbp_json[i]
        pbp_data = quarter_data['pla']
        for row in pbp_data:
            entry = []
            col_keys = list(row)
            for key in col_keys:
                col_data = row[key]
                entry.append(col_data)
            col_keys[1] = 'Clock'
            col_keys[2] = 'Description'
            col_keys[10] = 'TeamID'
            col_keys[11] = 'PlayerID'
            col_keys[12] = str(home_team)
            col_keys[13] = str(away_team)
            pbp_entry = pd.DataFrame(data=np.array(entry).reshape(1,17),columns=col_keys)
            pbp = pd.concat([pbp,pbp_entry],axis=0,ignore_index=True)
    col_keys[10], col_keys[11] = col_keys[11], col_keys[10]
    pbp = pbp[col_keys]
    filename = str(home_team)+'vs'+str(away_team)+'_PBP_'+str(month)+str(day)+str(year)+'.csv'
    
    # Parse through play-by-play data to determine which players 
    # have taken a shot or were subbed in using PBP data
    
    etype_list = ['1', '2', '3', '4', '5', '6', '7', '10']
    
    player_one = []
    player_two = []
    for i in range(len(pbp)):
        row = pbp.iloc[i]
        if row.etype == '8':
            subbed_out = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.PlayerID))]['Player'])[0]
            subbed_in = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.epid))]['Player'])[0]
            player_one.append(subbed_in)
            player_two.append(subbed_out)
        elif row.etype in etype_list and row.PlayerID != '0' and int(row.PlayerID) in keys:
            player_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.PlayerID))]['Player'])[0]
            player_one.append(player_name)
            if row.etype == '1' and row.epid != '':
                player_two_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.epid))]['Player'])[0]
                player_two.append(player_two_name)
            elif row.etype == '5' and 3 < len(row.opid) <10:
                player_two_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.opid))]['Player'])[0]
                player_two.append(player_two_name)
            elif row.etype == '6' and 3 < len(row.opid) <10:
                player_two_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.opid))]['Player'])[0]
                player_two.append(player_two_name)
            elif row.etype == '10' and 3 < len(row.opid) < 10:
                player_two_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.opid))]['Player'])[0]
                player_two.append(player_two_name)
            else:
                player_two.append('')
        else:
            player_one.append('')
            player_two.append('')
    player_one = pd.Series(data=player_one,name='Player One')
    player_two = pd.Series(data=player_two,name='Player Two')
    
    pbp_df = pd.concat([pbp,player_one,player_two],axis=1)
    
    pbp_df.to_csv(filename)
    
    return (pbp_df, filename, home_team, away_team)

def PBP_team_sort(pbp_df, home_team, away_team):

    player_one_list = [x for x in list(pbp_df['Player One']) if x != '' and x!= 'None']
    player_two_list = [x for x in list(pbp_df['Player Two']) if x != '' and x!= 'None']
    player_list = list(np.unique(player_one_list+player_two_list))
    
    Ai_df = pd.DataFrame()
    Aj_df = pd.DataFrame()
    for player in player_list:
        team = list(NBA_Legend[(NBA_Legend.Player == player)].Team)[0]
        p_id = list(NBA_Legend[(NBA_Legend.Player == player)].PlayerID)[0]
        if team == home_team:
            entry = pd.Series(data=[team, player,p_id]).values.reshape(1,3)
            entry_df = pd.DataFrame(data=entry,columns=['Team','Player','PlayerID'])
            Ai_df = pd.concat([Ai_df,entry_df],axis=0,ignore_index=True)
        if team == away_team:
            entry = pd.Series(data=[team, player,p_id]).values.reshape(1,3)
            entry_df = pd.DataFrame(data=entry, columns=['Team','Player','PlayerID'])
            Aj_df = pd.concat([Aj_df,entry_df],axis=0,ignore_index=True)
                
    Ai_df = Ai_df.set_index('Team')
    Aj_df = Aj_df.set_index('Team')
    
    players_df = pd.concat([Ai_df,Aj_df], axis=0)
    
    home_cols = sorted(list(players_df.loc[home_team]['Player']))
    
    away_cols = sorted(list(players_df.loc[away_team]['Player']))
    
    df_cols = home_cols+away_cols
    
    return (Ai_df, Aj_df,df_cols)

def StatusCheck(pbp_df, df_cols, home_team, away_team):
    in_out_df = pd.DataFrame()
    starters = []
    bench = []
    for player in df_cols:
        entry = []
        t_sub_count = 0
        p_out_count = 0
        p_in_count = 0
        quarter = 0
        for i in range(len(pbp_df)):
            row = pbp_df.iloc[i]
            if row.Description == 'Start Period':
                status = "Unknown"
                quarter += 1
            if row.etype != '8' and (player == row['Player One']) | (player == row['Player Two']):
                status = "In"
            if row.etype != '8' and len(row.opid) > 0:
                player_found = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.opid))].Player)[0]
                if player == player_found:
                    status = "In"
            if row.etype == '8':
                subbed_in = row['Player One']
                subbed_out = row['Player Two']
                t_sub_count += 1
                if player == subbed_in:
                    p_in_count += 1
                    status = "In"
                    if quarter == 1 and p_in_count == 1 and p_out_count == 0:
                        if player not in bench:
                            bench.append(player)
                if player == subbed_out:
                    p_out_count +=1
                    status = "Out"
                    if quarter == 1 and p_in_count == 0 and p_out_count == 1:
                        if player not in starters:
                            starters.append(player)
            if row.Description == 'End Period':
                status = entry[i-1]
                if status == "In" and p_in_count == 0 and quarter == 1:
                    if player not in bench and player not in starters:
                        starters.append(player)
            if row.Description == 'Game End':
                status = entry[i-1]
            entry.append(status)
        Q1_end = list(pbp_df['Description']).index('End Period')
        search_index = Q1_end+1
        if player not in starters:
            for i in range(search_index):
                if entry[i] == "Unknown":
                    entry[i] = "Out"
        if player in starters:
            for i in range(search_index):
                if entry[i] == "Unknown":
                    entry[i] = "In"
        entry = SearchCourt(pbp_df, player, entry)
        in_out_entry = pd.Series(data=entry,name=player)
        in_out_df = pd.concat([in_out_df,in_out_entry], axis=1)
    if len(starters) != 10:
        print('Count of starters could not be determined by parsing PBP data.')
    return(in_out_df, starters, bench)
    
def SearchCourt(pbp_df, player, entry):
    loop_count = 0
    while "Unknown" in entry:
        loop_count += 1
        index_one = entry.index('Unknown')
        n_events = list(pbp_df.iloc[index_one:].Description)
        if 'Start Period' in n_events:
            n_start_period = n_events.index('Start Period')
        if 'End Period' in n_events:
            n_end_period = n_events.index('End Period')
        # Determine whether the player was on floor at the start of the period.
        look_in = entry[index_one:]
        if "Out" in look_in:
            out_at = look_in.index('Out')
            out_row = pbp_df.iloc[index_one+out_at]
            is_out_val = True
        if "Out" not in look_in:
            is_out_val = False
        if "In" in look_in:
            in_at = look_in.index('In')
            in_row = pbp_df.iloc[index_one+in_at]
            is_in_val = True
        if "In" not in look_in:
            is_in_val = False
        if n_start_period == 0:
            if is_out_val and is_in_val:
                if out_at < in_at and out_row.etype == '8':
                    for i in range(index_one,index_one+out_at):
                        entry[i] = "In"
                if in_at < out_at < n_end_period and out_row["Player Two"] == player:
                    for i in range(index_one,index_one+in_at):
                        entry[i] = "In"
                if in_at < out_at < n_end_period and in_row["Player One"] == player:
                    for i in range(index_one,index_one+in_at):
                        entry[i] = "Out"
                if n_end_period < in_at < out_at and out_row.etype == '8':
                    for i in range(index_one,index_one+n_end_period+1):
                        entry[i] = "Out"
                if in_at < n_end_period < out_at and in_row.etype == '8':
                    for i in range(index_one,index_one+in_at):
                        entry[i] = "Out"
                if in_at < n_end_period < out_at and in_row.etype != '8':
                    for i in range(index_one,index_one+in_at):
                        entry[i] = "In"
                if in_at < out_at < n_end_period and in_row.etype != '8':
                    for i in range(index_one,index_one+in_at):
                        entry[i] = "In"
            if (is_in_val) & (is_out_val == False) and in_row.etype == '8':
                if in_at < n_end_period:
                    for i in range(index_one,index_one+in_at):
                        entry[i] = "Out"
                if in_at > n_end_period:
                    for i in range(index_one,index_one+n_end_period+1):
                        entry[i] = "Out"
            if (is_in_val) & (is_out_val == False) and in_row.etype != '8':
                if in_at < n_end_period:
                    for i in range(index_one,index_one+in_at):
                        entry[i] = "In"
                if in_at > n_end_period:
                    for i in range(index_one,index_one+n_end_period+1):
                        entry[i] = "Out"
            if (is_out_val) & (is_in_val == False):
                if out_at < n_end_period and out_row.etype == '8':
                    for i in range(index_one,index_one+out_at):
                        entry[i] = "In"
                if out_at > n_end_period:
                    for i in range(index_one,index_one+n_end_period+1):
                        entry[i] = "Out"
            if (is_in_val == False) & (is_out_val == False):
                for i in range(index_one,len(entry)):
                    entry[i] = "Out"
        if loop_count > len(entry)/2:
            break
    return(entry)
    
def CalcPerf(new_pbp_df, home_team, away_team):
    performance_list = []
    quarter = 0
    sub_count = 0
    for i in range(len(new_pbp_df)):
        row = new_pbp_df.iloc[i]
        if i < len(new_pbp_df)-1:
            next_row = new_pbp_df.iloc[i+1]
        home_score = int(row[home_team])
        away_score = int(row[away_team])
        spread = int(home_score-away_score)
        if row['Description'] == 'Start Period':
            quarter += 1
            performance_list.append("")
        elif next_row['etype'] == '8':
            sub_count += 1
            if sub_count == 1:
                performance = spread
                spread_pass = spread
                performance_list.append(performance)
            if sub_count > 1:
                prior_def = spread_pass
                performance = spread-prior_def
                spread_pass = spread
                performance_list.append(performance)
        elif row['Description'] == 'End Period':
            prior_def = spread_pass
            performance = spread-prior_def
            spread_pass = spread
            performance_list.append(performance)
        else:
            performance_list.append("")
            
    performance_df = pd.Series(performance_list,name="Performance")
    final_pbp_df = pd.concat([new_pbp_df, performance_df],axis=1)
    return(final_pbp_df)

def BoxScore(pbp_df, game_id, home_team, away_team):
    
    # BOX SCORE CODE
    
    # Iterate through DataFrame to determine points scored by quarter
    quarter = 1
    away_score = []
    home_score = []
    for i in range(len(pbp_df)):
        row = pbp_df.iloc[i]
        if row['etype'] == '12':
            if quarter == 1:
                print(home_team + ' vs ' + away_team)
        if row['etype'] == '13':
            if quarter == 1:
                away_scored = int(row[away_team])-np.sum(np.array(away_score,dtype=int))
                home_scored = int(row[home_team])-np.sum(np.array(home_score,dtype=int))
                away_score.append(away_scored)
                home_score.append(home_scored)
            
            if quarter != 1:
                away_scored = int(row[away_team])-np.sum(np.array(away_score,dtype=int))
                home_scored = int(row[home_team])-np.sum(np.array(home_score,dtype=int))
                away_score.append(away_scored)
                home_score.append(home_scored)
            quarter +=1
        if row['Description'] == 'Game End':
            if np.sum(home_score) > np.sum(away_score):
                print(home_team+' wins ' + str(np.sum(home_score)) + ' to ' + str(np.sum(away_score)))
            if np.sum(away_score) > np.sum(home_score):
                print(away_team+' wins ' + str(np.sum(away_score)) + ' to ' + str(np.sum(home_score)))
    
    #Create a box score from parsed data and normalize scoring to determine percentiles
    
    box_score = np.array([away_score,home_score],dtype=int)
    
    loc = np.mean(box_score)
    scale = np.std(box_score)
    
    away_percentile = round(stats.norm.cdf(np.mean(box_score[0]),loc,scale),3)
    home_percentile = round(stats.norm.cdf(np.mean(box_score[1]),loc,scale),3)
    
    print(home_team,round(home_percentile,3))
    print(away_team,round(away_percentile,3))
    
    # Create a new unweighted graph and add boxscore details from PBP data
    
    NBA = nx.Graph()
    
    NBA.add_edge(home_team,away_team,boxscore=box_score,h_mu=home_percentile,
                 a_mu=away_percentile,gameid=game_id[1])
    
    columns=['Team','Opp','Location','Q1','Q2','Q3','Q4','mu','std']
    
    box_score_df = pd.DataFrame(columns=columns)
    
    for x in list(NBA.edges(data=True)):
        h_mu = NBA[x[0]][x[1]]['a_mu']
        a_mu = NBA[x[0]][x[1]]['h_mu']
        score_mu = round(np.mean(NBA[x[0]][x[1]]['boxscore']),3)
        score_std = round(np.std(NBA[x[0]][x[1]]['boxscore']),3)
        print(score_mu,score_std)
        print('----------------')
        h_entry = [x[0],x[1],'Home',
                   NBA[x[0]][x[1]]['boxscore'][0][0],NBA[x[0]][x[1]]['boxscore'][0][1],
                   NBA[x[0]][x[1]]['boxscore'][0][2],NBA[x[0]][x[1]]['boxscore'][0][3],
                   h_mu,score_std]
        
        a_entry = [x[1],x[0],'Away',
                   NBA[x[0]][x[1]]['boxscore'][1][0],NBA[x[0]][x[1]]['boxscore'][1][1],
                   NBA[x[0]][x[1]]['boxscore'][1][2],NBA[x[0]][x[1]]['boxscore'][1][3],
                   a_mu,score_std]
        entry = pd.DataFrame([h_entry,a_entry],columns=columns)
        box_score_df = pd.concat([box_score_df,entry],axis=0)
    
    box_score_df = box_score_df.reset_index(drop=True)
    
    return(box_score_df, NBA)
