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
from NBA_Player_DB import GetDB

NBA_Legend = pd.read_csv('NBA Player DF - 2019.csv', delimiter = ',')
Team_Legend = pd.read_csv('NBA PBP - Team Legend.csv', delimiter = ',')
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
    home_id = Team_Legend[(Team_Legend.Code == home_team)].TeamID.iloc[0]
    away_id = Team_Legend[(Team_Legend.Code == away_team)].TeamID.iloc[0]
    home_team_name, home_team, home_team_df, home_coach_df, home_df = GetDB(home_id)
    away_team_name, away_team, away_team_df, away_coach_df, away_df = GetDB(away_id)
    current_rosters = pd.concat([home_df,away_df],axis=0, ignore_index=True)
    roster_keys = list(current_rosters.PlayerID)
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
            sub_out = int(row.PlayerID)
            sub_in = int(row.epid)
            if sub_out in keys:
                subbed_out = list(NBA_Legend[(NBA_Legend.PlayerID == sub_out)]['Player'])[0]
            elif sub_out not in keys and sub_out in roster_keys:
                subbed_out = list(current_rosters[(current_rosters.PlayerID == sub_out)]['Player'])[0]
            if sub_in in keys:
                subbed_in = list(NBA_Legend[(NBA_Legend.PlayerID == sub_in)]['Player'])[0]
            elif sub_in not in keys and sub_in in roster_keys:
                subbed_in = list(current_rosters[(current_rosters.PlayerID == sub_in)]['Player'])[0]
            else:
                subbed_in = str(sub_in)
                subbed_out = str(sub_out)
            player_one.append(subbed_in)
            player_two.append(subbed_out)
        elif row.etype in etype_list and row.PlayerID != '0':
            #Find player's name using pbp ID
            playerID = int(row.PlayerID)
            if playerID in keys:
                players_name = list(NBA_Legend[(NBA_Legend.PlayerID == playerID)]['Player'])[0]
            if playerID not in keys and playerID in roster_keys:
                players_name = list(current_rosters[(current_rosters.PlayerID == playerID)]['Player'])[0]
            
            #Player's name should be known at this point.
            if row.etype == '1' and row.epid != '':
                player_one.append(players_name)
                if int(row.epid) in keys:
                    player_two_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.epid))]['Player'])[0]
                    player_two.append(player_two_name)
                if int(row.epid) not in keys and int(row.epid) in roster_keys:
                    player_two_name = list(current_rosters[(current_rosters.PlayerID == int(row.epid))]['Player'])[0]
                    player_two.append(player_two_name)
                #Assist from one to two
            elif row.etype == '5':
                if row.opid == '':
                    player_one.append(players_name)
                    player_two.append('')
                    #Turnover by player one
                if 3 < len(row.opid) <10:
                    player_one.append(players_name)
                    if int(row.opid) in keys:
                        player_two_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.opid))]['Player'])[0]
                        player_two.append(player_two_name)
                    if int(row.opid) not in keys and int(row.opid) in roster_keys:
                        player_two_name = list(current_rosters[(current_rosters.PlayerID == int(row.opid))]['Player'])[0]
                        player_two.append(player_two_name)
                    #Turnover by player one and steal by player two
            elif row.etype == '6' and 3 < len(row.opid) <10:
                player_one.append(players_name)
                player_two_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.opid))]['Player'])[0]
                player_two.append(player_two_name)
            elif row.etype == '10' and 3 < len(row.opid) < 10:
                player_one.append(players_name)
                player_two_name = list(NBA_Legend[(NBA_Legend.PlayerID == int(row.opid))]['Player'])[0]
                player_two.append(player_two_name)
            else:
                player_one.append(players_name)
                player_two.append('')
        else:
            player_one.append('')
            player_two.append('')
    player_one = pd.Series(data=player_one, name='Player One')
    player_two = pd.Series(data=player_two, name='Player Two')
    pbp_df = pd.concat([pbp,player_one,player_two],axis=1, ignore_index=False)
    shots_made = len(pbp_df[(pbp_df.etype == '1')])
    shots_missed = len(pbp_df[(pbp_df.etype == '2')])
    fts_shot = len(pbp_df[(pbp_df.etype == '3')])
    fts_made = len(pbp_df[(pbp_df.etype == '3') & (pbp_df.opt1 == '1')])
    rebounds = len(pbp_df[(pbp_df.etype == '4') & (pbp_df.PlayerID != '0')])
    turnovers = len(pbp_df[(pbp_df.etype == '5')])
    fouls = len(pbp_df[(pbp_df.etype == '6')])
    substitutions = len(pbp_df[(pbp_df.etype == '8')])
    timeouts = len(pbp_df[(pbp_df.etype == '9')])
    game_summary = pd.Series([shots_made,shots_missed,fts_made, fts_shot, 
                              rebounds, turnovers, fouls, 
                              substitutions, timeouts]).values.reshape(1,9)
    sum_columns = ['Shots Made', 'Shots Missed', 'FTs', 'FTAs',
                   'Rebounds', 'Turnovers', 'Fouls', 'Subs', 'Timeouts']
    pbpsumdf = pd.DataFrame(data=game_summary, columns=sum_columns)
    pbp_df = pbp_df.replace(np.nan, '', regex=True)
    return (pbp_df, pbpsumdf, filename, home_team, away_team)

def PBP_team_sort(pbp_df):
    home_team, away_team = list(pbp_df)[12:14]
    home_id = Team_Legend[(Team_Legend.Code == home_team)].TeamID.iloc[0]
    away_id = Team_Legend[(Team_Legend.Code == away_team)].TeamID.iloc[0]
    home_team_name, home_team, home_team_df, home_coach_df, home_df = GetDB(home_id)
    away_team_name, away_team, away_team_df, away_coach_df, away_df = GetDB(away_id)
    current_rosters = pd.concat([home_df,away_df],axis=0, ignore_index=True)
    player_one_list = [x for x in list(pbp_df['Player One']) if x != '' and x!= 'None']
    player_two_list = [x for x in list(pbp_df['Player Two']) if x != '' and x!= 'None']
    player_list = list(np.unique(player_one_list+player_two_list))
    Ai_df = pd.DataFrame()
    Aj_df = pd.DataFrame()
    for player in player_list:
        if player not in list(NBA_Legend.Player):
            if player in list(current_rosters.Player):
                p_id = current_rosters[(current_rosters.Player == player)].PlayerID.values[0]
                team = current_rosters[(current_rosters.Player == player)].Team.values[0]
                if team == home_team:
                    entry = pd.Series(data=[team, player,p_id]).values.reshape(1,3)
                    entry_df = pd.DataFrame(data=entry,columns=['Team','Player','PlayerID'])
                    Ai_df = pd.concat([Ai_df,entry_df],axis=0,ignore_index=True)
                if team == away_team:
                    entry = pd.Series(data=[team, player,p_id]).values.reshape(1,3)
                    entry_df = pd.DataFrame(data=entry, columns=['Team','Player','PlayerID'])
                    Aj_df = pd.concat([Aj_df,entry_df],axis=0,ignore_index=True)
        elif player in list(NBA_Legend.Player):
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
            if team != home_team and team != away_team:
                print('According to your database, '+ player + ' is on '+team+"'s roster,")
                pbp_home = list(pbp_df)[12]
                pbp_away = list(pbp_df)[13]
                team = input('Which team is '+player+' playing for in this game: '+pbp_home+' or '+pbp_away+'? ')
                entry = pd.Series(data=[team, player,p_id]).values.reshape(1,3)
                entry_df = pd.DataFrame(data=entry, columns=['Team','Player','PlayerID'])
                if team == home_team:    
                    Ai_df = pd.concat([Ai_df,entry_df],axis=0,ignore_index=True)
                if team == away_team:    
                    Aj_df = pd.concat([Aj_df,entry_df],axis=0,ignore_index=True)
        else:
            print('NBA Player #'+player+' is missing from the team database.')
            pbp_home = list(pbp_df)[12]
            pbp_away = list(pbp_df)[13]
            team = input('Which team are they on? '+pbp_home+' or '+pbp_away+' ')
            p_id = int(player)
            player_name = input('Please input the correct name for this player ')
            entry = pd.Series(data=[team, player_name, p_id]).values.reshape(1,3)
            entry_df = pd.DataFrame(data=entry,columns=['Team','Player','PlayerID'])
            if team == home_team:    
                Ai_df = pd.concat([Ai_df,entry_df],axis=0,ignore_index=True)
            if team == away_team:    
                Aj_df = pd.concat([Aj_df,entry_df],axis=0,ignore_index=True)
    Ai_df = Ai_df.set_index('Team')
    Aj_df = Aj_df.set_index('Team')
    players_df = pd.concat([Ai_df,Aj_df], axis=0)
    home_cols = sorted(list(players_df.loc[home_team]['Player']))
    away_cols = sorted(list(players_df.loc[away_team]['Player']))
    df_cols = home_cols+away_cols
    return (players_df, df_cols)

def StatusCheck(pbp_df, df_cols, players_df):
    in_out_df = pd.DataFrame()
    starters = []
    bench = []
    for player in df_cols:
        entry = []
        p_id = players_df[(players_df['Player'] == player)].PlayerID[0]
        t_sub_count = 0
        p_out_count = 0
        p_in_count = 0
        quarter = 0
        for i in range(len(pbp_df)):
            row = pbp_df.iloc[i]
            if row.Description == 'Start Period':
                status = "Unknown"
                quarter += 1
            if row.etype != '8' and str(p_id) == row.PlayerID:
                status = "In"
            if row.etype != '8' and str(p_id) == row.epid: 
                status = "In"
            if row.etype != '8' and len(row.opid) > 0:
                player_index = list(players_df.PlayerID).index(int(row.opid))
                player_found = players_df.iloc[player_index].Player
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
                if 'End Period' in n_events and in_at < n_end_period:
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

def CalcPoss(pbp_df, team_id):
    team_fgas = len( pbp_df[(pbp_df.TeamID == str(team_id)) & ((pbp_df.etype == '1') | (pbp_df.etype == '2'))])
    team_fgs = len( pbp_df[(pbp_df.TeamID == str(team_id)) & (pbp_df.etype == '2')])
    team_3pts = len(pbp_df[(pbp_df.TeamID == str(team_id)) & (pbp_df.opt1 == '3') & (pbp_df.etype == '1')])
    team_turnovers = len(pbp_df[(pbp_df.etype == '5') & (pbp_df.TeamID == str(team_id))])
    team_fta = len(pbp_df[(pbp_df.etype == '3') & (pbp_df.TeamID == str(team_id))])
    team_ftm = len(pbp_df[(pbp_df.etype == '3') & (pbp_df.opt1 == '1') & (pbp_df.TeamID == str(team_id))])
    team_off_rbs = len(pbp_df[((pbp_df.opt1 != '0') & (pbp_df.etype == '4')) & (pbp_df.TeamID == str(team_id)) & (pbp_df.PlayerID != '0')])
    opp_drbs = len(pbp_df[((pbp_df.opt1 != '1') & (pbp_df.etype == '4')) & (pbp_df.TeamID != str(team_id)) & (pbp_df.PlayerID != '0')])
    team_drbs = len(pbp_df[((pbp_df.opt1 == '0') & (pbp_df.etype == '4')) & (pbp_df.TeamID == str(team_id)) & (pbp_df.PlayerID != '0')])
    possessions = team_fgas+(team_fta*.4)-(1.07*(team_off_rbs/(team_off_rbs+opp_drbs))*(team_fgas-team_fgs))+team_turnovers
    TO_perc = team_turnovers/possessions
    if team_ftm != 0:
        ft_fga_perc = round(team_ftm/team_fgas,3)
    else:
        ft_fga_perc = 0.0
    if team_3pts != 0:
        efg_perc = ((team_3pts*.5)+(team_fgs))/team_fgas
    else:
        efg_perc = 0.0
    orb_perc = round(team_off_rbs/(team_off_rbs+opp_drbs),3)
    summary_text = ['teamID','Possessions', 'EFGr', 'TOr', 'ORBr', 'FT_FGAr', 'FGsA', 'FGsM',
                    '3PTsM', 'TOs', 'FTsA','FTsM','ORBs','DRBs']
    summary_data = [team_id, round(possessions,3), round(efg_perc,3),round(TO_perc,3), orb_perc,ft_fga_perc, team_fgas, team_fgs, team_3pts, team_turnovers, team_fta,
                    team_ftm, team_off_rbs, team_drbs]
    summary_df = pd.DataFrame(data=np.array(summary_data,dtype=str).reshape(1,14), columns=summary_text)
    return(summary_df)
    
def CalcPerf(pbp_df, home_team, away_team):
    performance_list = []
    quarter = 0
    sub_count = 0
    spread_pass = 0
    for i in range(len(pbp_df)):
        row = pbp_df.iloc[i]
        if i < len(pbp_df)-1:
            next_row = pbp_df.iloc[i+1]
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
    final_pbp_df = pd.concat([pbp_df, performance_df],axis=1)
    return(final_pbp_df)
