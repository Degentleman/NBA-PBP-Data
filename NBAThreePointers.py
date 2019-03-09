#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 02:07:09 2018

@author: Degentleman
"""
from NBAPBPReader import PBP_Read as Reader
from NBA_Player_DB import GetDB
import BoxScores
import pandas as pd
import numpy as np

season = '2018'
NBA_Legend = pd.read_csv('NBA Player DF - 2019.csv', delimiter = ',')   
Team_Legend = pd.read_csv('NBA PBP - Team Legend.csv', delimiter = ',')

#Today's Schedule
schedule = [['GSW', 'DEN'],
            ['LAC', 'OKC']]

game_ids_list = []
columns = ['Player One', 'TeamID', 'locX', 'locY', 'mtype', 'etype']
threes_df = pd.DataFrame(columns=columns)
for matchup in schedule:
    home_key = matchup[0]
    away_key = matchup[1]
    home_game_ids, home_id = BoxScores.GetIDs(home_key)
    away_game_ids, away_id = BoxScores.GetIDs(away_key)
    home_team_name, home_team, home_team_df, home_coach_df, home_df = GetDB(home_id)
    away_team_name, away_team, away_team_df, away_coach_df, away_df = GetDB(away_id)
    game_ids = list(np.unique(home_game_ids[0:10]+away_game_ids[0:10]))
    for game_id in sorted(game_ids):
        pbp_df, pbpsumdf, filename, home_team, away_team = Reader(game_id, season)
        entry = pbp_df[(pbp_df.opt1 == '3')][columns]
        entry.reset_index(drop=True,inplace=True)
        threes_df = pd.concat([threes_df, entry], axis=0, ignore_index=True)

Results = []
Teams = []
for i in range(len(threes_df)):
    row = threes_df.iloc[i]
    team_id = int(row['TeamID'])
    team_name = Team_Legend[(Team_Legend.TeamID == team_id)]['Code'].iloc[0]
    if row['etype'] == '2':
        Result = 'Miss'
    if row['etype'] == '1':
        Result = 'Make'
    Teams.append(team_name)
    Results.append(Result)
    
Results_series = pd.Series(data=Results, name = 'Result')
Teams_series = pd.Series(data=Teams, name = 'Team')
results_df = pd.concat([Teams_series,threes_df[['Player One', 'locX', 'locY', 'mtype']], Results_series], axis=1)
NBA_Threes = results_df.sort_values(by=['Team'])
