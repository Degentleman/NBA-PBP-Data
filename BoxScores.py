#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:07:27 2018

@author: Degentleman
"""
import requests
import json
import pandas as pd
import numpy as np

Team_Legend = pd.read_csv('NBA PBP - Team Legend.csv', delimiter = ',')

def GetIDs(team):
    teamID = Team_Legend[(Team_Legend.Code == team)].TeamID.iloc[0]
    team_index = list(Team_Legend.TeamID).index(teamID)
    team_code = Team_Legend.iloc[team_index]['Code']
    team_mapping = {teamID:team_code}
    
    # HOME URL to SCRAPE
    team_url = 'https://stats.nba.com/stats/teamgamelog?DateFrom=&DateTo=&LeagueID=00&Season=2019-20&SeasonType=Regular+Season&TeamID={team}'.format(team=str(teamID))
    headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://stats.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
    } 
    response = requests.get(team_url, headers = headers)
    team_data = json.loads(response.content.decode())
    
    # Organize json data into DataFrame
    team_keys = list(team_data['resultSets'][0])
    team_dict = team_data['resultSets'][0]
    team_headers = team_dict[team_keys[1]]
    parsed_data = team_dict[team_keys[2]]
    team_df = pd.DataFrame(data=parsed_data, columns=team_headers)
    team_df = team_df.replace({'Team_ID': team_mapping})
    team_game_ids = list(team_df.Game_ID)
    return (team_df, team_game_ids, teamID)

def BoxScore(pbp_df):
    home_team = list(pbp_df)[12]
    away_team = list(pbp_df)[13]
    home_scores = []
    away_scores = []
    scores_list = []
    play_by_play = pbp_df
    quarter = 0
    for i in range(len(play_by_play)):
        event = play_by_play.iloc[i]
        event_type = event.etype
        home_pts = event[home_team]
        away_pts = event[away_team]
        if event_type == '12':
            quarter += 1
            if quarter > 4:
                print('***OVERTIME***')
        if event_type == '13':
            if quarter == 1:
                home_pts_scored = int(home_pts)
                away_pts_scored = int(away_pts)
            if quarter > 1:
                home_pts_scored = int(home_pts)-np.sum(home_scores)
                away_pts_scored = int(away_pts)-np.sum(away_scores)
            entry = [quarter,home_pts_scored, away_pts_scored]
            scores_list.append(entry)
            home_scores.append(home_pts_scored)
            away_scores.append(away_pts_scored)
    box_score = pd.DataFrame()
    for x in scores_list:
        q_score = {str(x[0]):[x[1],x[2]]}
        q_df = pd.DataFrame(q_score,index=[home_team,away_team])
        box_score = pd.concat([box_score,q_df],axis=1)
    
    box_score_cols = list(box_score)
    if len(box_score_cols) > 4:
        for quarter in box_score_cols:
            if quarter == '5':
                box_score_cols[4] = 'OT'
            if quarter == '6':
                box_score_cols[5] = 'D_OT'
            if quarter == '7':
                box_score_cols[6] = 'T_OT'
            if quarter == '8':
                box_score_cols[7] = 'Q_OT'
        box_score.columns = box_score_cols
    
    home_total = np.sum(box_score.loc[home_team])
    away_total = np.sum(box_score.loc[away_team])
    total_pts = {'Total':[home_total,away_total]}
    total_df = pd.DataFrame(data=total_pts,index=[home_team,away_team])
    box_score = pd.concat([box_score,total_df],axis=1)
    return(box_score)
