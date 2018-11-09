#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:20:00 2018

@author: Degentleman
"""
import requests
import json
import pandas as pd

Team_Legend = pd.read_csv('NBA PBP - Team Legend.csv', delimiter = ',')

NBA_Team_IDs = list(Team_Legend.TeamID)

NBA_df = pd.DataFrame()

for team_id in NBA_Team_IDs:
    
    row = Team_Legend[(Team_Legend.TeamID == team_id)]
    team_name = row.Team.iloc[0]
    team_code = row.Code.iloc[0]
    mapping = {team_id:team_code}
    
    # Scrape URL for Team
    url = 'https://stats.nba.com/stats/commonteamroster?LeagueID=00&Season=2018-19&TeamID={team_id}'.format(team_id=team_id)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    response = requests.get(url, headers = headers)
    
    if str(response) == '<Response [200]>':
    
        data = json.loads(response.content.decode())
        
        # Organize json data into DataFrame
        team_keys = list(data['resultSets'][0])
        coach_keys = list(data['resultSets'][1])
        team_dict = data['resultSets'][0]
        coach_dict = data['resultSets'][1]
        team_headers = team_dict[team_keys[1]]
        coach_headers = coach_dict[coach_keys[1]]
        team_data = team_dict[team_keys[2]]
        coach_data = coach_dict[coach_keys[2]]
        team_df = pd.DataFrame(data=team_data, columns=team_headers)
        coach_df = pd.DataFrame(data=coach_data, columns=coach_headers)
        
        new_df_cols = ['TeamID','PLAYER', 'HEIGHT', 'WEIGHT','POSITION', 'PLAYER_ID']
        
        player_df = team_df[new_df_cols]
        
        player_df = player_df.replace({'TeamID': mapping})
        
        new_col_head = ['Team', 'Player', 'Height', 'Weight','Position', 'PlayerID', ]
        
        player_df.columns = new_col_head
        
        filename = team_name+'_PlayerIDs.csv'
        
        NBA_df = pd.concat([NBA_df,player_df], axis=0, ignore_index=True)
        
        print(filename + ' added to NBA DF')
    else:
        print(str(response))
        print(team_name)
        print('!!!!!!!')
        
NBA_df.to_csv('NBA_Player_DB.csv', index=False)
