#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 18:46:47 2018

@author: Degentleman
"""
import json
import requests
import pandas as pd

Team_Legend = pd.read_csv('NBA PBP - Team Legend.csv', delimiter = ',')
NBA_Team_IDs = list(Team_Legend.TeamID)

def GetIDs(team):
    teamID = Team_Legend[(Team_Legend.Code == team)].TeamID.iloc[0]
    team_index = list(Team_Legend.TeamID).index(teamID)
    team_code = Team_Legend.iloc[team_index]['Code']
    team_mapping = {teamID:team_code}
    
    # HOME URL to SCRAPE
    team_url = 'https://stats.nba.com/stats/teamgamelog?DateFrom=&DateTo=&LeagueID=00&Season=2018-19&SeasonType=Regular+Season&TeamID={team}'.format(team=str(teamID))
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
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
    return (team_game_ids, teamID)