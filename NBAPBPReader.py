#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:34:43 2018

@author: Degentleman
"""
import urllib.request, json
import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx

# Specify which Game ID you need the Play-by-Play (PBP) Data For Below in url_id

url_id = '0021800016'
url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2018/scores/pbp/{urlid}_full_pbp.json'.format(urlid=url_id)
with urllib.request.urlopen(url) as url:
    data = json.loads(url.read().decode())
    
access_key = list(data)[0]

# Use JSON data to pull Matchup Details i.e. day,month, year

game_id = list(data[access_key].items())[1]
day = list(data[access_key].items())[2][1][6:8]
month = list(data[access_key].items())[2][1][4:6]
year = list(data[access_key].items())[2][1][0:4]
away_team = list(data[access_key].items())[2][1][-6:-3]
home_team = list(data[access_key].items())[2][1][-3:]

pbp_json = list(data[access_key].items())[4][1]

# Convert JSON data to DataFrame using for loop

pbp = pd.DataFrame()

for x in pbp_json:
    dict_keys = list(x)
    index = str(x[dict_keys[0]])
    pbp_data = x[dict_keys[1]]
    pbp_entry = []
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

filename = str(away_team)+'at'+str(home_team)+'_PBP_'+str(month)+str(day)+str(year)+'.csv'

# Parse through play-by-play data to determine which players 
# have taken a shot or were subbed in using PBP data

known_players = pd.read_csv('Default PBP - NBA Legend.csv', delimiter = ',')[['PlayerID','Player']]

players_in = []
players_out = []
for i in range(len(pbp)):
    row = pbp.iloc[i]
    keys = list(known_players['PlayerID'])
    matches = list(known_players['Player'])
    if (row.etype == '1') | (row.etype == '2'):
        if int(row['PlayerID']) in keys:
            player_name = matches[keys.index(int(row['PlayerID']))]
            players_in.append(player_name)
            players_out.append("None")
            #print(player_name + ' shot the ball ')
        else:
            player_name = row['PlayerID']
            players_in.append(player_name)
            players_out.append("None")
    elif (row['etype'] == '4') & (row['PlayerID'] != '0'):
        if int(row['PlayerID']) in keys:
            player_name = matches[keys.index(int(row['PlayerID']))]
            players_in.append(player_name)
            players_out.append("None")
            print(player_name + ' grabbed the rebound ')
        else:
            player_name = row['PlayerID']
            players_in.append(player_name)
            players_out.append("None")
    elif (row.etype == '8'):
        print('**Sub**')
        if int(row['PlayerID']) in keys:
            subbed_out = matches[keys.index(int(row['PlayerID']))]
            subbed_in = row['epid']
            if int(row['epid']) in keys:
                subbed_in = matches[keys.index(int(row['epid']))]
            players_in.append(subbed_in)
            players_out.append(subbed_out)
            print(subbed_out + ' was subbed out for ' + str(subbed_in))
        else:
            subbed_out = row['PlayerID']
            subbed_in = int(row['epid'])
            if int(row['epid']) in keys:
                subbed_in = matches[keys.index(int(row['epid']))]
            players_in.append(subbed_in)
            players_out.append(subbed_out)
            print(subbed_out + ' was subbed out for ' + str(subbed_in))
    else:
        players_in.append("None")
        players_out.append("None")
        

players_in = pd.Series(data=players_in,name='Player In')
players_out = pd.Series(data=players_out,name='Player Out')

pbp_df = pd.concat([pbp,players_in,players_out],axis=1)

print(pbp_df[pbp.etype == '8'][['Description','Player In','Player Out']])

# BOX SCORE CODE

# Iterate through DataFrame to determine points scored by quarter

quarter = 1
away_score = []
home_score = []
for i in range(len(pbp)):
    row = pbp.iloc[i]
    if row['etype'] == '12':
        if quarter == 1:
            print('Game Started')
            print(away_team + ' @ ' + home_team)
    if row['etype'] == '13':
        if quarter == 1:
            away_scored = int(row[away_team])-np.sum(np.array(away_score,dtype=int))
            home_scored = int(row[home_team])-np.sum(np.array(home_score,dtype=int))
            away_score.append(away_scored)
            home_score.append(home_scored)
            print('Quarter ' + str(quarter) + ' ended')
            print(away_team,away_score)
            print(home_team,home_score)
        if quarter != 1:
            away_scored = int(row[away_team])-np.sum(np.array(away_score,dtype=int))
            home_scored = int(row[home_team])-np.sum(np.array(home_score,dtype=int))
            away_score.append(away_scored)
            home_score.append(home_scored)
            print('Quarter ' + str(quarter) + ' ended')
            print(away_team,away_score)
            print(home_team,home_score)
        quarter +=1
    if row['Description'] == 'Game End':
        print('Game Over')
        
        if np.sum(away_score) > np.sum(home_score):
            print(away_team+' wins ' + str(np.sum(away_score)) + ' to ' + str(np.sum(home_score)))
        if np.sum(home_score) > np.sum(away_score):
            print(home_team+' wins ' + str(np.sum(home_score)) + ' to ' + str(np.sum(away_score)))

#Create a box score from parsed data and normalize scoring to determine percentiles

box_score = np.array([away_score,home_score],dtype=int)

loc = np.mean(box_score)
scale = np.std(box_score)

away_percentile = round(stats.norm.cdf(np.mean(box_score[0]),loc,scale),3)
home_percentile = round(stats.norm.cdf(np.mean(box_score[1]),loc,scale),3)

print(away_team,round(away_percentile,3))
print(home_team,round(home_percentile,3))

pbp_df.to_csv(filename)

# Create a new unweighted graph and add boxscore details from PBP data

NBA = nx.Graph()

NBA.add_edge(away_team,home_team,boxscore=box_score,a_mu=away_percentile,
             h_mu=home_percentile,gameid=game_id[1])

columns=['Team','Opp','Q1','Q2','Q3','Q4','mu','std']

box_score_df = pd.DataFrame(columns=columns)

for x in list(NBA.edges(data=True)):
    a_mu = NBA[x[0]][x[1]]['a_mu']
    h_mu = NBA[x[0]][x[1]]['h_mu']
    a_pts = np.sum(NBA[x[0]][x[1]]['boxscore'][0])
    h_pts = np.sum(NBA[x[0]][x[1]]['boxscore'][1])
    score_mu = round(np.mean(NBA[x[0]][x[1]]['boxscore']),3)
    score_std = round(np.std(NBA[x[0]][x[1]]['boxscore']),3)
    print(score_mu,score_std)
    print('----------------')
    a_entry = [x[0],x[1], 
               NBA[x[0]][x[1]]['boxscore'][0][0],NBA[x[0]][x[1]]['boxscore'][0][1],
               NBA[x[0]][x[1]]['boxscore'][0][2],NBA[x[0]][x[1]]['boxscore'][0][3],
               a_mu,score_std]
    
    h_entry = [x[1],x[0], 
               NBA[x[0]][x[1]]['boxscore'][1][0],NBA[x[0]][x[1]]['boxscore'][1][1],
               NBA[x[0]][x[1]]['boxscore'][1][2],NBA[x[0]][x[1]]['boxscore'][1][3],
               h_mu,score_std]
    entry = pd.DataFrame([a_entry,h_entry],columns=columns)
    box_score_df = pd.concat([box_score_df,entry],axis=0)

box_score_df = box_score_df.reset_index(drop=True)