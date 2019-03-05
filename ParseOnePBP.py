#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:20:00 2018

@author: Degentleman
"""
import NBAPBPReader as Reader
import NBAadvsyn as NSA
import pandas as pd
from BoxScores import BoxScore

game_id = '0021800164'
season = '2018'
iteration_limit = 100
#Parse the play-by-play data
pbp_df, pbpsumdf, pbp_file_name, home_team, away_team = Reader.PBP_Read(game_id, season)
players_df, df_cols = Reader.PBP_team_sort(pbp_df)
Ai_pID, Aj_pID = players_df[(players_df.index == home_team)],players_df[(players_df.index == away_team)]
team_Ai, team_Aj = list(Ai_pID.Player), list(Aj_pID.Player)
in_out_df, starters, bench = Reader.StatusCheck(pbp_df, df_cols, players_df)
new_pbp_df = pd.concat([pbp_df,in_out_df], axis=1)
final_pbp = Reader.CalcPerf(new_pbp_df, home_team, away_team)
pbp_perf = final_pbp[(final_pbp.Performance != '') & (final_pbp.etype != '8')].reset_index(drop=True,inplace=False)
lineup_df = NSA.lineups(pbp_perf)
print('All play-by-play data from Game id #'+str(game_id)+' has been parsed.')
#Save new file
lineup_file_name = pbp_file_name[0:9]+'Lineups'+pbp_file_name[-13:]
lineup_df.to_csv(lineup_file_name, index = False)
print(lineup_file_name+' has been saved to the current working directory.')
# Get the box score
box_score = BoxScore(pbp_df)
print(box_score)
