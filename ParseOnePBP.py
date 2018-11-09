#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:07:27 2018

@author: Degentleman
"""
import NBAPBPReader as Reader
import NBAadvsyn as NSA
import pandas as pd
from NBAmodel import NBAmodel
joint_lineups = pd.read_csv('Joint Reverse NBA Lineups.csv',delimiter=',')

NBA_Legend = pd.read_csv('NBA_Player_DB.csv', delimiter = ',')    
known_players = NBA_Legend[['PlayerID','Player']]
game_id = '0021800164'
season = '2018'

pbp_df, pbp_file_name, home_team, away_team = Reader.PBP_Read(game_id, season)
Ai_df, Aj_df, df_cols = Reader.PBP_team_sort(pbp_df, home_team, away_team)
in_out_df, starters, bench = Reader.StatusCheck(pbp_df, df_cols, home_team, away_team)
new_pbp_df = pd.concat([pbp_df,in_out_df], axis=1)
final_pbp = Reader.CalcPerf(new_pbp_df, home_team, away_team)
pbp_perf = final_pbp[(final_pbp.Performance != '') & (final_pbp.etype != '8')].reset_index(drop=True,inplace=False)
lineup_df = NSA.lineups(pbp_perf)
team_Ai, team_Aj = NSA.team_sort(lineup_df)

box_score, NBA = Reader.BoxScore(pbp_df, game_id, home_team, away_team)

G_Alpha, predictions_df, agent_capability, alpha_spreads, beta_spreads = NBAmodel(lineup_df, team_Ai, team_Aj, 500)
model_error, model_predictions = NSA.MeasureError(G_Alpha,lineup_df)

Player_Spread = NSA.ProjSpread(G_Alpha, team_Ai, team_Aj)

Ai_matrix_df, Ai_Syn_coef, Aj_Syn_coef, Ai_Aj_Adv_Syn_coef, R_Lineup_Spread, = NSA.LineupSyn(G_Alpha,lineup_df)

file_name = pbp_file_name[0:9]+'Ai Matrix DF'+pbp_file_name[-13:]