#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:20:00 2018

@author: Degentleman
"""
import NSAmodel as NSA
from NBAmodel import NBAmodel
import NBAPBPReader as Reader
import pandas as pd

NBA_Legend = pd.read_csv('NBA_Player_DB.csv', delimiter = ',')    
known_players = NBA_Legend[['PlayerID','Player']]

game_id = '0021800952'
season = '2018'

# Parse PBP Data and Create Lineup DataFrame
pbp_df, pbpsumdf, pbp_file_name, home_team, away_team = Reader.PBP_Read(game_id, season)
players_df, df_cols = Reader.PBP_team_sort(pbp_df)
in_out_df, starters, bench = Reader.StatusCheck(pbp_df, df_cols, players_df)
home_starters = starters[0:5]
away_starters = starters[5:10]
new_pbp_df = pd.concat([pbp_df,in_out_df], axis=1)
final_pbp = Reader.CalcPerf(new_pbp_df, home_team, away_team)
pbp_perf = final_pbp[(final_pbp.Performance != '') & (final_pbp.etype != '8')].reset_index(drop=True,inplace=False)
file_name = pbp_file_name[0:8]+' Lineups '+pbp_file_name[-12:]

# This includes players who are on the team's roster, not just who played.
home_roster = NBA_Legend[(NBA_Legend.Team == home_team)]
away_roster = NBA_Legend[(NBA_Legend.Team == away_team)]

# This only includes players who were found by the PBP script.
lineup_df = NSA.lineups(pbp_perf)
team_Ai, team_Aj = NSA.team_sort(lineup_df)

iterations = 100
#Simulate different graphs to determine optimal structure using performance.
G_Alpha, model_predictions_df, Agent_Cap_DF, Alpha_Model, explored_structures = NBAmodel(lineup_df, team_Ai, team_Aj, home_team, away_team, iterations)
Ai_matrix_df = NSA.LineupSyn(G_Alpha,lineup_df)
syn_df, adv_syn_df, Adv_Syn_Spread, Adv_Syn_Spread_mu = NSA.ProjSpread(G_Alpha, home_starters, away_starters, home_team, away_team)
