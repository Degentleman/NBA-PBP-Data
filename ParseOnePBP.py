#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:20:00 2018

@author: Degentleman
"""
import NBAPBPReader as Reader
import NBAadvsyn as NSA
import pandas as pd

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
