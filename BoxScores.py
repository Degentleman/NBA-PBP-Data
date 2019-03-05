#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:07:27 2018

@author: Degentleman
"""

import pandas as pd
import numpy as np

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