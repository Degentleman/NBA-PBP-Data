#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:20:00 2018

@author: Degentleman
"""
import NBAadvsyn as NSA

def NBAmodel(lineup_df, team_Ai, team_Aj, iteration_limit):

    D_Train, player_value = NSA.LearnCap(lineup_df)
    G_Alpha, alpha_player_value = NSA.SimGraph(D_Train, lineup_df)
    alpha_performance, a_predictions_df = NSA.MeasureError(G_Alpha, lineup_df)
    trials = 0
    stop_count = 0
    alpha_spreads = []
    beta_spreads = []
    explored_structures = []
    while stop_count < iteration_limit:
        
        if alpha_performance > 0:
            
            explored_structures.append(list(G_Alpha.edges))
            
            #Create a new Graph Structure
            
            G_Beta, beta_player_value = NSA.SimGraph(D_Train, lineup_df)
            
            if list(G_Beta.edges) in explored_structures:
                print('Identical structure randomly generated...generating a new graph')
                G_Beta, beta_player_value = NSA.SimGraph(D_Train, lineup_df)
            
            beta_performance, b_predictions_df = NSA.MeasureError(G_Beta, lineup_df)
            
            # Run K-weighted Adversarial Synergy Projection
            
            beta_proj_spread = round(NSA.ProjSpread(G_Beta,team_Ai,team_Aj),2)
            beta_spreads.append(beta_proj_spread)
            
            explored_structures.append(list(G_Beta.edges))
            
            trials +=1
            
            if round(beta_performance,5) < round(alpha_performance,5):
                
                print('Alpha Performance: ' + str(round(alpha_performance,4))+' Beta Performance: ' + str(round(beta_performance,3)))                
                print('Beta Graph changed to Alpha Graph on trial #' + str(trials))
                
                #Copy Beta information to Alpha model
                G_Alpha = G_Beta
                alpha_player_value = beta_player_value
                a_predictions_df = b_predictions_df
                
                # Run K-weighted Adversarial Synergy Projection
                
                alpha_proj_spread = round(NSA.ProjSpread(G_Alpha,team_Ai,team_Aj),2)
                alpha_spreads.append(round(alpha_proj_spread,2))
                
                alpha_performance = beta_performance
                stop_count = 0
            else:
                # Count increased if better structure isn't found
                stop_count +=1
            
    print(str(iterations)+' structures generated without improvement...terminating')
    
    return(G_Alpha,a_predictions_df, alpha_player_value,alpha_spreads,beta_spreads)
