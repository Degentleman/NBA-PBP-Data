#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:21:23 2018

@author: Degentleman
"""
import NSAmodel as NSA

def NBAmodel(lineup_df, team_Ai, team_Aj, home_team, away_team, iterations):
    D_Train, player_value = NSA.LearnCap(lineup_df)
    G_Alpha, Agent_Cap_DF, Alpha_Model = NSA.SimGraph(D_Train, lineup_df)
    alpha_performance, mu_performance, model_predictions_df = NSA.MeasureError(G_Alpha, lineup_df)
    trials = 0
    stop_count = 0
    iterations = iterations
    explored_structures = []
    explored_structures.append(list(G_Alpha.edges(data=True)))
    while stop_count < iterations:
        
        if alpha_performance > 1:
        
            #Create a new Graph Structure    
            G_Beta, Agent_Cap_DF_Beta, Beta_Model = NSA.SimGraph(D_Train, lineup_df)
            
            caveat = list(G_Beta.edges(data=True)) in explored_structures
            
            if caveat:
                duplicate_s = True
                print('Bob')
                print('Identical structure randomly generated...generating a new graph')
                while duplicate_s == True:
                    G_Beta, Agent_Cap_DF_Beta, Beta_Model = NSA.SimGraph(D_Train, lineup_df)
                    duplicate_s = list(G_Beta.edges(data=True)) in explored_structures

            beta_performance, beta_mu_performance, beta_model_predictions_df = NSA.MeasureError(G_Beta, lineup_df)
            
            explored_structures.append(list(G_Beta.edges))
            
            trials +=1
            
            if round(beta_performance,4) < round(alpha_performance,4):
                
                #Copy Beta information to Alpha model
                G_Alpha = G_Beta
                Agent_Cap_DF = Agent_Cap_DF_Beta
                Alpha_Model = Beta_Model
                model_predictions_df = beta_model_predictions_df
                alpha_performance = beta_performance
                stop_count = 0
            else:
                # Count increased if better structure isn't found
                stop_count +=1
            
    print(str(iterations)+' structures generated without improvement...terminating')
    
    return(G_Alpha, model_predictions_df, Agent_Cap_DF, Alpha_Model, explored_structures)
