import pandas as pd
import numpy as np

import re
import queue
import threading

def get_pitchers():
    pitchers = pd.read_csv('../data/pitchers.csv')
    pitchers['is_home_team']=True
    pitchers['is_home_team'][pitchers['home_away']=='away']=False
    pitchers.drop(columns='home_away', inplace=True)
    
    pitchers['is_starting_pitcher'] = False
    pitchers['is_starting_pitcher'][pitchers.inherited_runners.isna()]=True
    
    return pitchers


def get_pitching():
    pitching = pd.read_csv('../data/pitching.csv')
    pitching['is_home_team']=True
    pitching['is_home_team'][pitching['home_away']=='away']=False
    pitching.drop(columns='home_away', inplace=True)
    return pitching


def get_batting():
    batting = pd.read_csv('../data/batting.csv')
    batting['is_home_team']=True
    batting['is_home_team'][batting['home_away']=='away']=False
    batting.drop(columns='home_away', inplace=True)
    return batting


def get_games():
    games = pd.read_csv('../data/game_summaries.csv')
    games['date'] = pd.to_datetime(games.date).dt.date
    games['start_time'] = games['start_time'].apply(lambda x: start_times(x))
    games['is_night_game']=True
    games['is_night_game'][games['day_night'].isin(['Day Game'])] = False
    games.drop(columns='day_night', inplace=True)

    games['is_grass']=False
    games['is_grass'][games['field_type'].isin(['on grass'])]=True
    #fix for error in data collection
    games = games[games['field_type'].isin(['on grass','on turf'])].copy().reset_index(drop=True)
    games.drop(columns='field_type',inplace=True)

    games['spread'] = games.home_team_runs.astype('int') - games.away_team_runs
    
    return games

def start_times(x):
    '''
    cleanup routine for start times
    '''
    if re.findall('(\d+\:\d+)', x) == []:
        return np.nan
    else:
        return re.findall('(\d+\:\d+)', x)[0]