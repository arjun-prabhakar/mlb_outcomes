import pandas as pd
import numpy as np

import re
from scipy.stats import skew

def calc_stat(stat, stat_df, games_df, result_q):
    '''
    calculates season-to-date mean, stdev, skew for given stat from reference dataframe. fills na with 0
    
    params:
    -------
    - stat: column name in reference df from which to calc stat
    - stat_df: dataframe with end of game stats (like batting or pitching csvs gemerated by scrape.py)
    - games_df: dataframe with games of interest
    - result_q: queue object in which to store results
    
    returns:
    --------
    results_q gets a tuple
    - 0: name of created feature, like "batting_avg_mean"
    - 1: list of values corresponging to games ing games_df
    add the results back to games_df like this:
        key, result = result_q.get()
        df[key]=result
    '''
       
    hmean,amean = [],[]
    hstdev,astdev = [],[]
    hskew,askew = [],[]
    
    df_len = len(games_df)
    # merge in home team stat to df
    b = stat_df[['game_id',stat]][stat_df['is_home_team']==True].groupby('game_id').first().reset_index()
    games_df = pd.merge(left=games_df, right=b,on='game_id', how='left')
    games_df['home_'+stat] = games_df[stat]
    games_df.drop(columns=stat, inplace=True)
    
    #now moerge in away team stat
    b = stat_df[['game_id',stat]][stat_df['is_home_team']==False].groupby('game_id').first().reset_index()
    games_df = pd.merge(left=games_df, right=b, on='game_id', how='left')
    games_df['away_'+stat] = games_df[stat]
    games_df.drop(columns=stat, inplace=True)
    
    assert df_len == len(games_df)
    
    stats = {}
    for t in games_df.home_team_abbr.unique():stats[t]=[]
    for t in games_df.away_team_abbr.unique():stats[t]=[]
    
    for i, r in games_df.iterrows():
        
        #get distributions
        h = np.array(stats[r.home_team_abbr])
        a = np.array(stats[r.away_team_abbr])
        
        #calc stat  and append to dict
        hmean.append(h.mean())
        amean.append(a.mean())

        hstdev.append(h.std())
        astdev.append(a.std())

        hskew.append(skew(h))
        askew.append(skew(a))
        
        #update stats
        stats[r.home_team_abbr].append(r['home_'+stat])
        stats[r.away_team_abbr].append(r['away_'+stat])
    diff = np.array(hmean) - np.array(amean)
    
    names = ['home_'+stat+'_mean', 'away_'+stat+'_mean',
            'home_'+stat+'_stdev', 'away_'+stat+'_stdev',
            'home_'+stat+'_skew', 'away_'+stat+'_skew',
            stat+'_diff']
    lists = [hmean,amean,hstdev,astdev,hskew,askew,diff]
    for i in range(len(names)):
        result_q.put((names[i],lists[i]))

def calc_stat_worker(q,batting,df,result_q):
    #worker for threaded calc_stat
    while not q.empty():
        stat = q.get()
        calc_stat(stat,batting,df,result_q)
        print(stat,'Done!')
        q.task_done()


#######################
##   get/clean data  ##
#######################


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
    x = str(x)
    if re.findall('(\d+\:\d+)', x) == []:
        return np.nan
    else:
        return re.findall('(\d+\:\d+)', x)[0]

