import pandas as pd
import numpy as np

import re
from scipy.stats import skew

def add_season_rolling(stat_df, df, cols, team, name):
    '''
    add seasonal rolling statistical features to target dataframe
    
    params:
    ------
    stat_df: the dataframe with the game stats (like from batting.csv
    df: the target dataframe we're going to return with new columns
    cols: list of columns in stat_df containing the stats of interest
    team: binary whether this is a team stat (false if pitcher stat)
    name: the string appended to each feature name (like 'batting')
    '''
    stat_df['season'] = stat_df.game_id.str[3:7]
    for s in cols:
        if team:
            stat_df[s+'_mean'] = stat_df.groupby(['team', 'season'])[s].apply(lambda x:x.rolling(200, min_periods=1).mean())
            stat_df[s+'_stdev'] = stat_df.groupby(['team', 'season'])[s].apply(lambda x:x.rolling(200, min_periods=1).std())
            stat_df[s+'_skew'] = stat_df.groupby(['team', 'season'])[s].apply(lambda x:x.rolling(200, min_periods=1).skew())

            #shift the stats to the next game, in order to convert to pre-game stats
            stat_df[s+'_mean'] = stat_df.groupby(['team', 'season'])[s+'_mean'].shift()
            stat_df[s+'_stdev'] = stat_df.groupby(['team', 'season'])[s+'_stdev'].shift()
            stat_df[s+'_skew'] = stat_df.groupby(['team', 'season'])[s+'_skew'].shift()
        else:
            stat_df[s+'_mean'] = stat_df.groupby(['name', 'season'])[s].apply(lambda x:x.rolling(200, min_periods=1).mean())
            stat_df[s+'_stdev'] = stat_df.groupby(['name', 'season'])[s].apply(lambda x:x.rolling(200, min_periods=1).std())
            stat_df[s+'_skew'] = stat_df.groupby(['name', 'season'])[s].apply(lambda x:x.rolling(200, min_periods=1).skew())
        
            #shift the stats to the next game, in order to convert to pre-game stats
            stat_df[s+'_mean'] = stat_df.groupby(['name', 'season'])[s+'_mean'].shift()
            stat_df[s+'_stdev'] = stat_df.groupby(['name', 'season'])[s+'_stdev'].shift()
            stat_df[s+'_skew'] = stat_df.groupby(['name', 'season'])[s+'_skew'].shift()
        
    stat_cols = []
    for s in cols:
        stat_cols.append(s + '_mean')
        stat_cols.append(s + '_stdev')
        stat_cols.append(s + '_skew')
    stat_cols.append('game_id')

    df_len = len(df)
    b = stat_df[stat_cols][stat_df['is_home_team']==True].groupby('game_id').first().reset_index()
    df = pd.merge(left=df, right=b,on='game_id', how='left')

    for s in stat_cols:
        if s == 'game_id':continue
        df['home_'+name+'_'+s] = df[s]
        df.drop(columns=s, inplace=True)

    b = stat_df[stat_cols][stat_df['is_home_team']==False].groupby('game_id').first().reset_index()
    df = pd.merge(left=df, right=b,on='game_id', how='left')

    for s in stat_cols:
        if s == 'game_id':continue
        df['away_'+name+'_'+s] = df[s]
        df.drop(columns=s, inplace=True)

    assert df_len == len(df)

    # create diff stats
    for s in cols:
        if s == 'game_id':continue
        df[name+'_'+s+'_diff']= df['home_'+name+'_'+s+'_mean']-df['away_'+name+'_'+s+'_mean']

    return df

def add_10RA_rolling(stat_df, df, cols, team, name):
    '''
    add 10 period rolling statistical features to target dataframe
    
    params:
    ------
    stat_df: the dataframe with the game stats (like from batting.csv
    df: the target dataframe we're going to return with new columns
    cols: list of columns in stat_df containing the stats of interest
    team: binary whether this is a team stat (false if pitcher stat)
    name: the string appended to each feature name (like 'batting')
    '''
    #create stat
    for s in cols:
        if team:
            stat_df[s+'_10RA'] = stat_df.groupby('team')[s].apply(lambda x:x.rolling(10, min_periods=1).mean())
            #shift the stats to the next game, in order to convert to pre-game stats
            stat_df[s+'_10RA'] = stat_df.groupby('team')[s+'_10RA'].shift()
        else:
            stat_df[s+'_10RA'] = stat_df.groupby('name')[s].apply(lambda x:x.rolling(10, min_periods=1).mean())
            #shift the stats to the next game, in order to convert to pre-game stats
            stat_df[s+'_10RA'] = stat_df.groupby('name')[s+'_10RA'].shift()
        
    # add stat to target dataframe
    stat_cols = [x + '_10RA' for x in cols]
    stat_cols.append('game_id')
    
    #home team first
    df_len = len(df)
    b = stat_df[stat_cols][stat_df['is_home_team']==True].groupby('game_id').first().reset_index()
    df = pd.merge(left=df, right=b,on='game_id', how='left')

    for s in stat_cols:
        if s == 'game_id':continue
        df['home_'+name+'_'+s] = df[s]
        df.drop(columns=s, inplace=True)
    
    #now away team
    b = stat_df[stat_cols][stat_df['is_home_team']==False].groupby('game_id').first().reset_index()
    df = pd.merge(left=df, right=b,on='game_id', how='left')

    for s in stat_cols:
        if s == 'game_id':continue
        df['away_'+name+'_'+s] = df[s]
        df.drop(columns=s, inplace=True)
    
    assert df_len == len(df)
    
    # create diff stats
    for s in stat_cols:
        if s == 'game_id':continue
        df[name+'_'+s+'_diff']= df['home_'+name+'_'+s]-df['away_'+name+'_'+s]
    
    return df  

       
def get_stats_from_dist(dist):
    d = np.array(dist).astype('float')
    return d.mean(),d.std(),skew(d)

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

