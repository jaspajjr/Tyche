from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
import pandas as pd
import numpy as np
import pymc3 as pm


def transform_results_to_teams_data(results_df):
    '''
    Take the results dataframe containing the data to train the model.
    '''
    teams = results_df['home_team'].unique()
    teams = pd.DataFrame(teams, columns=['team'])
    teams['i'] = teams.index

    return teams

def transform_results_to_training_df(results_df, teams):

    training_df = pd.merge(results_df, teams, left_on='home_team', right_on='team', how='left')
    training_df = training_df.rename(columns={'i': 'i_home'}).drop('team', 1)
    training_df = pd.merge(training_df, teams, left_on='away_team', right_on='team', how='left')
    training_df = training_df.rename(columns={'i': 'i_away'}).drop('team', 1)

    return training_df


def specify_model(training_df):
    '''
    This function sets up some basic parameters from the training_df, and then
    uses PyMC to fit the Dixon-Coles model.
    '''

    teams = training_df['home_team'].unique()
    teams = pd.DataFrame(teams, columns=['team'])
    teams['i'] = teams.index

    observed_home_goals = training_df['home_score'].values
    observed_away_goals = training_df['away_score'].values

    home_team = training_df['i_home'].values
    away_team = training_df['i_away'].values

    num_teams = len(training_df['i_home'].drop_duplicates())
    num_games = len(home_team)

    g = training_df.groupby('i_away')
    att_starting_points = np.log(g.away_score.mean())
    g = training_df.groupby('i_home')
    def_starting_points = -np.log(g.away_score.mean())

    print('Specifying Model')
    # specify model
    with pm.Model() as model:
        # global model parameters
        home = pm.Normal('home', 0, 0.0001)
        tau_att = pm.Gamma('tau_att', .1, .1)
        tau_def = pm.Gamma('tau_def', .1, .1)
        intercept = pm.Normal('intercept', 0, 0.0001)

        # team specific parameters
        atts_star = pm.Normal('atts_star', mu=0, tau=tau_att, shape=num_teams)
        defs_star = pm.Normal('defs_star', mu=0, tau=tau_def, shape=num_teams)

        #
        atts = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
        defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))

        home_theta = tt.exp(intercept + home + atts[home_team] + defs[away_team])
        away_theta = tt.exp(intercept + atts[away_team] + defs[home_team])

        # likelihood of observed data
        home_goals = pm.Poisson('home_goals', mu=home_theta, observed=observed_home_goals)
        away_goals = pm.Poisson('away_goals', mu=away_theta, observed=observed_away_goals)

    return model

def run_model(model):
    '''
    '''
    n_iterations = 10000
    print('Running Model with {0} iterations.'.format(n_iterations))
    with model:

        start = pm.find_MAP()
        step = pm.NUTS(state=start)
        trace = pm.sample(n_iterations, step, start=start, njobs=1)

    pm.trace
    return trace

def create_param_df(trace, training_df):
    '''
    Unpacks the trace into a usable dataframe of usable parameters.
    '''

    teams = transform_results_to_teams_data(training_df)
    atts_df = pd.DataFrame(trace.get_values('atts'), columns=teams.team.values.tolist())
    defs_df = pd.DataFrame(trace.get_values('defs'), columns=teams.team.values.tolist())
    home_df = pd.DataFrame(trace.get_values('home'), columns=['home']) #, columns=teams.team.values.tolist())
    intercept_df = pd.DataFrame(trace.get_values('intercept'), columns=['intercept']) #, columns=teams.team.values.tolist())
    atts_df.columns = ['atts_{}'.format(name) for name in atts_df.columns]
    defs_df.columns = ['defs_{}'.format(name) for name in defs_df.columns]
    param_df = pd.concat([atts_df, defs_df, home_df, intercept_df], axis=1, join='inner')

    return param_df
