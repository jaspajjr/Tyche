import pandas as pd
import numpy as np
from scipy.stats import poisson

def tau(xx, yy, lambda_param, mu, rho):
    if xx == 0 & yy == 0:
        return 1 - (lambda_param*mu*rho)
    elif xx == 0 & yy== 1:
        return 1 + (lambda_param * rho)
    elif xx == 1 & yy == 0:
        return 1 + (1 + (mu * rho))
    elif xx == 1 & yy == 1:
        return 1 - rho
    else:
        return 1

def calculate_mu(home_team_name, away_team_name, param_df, row_num):
    '''
    This function calculates the mu value for the given combination of home team and away team,
    given the param_df
    '''
    home_atts = param_df['atts_{0}'.format(home_team_name)].iloc[row_num]
    home_defs = param_df['defs_{0}'.format(home_team_name)].iloc[row_num]
    away_atts = param_df['atts_{0}'.format(away_team_name)].iloc[row_num]
    away_defs = param_df['defs_{0}'.format(away_team_name)].iloc[row_num]
    intercept = param_df['intercept'].iloc[row_num]
    home = param_df['home'].iloc[row_num]
    mu = np.exp(home_atts + away_defs + home + intercept)
    return mu


def calculate_lambda_val(home_team_name, away_team_name, param_df, row_num):
    '''
    This function calculates the lambda value for the given combination of home team and away team,
    given the param_df
    '''
    home_atts = param_df['atts_{0}'.format(home_team_name)].iloc[row_num]
    home_defs = param_df['defs_{0}'.format(home_team_name)].iloc[row_num]
    away_atts = param_df['atts_{0}'.format(away_team_name)].iloc[row_num]
    away_defs = param_df['defs_{0}'.format(away_team_name)].iloc[row_num]
    intercept = param_df['intercept'].iloc[row_num]
    lambda_val = np.exp(away_atts + home_defs + intercept)
    return lambda_val

def create_scaling_matrix(mu, lambda_val, rho):
    '''
    '''
    results = []
    for xx, yy in zip([0, 1, 0, 1], [0, 0, 1, 1]):
        results.append(tau(xx, yy, lambda_param=lambda_val, mu=mu, rho=rho))

    result_array = np.array(results)
    result_matrix = result_array.reshape(2, 2)
    result_matrix = result_matrix.transpose()
    return result_matrix


def create_likelihood_matrix(mu, lambda_val, max_goals=6):
    '''
    '''
    home_dist = poisson(mu).pmf([x for x in xrange(max_goals)])
    away_dist = poisson(lambda_val).pmf([x for x in xrange(max_goals)])
    #
    # need to clearl define the shapes for home_dist and away_dist
    home_dist.shape = (6, 1)
    away_dist.shape = (6, 1)

    result_matrix = np.matmul(home_dist, away_dist.transpose())

    return result_matrix

def get_fixture_result(result_matrix, scaling_matrix):
    '''
    This function takes the result matric and the scaling matrix for a given
    fixture and returns a dictionary with the home / draw / away probabilities.
    '''
    result_matrix[0:2, 0:2] = np.dot(result_matrix[0:2, 0:2], scaling_matrix.transpose())
    home_abs = np.sum(np.tril(result_matrix))
    draw_abs = np.sum(np.diag(result_matrix))
    away_abs = np.sum(np.triu(result_matrix))

    # transform to probabilities
    total_absolute = home_abs + draw_abs + away_abs
    home_prob = ((home_abs / total_absolute) * 100)
    draw_prob = ((draw_abs / total_absolute) * 100)
    away_prob = ((away_abs / total_absolute) * 100)

    result_dict = {
        'home_prob': home_prob,
        'draw_prob': draw_prob,
        'away_prob': away_prob}

    return result_dict

def calculate_fixture_result(fixture_row, param_df, home_team_name,
    away_team_name):
    '''
    Using the param_df, calculate the home_prob, draw_prob, and away_prob.
    '''
    row_num = 1
    mu = calculate_mu(home_team_name, away_team_name, param_df, row_num)
    lambda_val = calculate_lambda_val(home_team_name, away_team_name,
        param_df, row_num)
    result_matrix = create_likelihood_matrix(mu, lambda_val)

    rho = param_df['home'].iloc[row_num]
    scaling_matrix = create_scaling_matrix(mu, lambda_val, rho)
    result_dict = get_fixture_result(result_matrix, scaling_matrix)
    result_dict['home_team_name'] = home_team_name
    result_dict['away_team_name'] = away_team_name

    return result_dict

def make_predictions_from_fixture_list(fixture_df, param_df, row_num):
    '''
    From a list of fixtures, and a dataframe of prediction parameters, and the
    row_num to sample from, return a dataframe with a row for each fixture.
    This must contain home team,away team, home win probability, draw
    probability, and away probability.
    '''
    home_team_list = []
    away_team_list = []
    home_prob_list = []
    draw_prob_list = []
    away_prob_list = []
    for row in fixture_df.iterrows():
        fixture_row = row[1]
        home_team_name = fixture_row['home_team']
        away_team_name = fixture_row['away_team']
        fixture_result_dict = calculate_fixture_result(fixture_row, param_df,
            home_team_name, away_team_name)
        home_team_list.append(fixture_result_dict['home_team_name'])
        away_team_list.append(fixture_result_dict['away_team_name'])
        home_prob_list.append(fixture_result_dict['home_prob'])
        draw_prob_list.append(fixture_result_dict['draw_prob'])
        away_prob_list.append(fixture_result_dict['away_prob'])

    data = {'home_team': home_team_list,
            'away_team': away_team_list,
            'home_prob': home_prob_list,
            'draw_prob': draw_prob_list,
            'away_prob': away_prob_list
            }
    predictions_df = pd.DataFrame(data)
    return predictions_df

def get_actual_result(row, results_df):
    '''
    Takes a row containing 'home_team' and 'away_team', and looks up the fixture
    in the results_df to find out who won.
    '''

    home_team = row['home_team']
    away_team = row['away_team']
    result = results_df[(results_df['home_team'] == home_team) &
        (results_df['away_team'] == away_team)]
    assert len(result) == 1
    result = result.iloc[0]
    if result['home_score'] > result['away_score']:
        return 'H'

    if result['home_score'] == result['away_score']:
        return 'D'

    if result['home_score'] < result['away_score']:
        return 'L'
