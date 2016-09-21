from __future__ import division
import Tyche.fitting as fitting
import Tyche.pipeline_utils as pipeline_utils
import pandas as pd

def get_results():
    '''
    This function will read the results.csv file.
    Which is the training data.
    This file must contain the fields:
        home_team,
        away_team,
        home_score,
        away_score
    '''
    df = pd.read_csv('./results.csv') #, sep='\t')

    return df

def get_fixtures():
    '''
    This function will read the fixtures.csv file.
    This file must contain the fields:
        home_team,
        away_team
    '''
    df = pd.read_csv('./fixtures.csv')

    return df


def get_param_df():
    '''
    This starts the process.
    '''
    training_df = get_results()

    teams = training_df['home_team'].unique()
    teams = pd.DataFrame(teams, columns=['team'])
    teams['i'] = teams.index

    results_df = get_results()

    training_df = fitting.transform_results_to_training_df(results_df,
        fitting.transform_results_to_teams_data(results_df))

    model = fitting.specify_model(training_df)

    trace = fitting.run_model(model)

    param_df = fitting.create_param_df(trace, training_df)

    return param_df


if __name__ == '__main__':
    param_df = get_param_df()

    fixture_df = get_fixtures()

    #print param_df.head()
    #import time

    #print("Sleeping")
    #time.sleep(100)


    predictions_df = pipeline_utils.make_predictions_from_fixture_list(fixture_df,
        param_df, 2500)

    print("Done")

    predictions_df.to_csv('./predictions.csv')
