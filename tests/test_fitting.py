from __future__ import division
import pytest
import pandas as pd
from Tyche.fitting import transform_results_to_teams_data, transform_results_to_training_df


def test_transform_results_to_teams_data_returns_df():
    # arrange
    results_df = pd.DataFrame({ 'id': [1, 11, 111],
                                'home_team': [1, 1, 1],
                                'away_team': [2, 2, 2]})
    # act
    teams_df = transform_results_to_teams_data(results_df)

    # assert
    assert isinstance(teams_df, pd.DataFrame)

def test_transform_results_to_training_df_returns_df():
    # arrange
    results_df = pd.DataFrame({ 'id': [1, 11, 111],
                                'home_team': [1, 1, 1],
                                'away_team': [2, 2, 2]})

    teams = transform_results_to_teams_data(results_df)
    # act
    training_df = transform_results_to_training_df(results_df, teams)

    # assert
    assert isinstance(training_df, pd.DataFrame)
