import os
import pytest
from .kt_model import KTModel
from collections import deque
mini_board_state = [[4, 1, 4], [1, 3, 1], [4, 1, 4]]


@pytest.fixture()
def mini_model():
    test_model = KTModel(mkdir=False)
    test_model.play_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    return test_model


def test_model_initial_states(mini_model):
    assert mini_model.n_labels == 2420


def test_model_load_best(mini_model):
    model_dir = r'C:\Users\chris\OneDrive\Documents\GitHub\machine-learning\projects\capstone\kings_table\data\model'
    assert mini_model.config_path == r'{}\model_best_config.json'.format(model_dir)
    assert mini_model.weight_path == r'{}\model_best_weight.h5'.format(model_dir)


def test_model_load_next_gen(mini_model):
    test_model = KTModel(mkdir=False, model='new')
    model_dir = r'C:\Users\chris\OneDrive\Documents\GitHub\machine-learning\projects\capstone\kings_table\data\model\next_generation'
    assert test_model.config_path.startswith(model_dir)
    assert test_model.config_path.endswith('.json')
    assert test_model.weight_path.startswith(model_dir)
    assert test_model.weight_path.endswith('.h5')


def test_model_flatten(mini_model):
    assert mini_model.flatten_env(mini_board_state) == '4 1 4 1 3 1 4 1 4'


def test_model_get_filenames_no_files():
    test_model = KTModel(mkdir=False)
    test_model.play_data_dir = os.path.join(os.path.abspath(__file__), 'non_existent_test_data')
    assert test_model.get_game_data_filenames() == []


def test_model_get_filenames(mini_model):
    assert os.path.isdir(mini_model.play_data_dir)
    assert len(mini_model.get_game_data_filenames()) == 2


def test_fill_queue(mini_model):
    mini_model.filenames = deque(mini_model.get_game_data_filenames())
    mini_model.fill_queue()
    assert len(mini_model.dataset) == 4
    assert len(mini_model.dataset[0]) == 3
    assert len(mini_model.dataset[0][0]) == 11
    assert len(mini_model.dataset[0][1]) == 1
    assert mini_model.dataset[0][2] == -1
    assert len(mini_model.dataset[1]) == 3


def test_collect_all_loaded_data(mini_model):
    mini_model.filenames = deque(mini_model.get_game_data_filenames())
    mini_model.fill_queue()

    # make sure we haven't double filled the dataset from running two consecutive tests
    expected_game_steps = 4
    assert len(mini_model.dataset) == expected_game_steps

    state_ary, policy_ary, value_ary = mini_model.collect_all_loaded_data()
    assert state_ary.shape == (expected_game_steps, 11, 11)
    assert policy_ary.shape == (expected_game_steps, 1)
    assert value_ary.shape == (expected_game_steps, )
