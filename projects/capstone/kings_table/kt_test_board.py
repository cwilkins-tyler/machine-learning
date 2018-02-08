import pytest
from .kt_simulator import Board, Simulator, INITIAL_STATE
mini_board_state = [[4, 1, 4], [1, 3, 1], [4, 1, 4]]


@pytest.fixture(scope='module')
def mini_board():
    test_board = Board(mini_board_state)
    test_board.initialize_groups()
    test_board.initialize_pieces()
    return test_board


class TestSimulator:
    def test_initial_sim(self):
        test_sim = Simulator()
        assert test_sim.board.get_current_state() == INITIAL_STATE


def test_board_setup():
    test_board = Board(INITIAL_STATE)
    assert test_board.dim == len(INITIAL_STATE)


def test_board_non_standard_setup():
    test_board = Board(mini_board_state)
    assert test_board.dim == len(mini_board_state)


def test_board_initialise_pieces(mini_board):
    assert len(mini_board.Attackers) == 4
    assert len(mini_board.Defenders) == 1
    assert len(mini_board.Kings) == 1
    assert len(mini_board.Pieces) == 5


def test_board_has_piece(mini_board):
    assert mini_board.cell_has_piece((1, 1))
    assert not mini_board.cell_has_piece((0, 2))


def test_board_has_attacking_piece(mini_board):
    assert mini_board.cell_has_attacking_piece((0, 1))
    assert not mini_board.cell_has_attacking_piece((1, 1))
