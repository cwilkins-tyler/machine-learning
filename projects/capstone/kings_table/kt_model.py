import os
import json
import hashlib
import itertools
import numpy as np
from glob import glob
from datetime import datetime
from collections import deque, defaultdict
from random import shuffle
from logging import getLogger
from threading import Lock
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from kt_simulator import *

logger = getLogger(__name__)


# these are from AGZ nature paper
class VisitStats:
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0

    def __str__(self):
        return 'VisitStats: {}, {}'.format(self.a, self.sum_n)


class ActionStats:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

    def __str__(self):
        return 'ActionStats: {}, {}, {}, {}'.format(self.n, self.w, self.q, self.p)

    def __repr__(self):
        return 'Class ActionStats {}, {}, {}, {}'.format(self.n, self.w, self.q, self.p)


class KTModel:
    def __init__(self, mkdir=True, model='best'):
        self.model = None
        self.digest = None
        self.filenames = None
        self.sim = None
        self.tree = None
        self.node_lock = defaultdict(Lock)
        self.data = []
        self.loss_weights = [1.25, 1.0]
        self.batch_size = 384
        self.epoch_to_checkpoint = 1
        self.games_in_file = 1
        self.data_dir = os.environ.get("DATA_DIR", self._data_dir)
        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"
        self.simulation_num_per_move = 10  # up this later
        self.cnn_filter_num = 256
        self.cnn_first_filter_size = 5
        self.cnn_filter_size = 3
        self.res_layer_num = 7
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.distributed = False
        self.input_depth = 18
        self.dataset_size = 100000
        self.dataset = deque()
        self.labels = self.create_uci_labels()
        self.n_labels = int(len(self.labels))
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")

        if model == 'best':
            self.config_path = os.path.join(self.model_dir, "model_best_config.json")
            self.weight_path = os.path.join(self.model_dir, "model_best_weight.h5")
        elif model == 'new':
            # use the latest next generation model
            all_model_dirs = []
            for d in os.listdir(self.next_generation_model_dir):
                if os.path.isdir(os.path.join(self.next_generation_model_dir, d)):
                    all_model_dirs.append(os.path.join(self.next_generation_model_dir, d))
            latest_dir = max(all_model_dirs, key=os.path.getmtime)
            self.config_path = os.path.join(self.next_generation_model_dir, latest_dir, self.next_generation_model_config_filename)
            self.weight_path = os.path.join(self.next_generation_model_dir, latest_dir, self.next_generation_model_weight_filename)
        else:
            # create a new model
            model_dir = os.path.join(self.next_generation_model_dir, self.next_generation_model_dirname_tmpl % model_id)
            if mkdir:
                os.makedirs(model_dir, exist_ok=True)
            self.config_path = os.path.join(model_dir, self.next_generation_model_config_filename)
            self.weight_path = os.path.join(model_dir, self.next_generation_model_weight_filename)

    def self_play(self, num_games):
        logger.debug('Starting Self-Play Mode')
        if not self.load(self.config_path, self.weight_path):
            self.build()

        game_details = []
        for i in range(num_games):
            self.sim = Simulator(visualise=False)
            print('Playing game: {}'.format(i))
            game_details.append(self.play_game())
            print('Game {} complete'.format(i))
            if i % self.games_in_file == 0:
                self.save_game_data_to_file()
                self.data = []

        return game_details

    def play_game(self):
        game_over = False

        while not game_over:
            state = self.sim.get_state()
            chosen_action = self.action(state)
            # print('Model chose action: {}'.format(chosen_action))
            self.sim.step(chosen_action)
            # print('Step complete')
            if self.sim.move.game_over:
                logger.debug('Game over in {} turns'.format(self.sim.round_number))
                game_over = True
                attacker_win = self.sim.game_over()
                for move in self.data:
                    move += [attacker_win]

                return attacker_win, self.sim.round_number

    def action(self, env, can_stop=False) -> str:
        self.reset()

        # for tl in range(self.play_config.thinking_loop):
        root_value, naked_value = self.search_moves(env)
        policy = self.calc_policy(env)
        my_action = int(np.random.choice(range(self.n_labels), p=self.apply_temperature(policy,
                                                                                        self.sim.round_number)))

        while self.labels[my_action] not in self.sim.board.get_all_valid_actions(self.sim.move.a_turn):
            my_action = int(np.random.choice(range(self.n_labels), p=self.apply_temperature(policy,
                                                                                            self.sim.round_number)))

        if can_stop and root_value <= -0.8 and self.sim.round_number > 5:
            # noinspection PyTypeChecker
            print('Root value: {}'.format(root_value))
            print('Round number: {}'.format(self.sim.round_number))
            return None
        else:
            if self.labels[my_action] not in self.sim.board.get_all_valid_actions(self.sim.move.a_turn):
                print('Selected invalid action *****************')

            self.data.append([env, list(policy)])
            return self.labels[my_action]

    def search_moves(self, env) -> (float, float):
        vals = []
        for i in range(self.simulation_num_per_move):
            # print('Simulating move: {}'.format(i))
            temp_sim = Simulator(visualise=False, state=env)
            vals.append(self.search_my_move(temp_sim, env, is_root_node=True))

        return np.max(vals), vals[0]  # vals[0] is kind of racy

    def search_my_move(self, simulator, env, is_root_node=False) -> float:
        """
        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)
        :return: leaf value
        """
        if simulator.move.game_over:
            #print('We have a winner')
            return -1

        state = self.flatten_env(env)
        #print('Current board: ')
        #print(simulator.board.dump_board())

        with self.node_lock[state]:
            if state not in self.tree:
                # print('State not in tree, returning')
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v  # I'm returning everything from the POV of side to move
            # SELECT STEP
            action_t = self.select_action_q_and_u(simulator, state, is_root_node)
            #print('Action chosen in search: {}'.format(action_t))
            virtual_loss = 3
            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        simulator.step(action_t)
        leaf_v = self.search_my_move(simulator, env)  # next move from enemy POV
        leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env) -> (np.ndarray, float):
        """ expand new leaf, this is called only once per state
        this is called with state locked
        insert P(a|s), return leaf_v
        """
        leaf_p, leaf_v = self.model.predict(np.array([[env]]))
        return leaf_p[0], leaf_v

    def select_action_q_and_u(self, simulator, env, is_root_node):
        # this method is called with state locked
        my_visitstats = self.tree[env]

        current_valid_actions = simulator.board.get_all_valid_actions(simulator.move.a_turn)
        if my_visitstats.p is not None:  # push p to edges
            tot_p = 1e-8
            for mov in current_valid_actions:
                mov_i = self.labels.index(mov)
                mov_p = my_visitstats.p[mov_i]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = 0.25
        c_puct = 1.5
        dir_alpha = 0.3

        best_s = -9999
        best_a = None

        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1 - e) * p_ + e * np.random.dirichlet([dir_alpha])
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s and action in current_valid_actions:
                best_s = b
                best_a = action

        if not best_a:
            print('No best action found, selecting random valid action')
            print(my_visitstats.a.keys())
            print(current_valid_actions)
            print(simulator.board.dump_board())
            return random.choice(current_valid_actions)

        if best_a not in current_valid_actions:
            print('********************  Selected an invalid action')

        return best_a

    def calc_policy(self, env):
        """calc π(a|s0)
        :return:
        """
        state = self.flatten_env(env)

        my_visitstats = self.tree[state]
        policy = np.zeros(self.n_labels)
        for action, a_s in my_visitstats.a.items():
            policy[self.labels.index(action)] = a_s.n

        policy /= np.sum(policy)
        return policy

    def apply_temperature(self, policy, turn):
        tau_decay_rate = 0.99
        tau = np.power(tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.n_labels)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret

    def reset(self):
        self.tree = defaultdict(VisitStats)

    def start(self):
        if not self.load(self.config_path, self.weight_path):
            self.build()
            self.save(self.config_path, self.weight_path)
        self.training()

    def training(self):
        self.compile_model()

        self.filenames = deque(self.get_game_data_filenames())
        shuffle(self.filenames)
        total_steps = 0

        while True:
            self.fill_queue()
            steps = self.train_epoch(self.epoch_to_checkpoint)
            total_steps += steps
            self.save_current_model()
            a = self.dataset
            while len(a) > self.dataset_size / 2:
                a.popleft()

    def fill_queue(self):
        if len(self.filenames) == 0:
            return

        while self.filenames:
            filename = self.filenames.popleft()
            print('Parsing file: {}'.format(filename))
            game_data = self.load_game_data_from_file(filename)
            if len(self.dataset) < self.dataset_size:
                for game_step in game_data:
                    self.dataset.append(game_step)
                    assert len(game_step) == 3, game_step

                #if len(self.filenames) > 0:
                #    filename = self.filenames.popleft()
                #    print('Continuing with {}'.format(filename))

    def train_epoch(self, epochs):
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        state_ary = state_ary.reshape((state_ary.shape[0], 1, 11, 11))
        tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=self.batch_size, histogram_freq=1)
        self.model.fit(state_ary, [policy_ary, value_ary], batch_size=self.batch_size, epochs=epochs,
                       shuffle=True, validation_split=0.02, callbacks=[tensorboard_cb])
        steps = (state_ary.shape[0] // self.batch_size) * epochs
        return steps

    def collect_all_loaded_data(self):
        state_array = []
        policy_array = []
        value_array = []
        for game_step in self.dataset:
            state, policy, value = game_step
            assert len(state) == 11
            state_array.append(state)
            policy_array.append(policy)
            value_array.append(value)

        state_array_np = np.array(state_array, dtype=np.float32)
        policy_array_np = np.asarray(policy_array, dtype=np.float32)
        value_array_np = np.asarray(value_array, dtype=np.float32)
        return state_array_np, policy_array_np, value_array_np

    def compile_model(self):
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']  # avoid overfit for supervised
        self.model.compile(optimizer=opt, loss=losses, loss_weights=self.loss_weights)

    def build(self):
        in_x = x = Input((1, 11, 11))

        # (batch, channels, height, width)
        x = Conv2D(filters=self.cnn_filter_num, kernel_size=self.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg),
                   name="input_conv-" + str(self.cnn_first_filter_size) + "-" + str(self.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(self.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(self.l2_reg),
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(self.n_labels, kernel_regularizer=l2(self.l2_reg), activation="softmax",
                           name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(self.l2_reg),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(self.value_fc_size, kernel_regularizer=l2(self.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(self.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x, index):
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=self.cnn_filter_num, kernel_size=self.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg),
                   name=res_name + "_conv1-" + str(self.cnn_filter_size) + "-" + str(self.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=self.cnn_filter_num, kernel_size=self.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg),
                   name=res_name + "_conv2-" + str(self.cnn_filter_size) + "-" + str(self.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    @staticmethod
    def create_uci_labels():
        labels_array = []
        letters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for l1 in range(len(letters)):
            for n1 in range(len(numbers)):
                destinations = [(t, n1) for t in range(len(numbers))] + \
                               [(l1, t) for t in range(len(letters))]
                for (l2, n2) in destinations:
                    if (l1, n1) != (l2, n2) and l2 in range(11) and n2 in range(11):
                        move = (letters[l1], numbers[n1], letters[l2], numbers[n2])
                        labels_array.append(move)

        logger.debug('Total potential moves: {}'.format(len(labels_array)))
        return labels_array

    @staticmethod
    def flatten_env(env):
        return ' '.join([str(x) for x in list(itertools.chain.from_iterable(env))])

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    @staticmethod
    def _project_dir():
        d = os.path.dirname
        return d(os.path.abspath(__file__))

    @property
    def _data_dir(self):
        return os.path.join(self._project_dir(), "data")

    def get_game_data_filenames(self):
        pattern = os.path.join(self.play_data_dir, self.play_data_filename_tmpl % "*")
        files = list(sorted(glob(pattern)))
        return files

    def save_game_data_to_file(self):
        game_id = datetime.now().strftime('%Y%m%d-%H%M%S.%f')
        target_game_file = os.path.join(self.play_data_dir, self.play_data_filename_tmpl % game_id)
        logger.info('Saving game data to {}'.format(target_game_file))
        with open(target_game_file, 'wt') as f:
            json.dump(self.data, f)

    @staticmethod
    def load_game_data_from_file(filename):
        with open(filename, 'rt') as f:
            return json.load(f)

    def save_current_model(self):
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(self.next_generation_model_dir, self.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, self.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, self.next_generation_model_weight_filename)
        self.save(config_path, weight_path)

    def save_as_best_model(self):
        self.save(self.config_path, self.weight_path)

    def load(self, config_path, weight_path):
        if os.path.exists(config_path) and os.path.exists(weight_path):
            print('Loading model from {}'.format(config_path))
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.model._make_predict_function()
            self.digest = self.fetch_digest(weight_path)
            return True
        else:
            print('No model found at {} or {}'.format(config_path, weight_path))
            exit(0)
            return False

    def save(self, config_path, weight_path):
        logger.debug('Saving model to {}'.format(config_path))
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
