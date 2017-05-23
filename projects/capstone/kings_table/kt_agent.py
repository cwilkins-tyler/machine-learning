import os
import csv
import time
import random
import klepto
import pickle
import logging
import datetime
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from collections import deque
from argparse import ArgumentParser
from kt_simulator import Simulator
from kt_environment import Environment

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
logging.basicConfig(format='%(asctime)s %(message)s', filename='kt_{}.log'.format(start_time))


class LearningAgent():
    MEMORY_SIZE = 500000  # number of observations to remember
    OBSERVATIONS = 50000  # number of games to play before starting to train
    MINI_BATCH_SIZE = 100  # size of mini batches
    MAX_MOVES = 200

    def __init__(self, env):
        self.epsilon = 0.9
        self.env = env
        self._checkpoint_path = 'kings_table_networks'
        self._start_time = start_time
        self.input_layer, self.output_layer = self._create_network()
        self._session = tf.Session()
        self.action = tf.placeholder("float", [None, self.MAX_MOVES])
        self.target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self.action), reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.target - readout_action))
        tf.summary.scalar('cost', self.cost)
        self.train_operation = tf.train.AdamOptimizer(0.1).minimize(self.cost)
        self._observations = deque()
        self.action_history = []  # The list of historical actions taken

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('board/train', self._session.graph)
        self.test_writer = tf.summary.FileWriter('board/test')

        self._session.run(tf.global_variables_initializer())

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)
        self._saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)

    def _create_network(self):
        # network weights
        convolution_weights_1 = tf.Variable(tf.truncated_normal([self.env.grid_width, self.env.grid_height, 1, 32],
                                                                stddev=0.01))
        mean = tf.reduce_mean(convolution_weights_1)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(convolution_weights_1 - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(convolution_weights_1))
        tf.summary.scalar('min', tf.reduce_min(convolution_weights_1))
        tf.summary.histogram('histogram', convolution_weights_1)

        feed_forward_weights_1 = tf.Variable(tf.truncated_normal([288, self.MAX_MOVES], stddev=0.01))

        input_layer = tf.placeholder("float", [None, self.env.grid_width, self.env.grid_height, 1])

        hidden_convolutional_layer_1 = tf.nn.relu(tf.nn.conv2d(input_layer, convolution_weights_1,
                                                               strides=[1, 4, 4, 1], padding="SAME"))

        hidden_convolutional_layer_1_flat = tf.reshape(hidden_convolutional_layer_1, [-1, 288])

        output_layer = tf.matmul(hidden_convolutional_layer_1_flat, feed_forward_weights_1)

        return input_layer, output_layer

    def choose_next_action(self, state, sim):
        new_action = np.zeros([self.MAX_MOVES])
        all_moves = sim.get_all_valid_actions()
        assert len(all_moves) <= self.MAX_MOVES, len(all_moves)
        input_state = np.reshape(np.array(state), (self.env.grid_width, self.env.grid_height, 1))
        allQ = self._session.run([self.output_layer], feed_dict={self.input_layer: [input_state]})
        valid_q = allQ[0]
        all_valid_moves = valid_q[0][:len(all_moves)]
        if self.epsilon > random.random():
            action_index = random.randrange(len(all_moves))
        else:
            action_index = np.argmax(all_valid_moves)

        try:
            chosen_action = all_moves[action_index]
        except IndexError:
            print(action_index, len(all_moves))
            raise

        new_action[action_index] = 1
        return chosen_action, new_action

    def train(self, game_number):
        # sample a mini_batch to train on
        mini_batch = random.sample(self._observations, self.MINI_BATCH_SIZE)

        states = [d[0] for d in mini_batch]
        actions = [d[1] for d in mini_batch]
        rewards = [d[2] for d in mini_batch]
        new_states = [d[3] for d in mini_batch]
        agents_expected_reward = []

        # this gives us the agents expected reward for each action we might take
        agents_reward_per_action = self._session.run(self.output_layer, feed_dict={self.input_layer: new_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][4]:
                # this was a terminal frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(rewards[i] + 0.99 * np.max(agents_reward_per_action[i]))

        # learn that these actions in these states lead to this reward
        run_metadata = tf.RunMetadata()
        _, summary = self._session.run([self.train_operation, self.merged], feed_dict={self.input_layer: states,
                                                                                       self.target: agents_expected_reward,
                                                                                       self.action: actions},
                                       run_metadata=run_metadata)
        self.train_writer.add_run_metadata(run_metadata, 'step{:02d}'.format(game_number))
        self.train_writer.add_summary(summary, '{:02d}'.format(game_number))
        game_number += 1
        return game_number

    def end_epoch(self, durations, average_durations, epoch_number):
        progress_csv = os.path.join(os.path.abspath(os.path.dirname(__file__)), self._checkpoint_path,
                                    'progress_{}.csv'.format(self._start_time))

        av_duration = float(sum(durations))/float(len(durations))
        print('Average game duration after epoch {}: {:.1f}'.format(epoch_number, av_duration))
        average_durations.append(av_duration)
        fields = [epoch_number, av_duration]
        headers = ['Epoch', 'AvDuration']
        if os.path.isfile(progress_csv):
            with open(progress_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        else:
            with open(progress_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerow(fields)

        self._saver.save(self._session, self._checkpoint_path + '/network', global_step=epoch_number)

    def calculate_rewards(self, attacker_win):
        game_observations = []
        history_iterator = 0
        for history_item in self.action_history:
            history_iterator += 1
            history_state, history_actions, history_new_state = history_item
            reward_multiplier = history_iterator / len(self.action_history)
            if attacker_win:
                reward = 1
            else:
                reward = -1

            step_reward = reward * reward_multiplier

            if history_iterator == len(self.action_history):
                game_over = True
            else:
                game_over = False

            experience = (history_state, history_actions, step_reward, history_new_state, game_over)
            game_observations.append(experience)

        return game_observations

    def play_game(self, env, training_mode, game_number, visualise_screen=False):
        game_over = False
        sim = Simulator(visualise=visualise_screen)
        state = sim.get_state()

        while not game_over:
            logger.debug('Choosing action')
            chosen_action, all_actions = self.choose_next_action(state, sim)

            logger.debug('Action chosen, starting step')
            move, new_state, reward = sim.step(chosen_action)
            logger.debug('Step taken')
            input_state = np.reshape(np.array(state), (env.grid_width, env.grid_height, 1))
            input_new_state = np.reshape(np.array(new_state), (env.grid_width, env.grid_height, 1))
            self.action_history.append((input_state, all_actions, input_new_state))
            tf.summary.image('game_state', input_new_state)
            if move.game_over:
                print('Game over in {} turns'.format(sim.round_number))
                logger.debug('Game over in {} turns'.format(sim.round_number))
                game_observations = self.calculate_rewards(move.king_killed)
                game_over = True
                sim.game_over()

                for experience in game_observations:
                    self._observations.append(experience)

                logger.debug('Observations stored')
                self.action_history = []

            if len(self._observations) > self.MEMORY_SIZE:
                self._observations.popleft()

            # only train if done observing
            if training_mode:
                logger.debug('Starting train')
                game_number = self.train(game_number)
                logger.debug('Training complete')

        return sim.round_number, game_number


def nn_run(test_mode, number_of_games, visualise_screen):
    env = Environment(verbose=True)
    print('Creating NN agent')
    logger.debug("Creating agent")
    agent = env.create_agent(LearningAgent)
    durations = deque()
    average_durations = []
    if test_mode:
        agent.epsilon = 0
        for i in range(number_of_games):
            print('Starting test game {}'.format(i))
            logger.debug('Starting test game {}'.format(i))
            agent.play_game(env, False, i, visualise_screen=visualise_screen)
    else:
        observations_file = 'observations.pkl'

        # if we have existing observations, use them
        if os.path.isfile(observations_file):
            print('Using stored observations')
            logger.debug('Using stored observations')
            ob_file = open(observations_file, 'rb')
            ob_contents = ob_file.read()
            observations = pickle.loads(ob_contents)
            agent._observations = observations['results']
        else:
            # observe for a number of games, using only random actions
            agent.epsilon = 1
            for i in range(agent.OBSERVATIONS):
                print('Starting observation game {}'.format(i))
                logger.debug('Starting observation game {}'.format(i))
                agent.play_game(env, False, i)
                tf.reset_default_graph()

            # save the observations
            ob_dir = klepto.archives.file_archive(observations_file, cached=True, serialized=True)
            ob_dir['results'] = agent._observations
            ob_dir.dump()

        global_step = 1
        for epoch in range(number_of_games + 1):
            print('Starting training game {} of {} using epsilon: {}'.format(epoch, number_of_games, agent.epsilon))
            logger.debug('Starting training game {}'.format(epoch))
            game_length, global_step = agent.play_game(env, True, global_step)
            tf.reset_default_graph()
            durations.append(game_length)
            if len(durations) > 10:
                durations.popleft()

            agent.epsilon -= (1 - 0.05) / number_of_games

            # every 5 games print the average duration and save the model
            if epoch % 5 == 0 and epoch != 0:
                agent.end_epoch(durations, average_durations, epoch)

            # every 1000 games run some test games with epsilon of zero
            if epoch % 1000 == 0 and epoch != 0:
                old_epsilon = agent.epsilon
                agent.epsilon = 0
                for i in range(10):
                    print('Starting test game {}'.format(i))
                    logger.debug('Starting test game {}'.format(i))
                    agent.play_game(env, False, global_step, visualise_screen=visualise_screen)

                # reset epsilon back to the old value
                agent.epsilon = old_epsilon

        # run some test games after training, and record stats


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-t", "--test", dest="test_mode",
                        action="store_true", help="Run the agent in test mode")

    parser.add_argument("-n", "--number-of-games", dest="number_of_games",
                        help="Number of games to run", type=int)

    parser.add_argument("-v", "--visualise_screen", dest="visualise",
                        action="store_true", help="Number of games to run")

    args = parser.parse_args()
    start_time = time.time()
    nn_run(args.test_mode, args.number_of_games, args.visualise)
    end_time = time.time()
    print(end_time - start_time)