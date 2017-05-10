import os
import csv
import random
import datetime
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from collections import deque
from optparse import OptionParser
from kt_simulator import Simulator
from kt_environment import Environment


class Agent(object):
    """Base class for all agents."""

    def __init__(self):
        self.state = None
        self.primary_agent = False

    def reset(self, destination=None, testing=False):
        pass

    def get_state(self):
        return self.state


class LearningAgent(Agent):
    MEMORY_SIZE = 500000  # number of observations to remember
    MINI_BATCH_SIZE = 5  # size of mini batches
    MAX_MOVES = 160

    def __init__(self, env):
        super(LearningAgent, self).__init__()     # Set the agent in the environment

        self.epsilon = 0.9
        self.env = env
        self._checkpoint_path = 'kings_table_networks'
        self._start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.input_layer, self.output_layer = self._create_network()
        self._session = tf.Session()
        self.action = tf.placeholder("float", [None, self.MAX_MOVES])
        self.target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self.action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self.target - readout_action))
        self.train_operation = tf.train.AdamOptimizer(0.1).minimize(cost)
        self._observations = deque()

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
        valid_q = allQ[0][:len(all_moves) + 1]

        if self.epsilon > random.random():
            action_index = random.randrange(len(all_moves))
        else:
            action_index = np.argmax(valid_q)

        try:
            chosen_action = all_moves[action_index]
        except IndexError:
            print(action_index, len(all_moves))
            raise

        new_action[action_index] = 1
        return chosen_action, new_action

    def train(self):
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
        self._session.run(self.train_operation, feed_dict={self.input_layer: states,
                                                           self.target: agents_expected_reward,
                                                           self.action: actions})

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


def nn_run(test_mode, number_of_games):
    env = Environment(verbose=True)
    print('Creating NN agent')
    agent = env.create_agent(LearningAgent)
    durations = deque()
    average_durations = []
    if test_mode:
        agent.epsilon = 0

    for i in range(number_of_games):
        print('Starting game {}'.format(i))
        game_over = False
        sim = Simulator()
        state = sim.get_state()

        while not game_over:
            chosen_action, all_actions = agent.choose_next_action(state, sim)

            move, new_state, reward = sim.step(chosen_action)
            input_state = np.reshape(np.array(state), (env.grid_width, env.grid_height, 1))
            input_new_state = np.reshape(np.array(new_state), (env.grid_width, env.grid_height, 1))
            if move.game_over:
                print('Game over in {} turns'.format(sim.round_number))
                game_over = True
                sim.game_over()
                durations.append(sim.round_number)
                if len(durations) > 10:
                    durations.popleft()

            experience = (input_state, all_actions, reward, input_new_state, game_over)
            agent._observations.append(experience)

            if len(agent._observations) > agent.MEMORY_SIZE:
                agent._observations.popleft()

            # only train if done observing
            if len(agent._observations) > 5:
                agent.train()

        # every 5 games print the average duration and save the model
        if i % 5 == 0 and i != 0:
            agent.end_epoch(durations, average_durations, i)

    # save the model again after finishing the last epoch
    agent.end_epoch(durations, average_durations, i)


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-t", "--test", dest="test_mode",
                      action="store_true", help="Run the agent in test mode")

    parser.add_option("-n", "--number-of-games", dest="number_of_games",
                      help="Number of games to run")

    (options, args) = parser.parse_args()

    nn_run(options.test_mode, options.number_of_games)