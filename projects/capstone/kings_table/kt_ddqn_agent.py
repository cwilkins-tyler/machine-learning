import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from kt_simulator import Simulator
from kt_environment import Environment

EPISODES = 300


# Double DQN Agent for kings table
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these are hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_ddqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        for layer in model.layers:
            print(layer.name, len(layer.inbound_nodes), len(layer.outbound_nodes))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state, sim, env):
        new_action = np.zeros([self.action_size])
        all_moves = sim.get_all_valid_actions()
        assert len(all_moves) <= self.action_size, len(all_moves)
        input_state = np.reshape(np.array(state), (1, env.grid_width, env.grid_height))
        q_values = self.model.predict(input_state)

        valid_q = q_values[0]
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

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self, env):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, env.grid_width, env.grid_height))
        update_target = np.zeros((batch_size, env.grid_width, env.grid_height))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)
        print(target.shape, target_next.shape, target_val.shape)
        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                print(i, a)
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    env = Environment(verbose=True)

    # get size of state and action from environment
    state_size = [env.grid_width, env.grid_height]
    action_size = 200

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        sim = Simulator(visualise=False)
        state = sim.get_state()
        print('Episode: {}'.format(e))
        while not done:
            # get action for the current state and go one step in environment
            chosen_action, all_actions = agent.get_action(state, sim, env)
            print('Action: {}'.format(chosen_action))
            move, new_state, reward = sim.step(chosen_action)
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, all_actions, reward, new_state, done)
            # every time step do the training
            agent.train_model(env)
            score += reward
            state = new_state

            if move.game_over:
                # every episode update the target model to be same as model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
                done = True

        # save the model
        #if e % 50 == 0:
        #    agent.model.save_weights("./save_model/kt_ddqn.h5")