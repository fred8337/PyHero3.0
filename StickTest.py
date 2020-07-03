import gym
import tensorflow as tf
import random
import numpy as np

from tensorflow import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from collections import deque

#HyperParameters
MEMORY_SIZE = 50_000
STEPS_BEFORE_UPDATE = 5
EPSILON_MINIMUM = 0.1
EPSILON_DECAY = 0.995
EPSILON = 1
DISCOUNT = 0.95



# Own Tensorboard class. From SentDex
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class dqn_cart:
    def __init__(self, model=None):
        # Initialize environment
        self.env = gym.make('CartPole-v0')

        # Specific to cartPole
        self.scores = []

        # HyperParameters
        self.epsilon = 1
        self.lr = 0.001
        self.lr_decay = 0.01

        # Create model and target model
        self.model = self.create_model(model)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Initialize replay memory
        self.replay_memory = deque(maxlen=MEMORY_SIZE)
        # print(self.env.action_space.n) #[Output: ] Discrete(2)
        # print(self.env.observation_space) # [Output: ] Box(4,)

    def create_model(self, model=None):
        if model is not None:
            model = keras.models.load_model(model)
        else:
            model = Sequential()
            model.add(Dense(16, input_dim=4, activation="relu"))
            model.add(Dense(16, activation="relu"))
            model.add(Dense(2, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=self.lr), metrics=[])
        return model

    def preprocess_state(self, state):
        return np.array(state).reshape(1, 4)

    def save_transition(self, transition):
        self.replay_memory.append(transition)

    def train(self, episodes=1000):
        steps = 0
        best_score = 0
        for episode in range(episodes):
            state = self.preprocess_state(self.env.reset())
            # print(self.preprocess_state(state))
            done = False
            score = 0
            while not done:
                if random.random()<=self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.model.predict(state))
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.preprocess_state(new_state)
                self.save_transition((state, action, reward, new_state, done))
                state = new_state
                score += 1
                # check = steps%STEPS_BEFORE_UPDATE
            steps += 1
            if steps%STEPS_BEFORE_UPDATE ==0:
                self.target_model.set_weights(self.model.get_weights())

            if score >= best_score:
                best_score = score
                self.model.save("my_model")
            print("Episode: "+str(episode)+" Score: "+str(score)+" eps: "+str(self.epsilon))
            self.experience_replay()
            self.scores.append(score)

    def experience_replay(self, batch_size=32):
        X = []
        y = []
        mini_batch = random.sample(self.replay_memory, min(batch_size, len(self.replay_memory)))
        for state, action, reward, new_state, done in mini_batch:
            y_target = self.target_model.predict(state)
            y_target[0][action] = reward if done else reward+DISCOUNT*np.max(self.target_model.predict(new_state)[0])
            X.append(state[0])
            y.append(y_target[0])

        self.model.fit(np.array(X), np.array(y), batch_size=len(X), epochs=1, verbose=0, shuffle=False)

        if self.epsilon > EPSILON_MINIMUM:
            self.epsilon *= EPSILON_DECAY







if __name__ == "__main__":
    solver = dqn_cart(model="my_model")
    solver.train(10000)

