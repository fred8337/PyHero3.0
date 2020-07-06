import gym
import tensorflow as tf
import random
import numpy as np
import cv2

from tensorflow import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
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
        self.lr = 0.01
        self.lr_decay = 0.01

        # Initialize replay memory
        self.replay_memory = deque(maxlen=MEMORY_SIZE)
        # print(self.env.action_space.n) #[Output: ] Discrete(2)
        # print(self.env.observation_space) # [Output: ] Box(4,)
        self.steps_to_remember = 4
        self.rows = 32
        self.cols = 48
        self.state_memory = np.zeros((self.steps_to_remember, self.rows, self.cols))

        # Create model and target model
        self.model = self.create_model(model)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self, model=None):
        if model is not None:
            model = keras.models.load_model(model)
        else:
            input_shape = (4, self.rows, self.cols)
            input = Input

            model = Sequential()

            model.add(Conv2D(64, 5, (3, 3), padding="valid", input_shape=input_shape, data_format="channels_first"))
            model.add(Activation("relu"))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(64, 4, (2, 2), padding="valid", data_format="channels_first"))
            model.add(Activation("relu"))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            # model.add(Dropout(0.2))

            model.add(Conv2D(64, 3, (1, 1), padding="valid", data_format="channels_first"))
            model.add(Activation("relu"))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            # model.add(Dropout(0.2))

            model.add(Flatten())

            model.add(Dense(512, input_dim=4, activation="relu", kernel_initializer="he_uniform"))
            model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
            model.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
            model.add(Dense(2, activation="linear", kernel_initializer="he_uniform"))
            model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=[])
        return model

    def preprocess_state(self, state):
        return np.array(state).reshape(1, 4)

    def save_transition(self, transition):
        self.replay_memory.append(transition)

    def show_state(self, state=None):
        #Dead
        if state is None:
            for i in range(4):
                cv2.imshow('image '+str(i), self.state_memory[:,:,i])
                cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            state = state[0]
            for i in range(4):
                img = state[:,:,i]
                cv2.imshow('image ' + str(i), img)
                cv2.waitKey(0)
            cv2.destroyAllWindows()


    def get_image_state(self):
        img = self.env.render(mode="rgb_array")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized<255] = 0
        img_rgb_resized = img_rgb_resized/255

        # print(self.state_memory.shape)

        self.state_memory = np.roll(self.state_memory, 1, axis=0)
        self.state_memory[0,:,:] = img_rgb_resized

        # res = self.state_memory.reshape((4,self.rows, self.cols))
        res = np.expand_dims(self.state_memory, axis=0)

        # cv2.imshow('image', self.state_memory[:,:,0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(res.shape)
        return res

    def train(self, episodes=1000):
        steps = 0
        best_score = 0
        scores = []
        for episode in range(episodes):
            self.env.reset()
            state = self.get_image_state()
            # self.show_state(state)
            # print(self.preprocess_state(state))
            done = False
            score = 0
            while not done:
                if random.random()<=self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.model.predict(state))
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -100
                new_state = self.get_image_state()
                # self.show_state(new_state)
                self.save_transition((state, action, reward, new_state, done))
                state = new_state
                score += 1
                # check = steps%STEPS_BEFORE_UPDATE
            steps += 1
            if steps%STEPS_BEFORE_UPDATE ==0:
                self.target_model.set_weights(self.model.get_weights())
            scores.append(score)
            if score>best_score:
                best_score = score
                self.model.save("my_model_CNN")
            print("Episode: "+str(episode)+" Score: "+str(score)+" Avg:"+str(np.mean(scores[-50:]))+" eps: "+str(self.epsilon))
            self.experience_replay()
            self.scores.append(score)


    def experience_replay(self, batch_size=32):
        X = []
        y = []
        mini_batch = random.sample(self.replay_memory, min(batch_size, len(self.replay_memory)))
        for state, action, reward, new_state, done in mini_batch:
            # print(state.shape)
            y_target = self.target_model.predict(state)
            y_target[0][action] = reward if done else reward+DISCOUNT*np.max(self.target_model.predict(new_state)[0])
            # print(state[0].shape)
            # cv2.imshow('image', state[0][:,:,0])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            X.append(state[0])
            y.append(y_target[0])

        self.model.fit(np.array(X), np.array(y), batch_size=len(X), epochs=1, verbose=1, shuffle=False)

        if self.epsilon > EPSILON_MINIMUM:
            self.epsilon *= EPSILON_DECAY







if __name__ == "__main__":
    solver = dqn_cart()
    for episode in range(10):
    # print(solver.get_image_state())
        solver.train(1000)

