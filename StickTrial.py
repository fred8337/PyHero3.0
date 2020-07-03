import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras


def preprocess_state(state):
    return np.array(state).reshape(1, 4)


def test_model(episodes=1000):
    model = keras.models.load_model("my_model")
    env = gym.make('CartPole-v0')
    scores = []
    for episode in range(episodes):
        state = preprocess_state(env.reset())
        score = 0
        done = False
        while not done:
            action = np.argmax(model.predict(state))
            # env.render()
            new_state, reward, done, _ = env.step(action)
            state = preprocess_state(new_state)
            score += 1
        scores.append(score)
        print(score)

    print("The average score was: "+str(np.mean(np.array(scores))))


if __name__ == "__main__":
    test_model(100)
