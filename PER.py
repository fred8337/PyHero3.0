import numpy as np


class PER:
    pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(self.capacity, dtype=object)

        self.tree = np.zeros(2*self.capacity-1)

    def add(self, priority, data):
        tree_index = self.pointer + self.capacity -1
        self.data[self.pointer] = data

        if self.pointer >= self.capacity:
            self.pointer = 0

    def update(self, index, priority):