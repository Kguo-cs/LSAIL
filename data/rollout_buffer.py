import random
from collections import deque

class RolloutBuffer():
    def __init__(self) :

        self.state_list=deque(maxlen=50)

    def __len__(self) -> int:
        return len(self.state_list)

    def append(self,experience):
        self.state_list.append(experience)

    def sample_state(self):

        batch = random.sample(self.state_list, 1)

        return batch[0]
