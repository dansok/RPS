import random
from enum import Enum


class Action(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

    @classmethod
    def list_actions(cls):
        return list(cls)

    @classmethod
    def get_random(cls):
        """Return a random action."""
        return random.choice(cls.list_actions())

    @classmethod
    def length(cls):
        return len(cls)

    def __len__(self):
        return len(self.list_actions())
