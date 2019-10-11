

from abc import ABCMeta, abstractmethod

class agent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, env):
        pass

    @abstractmethod
    def act(self, observation):
        pass

    @abstractmethod
    def learn(self, *args):
        pass


class random(agent):
    """The world's simplest agent!"""
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, observation):
        return self.action_space.sample()

    def learn(self, *args):
        pass




