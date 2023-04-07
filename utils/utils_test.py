import d4rl
import gym
import numpy as np
import torch

from utils import get_dataset

def test_get_dataset():
    dataset = get_dataset('antmaze-large-diverse-v1', 40, 1, 0.2)

if __name__ == '__main__' :
    test_get_dataset()