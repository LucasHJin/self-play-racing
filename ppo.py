"""
Needed functions/logic:

parse_args
make_env
Agent class
train
    collect_roolout
    compute_advantages
    ppo_update
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical