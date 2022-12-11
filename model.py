import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Callable

class NeRF():
    def __init__(self):
        self.layer_count = 8
        self.hidden_dimension = 256
        