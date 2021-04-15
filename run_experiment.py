import sys

import torch

from src.argparser import read_json
from src.experiment import Experiment


if len(sys.argv) != 2:
    raise Exception("Filename must be specified as argument.")

# Read parameters.
name = sys.argv[1]
params = read_json('params/'+name+'.json')

torch.manual_seed(1234)
# Run experiment.
exp = Experiment(params)
exp.train()
print(exp.losses)