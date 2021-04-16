import sys
import glob
import torch

from src.argparser import read_json
from src.experiment import Experiment


###  MORTEN COMMENT OUT
#if len(sys.argv) != 2:
#    raise Exception("Filename must be specified as argument.")

# Read parameters.
# name = sys.argv[1]
### 

for filename in glob.glob("./params/*"):
    params = read_json(filename)
    torch.manual_seed(1234)
    # Run experiment.
    exp = Experiment(params)
    exp.run_multiple()
    exp.save_experiment()