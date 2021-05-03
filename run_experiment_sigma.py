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

# Read all parameters into memory.
all_params = []
for filename in glob.glob("./params/sigma/*"):
    all_params.append(read_json(filename))

for params in all_params:
    torch.manual_seed(1234)
    if params['measure'] != 'gaussian':
        continue
    for sigma in [500, 800, 1000, 1500, 2000]:
        params['sigma'] = sigma
        print(f"Running for init type '{params['init_type']}' and sigma = {sigma}.", flush=True)
        # Run experiment.
        exp = Experiment(params, verbose=False)
        exp.run_multiple()
        exp.save_experiment()