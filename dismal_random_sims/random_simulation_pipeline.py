from ruffus import *
import yaml
from joblib import Parallel, delayed
import json
import datetime
import secrets
from argparse import ArgumentParser
import tqdm
import math
import numpy as np
from dismal_random_sims.simulate import Simulation
from dismal_random_sims.functions import random_param_value
from dismal.models import three_epoch_gim, three_epoch_iim, three_epoch_sec, three_epoch_iso


# new pipeline design:
# 1. Read YAML
# 2. @transform YAML into random parameter values for n replicates; save as npy where each row is a parameter set
# 3. @split into each simulation replicate, saving a compressed simulation.npz ON SCRATCH containing [params, s1, s2, s3]
# 4. @transform each replicate into a compressed modelled.npz containing [true_params, inferred_params, s1, s2, s3, expected_s1, expected_s2, expected_s3]
# 5. @merge all outputs into single  
#
# cleanup: delete files on scratch

parser = ArgumentParser()
parser.add_argument("--yaml-spec", help="YAML specifying parameters for random simulations")
parser.add_argument("--threads", help="Number of threads to use; use -1 for all threads. Defaults to 1 (no parallelisation).", default=1, type=int)
parser.add_argument("--simulation-id", help="Simulation ID, either to control name of simulation or to rerun failed sim. Default is {DATE}_{UNIQUE_HEX}", default=None)
args = parser.parse_args()

if args.simulation_id is None:
    date = datetime.datetime.now().strftime("%Y%m%d")
    unique_hex = secrets.token_hex(5)
    SIMULATION_ID = f"{date}_{unique_hex}"
else:
    SIMULATION_ID = args.simulation_id

@split(args.yaml_spec, [f"{SIMULATION_ID}.s1.npz",
                        f"{SIMULATION_ID}.s2.npz",
                        f"{SIMULATION_ID}.s3.npz",
                        f"{SIMULATION_ID}.sim_params.json"], 
                        args.threads)
def simulate(yaml_spec, outfiles, threads):
    
    with open(yaml_spec, "r") as f:
        yaml_spec = yaml.safe_load(f)

    def _sim_wrapper(yaml_spec):

        block_thetas = [random_param_value(yaml_spec["thetas"]["distribution"], distr_params) 
                for distr_params in 
                [yaml_spec["thetas"]["epoch1"]["pop1"], yaml_spec["thetas"]["epoch1"]["pop2"],
                yaml_spec["thetas"]["epoch2"]["pop1"], yaml_spec["thetas"]["epoch2"]["pop2"],
                yaml_spec["thetas"]["epoch3"]["pop1"]]]
        
        epoch_durations = [random_param_value(yaml_spec["epoch_durations"]["distribution"],
                                            yaml_spec["epoch_durations"]["epoch1"]),
                            random_param_value(yaml_spec["epoch_durations"]["distribution"],
                                            yaml_spec["epoch_durations"]["epoch2"])]
        
        migration_rates = [random_param_value(yaml_spec["migration_rates"]["distribution"], distr_params) 
                for distr_params in 
                [yaml_spec["migration_rates"]["epoch1"]["rate1"], yaml_spec["migration_rates"]["epoch1"]["rate2"],
                yaml_spec["migration_rates"]["epoch2"]["rate1"], yaml_spec["migration_rates"]["epoch2"]["rate2"]]]

        return Simulation(block_thetas, epoch_durations, migration_rates,
                    yaml_spec["blocklen"], yaml_spec["mutation_rate"], 
                    blocks_per_state=yaml_spec["blocks_per_state"], recombination_rate=yaml_spec["recombination_rate"])

    sims = Parallel(n_jobs=threads, prefer="threads")(
        delayed(_sim_wrapper)(yaml_spec) 
        for _ in tqdm.tqdm(range(yaml_spec["num_replicates"])))
    
    np.savez(outfiles[0], [sim.s1 for sim in sims])
    np.savez(outfiles[1], [sim.s2 for sim in sims])
    np.savez(outfiles[2], [sim.s3 for sim in sims])

    simulated_params = [{"block_thetas": list(sim.block_thetas),
         "epoch_durations": list(sim.epoch_durations),
         "migration_rates_fraction": list(sim.migration_rates_fraction),
         "recombination_rate": sim.recombination_rate} for sim in sims]
    
    with open(outfiles[3], "w") as f:
        json.dump(simulated_params, f)
    

@merge(simulate,
        f"{SIMULATION_ID}_results.json", args.yaml_spec, args.threads)
def infer(infiles, outfile, yaml_spec, threads):

    with open(yaml_spec, "r") as f:
        yaml_spec = yaml.safe_load(f)

    model_type = yaml_spec["infer_with"]
    blocklen = yaml_spec["blocklen"]

    if model_type.lower() == "iso":
        mod = three_epoch_iso()
    elif model_type.lower() == "iim":
        mod = three_epoch_iim()
    elif model_type.lower() == "sec":
        mod = three_epoch_sec()
    elif model_type.lower() == "gim":
        mod = three_epoch_gim()
    else:
        raise ValueError(f"{model_type} not recognised as valid model")
        
    s1s, s2s, s3s = [np.load(infiles[i])["arr_0"] for i in range(3)]
    assert s1s.shape[0] == s2s.shape[0] == s3s.shape[0]
    assert s1s.shape[0] == yaml_spec["num_replicates"]

    try:

        if threads == 1:
            mods = [mod.fit(s1s[i], s2s[i], s3s[i], 
                            yaml_spec["blocklen"], None, None, None, False) 
                            for i in tqdm.tqdm(range(yaml_spec["num_replicates"]))]
        else:
            mods = Parallel(n_jobs=threads, prefer="threads")(
            delayed(mod.fit)(s1s[i], s2s[i], s3s[i], 
                            yaml_spec["blocklen"], None, None, None, False) 
                            for i in tqdm.tqdm(range(yaml_spec["num_replicates"])))
            
    except Exception:
        pass
        
    results = [{
            "thetas_block": list(mod.thetas_block),
            "thetas_site": list(mod.thetas_site),
            "migration_rates": list(mod.migration_rates),
            "epoch_durations": list(mod.ts_2n),
            "fitted_s1": list(mod.fitted_s1),
            "fitted_s2": list(mod.fitted_s2),
            "fitted_s3": list(mod.fitted_s3),
            "optimiser": mod.optimiser,
            "negll": mod.negll,
            "claic": mod.claic
        } for mod in mods]
    
    with open(outfile, "w") as f:
        json.dump(results, f)


def main():
    pipeline_run()

if __name__ == "__main__":
    main()
        

# @merge
# def analyse_results()