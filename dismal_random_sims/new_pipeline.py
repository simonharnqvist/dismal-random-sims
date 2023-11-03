# new pipeline design:
# 1. Read YAML
# 2. @transform YAML into random parameter values for n replicates; save as npy where each row is a parameter set
# 3. @split into each simulation replicate, saving a compressed simulation.npz ON SCRATCH containing [params, s1, s2, s3]
# 4. @transform each replicate into a compressed modelled.npz containing [true_params, inferred_params, s1, s2, s3, expected_s1, expected_s2, expected_s3]
# merge outputs
# 5. @split all outputs into single to generate
#       * DataFrame [param, epoch, true_val, inferred_val, abs_bias, rel_bias]
#       * DataFrame [*parameters, relative entropy (normalise distributions for this), sum_relative_bias]
# 6. @transform summarised output to key plots
#       * Rel bias by parameter value (scatterplot)
#       * D_KL for each mig parameter against T (heatmap)
#       * Rel bias for each mig parameter against T (heatmap)
#       * Sum(m) vs sum(t) (heatmap)
# cleanup: delete files on scratch (give option to retain)


@transform(input=args.yaml_spec, filter=suffix(".yaml"), output="params.npy", tmpdir)
def sample_random_parameters(yaml_spec, np_params, tmpdir):
    """Read YAML file and randomly sample parameters for each simulation replicate"""

    # allow option to fix parameters - needs to be implemented in YAML reader

@split(input=sample_random_parameters, f"{simulation_unique_id}_sim.npz")
def simulate_replicate()
    """Simulate demography and mutations with msprime"""

    for param_set in Parallel(sample_random_parameters): # is this the best way to parallelise?
        sim = Simulation()
        np.savez(
            [sim.s1, sim.s2, sim.s3, (SIM_PARAMS)]
    )

@transform(input=simulate_replicate, filter=suffix("_sim.npz"), output="_inferred.npz", tmpdir)
def fit_model(infile, outfile, tmpdir):
    """Fit model with DISMaL"""
    np.load(infile)
    mod = three_epoch_gim(s1, s2, s3)
    np.savez(
        [mod.s1, mod.s2, mod.s3, s1, s2, s3, sim_params, inferred_params, model_info]  # give arrays names
    )

@merge(fit_model, "_results.npz")
def merge_output(models, outfile):
    """Merge output from each model to single file"""
    np.concat -> np.savez()

@split(merge_output, ["parameters.csv", "models.csv"])
def generate_output_data(infile, outfiles):
    """Process output data into tables"""

    #       * DataFrame [param, epoch, true_val, inferred_val, abs_bias, rel_bias]
#       * DataFrame [*parameters, relative entropy (normalise distributions for this), sum_relative_bias]


@transform
def generate_plots(generate_output_data, [""])
    """Produce plots"""
#       * Rel bias by parameter value (scatterplot)
#       * D_KL for each mig parameter against T (heatmap)
#       * Rel bias for each mig parameter against T (heatmap)
#       * Sum(m) vs sum(t) (heatmap)


def cleanup_tmp():
    """Delete files created in temp dir"""
    pass

