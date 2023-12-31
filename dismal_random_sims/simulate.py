import msprime
import numpy as np
import random
import itertools

class Simulation:

    def __init__(self, block_thetas, epoch_durations, migration_rates_fraction,
                 blocklen, mutation_rate, blocks_per_state=20_000, recombination_rate=0):
        """Msprime simulation

        Args:
            deme_ids (iterable): Names of demes, in DISMaL order (backwards in time)
            block_theta (iterable): Deme sizes in number of haploid individuals, in DISMaL order (backwards in time)
            epoch_durations (iterable): Durations of epoch 0 and 1 in 2 Ne generations
            migration_rates_fraction (iterable): Migration rates as fraction of population
            blocklen (int): Length of simulated blocks
            mutation_rate (float): Mutation rate per base
            blocks_per_state (int, optional): Number of block simulations per simulation. Defaults to 20_000.
            recombination_rate (int, optional): Recombination rate per base per generation. Defaults to 0.
        """

        # original parameters
        self.block_thetas = block_thetas
        self.epoch_durations = epoch_durations
        self.migration_rates_fraction = migration_rates_fraction
        self.blocklen = blocklen
        self.mutation_rate = mutation_rate
        self.blocks_per_state = blocks_per_state
        self.recombination_rate = recombination_rate
        self.deme_ids = ["pop1", "pop2", "pop1_anc", "pop2_anc", "ancestral"]

        # converted parameters for msprime
        self.site_theta = np.array(self.block_thetas)/self.blocklen
        self.deme_sizes_2N = np.array(self.site_theta)/(2*self.mutation_rate)
        self.epoch_durations_generations = np.array(self.epoch_durations) * self.deme_sizes_2N[2]
        self.split_times_generations = self.epoch_durations_generations[0], np.sum(self.epoch_durations_generations)

        self.demography = self.create_demography()
        self.tree_sequences = self.create_treesequences()
        self.s1, self.s2, self.s3 = [self.add_mutations(ts) for ts in self.tree_sequences]


    def create_demography(self):
        demography = msprime.Demography()

        for deme_idx, deme_id in enumerate(self.deme_ids):
            demography.add_population(name=deme_id, initial_size=self.deme_sizes_2N[deme_idx])

        demography.add_population_split(self.split_times_generations[0], 
                                        derived=[self.deme_ids[0]], 
                                        ancestral=self.deme_ids[2])
        demography.add_population_split(self.split_times_generations[0], 
                                        derived=[self.deme_ids[1]], 
                                        ancestral=self.deme_ids[3])
        demography.add_population_split(self.split_times_generations[1], 
                                        derived=[self.deme_ids[2], 
                                                 self.deme_ids[3]], 
                                                 ancestral=self.deme_ids[4])

        demography.set_migration_rate(self.deme_ids[1], self.deme_ids[0], self.migration_rates_fraction[0])
        demography.set_migration_rate(self.deme_ids[0], self.deme_ids[1], self.migration_rates_fraction[1])
        demography.set_migration_rate(self.deme_ids[3], self.deme_ids[2], self.migration_rates_fraction[2])
        demography.set_migration_rate(self.deme_ids[2], self.deme_ids[3], self.migration_rates_fraction[3])

        demography.sort_events()

        return demography
    
    def create_treesequences(self):
        ts_state1 = msprime.sim_ancestry(samples={self.deme_ids[0]: 2,
                                                  self.deme_ids[1]: 0},
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen,
                                                  recombination_rate=self.recombination_rate,
                                                  num_replicates=self.blocks_per_state, 
                                                  ploidy=2)
        ts_state2 = msprime.sim_ancestry(samples={self.deme_ids[0]: 0,
                                                  self.deme_ids[1]: 2}, 
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen, 
                                                  recombination_rate=self.recombination_rate,
                                                  num_replicates=self.blocks_per_state, 
                                                  ploidy=2)
        ts_state3 = msprime.sim_ancestry(samples={self.deme_ids[0]: 1, 
                                                  self.deme_ids[1]: 1}, 
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen, 
                                                  recombination_rate=self.recombination_rate,
                                                  num_replicates=self.blocks_per_state, ploidy=2)
        
        return ts_state1, ts_state2, ts_state3
    
    def add_mutations(self, ts):
        sim = np.zeros(self.blocks_per_state)
        for replicate_index, ts in enumerate(ts):
            ts_muts = msprime.sim_mutations(
                ts, rate=self.mutation_rate, discrete_genome=False)
            sim[replicate_index] = ts_muts.divergence(
                sample_sets=[[0], [2]], span_normalise=False)
            
        return sim


    @staticmethod    
    def M2m(Ms, twoNs):
        """Convert big M (number of migrants) to small m (migrant fraction of population)"""
        Ms = np.array(Ms)
        twoNs = np.array(twoNs)

        return Ms/(2*twoNs)


    @staticmethod
    def m2M(ms, twoNs):
        """Convert small m (migrant fraction of population) to big M (number of migrants)"""
        ms = np.array(ms)
        twoNs = np.array(twoNs)

        return 2 * twoNs * ms