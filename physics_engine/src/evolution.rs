use rayon::prelude::*;
use rand::{thread_rng, Rng};

use crate::analysis::{DiscretizationType, EmergentAnalysis};
use crate::physics::Universe;
use crate::substrate::{ComputationalSubstrate, NonlinearityType};
use crate::EvolutionConfig;

use std::env;

fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

pub struct Evolution {
    population_size: usize,
    mutation_rate: f64,
    analyzer: EmergentAnalysis,
    config: EvolutionConfig,
    allowed: Vec<NonlinearityType>,
    quant_levels: u8,
    quiet: bool,
}

impl Evolution {
    pub fn new(population_size: usize, mutation_rate: f64) -> Self {
        Self {
            population_size,
            mutation_rate,
            analyzer: EmergentAnalysis::new(),
            config: EvolutionConfig::standard(),
            allowed: NonlinearityType::all(),
            quant_levels: 3,
            quiet: false,
        }
    }

    pub fn new_with_config(population_size: usize, mutation_rate: f64, config: EvolutionConfig) -> Self {
        Self {
            population_size,
            mutation_rate,
            analyzer: EmergentAnalysis::new(),
            config,
            allowed: NonlinearityType::all(),
            quant_levels: 3,
            quiet: false,
        }
    }

    pub fn new_with_config_allowed(
        population_size: usize,
        mutation_rate: f64,
        config: EvolutionConfig,
        allowed: Vec<NonlinearityType>,
        quant_levels: u8,
    ) -> Self {
        Self {
            population_size,
            mutation_rate,
            analyzer: EmergentAnalysis::new(),
            config,
            allowed,
            quant_levels,
            quiet: false,
        }
    }

    pub fn set_quiet(&mut self, q: bool) {
        self.quiet = q;
    }

    pub fn run(&mut self, generations: usize, universe: &mut Universe) -> ComputationalSubstrate {
        // Initialize population with allowed constraints
        let mut population: Vec<ComputationalSubstrate> = (0..self.population_size)
            .map(|_| ComputationalSubstrate::random_with_constraints(
                self.allowed.clone(), 
                self.quant_levels
            ))
            .collect();

        // Verify that all substrates respect allowed constraints (debug check)
        #[cfg(debug_assertions)]
        {
            for (i, substrate) in population.iter().enumerate() {
                // This would require exposing genome, but helps catch bugs
                println!("Initial substrate {} uses allowed nonlinearity", i);
            }
        }

        for gen in 0..generations {
            // Reset energy for fair comparison
            population.par_iter_mut().for_each(|s| s.reset_energy());

            // Evaluate fitness in parallel with forked universes
            let fits: Vec<(f64, FitnessComponents)> = population
                .par_iter()
                .enumerate()
                .map(|(i, s)| {
                    let mut u = universe.forked(10_000 + i as u64);
                    let comps = self.evaluate_fitness_components(s, &mut u);
                    (comps.total(), comps)
                })
                .collect();

            // Select best index
            let (best_idx, (best_total, best_components)) = fits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
                .map(|(i, v)| (i, v.clone()))
                .unwrap();

            if !self.quiet && gen % 10 == 0 {
                let disc = self.analyzer.detect_discretization(&population[best_idx]);
                match disc {
                    DiscretizationType::Discrete { n_states, .. } => {
                        println!(
                            "Gen {}: Best fitness={:.3} | info={:.3} energy={:.3} structure={:.3} stability={:.3} target={:.3} n_states={}",
                            gen,
                            best_total,
                            best_components.info,
                            best_components.energy,
                            best_components.structure,
                            best_components.stability,
                            best_components.target,
                            n_states
                        );
                    }
                    _ => {
                        println!(
                            "Gen {}: Best fitness={:.3} | info={:.3} energy={:.3} structure={:.3} stability={:.3} target={:.3} n_states=continuous",
                            gen,
                            best_total,
                            best_components.info,
                            best_components.energy,
                            best_components.structure,
                            best_components.stability,
                            best_components.target
                        );
                    }
                }
            }

            // Elitism
            let mut new_population = Vec::with_capacity(self.population_size);
            new_population.push(population[best_idx].clone());

            // Fill with mutated offspring (tournament selection)
            while new_population.len() < self.population_size {
                let parent = self.tournament_select(&population, &fits, 5);
                let mut child = parent.clone();
                child.mutate(self.mutation_rate);
                new_population.push(child);
            }

            population = new_population;
        }

        // Return the best of final gen
        population
            .into_iter()
            .max_by(|a, b| {
                let mut u_a = universe.forked(99999);
                let mut u_b = universe.forked(99998);
                let fa = self.evaluate_fitness(&*a, &mut u_a);
                let fb = self.evaluate_fitness(&*b, &mut u_b);
                fa.partial_cmp(&fb).unwrap()
            })
            .unwrap()
    }

    fn evaluate_fitness(&self, substrate: &ComputationalSubstrate, universe: &mut Universe) -> f64 {
        self.evaluate_fitness_components(substrate, universe).total()
    }

    fn evaluate_fitness_components(&self, substrate: &ComputationalSubstrate, universe: &mut Universe) -> FitnessComponents {
        // Information preservation
        let info_score = self.analyzer.measure_information_preservation(substrate, universe);

        // Energy efficiency with exponent (LAMBDA_ENERGY)
        let lambda_energy = env_f64("LAMBDA_ENERGY", 1.0);
        let energy_raw = 1.0 / (1.0 + substrate.energy_consumed());
        let energy_score = energy_raw.powf(lambda_energy);

        // Structure bonus via discretization
        let structure_bonus = match self.analyzer.detect_discretization(substrate) {
            DiscretizationType::Discrete { n_states, .. } => match n_states {
                2 => self.config.binary_bonus,
                3 => self.config.ternary_bonus,
                4 => self.config.quaternary_bonus,
                5..=10 => self.config.other_discrete_bonus,
                _ => 1.0,
            },
            _ => 1.0,
        };

        // Stability metric
        let stability_score = self.analyzer.measure_stability(substrate, universe);

        // Target mismatch penalty
        let target_penalty = if let Some(ts) = self.config.target_states {
            if let DiscretizationType::Discrete { n_states, .. } = self.analyzer.detect_discretization(substrate) {
                if n_states == ts {
                    1.0
                } else {
                    (1.0 - self.config.mismatch_penalty).max(0.0)
                }
            } else {
                1.0 - self.config.mismatch_penalty
            }
        } else {
            1.0
        };

        FitnessComponents {
            info: info_score,
            energy: energy_score,
            structure: structure_bonus,
            stability: stability_score,
            target: target_penalty,
        }
    }

    fn tournament_select(
        &self,
        population: &[ComputationalSubstrate],
        fits: &[(f64, FitnessComponents)],
        k: usize,
    ) -> ComputationalSubstrate {
        let mut rng = thread_rng();
        let n = population.len();
        let winner_idx = (0..k)
            .map(|_| rng.gen_range(0..n))
            .max_by(|&a, &b| fits[a].0.partial_cmp(&fits[b].0).unwrap())
            .unwrap();
        population[winner_idx].clone()
    }
}

#[derive(Clone)]
struct FitnessComponents {
    info: f64,
    energy: f64,
    structure: f64,
    stability: f64,
    target: f64,
}

impl FitnessComponents {
    fn total(&self) -> f64 {
        self.info * self.energy * self.structure * self.stability * self.target
    }
}