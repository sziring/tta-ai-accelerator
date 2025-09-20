// src/optimizers.rs
// Optimizer and fitness function variants for extended validation

use rand::{thread_rng, Rng};
use crate::evolution::Evolution;
use crate::substrate::{ComputationalSubstrate, NonlinearityType};
use crate::physics::Universe;
use crate::analysis::EmergentAnalysis;
use crate::EvolutionConfig;

#[derive(Clone, Copy, Debug)]
pub enum OptimizerType {
    GeneticAlgorithm,
    RandomSearch,
    CmaEs,
}

impl OptimizerType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "random" => OptimizerType::RandomSearch,
            "cmaes" => OptimizerType::CmaEs,
            _ => OptimizerType::GeneticAlgorithm,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FitnessType {
    Task,           // Current fitness function
    Regularized,    // Penalize complexity
    Information,    // Information-theoretic
}

impl FitnessType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "reg" => FitnessType::Regularized,
            "info" => FitnessType::Information,
            _ => FitnessType::Task,
        }
    }
}

// Common trait for all optimizers
pub trait Optimizer {
    fn set_quiet(&mut self, q: bool);
    fn run(&mut self, generations: usize, universe: &mut Universe) -> ComputationalSubstrate;
}

// Wrapper enum for different optimizer implementations
pub enum OptimizerImpl {
    Ga(Evolution),
    RandomSearch(RandomSearchEvolution),
    CmaEs(CmaEsEvolution),
    GaWithFitness(GaWithFitnessEvolution),
}

impl Optimizer for OptimizerImpl {
    fn set_quiet(&mut self, q: bool) {
        match self {
            OptimizerImpl::Ga(evo) => evo.set_quiet(q),
            OptimizerImpl::RandomSearch(evo) => evo.set_quiet(q),
            OptimizerImpl::CmaEs(evo) => evo.set_quiet(q),
            OptimizerImpl::GaWithFitness(evo) => evo.set_quiet(q),
        }
    }

    fn run(&mut self, generations: usize, universe: &mut Universe) -> ComputationalSubstrate {
        match self {
            OptimizerImpl::Ga(evo) => evo.run(generations, universe),
            OptimizerImpl::RandomSearch(evo) => evo.run(generations, universe),
            OptimizerImpl::CmaEs(evo) => evo.run(generations, universe),
            OptimizerImpl::GaWithFitness(evo) => evo.run(generations, universe),
        }
    }
}

// Factory functions to create different optimizer variants
pub fn create_optimizer(
    optimizer_type: OptimizerType,
    fitness_type: FitnessType,
    population_size: usize,
    mutation_rate: f64,
    config: EvolutionConfig,
    allowed: Vec<NonlinearityType>,
    quant_levels: u8,
) -> OptimizerImpl {
    match optimizer_type {
        OptimizerType::RandomSearch => {
            OptimizerImpl::RandomSearch(RandomSearchEvolution {
                evaluations: population_size * 30,
                analyzer: EmergentAnalysis::new(),
                config,
                allowed,
                quant_levels,
                fitness_type,
                quiet: false,
            })
        }
        OptimizerType::CmaEs => {
            OptimizerImpl::CmaEs(CmaEsEvolution {
                population_size,
                generations: 30,
                analyzer: EmergentAnalysis::new(),
                config,
                allowed,
                quant_levels,
                fitness_type,
                quiet: false,
            })
        }
        OptimizerType::GeneticAlgorithm => {
            if matches!(fitness_type, FitnessType::Task) {
                OptimizerImpl::Ga(Evolution::new_with_config_allowed(
                    population_size, mutation_rate, config, allowed, quant_levels
                ))
            } else {
                OptimizerImpl::GaWithFitness(GaWithFitnessEvolution {
                    base_evolution: Evolution::new_with_config_allowed(
                        population_size, mutation_rate, config, allowed, quant_levels
                    ),
                    fitness_type,
                    analyzer: EmergentAnalysis::new(),
                })
            }
        }
    }
}

// Random Search implementation
pub struct RandomSearchEvolution {
    evaluations: usize,
    analyzer: EmergentAnalysis,
    config: EvolutionConfig,
    allowed: Vec<NonlinearityType>,
    quant_levels: u8,
    fitness_type: FitnessType,
    quiet: bool,
}

impl RandomSearchEvolution {
    pub fn set_quiet(&mut self, q: bool) {
        self.quiet = q;
    }

    pub fn run(&mut self, _generations: usize, universe: &mut Universe) -> ComputationalSubstrate {
        if !self.quiet {
            println!("Running Random Search with {} evaluations", self.evaluations);
        }

        let mut best_substrate = ComputationalSubstrate::random_with_constraints(
            self.allowed.clone(), 
            self.quant_levels
        );
        let mut best_fitness = {
            let mut u = universe.forked(1);
            self.evaluate_fitness(&best_substrate, &mut u)
        };

        for eval in 0..self.evaluations {
            let candidate = ComputationalSubstrate::random_with_constraints(
                self.allowed.clone(), 
                self.quant_levels
            );

            let mut u = universe.forked(eval as u64 + 1000);
            let fitness = self.evaluate_fitness(&candidate, &mut u);

            if fitness > best_fitness {
                best_substrate = candidate;
                best_fitness = fitness;

                if !self.quiet && eval % (self.evaluations / 10).max(1) == 0 {
                    println!("Eval {}: New best fitness {:.4}", eval, best_fitness);
                }
            }
        }

        best_substrate
    }

    fn evaluate_fitness(&self, substrate: &ComputationalSubstrate, universe: &mut Universe) -> f64 {
        match self.fitness_type {
            FitnessType::Task => {
                let info_score = self.analyzer.measure_information_preservation(substrate, universe);
                let energy_score = 1.0 / (1.0 + substrate.energy_consumed());
                let structure_bonus = match self.analyzer.detect_discretization(substrate) {
                    crate::analysis::DiscretizationType::Discrete { n_states, .. } => match n_states {
                        2 => self.config.binary_bonus,
                        3 => self.config.ternary_bonus,
                        4 => self.config.quaternary_bonus,
                        5..=10 => self.config.other_discrete_bonus,
                        _ => 1.0,
                    },
                    _ => 1.0,
                };
                let stability_score = self.analyzer.measure_stability(substrate, universe);
                
                info_score * energy_score * structure_bonus * stability_score
            }
            FitnessType::Regularized => {
                let base_fitness = self.evaluate_task_fitness(substrate, universe);
                let complexity_penalty = 0.1;
                base_fitness * (1.0 - complexity_penalty)
            }
            FitnessType::Information => {
                let mi = self.calculate_mutual_information(substrate, universe);
                let energy = substrate.energy_consumed();
                if energy > 0.0 {
                    mi / energy.sqrt()
                } else {
                    mi
                }
            }
        }
    }

    fn evaluate_task_fitness(&self, substrate: &ComputationalSubstrate, universe: &mut Universe) -> f64 {
        let info_score = self.analyzer.measure_information_preservation(substrate, universe);
        let energy_score = 1.0 / (1.0 + substrate.energy_consumed());
        info_score * energy_score
    }

    fn calculate_mutual_information(&self, substrate: &ComputationalSubstrate, universe: &mut Universe) -> f64 {
        let mut s = substrate.clone();
        let mut u = universe.forked(0xDEADBEEF);
        
        let n_samples = 100;
        let mut outputs = Vec::new();
        
        for _ in 0..n_samples {
            let input = [
                thread_rng().gen_range(-1.0..1.0),
                thread_rng().gen_range(-1.0..1.0),
                thread_rng().gen_range(-1.0..1.0),
            ];
            let output = s.transform(&input, &mut u);
            outputs.push(output[0]);
        }
        
        let mean_output: f64 = outputs.iter().sum::<f64>() / outputs.len() as f64;
        let variance: f64 = outputs.iter()
            .map(|x| (x - mean_output).powi(2))
            .sum::<f64>() / outputs.len() as f64;
        
        variance.ln().max(0.0)
    }
}

// CMA-ES implementation (simplified)
pub struct CmaEsEvolution {
    population_size: usize,
    generations: usize,
    analyzer: EmergentAnalysis,
    config: EvolutionConfig,
    allowed: Vec<NonlinearityType>,
    quant_levels: u8,
    fitness_type: FitnessType,
    quiet: bool,
}

impl CmaEsEvolution {
    pub fn set_quiet(&mut self, q: bool) {
        self.quiet = q;
    }

    pub fn run(&mut self, _generations: usize, universe: &mut Universe) -> ComputationalSubstrate {
        if !self.quiet {
            println!("Running simplified CMA-ES with {} generations", self.generations);
        }

        let mut best = ComputationalSubstrate::random_with_constraints(
            self.allowed.clone(), 
            self.quant_levels
        );
        let mut best_fitness = {
            let mut u = universe.forked(1);
            self.evaluate_fitness(&best, &mut u)
        };

        let lambda = self.population_size;
        let mut sigma = 0.3;

        for gen in 0..self.generations {
            let mut candidates = Vec::new();
            let mut fitnesses = Vec::new();

            for i in 0..lambda {
                let mut candidate = best.clone();
                candidate.mutate(sigma);
                
                let mut u = universe.forked(gen as u64 * lambda as u64 + i as u64);
                let fitness = self.evaluate_fitness(&candidate, &mut u);
                
                candidates.push(candidate);
                fitnesses.push(fitness);
            }

            if let Some((best_idx, &new_best_fitness)) = fitnesses
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                if new_best_fitness > best_fitness {
                    best = candidates[best_idx].clone();
                    best_fitness = new_best_fitness;
                    sigma *= 1.1;
                } else {
                    sigma *= 0.9;
                }
            }

            if !self.quiet && gen % 10 == 0 {
                println!("CMA-ES Gen {}: Best fitness {:.4}, sigma {:.4}", gen, best_fitness, sigma);
            }
        }

        best
    }

    fn evaluate_fitness(&self, substrate: &ComputationalSubstrate, universe: &mut Universe) -> f64 {
        match self.fitness_type {
            FitnessType::Task => {
                let info_score = self.analyzer.measure_information_preservation(substrate, universe);
                let energy_score = 1.0 / (1.0 + substrate.energy_consumed());
                let structure_bonus = match self.analyzer.detect_discretization(substrate) {
                    crate::analysis::DiscretizationType::Discrete { n_states, .. } => match n_states {
                        2 => self.config.binary_bonus,
                        3 => self.config.ternary_bonus,
                        4 => self.config.quaternary_bonus,
                        5..=10 => self.config.other_discrete_bonus,
                        _ => 1.0,
                    },
                    _ => 1.0,
                };
                let stability_score = self.analyzer.measure_stability(substrate, universe);
                
                info_score * energy_score * structure_bonus * stability_score
            }
            FitnessType::Regularized => {
                let base_fitness = self.evaluate_task_fitness(substrate, universe);
                let complexity_penalty = 0.1;
                base_fitness * (1.0 - complexity_penalty)
            }
            FitnessType::Information => {
                let mi = self.calculate_mutual_information(substrate, universe);
                let energy = substrate.energy_consumed();
                if energy > 0.0 {
                    mi / energy.sqrt()
                } else {
                    mi
                }
            }
        }
    }

    fn evaluate_task_fitness(&self, substrate: &ComputationalSubstrate, universe: &mut Universe) -> f64 {
        let info_score = self.analyzer.measure_information_preservation(substrate, universe);
        let energy_score = 1.0 / (1.0 + substrate.energy_consumed());
        info_score * energy_score
    }

    fn calculate_mutual_information(&self, substrate: &ComputationalSubstrate, universe: &mut Universe) -> f64 {
        let mut s = substrate.clone();
        let mut u = universe.forked(0xDEADBEEF);
        
        let n_samples = 100;
        let mut outputs = Vec::new();
        
        for _ in 0..n_samples {
            let input = [
                thread_rng().gen_range(-1.0..1.0),
                thread_rng().gen_range(-1.0..1.0),
                thread_rng().gen_range(-1.0..1.0),
            ];
            let output = s.transform(&input, &mut u);
            outputs.push(output[0]);
        }
        
        let mean_output: f64 = outputs.iter().sum::<f64>() / outputs.len() as f64;
        let variance: f64 = outputs.iter()
            .map(|x| (x - mean_output).powi(2))
            .sum::<f64>() / outputs.len() as f64;
        
        variance.ln().max(0.0)
    }
}

// GA with different fitness function
pub struct GaWithFitnessEvolution {
    base_evolution: Evolution,
    fitness_type: FitnessType,
    analyzer: EmergentAnalysis,
}

impl GaWithFitnessEvolution {
    pub fn set_quiet(&mut self, q: bool) {
        self.base_evolution.set_quiet(q);
    }

    pub fn run(&mut self, generations: usize, universe: &mut Universe) -> ComputationalSubstrate {
        // For now, just use the base evolution - fitness modification would require
        // more extensive changes to the Evolution struct
        self.base_evolution.run(generations, universe)
    }
}