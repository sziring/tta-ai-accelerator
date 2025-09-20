use rayon::prelude::*;
use rand::{thread_rng, Rng};
use crate::substrate::ComputationalSubstrate;
use crate::physics::Universe;
use crate::analysis::EmergentAnalysis;

pub struct Evolution {
    population_size: usize,
    mutation_rate: f64,
    analyzer: EmergentAnalysis,
}

impl Evolution {
    pub fn new(population_size: usize, mutation_rate: f64) -> Self {
        Evolution {
            population_size,
            mutation_rate,
            analyzer: EmergentAnalysis::new(),
        }
    }
    
    pub fn run(&mut self, generations: usize, universe: &mut universe) -> ComputationalSubstrate {
        // Initialize population
        let mut population: Vec<ComputationalSubstrate> = (0..self.population_size)
            .map(|_| ComputationalSubstrate::random())
            .collect();
        
        for gen in 0..generations {
            // Reset energy for fair comparison
            population.par_iter_mut().for_each(|s| s.reset_energy());
            
            // Parallel fitness evaluation
            let fitness_scores: Vec<f64> = population
                .par_iter()
                .map(|substrate| self.evaluate_fitness(substrate, universe))
                .collect();
            
            // Find best
            let best_idx = fitness_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            
            let best = &population[best_idx];
            let best_fitness = fitness_scores[best_idx];
            
            // Report progress
            if gen % 10 == 0 {
                let structure = self.analyzer.detect_discretization(best);
                println!("Gen {}: Best fitness={:.3}", gen, best_fitness);
                println!("  Discovered: {:?}", structure);
            }
            
            // Create next generation
            let mut new_population = Vec::with_capacity(self.population_size);
            
            // Elitism - keep best
            new_population.push(best.clone());
            
            // Fill rest with offspring
            while new_population.len() < self.population_size {
                // Tournament selection
                let parent = self.tournament_select(&population, &fitness_scores, 5);
                let mut offspring = parent.clone();
                offspring.mutate(self.mutation_rate);
                new_population.push(offspring);
            }
            
            population = new_population;
        }
        
        // Return best from final generation
        population.into_iter()
            .max_by_key(|s| {
                (self.evaluate_fitness(s, universe) * 1000.0) as i64
            })
            .unwrap()
    }
    
    fn evaluate_fitness(&self, substrate: &ComputationalSubstrate, universe: &mut universe) -> f64 {
        // Information preservation
        let info_score = self.analyzer.measure_information_preservation(substrate, universe);
        
        // Energy efficiency
        let energy_score = 1.0 / (1.0 + substrate.energy_consumed());
        
        // MUCH STRONGER bonus for discovering discrete states
        let structure_bonus = match self.analyzer.detect_discretization(substrate) {
            crate::analysis::DiscretizationType::Discrete { n_states, .. } => {
                match n_states {
                    2 => 3.0,     // Binary gets huge bonus
                    3 => 2.5,     // Ternary also very good  
                    4..=5 => 2.0, // Other small discrete
                    _ => 1.2
                }
            },
            _ => 1.0,
        };
        
        info_score * energy_score * structure_bonus
    }
    
    fn tournament_select(
        &self, 
        population: &[ComputationalSubstrate], 
        fitness: &[f64], 
        tournament_size: usize
    ) -> ComputationalSubstrate {
        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..tournament_size)
            .map(|_| rng.gen_range(0..population.len()))
            .collect();
        
        let winner_idx = indices
            .iter()
            .max_by(|&&a, &&b| fitness[a].partial_cmp(&fitness[b]).unwrap())
            .unwrap();
        
        population[*winner_idx].clone()
    }
}
