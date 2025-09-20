use rand::{thread_rng, Rng};
use crate::physics::Universe;

#[derive(Clone)]
pub struct ComputationalSubstrate {
    genome: Genome,
    state: Option<Vec<f64>>,
    energy_consumed: f64,
}

#[derive(Clone)]
pub struct Genome {
    pub nonlinearity: NonlinearityType,
    pub feedback: f64,
    pub memory_factor: f64,
    pub weights: Vec<f64>,
    pub threshold: f64,  // Add threshold for discretization
    pub hysteresis: f64, // Add hysteresis for stability
}

#[derive(Clone, Copy, Debug)]
pub enum NonlinearityType {
    Linear,
    Tanh,
    Relu,
    Compressive,
    Expansive,
    Step,        // NEW: Hard step function
    Schmitt,     // NEW: Schmitt trigger (with hysteresis)
    Saturating,  // NEW: Saturating function
    Quantizer,   // NEW: Multi-level quantizer
}

impl ComputationalSubstrate {
    pub fn new(genome: Genome) -> Self {
        ComputationalSubstrate {
            genome,
            state: None,
            energy_consumed: 0.0,
        }
    }
    
    pub fn random() -> Self {
        let mut rng = thread_rng();
        
        let genome = Genome {
            nonlinearity: match rng.gen_range(0..9) {
                0 => NonlinearityType::Linear,
                1 => NonlinearityType::Tanh,
                2 => NonlinearityType::Relu,
                3 => NonlinearityType::Compressive,
                4 => NonlinearityType::Expansive,
                5 => NonlinearityType::Step,
                6 => NonlinearityType::Schmitt,
                7 => NonlinearityType::Saturating,
                _ => NonlinearityType::Quantizer,
            },
            feedback: rng.gen_range(-1.0..1.0),
            memory_factor: rng.gen_range(0.0..1.0),
            weights: (0..3).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            threshold: rng.gen_range(-0.5..0.5),
            hysteresis: rng.gen_range(0.0..0.3),
        };
        
        ComputationalSubstrate::new(genome)
    }
    
    pub fn transform(&mut self, input: &[f64], universe: &mut universe) -> Vec<f64> {
        let mut x = input.to_vec();
        
        // Apply nonlinearity
        x = self.apply_nonlinearity(&x);
        
        // Apply feedback if state exists
        if let Some(ref state) = self.state {
            for (xi, si) in x.iter_mut().zip(state.iter()) {
                *xi += self.genome.feedback * si;
            }
        }
        
        // Apply physics
        x = universe.decay(x);
        x = universe.noisy(x);
        
        // Update state
        if self.genome.memory_factor > 0.0 {
            self.state = Some(x.iter().map(|&v| v * self.genome.memory_factor).collect());
        }
        
        // Track energy (discrete states use less energy!)
        let is_discrete = matches!(
            self.genome.nonlinearity, 
            NonlinearityType::Step | NonlinearityType::Schmitt | NonlinearityType::Quantizer
        );
        let energy_multiplier = if is_discrete { 0.5 } else { 1.0 };
        
        let change: Vec<f64> = x.iter().zip(input.iter()).map(|(a, b)| a - b).collect();
        self.energy_consumed += universe.energy_cost(&change) * energy_multiplier;
        
        x
    }
    
    fn apply_nonlinearity(&self, x: &[f64]) -> Vec<f64> {
        match self.genome.nonlinearity {
            NonlinearityType::Linear => x.to_vec(),
            NonlinearityType::Tanh => x.iter().map(|&v| v.tanh()).collect(),
            NonlinearityType::Relu => x.iter().map(|&v| v.max(0.0)).collect(),
            NonlinearityType::Compressive => {
                x.iter().map(|&v| v.signum() * v.abs().sqrt()).collect()
            },
            NonlinearityType::Expansive => x.iter().map(|&v| v.powi(3)).collect(),
            NonlinearityType::Step => {
                // Binary step function
                x.iter().map(|&v| if v > self.genome.threshold { 1.0 } else { -1.0 }).collect()
            },
            NonlinearityType::Schmitt => {
                // Schmitt trigger with hysteresis
                x.iter().enumerate().map(|(i, &v)| {
                    let prev = self.state.as_ref()
                        .and_then(|s| s.get(i))
                        .copied()
                        .unwrap_or(0.0);
                    
                    if prev > 0.0 {
                        // Currently high, need to go below threshold - hysteresis
                        if v < self.genome.threshold - self.genome.hysteresis {
                            -1.0
                        } else {
                            1.0
                        }
                    } else {
                        // Currently low, need to go above threshold + hysteresis  
                        if v > self.genome.threshold + self.genome.hysteresis {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                }).collect()
            },
            NonlinearityType::Saturating => {
                // Soft saturation that approaches discrete states
                x.iter().map(|&v| {
                    let scaled = v * 3.0;
                    scaled.tanh() * 1.1  // Slightly over 1.0 to encourage discrete
                }).collect()
            },
            NonlinearityType::Quantizer => {
                // Multi-level quantizer (ternary or more)
                x.iter().map(|&v| {
                    if v < -self.genome.threshold {
                        -1.0
                    } else if v > self.genome.threshold {
                        1.0
                    } else {
                        0.0  // Ternary!
                    }
                }).collect()
            },
        }
    }
    
    pub fn mutate(&mut self, mutation_rate: f64) {
        let mut rng = thread_rng();
        
        // Mutate nonlinearity
        if rng.gen::<f64>() < mutation_rate {
            self.genome.nonlinearity = match rng.gen_range(0..9) {
                0 => NonlinearityType::Linear,
                1 => NonlinearityType::Tanh,
                2 => NonlinearityType::Relu,
                3 => NonlinearityType::Compressive,
                4 => NonlinearityType::Expansive,
                5 => NonlinearityType::Step,
                6 => NonlinearityType::Schmitt,
                7 => NonlinearityType::Saturating,
                _ => NonlinearityType::Quantizer,
            };
        }
        
        // Mutate numerical parameters
        if rng.gen::<f64>() < mutation_rate {
            self.genome.feedback += rng.gen_range(-0.2..0.2);
            self.genome.feedback = self.genome.feedback.clamp(-2.0, 2.0);
        }
        
        if rng.gen::<f64>() < mutation_rate {
            self.genome.memory_factor += rng.gen_range(-0.1..0.1);
            self.genome.memory_factor = self.genome.memory_factor.clamp(0.0, 1.0);
        }
        
        if rng.gen::<f64>() < mutation_rate {
            self.genome.threshold += rng.gen_range(-0.1..0.1);
            self.genome.threshold = self.genome.threshold.clamp(-1.0, 1.0);
        }
        
        if rng.gen::<f64>() < mutation_rate {
            self.genome.hysteresis += rng.gen_range(-0.05..0.05);
            self.genome.hysteresis = self.genome.hysteresis.clamp(0.0, 0.5);
        }
        
        // Mutate weights
        for w in &mut self.genome.weights {
            if rng.gen::<f64>() < mutation_rate {
                *w += rng.gen_range(-0.3..0.3);
                *w = w.clamp(-3.0, 3.0);
            }
        }
    }
    
    pub fn energy_consumed(&self) -> f64 {
        self.energy_consumed
    }
    
    pub fn reset_energy(&mut self) {
        self.energy_consumed = 0.0;
    }
}
