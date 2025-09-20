use rand::{thread_rng, Rng};
use crate::physics::Universe;

#[derive(Clone)]
pub struct ComputationalSubstrate {
    genome: Genome,
    state: Option<Vec<f64>>,
    energy_consumed: f64,
    allowed_types: Vec<NonlinearityType>,
    quant_levels: u8,
}

#[derive(Clone)]
pub struct Genome {
    pub nonlinearity: NonlinearityType,
    pub feedback: f64,
    pub memory_factor: f64,
    pub weights: Vec<f64>,
    pub threshold: f64,
    pub hysteresis: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonlinearityType {
    Linear,
    Tanh,
    Relu,
    Compressive,
    Expansive,
    Step,
    Schmitt,
    Saturating,
    Quantizer,
}

impl NonlinearityType {
    pub fn all() -> Vec<NonlinearityType> {
        vec![
            NonlinearityType::Linear,
            NonlinearityType::Tanh,
            NonlinearityType::Relu,
            NonlinearityType::Compressive,
            NonlinearityType::Expansive,
            NonlinearityType::Step,
            NonlinearityType::Schmitt,
            NonlinearityType::Saturating,
            NonlinearityType::Quantizer,
        ]
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            NonlinearityType::Linear => "linear",
            NonlinearityType::Tanh => "tanh",
            NonlinearityType::Relu => "relu",
            NonlinearityType::Compressive => "compressive",
            NonlinearityType::Expansive => "expansive",
            NonlinearityType::Step => "step",
            NonlinearityType::Schmitt => "schmitt",
            NonlinearityType::Saturating => "saturating",
            NonlinearityType::Quantizer => "quantizer",
        }
    }
}

impl ComputationalSubstrate {
    pub fn new(genome: Genome, allowed_types: Vec<NonlinearityType>, quant_levels: u8) -> Self {
        Self {
            genome,
            state: None,
            energy_consumed: 0.0,
            allowed_types,
            quant_levels,
        }
    }

    pub fn random() -> Self {
        Self::random_with_constraints(NonlinearityType::all(), 3)
    }

    pub fn random_with_constraints(allowed_types: Vec<NonlinearityType>, quant_levels: u8) -> Self {
        let mut rng = thread_rng();
        
        let nl = if allowed_types.is_empty() {
            NonlinearityType::Linear
        } else {
            allowed_types[rng.gen_range(0..allowed_types.len())]
        };

        let genome = Genome {
            nonlinearity: nl,
            feedback: rng.gen_range(-1.0..1.0),
            memory_factor: rng.gen_range(0.0..1.0),
            weights: (0..3).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            threshold: rng.gen_range(-0.5..0.5),
            hysteresis: rng.gen_range(0.0..0.3),
        };

        Self::new(genome, allowed_types, quant_levels)
    }

    pub fn transform(&mut self, input: &[f64], universe: &mut Universe) -> Vec<f64> {
        // Base linear combine (weights) + feedback
        let mut x: Vec<f64> = input
            .iter()
            .zip(self.genome.weights.iter().cycle())
            .map(|(v, w)| v * w)
            .collect();

        // Apply nonlinearity
        x = self.apply_nonlinearity(&x);

        // Feedback from last state
        if let Some(ref s) = self.state {
            for (xi, si) in x.iter_mut().zip(s.iter()) {
                *xi += self.genome.feedback * *si;
            }
        }

        // Physics per element: decay + noise
        for xi in x.iter_mut() {
            *xi = universe.step_scalar(*xi);
        }

        // Snapshot output
        let x_next = x;

        // Energy accounting
        let prev_state = self.state.as_ref();

        // Holding energy for this step
        let e_hold: f64 = x_next
            .iter()
            .map(|&v| universe.holding_cost(v))
            .sum::<f64>()
            * universe.energy_hold_scale;

        // Switching energy (movement vs previous)
        let e_switch: f64 = if let Some(prev) = prev_state {
            prev.iter()
                .zip(x_next.iter())
                .map(|(&p, &n)| universe.switching_cost(n - p))
                .sum::<f64>()
                * universe.energy_switch_scale
        } else {
            0.0
        };

        // Discrete implementations get a small discount
        let is_discrete = matches!(
            self.genome.nonlinearity,
            NonlinearityType::Step | NonlinearityType::Schmitt | NonlinearityType::Quantizer
        );
        let discrete_discount = if is_discrete { 0.5 } else { 1.0 };

        self.energy_consumed += (e_hold + e_switch) * discrete_discount;

        // State update after energy charge
        if self.genome.memory_factor > 0.0 {
            self.state = Some(
                x_next
                    .iter()
                    .map(|&v| v * self.genome.memory_factor)
                    .collect(),
            );
        } else {
            self.state = None;
        }

        x_next
    }

    fn apply_nonlinearity(&self, x: &[f64]) -> Vec<f64> {
        match self.genome.nonlinearity {
            NonlinearityType::Linear => x.to_vec(),
            NonlinearityType::Tanh => x.iter().map(|&v| v.tanh()).collect(),
            NonlinearityType::Relu => x.iter().map(|&v| v.max(0.0)).collect(),
            NonlinearityType::Compressive => x.iter().map(|&v| v.signum() * v.abs().sqrt()).collect(),
            NonlinearityType::Expansive => x.iter().map(|&v| v.powi(3)).collect(),

            NonlinearityType::Step => x
                .iter()
                .map(|&v| if v > self.genome.threshold { 1.0 } else { -1.0 })
                .collect(),

            NonlinearityType::Schmitt => x
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let prev = self
                        .state
                        .as_ref()
                        .and_then(|s| s.get(i))
                        .copied()
                        .unwrap_or(0.0);
                    if prev > 0.0 {
                        if v < self.genome.threshold - self.genome.hysteresis {
                            -1.0
                        } else {
                            1.0
                        }
                    } else {
                        if v > self.genome.threshold + self.genome.hysteresis {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                })
                .collect(),

            NonlinearityType::Saturating => x
                .iter()
                .map(|&v| {
                    let scaled = v * 3.0;
                    (scaled.tanh() * 1.1).clamp(-1.2, 1.2)
                })
                .collect(),

            NonlinearityType::Quantizer => {
                // Now properly uses quant_levels to generate N equally-spaced levels
                let n = self.quant_levels as usize;
                if n <= 1 {
                    x.to_vec()
                } else {
                    x.iter()
                        .map(|&v| {
                            // Map input to [0, n-1] index
                            let clamped = v.clamp(-1.0, 1.0);
                            let idx = ((clamped + 1.0) * (n as f64) / 2.0).floor() as usize;
                            let idx = idx.min(n - 1);
                            
                            // Map back to equally spaced levels in [-1, 1]
                            if n == 2 {
                                if idx == 0 { -1.0 } else { 1.0 }
                            } else {
                                -1.0 + (2.0 * idx as f64) / ((n - 1) as f64)
                            }
                        })
                        .collect()
                }
            }
        }
    }

    pub fn mutate(&mut self, mutation_rate: f64) {
        let mut rng = thread_rng();

        // Mutate nonlinearity - now respects allowed_types
        if rng.gen::<f64>() < mutation_rate && !self.allowed_types.is_empty() {
            self.genome.nonlinearity = self.allowed_types[rng.gen_range(0..self.allowed_types.len())];
        }
#[cfg(debug_assertions)]
{
    assert!(self.allowed_types.contains(&self.genome.nonlinearity), 
        "Mutation violated constraint: {:?} not in allowed {:?}", 
        self.genome.nonlinearity, self.allowed_types);
}
        // Mutate numeric parameters
        if rng.gen::<f64>() < mutation_rate {
            self.genome.feedback = (self.genome.feedback + rng.gen_range(-0.2..0.2)).clamp(-2.0, 2.0);
        }
        if rng.gen::<f64>() < mutation_rate {
            self.genome.memory_factor = (self.genome.memory_factor + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < mutation_rate {
            self.genome.threshold = (self.genome.threshold + rng.gen_range(-0.1..0.1)).clamp(-1.0, 1.0);
        }
        if rng.gen::<f64>() < mutation_rate {
            self.genome.hysteresis = (self.genome.hysteresis + rng.gen_range(-0.05..0.05)).clamp(0.0, 0.5);
        }

        // Mutate weights
        for w in &mut self.genome.weights {
            if rng.gen::<f64>() < mutation_rate {
                *w = (*w + rng.gen_range(-0.3..0.3)).clamp(-3.0, 3.0);
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