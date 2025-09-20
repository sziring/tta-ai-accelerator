use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::env;

// Helper: read f64 env var with default
fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy, Debug)]
pub enum PhysicsRegime {
    Baseline,              // Standard Gaussian noise
    AsymmetricEnergy,      // Zero state is cheap
    AmplitudeScaledNoise,  // Noise scales with signal magnitude
    CorrelationOnly,       // Can only observe correlations
    RoundedArithmetic,     // All ops quantized
    DecayFeature,          // Signals decay toward zero
    NegativeInformation,   // Some interactions erase info
    ProbabilisticSuperposition, // Nodes hold distributions
}
#[derive(Clone, Copy, Debug)]
pub enum EnergyModel {
    Base,     // Current standard model
    Asym,     // Asymmetric: zero state cheap, switching costs different  
    Leak,     // Idle cost: continuous power drain
}

pub struct Universe {
    pub base_sigma: f64,
    pub decay_per_step: f64,
    pub quantum_uncertainty: f64,
    rng: StdRng,

    // Energy parameters
    pub energy_zero: f64,
    pub energy_abs1: f64,
    pub energy_model: EnergyModel,

    // Tunable scales via env
    pub energy_hold_scale: f64,
    pub energy_switch_scale: f64,

    // Counterfactual physics parameters
    pub regime: PhysicsRegime,
    pub amp_noise_k: f64,      // AmplitudeScaledNoise coefficient
    pub rounding_quantum: f64,  // RoundedArithmetic quantum
    pub decay_half_life: f64,   // DecayFeature half-life
    pub p_erase: f64,           // NegativeInformation erasure probability
    pub temperature: f64,        // ProbabilisticSuperposition temperature
}

impl Universe {
    pub fn new(sigma: f64, decay: f64) -> Self {
        Self {
            base_sigma: sigma,
            decay_per_step: decay,
            quantum_uncertainty: 0.0,
            rng: StdRng::seed_from_u64(42),
            energy_zero: 0.5,
            energy_abs1: 1.0,
            energy_model: EnergyModel::Base,

            energy_hold_scale: env_f64("ENERGY_HOLD_SCALE", 1.0),
            energy_switch_scale: env_f64("ENERGY_SWITCH_SCALE", 1.0),
            regime: PhysicsRegime::Baseline,
            amp_noise_k: 0.5,
            rounding_quantum: 0.1,
            decay_half_life: 1000.0,
            p_erase: 0.05,
            temperature: 0.5,
        }
    }


pub fn with_energy_model(mut self, model: EnergyModel) -> Self {
    self.energy_model = model;
    self
}

    pub fn with_energy(mut self, e0: f64, eabs1: f64) -> Self {
        self.energy_zero = e0;
        self.energy_abs1 = eabs1;
        self.energy_hold_scale = env_f64("ENERGY_HOLD_SCALE", self.energy_hold_scale);
        self.energy_switch_scale = env_f64("ENERGY_SWITCH_SCALE", self.energy_switch_scale);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    pub fn with_regime(mut self, regime: PhysicsRegime) -> Self {
        self.regime = regime;
        // Adjust defaults based on regime
        match regime {
            PhysicsRegime::AsymmetricEnergy => {
                self.energy_zero = 0.1;  // Very cheap
                self.energy_abs1 = 10.0;  // Very expensive
            }
            PhysicsRegime::AmplitudeScaledNoise => {
                self.amp_noise_k = 1.0;  // Strong amplitude dependency
            }
            PhysicsRegime::RoundedArithmetic => {
                self.rounding_quantum = 0.1;
            }
            PhysicsRegime::DecayFeature => {
                self.decay_half_life = 100.0;  // Fast decay
            }
            PhysicsRegime::NegativeInformation => {
                self.p_erase = 0.1;  // 10% chance of erasure
            }
            PhysicsRegime::ProbabilisticSuperposition => {
                self.temperature = 1.0;
            }
            _ => {}
        }
        self
    }

    pub fn forked(&self, seed: u64) -> Self {
        let mut u = Self::new(self.base_sigma, self.decay_per_step);
        u.quantum_uncertainty = self.quantum_uncertainty;
        u.energy_zero = self.energy_zero;
        u.energy_abs1 = self.energy_abs1;
            u.energy_model = self.energy_model;  // ADD THIS LINE

        u.energy_hold_scale = self.energy_hold_scale;
        u.energy_switch_scale = self.energy_switch_scale;
        u.regime = self.regime;
        u.amp_noise_k = self.amp_noise_k;
        u.rounding_quantum = self.rounding_quantum;
        u.decay_half_life = self.decay_half_life;
        u.p_erase = self.p_erase;
        u.temperature = self.temperature;
        u.rng = StdRng::seed_from_u64(seed);
        u
    }

    // Core physics hooks
    pub fn decay(&self, x: f64) -> f64 {
        match self.regime {
            PhysicsRegime::DecayFeature => {
                // Exponential decay toward zero with specified half-life
                let decay_rate = 0.693 / self.decay_half_life;
                x * (-decay_rate).exp()
            }
            _ => {
                // Standard decay
                x * (1.0 - self.decay_per_step).clamp(0.0, 1.0)
            }
        }
    }

    pub fn noisy(&mut self, x: f64) -> f64 {
        // Box-Muller for Gaussian noise
        let u1 = (self.rng.next_u64() as f64 / u64::MAX as f64).max(1e-12);
        let u2 = self.rng.next_u64() as f64 / u64::MAX as f64;
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        
        let noise_sigma = match self.regime {
            PhysicsRegime::AmplitudeScaledNoise => {
                // Noise scales with signal magnitude
                self.base_sigma * (1.0 + self.amp_noise_k * x.abs())
            }
            _ => self.base_sigma
        };
        
        let noisy_x = x + noise_sigma * z0;
        
        // Apply rounding if in RoundedArithmetic regime
        match self.regime {
            PhysicsRegime::RoundedArithmetic => {
                self.round_quantum(noisy_x, self.rounding_quantum)
            }
            _ => noisy_x
        }
    }

    pub fn round_quantum(&self, x: f64, quantum: f64) -> f64 {
        if quantum <= 0.0 {
            x
        } else {
            (x / quantum).round() * quantum
        }
    }

    pub fn step_scalar(&mut self, x: f64) -> f64 {
        let d = self.decay(x);
        let n = self.noisy(d);
        
        match self.regime {
            PhysicsRegime::ProbabilisticSuperposition => {
                // Add probabilistic jitter based on temperature
                let jitter = self.temperature * (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5);
                n + jitter
            }
            _ => {
                if self.quantum_uncertainty > 0.0 {
                    self.round_quantum(n, self.quantum_uncertainty)
                } else {
                    n
                }
            }
        }
    }

    pub fn interact_pair(&mut self, a: f64, b: f64) -> (f64, f64) {
        match self.regime {
            PhysicsRegime::NegativeInformation => {
                // Sometimes erase information
                if (self.rng.next_u64() as f64 / u64::MAX as f64) < self.p_erase {
                    (0.0, 0.0)  // Information erased
                } else {
                    let k = 0.05;
                    (a + k * b, b + k * a)
                }
            }
            PhysicsRegime::CorrelationOnly => {
                // Return correlation-based values
                let corr = a * b;
                (corr, corr)
            }
            _ => {
                // Standard coupling
                let k = 0.05;
                (a + k * b, b + k * a)
            }
        }
    }

    pub fn observe_pair(&self, a: f64, b: f64) -> f64 {
        match self.regime {
            PhysicsRegime::CorrelationOnly => {
                // Can only observe correlations
                a * b
            }
            _ => a
        }
    }

    pub fn soft_bit(&mut self, logit: f64) -> f64 {
        match self.regime {
            PhysicsRegime::ProbabilisticSuperposition => {
                // Temperature-scaled sigmoid
                1.0 / (1.0 + (-logit / self.temperature).exp())
            }
            _ => 1.0 / (1.0 + (-logit).exp())
        }
    }

    // Energy model
pub fn holding_cost(&self, x: f64) -> f64 {
    let a = x.abs().min(1.0);
    
    match self.energy_model {
        EnergyModel::Asym => {
            // Zero state is dramatically cheaper
            if a < 0.1 {
                self.energy_zero * 0.01  // Nearly free
            } else {
                self.energy_abs1 * 2.0   // Expensive ±1 states
            }
        }
        EnergyModel::Leak => {
            // Add constant idle power consumption
            let base_cost = (1.0 - a) * self.energy_zero + a * self.energy_abs1;
            base_cost + 0.1  // Always burning 0.1 units
        }
        _ => {
            // Base model (current)
            (1.0 - a) * self.energy_zero + a * self.energy_abs1
        }
    }
}

    pub fn switching_cost(&self, dx: f64) -> f64 {
        match self.regime {
            PhysicsRegime::AsymmetricEnergy => {
                // Switching to/from zero is cheap, between ±1 is expensive
                if dx.abs() < 0.1 {
                    0.01  // Almost free to stay near zero
                } else {
                    dx.abs() * self.energy_abs1
                }
            }
            _ => {
                // Standard switching cost
                dx.abs() * 0.5 * (self.energy_zero + self.energy_abs1)
            }
        }
    }
}