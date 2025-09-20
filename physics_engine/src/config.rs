use clap::{Parser, ValueEnum};

/// Which "universe" (counterfactual physics regime) to simulate.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum Regime {
    /// Baseline: additive Gaussian noise, symmetric energy.
    Baseline,
    /// State 0 is cheap; |±1| are expensive.
    AsymmetricEnergy,
    /// Noise standard deviation grows with |value|: sigma = base_sigma + k*|value|.
    AmplitudeScaledNoise,
    /// You can only observe correlations (pairwise products), not raw values.
    CorrelationOnly,
    /// All operations round to nearest quantum (e.g., 0.1).
    RoundedArithmetic,
    /// Every value decays toward 0 with a fixed half-life.
    DecayFeature,
    /// Certain interactions erase information with probability p_erase.
    NegativeInformation,
    /// Nodes hold probability distributions; updates are noisy soft decisions (temperature).
    ProbabilisticSuperposition,
}

#[derive(Parser, Debug)]
#[command(name = "ai-ground-up", version, about = "Alternate-physics evolutionary simulator")]
pub struct Cli {
    /// Pick the physics regime / universe.
    #[arg(long, value_enum, default_value_t = Regime::Baseline)]
    pub regime: Regime,

    /// Base Gaussian noise (all regimes that use additive noise).
    #[arg(long, default_value_t = 0.1)]
    pub base_sigma: f64,

    /// AsymmetricEnergy: energy cost for holding/transitioning to state 0.
    #[arg(long, default_value_t = 1.0)]
    pub energy_e0: f64,
    /// AsymmetricEnergy: energy cost for holding/transitioning to |±1|.
    #[arg(long, default_value_t = 10.0)]
    pub energy_e1: f64,

    /// AmplitudeScaledNoise: coefficient k in sigma = base_sigma + k*|x|.
    #[arg(long, default_value_t = 0.5)]
    pub amp_noise_k: f64,

    /// RoundedArithmetic: rounding quantum (e.g., 0.1 or 1.0 for integers).
    #[arg(long, default_value_t = 0.1)]
    pub quantum: f64,

    /// DecayFeature: half-life in steps (value -> value/2 after this many steps).
    #[arg(long, default_value_t = 1000.0)]
    pub half_life: f64,

    /// NegativeInformation: probability that a pair interaction erases info.
    #[arg(long, default_value_t = 0.05)]
    pub p_erase: f64,

    /// ProbabilisticSuperposition: temperature for soft decisions (higher = noisier).
    #[arg(long, default_value_t = 0.5)]
    pub temperature: f64,
}

