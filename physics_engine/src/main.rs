#![allow(dead_code)]
mod physics;
mod substrate;
mod analysis;
mod evolution;
mod sweep;
mod sweep_extended;  // New extended sweep module
mod optimizers;      // Your new optimizers module

use clap::{Arg, ArgAction, Command};
use crate::physics::{Universe, PhysicsRegime};
use crate::evolution::Evolution;
use crate::substrate::NonlinearityType;

/// Fitness config carried into Evolution.
#[derive(Clone)]
pub struct EvolutionConfig {
    pub binary_bonus: f64,
    pub ternary_bonus: f64,
    pub quaternary_bonus: f64,
    pub other_discrete_bonus: f64,
    pub target_states: Option<usize>,
    pub mismatch_penalty: f64,
}

impl EvolutionConfig {
    pub fn standard() -> Self {
        Self {
            binary_bonus: 1.25,
            ternary_bonus: 1.25,
            quaternary_bonus: 1.10,
            other_discrete_bonus: 1.05,
            target_states: None,
            mismatch_penalty: 0.5,
        }
    }
}

/// Pretty-print state vectors, normalizing tiny values
fn pretty_vals(vals: &[f64]) -> String {
    let v: Vec<f64> = vals
        .iter()
        .map(|&x| if x.abs() < 1e-12 { 0.0 } else { x })
        .collect();
    format!("{:?}", v)
}

/// Parse `--allow` flag into a set of nonlinearities
fn parse_allowed_from_args() -> Option<Vec<NonlinearityType>> {
    let mut args = std::env::args().skip_while(|a| a != "--allow");
    let _ = args.next()?;
    let spec = args.next()?;

    match spec.as_str() {
        "binary_only" => Some(vec![
            NonlinearityType::Step, 
            NonlinearityType::Schmitt
        ]),
        "discrete_only" => Some(vec![
            NonlinearityType::Step,
            NonlinearityType::Schmitt,
            NonlinearityType::Quantizer,
        ]),
        "continuous_only" => Some(vec![
            NonlinearityType::Linear,
            NonlinearityType::Tanh, 
            NonlinearityType::Relu,
            NonlinearityType::Compressive,
            NonlinearityType::Expansive,
            NonlinearityType::Saturating,
        ]),
        _ => {
            // Custom comma-separated list
            let mut out = Vec::new();
            for tok in spec.split(',') {
                let t = tok.trim().to_lowercase();
                let v = match t.as_str() {
                    "linear" => NonlinearityType::Linear,
                    "tanh" => NonlinearityType::Tanh,
                    "relu" => NonlinearityType::Relu,
                    "compressive" => NonlinearityType::Compressive,
                    "expansive" => NonlinearityType::Expansive,
                    "step" => NonlinearityType::Step,
                    "schmitt" => NonlinearityType::Schmitt,
                    "saturating" => NonlinearityType::Saturating,
                    "quantizer" => NonlinearityType::Quantizer,
                    _ => continue,
                };
                out.push(v);
            }
            if out.is_empty() { None } else { Some(out) }
        }
    }
}

/// Parse regime from args
fn parse_regime_from_args() -> Option<PhysicsRegime> {
    let mut args = std::env::args().skip_while(|a| a != "--regime");
    let _ = args.next()?;
    let spec = args.next()?;
    
    match spec.as_str() {
        "baseline" => Some(PhysicsRegime::Baseline),
        "asymmetric" => Some(PhysicsRegime::AsymmetricEnergy),
        "amplitude_noise" => Some(PhysicsRegime::AmplitudeScaledNoise),
        "correlation" => Some(PhysicsRegime::CorrelationOnly),
        "rounded" => Some(PhysicsRegime::RoundedArithmetic),
        "decay" => Some(PhysicsRegime::DecayFeature),
        "negative" => Some(PhysicsRegime::NegativeInformation),
        "probabilistic" => Some(PhysicsRegime::ProbabilisticSuperposition),
        _ => None
    }
}

fn print_header() {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("AI Ground-Up Computing Research");
    println!("================================");
    println!("Using {} CPU cores\n", cores);
    println!(
        "energy_knobs: ENERGY_HOLD_SCALE={} ENERGY_SWITCH_SCALE={} LAMBDA_ENERGY={}",
        std::env::var("ENERGY_HOLD_SCALE").unwrap_or_else(|_| "1.0".into()),
        std::env::var("ENERGY_SWITCH_SCALE").unwrap_or_else(|_| "1.0".into()),
        std::env::var("LAMBDA_ENERGY").unwrap_or_else(|_| "1.0".into()),
    );
    println!(
        "Command/s used: {}\n--------------------------------------------------",
        std::env::args().collect::<Vec<_>>().join(" ")
    );
}

fn cli() -> Command {
    Command::new("ai-ground-up")
        .about("Alternate physics evolution runner")
        .arg(Arg::new("experiment")
            .value_parser(["baseline", "binary", "high_noise", "extreme_noise", "sweep", "sweep-extended"])
            .required(true))
        .arg(Arg::new("allow").long("allow").num_args(1))
        .arg(Arg::new("regime").long("regime").num_args(1))
        .arg(Arg::new("target-states").long("target-states").num_args(1))
        .arg(Arg::new("mismatch-penalty").long("mismatch-penalty").num_args(1))
        .arg(Arg::new("quant-levels").long("quant-levels").num_args(1))
        
        // Original sweep args
        .arg(Arg::new("sweep-fast").long("sweep-fast").action(ArgAction::SetTrue))
        .arg(Arg::new("sweep-stdout").long("sweep-stdout").num_args(1))
        .arg(Arg::new("sweep-csv").long("sweep-csv").num_args(1))
        .arg(Arg::new("sweep-sigmas").long("sweep-sigmas").num_args(1))
        .arg(Arg::new("sweep-energy-zero").long("sweep-energy-zero").num_args(1))
        .arg(Arg::new("sweep-energy-abs1").long("sweep-energy-abs1").num_args(1))
        .arg(Arg::new("sweep-allowed").long("sweep-allowed").num_args(1))
        .arg(Arg::new("sweep-quant-levels").long("sweep-quant-levels").num_args(1))
        .arg(Arg::new("sweep-target-states").long("sweep-target-states").num_args(1))
        .arg(Arg::new("sweep-mismatch-penalty").long("sweep-mismatch-penalty").num_args(1))
        .arg(Arg::new("sweep-gens").long("sweep-gens").num_args(1))
        .arg(Arg::new("sweep-pop").long("sweep-pop").num_args(1))
        .arg(Arg::new("sweep-seeds").long("sweep-seeds").num_args(1))
        
        // Energy model support
        .arg(Arg::new("energy-model").long("energy-model").num_args(1)
            .value_parser(["base", "asym", "leak"])
            .help("Energy model variant: base=standard, asym=asymmetric switching, leak=idle cost"))
        
        // Extended sweep args (only used with sweep-extended)
        .arg(Arg::new("sweep-optimizer").long("sweep-optimizer").num_args(1)
            .value_parser(["ga", "random", "cmaes"])
            .help("Optimizer for extended sweep experiments"))
        .arg(Arg::new("sweep-fitness").long("sweep-fitness").num_args(1)
            .value_parser(["task", "reg", "info"])
            .help("Fitness function for extended sweep experiments"))
}

fn main() {
    print_header();
    let matches = cli().get_matches();

    let experiment = matches.get_one::<String>("experiment").unwrap().to_string();

    // Handle extended sweep path with new optimizer/fitness features
    if experiment == "sweep-extended" {
        sweep_extended::run_sweep_extended(matches);
        return;
    }

    // Handle original sweep path (unchanged for backward compatibility)
    if experiment == "sweep" {
        sweep::run_sweep(matches);
        return;
    }

    // Single-run experiments (baseline, binary, etc.)
    // Evolution config for single-run modes
    let mut evolution_config = EvolutionConfig::standard();

    if let Some(ts) = matches.get_one::<String>("target-states") {
        if ts != "none" {
            evolution_config.target_states = ts.parse::<usize>().ok();
        }
    }
    if let Some(mp) = matches.get_one::<String>("mismatch-penalty") {
        evolution_config.mismatch_penalty = mp.parse::<f64>().unwrap_or(0.5);
    }

    // Allowed nonlinearities
    let allowed = parse_allowed_from_args().unwrap_or_else(|| {
        vec![
            NonlinearityType::Step,
            NonlinearityType::Schmitt,
            NonlinearityType::Quantizer,
        ]
    });

    // Physics regime - parse from matches, not raw args
    let regime = matches
        .get_one::<String>("regime")
        .and_then(|s| match s.as_str() {
            "baseline" => Some(PhysicsRegime::Baseline),
            "asymmetric" => Some(PhysicsRegime::AsymmetricEnergy),
            "amplitude_noise" => Some(PhysicsRegime::AmplitudeScaledNoise),
            "correlation" => Some(PhysicsRegime::CorrelationOnly),
            "rounded" => Some(PhysicsRegime::RoundedArithmetic),
            "decay" => Some(PhysicsRegime::DecayFeature),
            "negative" => Some(PhysicsRegime::NegativeInformation),
            "probabilistic" => Some(PhysicsRegime::ProbabilisticSuperposition),
            _ => None
        })
        .unwrap_or(PhysicsRegime::Baseline);

    // Quantization levels
    let quant_levels: u8 = matches
        .get_one::<String>("quant-levels")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    // Universe for single-run modes
    let mut universe = match experiment.as_str() {
        "high_noise" => Universe::new(0.3, 0.01).with_energy(0.5, 1.0).with_seed(42).with_regime(regime),
        "extreme_noise" => Universe::new(0.5, 0.01).with_energy(0.5, 1.0).with_seed(42).with_regime(regime),
        "binary" | "baseline" | _ => Universe::new(0.1, 0.01).with_energy(0.5, 1.0).with_seed(42).with_regime(regime),
    };

    println!("Running with physics regime: {:?}", regime);
    
    // Single-run evolution (using original GA evolution)
    let mut evolution = Evolution::new_with_config_allowed(100, 0.1, evolution_config, allowed, quant_levels);
    evolution.set_quiet(true);  // Suppress generation output
    let best = evolution.run(1000, &mut universe);

    // Report results
    let disc = analysis::EmergentAnalysis::new().detect_discretization(&best);
    println!();
    match disc {
        analysis::DiscretizationType::Discrete { n_states, state_values, .. } => {
            println!("Discrete computation emerged!");
            println!("Number of states: {}", n_states);
            println!("State values    : {}", pretty_vals(&state_values));
            match n_states {
                2 => println!("Converged on BINARY (±1)"),
                3 => println!("Converged on TERNARY (≈ [-1, 0, 1])"),
                4 => println!("Converged on QUATERNARY – 2-bit equivalent"),
                _ => println!("Converged on {}-level (nonstandard) {}", n_states, pretty_vals(&state_values)),
            }
        }
        _ => {
            println!("No discrete structure detected (analog/continuous).");
        }
    }
}