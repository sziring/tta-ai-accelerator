// src/sweep_extended.rs
// Extended sweep functionality with optimizer and fitness variants
// Keeps original sweep.rs unchanged for backward compatibility

use crate::optimizers::{OptimizerType, FitnessType, create_optimizer, Optimizer};
use crate::physics::Universe;
use crate::substrate::NonlinearityType;

use clap::ArgMatches;

/// Parse a --sweep-allowed spec into a set of nonlinearities.
fn parse_allowed(spec: &str) -> Vec<NonlinearityType> {
    match spec {
        "binary_only" => vec![NonlinearityType::Step, NonlinearityType::Schmitt],
        "discrete_only" => vec![
            NonlinearityType::Step,
            NonlinearityType::Schmitt,
            NonlinearityType::Quantizer,
        ],
        "continuous_only" => vec![
            NonlinearityType::Linear,
            NonlinearityType::Tanh,
            NonlinearityType::Relu,
            NonlinearityType::Compressive,
            NonlinearityType::Expansive,
            NonlinearityType::Saturating,
        ],
        _ => {
            let mut v = Vec::new();
            for tok in spec.split(',') {
                let t = tok.trim().to_lowercase();
                let n = match t.as_str() {
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
                v.push(n);
            }
            if v.is_empty() {
                NonlinearityType::all()
            } else {
                v
            }
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

pub fn run_sweep_extended(m: ArgMatches) {
    let fast = m.get_flag("sweep-fast");
    let stdout_mode = m
        .get_one::<String>("sweep-stdout")
        .map(|s| s.as_str())
        .unwrap_or("summary");

    let sigmas: Vec<f64> = m
        .get_one::<String>("sweep-sigmas")
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.1, 0.3]);

    let energy_zero: Vec<f64> = m
        .get_one::<String>("sweep-energy-zero")
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.2, 0.8]);

    let energy_abs1: Vec<f64> = m
        .get_one::<String>("sweep-energy-abs1")
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1.0, 1.5]);

    let allowed: Vec<NonlinearityType> = m
        .get_one::<String>("sweep-allowed")
        .map(|s| parse_allowed(s))
        .unwrap_or_else(|| {
            vec![
                NonlinearityType::Step,
                NonlinearityType::Schmitt,
                NonlinearityType::Quantizer,
            ]
        });

    let quant_levels: u8 = m
        .get_one::<String>("sweep-quant-levels")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let gens: usize = m
        .get_one::<String>("sweep-gens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(if fast { 120 } else { 300 });

    let pop: usize = m
        .get_one::<String>("sweep-pop")
        .and_then(|s| s.parse().ok())
        .unwrap_or(if fast { 40 } else { 120 });

    // Extended parameters
    let optimizer = m
        .get_one::<String>("sweep-optimizer")
        .map(|s| s.as_str())
        .unwrap_or("ga");
    
    let fitness = m
        .get_one::<String>("sweep-fitness")
        .map(|s| s.as_str())
        .unwrap_or("task");

    let energy_model_str = m
        .get_one::<String>("energy-model")
        .map(|s| s.as_str())
        .unwrap_or("base");

    // Convert to enum types
    let optimizer_type = OptimizerType::from_str(optimizer);
    let fitness_type = FitnessType::from_str(fitness);

    // Optional targeting knobs
    let target_states: Option<usize> = m
        .get_one::<String>("sweep-target-states")
        .and_then(|s| if s == "none" { None } else { s.parse().ok() });

    let mismatch_penalty: f64 = m
        .get_one::<String>("sweep-mismatch-penalty")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);

    println!("=== Extended Phase sweep ===");
    println!(
        "Repro command: {}",
        std::env::args().collect::<Vec<_>>().join(" ")
    );
    println!(
        "gens={} pop={} allowed={:?} quant_levels={} target_states={:?} mismatch_penalty={} energy_model={} optimizer={} fitness={}",
        gens,
        pop,
        allowed.iter().map(|n| n.as_str()).collect::<Vec<_>>(),
        quant_levels,
        target_states,
        mismatch_penalty,
        energy_model_str,
        optimizer,
        fitness
    );
    println!(
        "sigmas={:?}\nenergy_zero={:?}  energy_abs1={:?}",
        sigmas, energy_zero, energy_abs1
    );

    let cells = sigmas.len() * energy_zero.len() * energy_abs1.len();
    println!("cells={}", cells);
    println!("==============================================================");
    println!("CSV_HEADER,sigma,energy_zero,energy_abs1,n_states,state_values");

    let mut rows: Vec<(f64, f64, f64, usize, Vec<f64>)> = Vec::new();

    let mut completed = 0usize;
    for &sigma in &sigmas {
        for &e0 in &energy_zero {
            for &ea in &energy_abs1 {
                // Parse number of seeds from command line
                let seeds: usize = m.get_one::<String>("sweep-seeds")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1);

                // Parse energy model for this sweep
                let energy_model = match energy_model_str {
                    "asym" => crate::physics::EnergyModel::Asym,
                    "leak" => crate::physics::EnergyModel::Leak,
                    _ => crate::physics::EnergyModel::Base,
                };

                // Run multiple seeds per cell
                let mut seed_results = Vec::new();
                for seed in 0..seeds {
                    let mut universe = Universe::new(sigma, 0.01)
                        .with_energy(e0, ea)
                        .with_energy_model(energy_model)
                        .with_seed(42 + seed as u64);

                    let mut cfg = crate::EvolutionConfig::standard();
                    cfg.target_states = target_states;
                    cfg.mismatch_penalty = mismatch_penalty;

                    // Use factory function to create optimizer
                    let mut evo = create_optimizer(
                        optimizer_type,
                        fitness_type,
                        pop,
                        0.1,
                        cfg,
                        allowed.clone(),
                        quant_levels,
                    );
                    evo.set_quiet(true);

                    let best = evo.run(gens, &mut universe);
                    let disc = crate::analysis::EmergentAnalysis::new().detect_discretization(&best);
                    let has_hysteresis = crate::analysis::EmergentAnalysis::new().detect_hysteresis_artifact(&best);
                    
                    seed_results.push((disc, has_hysteresis));
                }

                // Aggregate results - take most common n_states
                let mode_result = find_mode_result(&seed_results);
                
                match mode_result.0 {
                    crate::analysis::DiscretizationType::Discrete {
                        n_states,
                        state_values,
                        ..
                    } => {
                        rows.push((sigma, e0, ea, n_states, state_values.clone()));
                        if stdout_mode == "summary" {
                            let hysteresis_flag = if mode_result.1 { " [H]" } else { "" };
                            let seed_info = if seeds > 1 { format!(" ({}/{})", count_matching(&seed_results, n_states), seeds) } else { String::new() };
                            println!(
                                "CSV_ROW,{:.3},{:.3},{:.3},{},{}{}{}",
                                sigma, e0, ea, n_states,
                                pretty_vals(&state_values),
                                seed_info,
                                hysteresis_flag
                            );
                        }
                    }
                    _ => {
                        rows.push((sigma, e0, ea, 0, vec![]));
                        if stdout_mode == "summary" {
                            let seed_info = if seeds > 1 { format!(" ({}/{})", count_matching(&seed_results, 0), seeds) } else { String::new() };
                            println!("CSV_ROW,{:.3},{:.3},{:.3},0,{}{}", sigma, e0, ea, "[]", seed_info);
                        }
                    }
                }

                completed += 1;
                let step = (cells.max(1) / (if cells >= 8 { 8 } else { 1 })).max(1);
                if completed % step == 0 || completed == cells {
                    println!("[{}/{}] completed", completed, cells);
                }
            }
        }
    }

    println!("==============================================================");
    // Quick histogram
    let mut b = 0;
    let mut t = 0;
    let mut q = 0;
    let mut n = 0;
    let mut a = 0;
    for &(_, _, _, ns, _) in &rows {
        match ns {
            0 => a += 1,
            2 => b += 1,
            3 => t += 1,
            4 => q += 1,
            _ => {
                if ns > 0 {
                    n += 1
                } else {
                    a += 1
                }
            }
        }
    }
    println!(
        "Summary (winners by cell): B={}  T={}  Q={}  N={}  A={}",
        b, t, q, n, a
    );
    println!("--- sample rows ---");
    for r in rows.iter().take(8) {
        println!(
            "CSV_ROW,{:.3},{:.3},{:.3},{},{}",
            r.0,
            r.1,
            r.2,
            r.3,
            pretty_vals(&r.4)
        );
    }

    println!(
        "Legend: B=binary(2), T=ternary(3), Q=quaternary(4), N=other discrete, A=analog/continuous"
    );
}

// Helper functions for multi-seed aggregation
fn find_mode_result(results: &[(crate::analysis::DiscretizationType, bool)]) -> (crate::analysis::DiscretizationType, bool) {
    use std::collections::HashMap;
    let mut counts = HashMap::new();
    
    for (disc, _) in results {
        let n_states = match disc {
            crate::analysis::DiscretizationType::Discrete { n_states, .. } => *n_states,
            _ => 0,
        };
        *counts.entry(n_states).or_insert(0) += 1;
    }
    
    let mode_n_states = counts.into_iter().max_by_key(|&(_, count)| count).unwrap().0;
    
    // Return first result matching the mode
    for (disc, hyst) in results {
        let n_states = match disc {
            crate::analysis::DiscretizationType::Discrete { n_states, .. } => *n_states,
            _ => 0,
        };
        if n_states == mode_n_states {
            return (disc.clone(), *hyst);
        }
    }
    
    (results[0].0.clone(), results[0].1)
}

fn count_matching(results: &[(crate::analysis::DiscretizationType, bool)], target_n_states: usize) -> usize {
    results.iter().filter(|(disc, _)| {
        let n_states = match disc {
            crate::analysis::DiscretizationType::Discrete { n_states, .. } => *n_states,
            _ => 0,
        };
        n_states == target_n_states
    }).count()
}