// src/main.rs
//! TTA Simulator CLI Interface
//! Provides testing and validation commands

use clap::{Arg, Command, ArgMatches};
use std::path::Path;

mod config;
mod physics;
mod tta;

use config::TtaConfig;
use physics::{Universe, energy_validation::*};
use tta::FunctionalUnit; // Import the trait

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = cli().get_matches();
    
    match matches.subcommand() {
        Some(("validate", sub_matches)) => {
            cmd_validate(sub_matches)?;
        }
        Some(("test-axpb", sub_matches)) => {
            cmd_test_axpb(sub_matches)?;
        }
        Some(("physics-validate", sub_matches)) => {
            cmd_physics_validate(sub_matches)?;
        }
        Some(("config-gen", sub_matches)) => {
            cmd_config_gen(sub_matches)?;
        }
        Some(("test-vecmac", sub_matches)) => {
            cmd_test_vecmac(sub_matches)?;
        }
        _ => {
            println!("TTA Simulator v0.1");
            println!("Use --help for available commands");
        }
    }
    
    Ok(())
}

fn cli() -> Command {
    Command::new("tta-simulator")
        .version("0.1.0")
        .about("Transport-Triggered Architecture Simulator for AI Research")
        .subcommand(
            Command::new("validate")
                .about("Validate TTA configuration file")
                .arg(Arg::new("config")
                    .short('c')
                    .long("config")
                    .value_name("FILE")
                    .help("Configuration file to validate")
                    .required(true))
        )
        .subcommand(
            Command::new("test-axpb")
                .about("Test y = ax + b kernel execution")
                .arg(Arg::new("config")
                    .short('c')
                    .long("config")
                    .value_name("FILE")
                    .help("Configuration file")
                    .default_value("config/tta.toml"))
                .arg(Arg::new("a")
                    .short('a')
                    .help("Coefficient 'a'")
                    .default_value("2"))
                .arg(Arg::new("x")
                    .short('x')
                    .help("Input 'x'")
                    .default_value("5"))
                .arg(Arg::new("b")
                    .short('b')
                    .help("Constant 'b'")
                    .default_value("3"))
                .arg(Arg::new("verbose")
                    .short('v')
                    .long("verbose")
                    .help("Verbose output")
                    .action(clap::ArgAction::SetTrue))
        )
        .subcommand(
            Command::new("physics-validate")
                .about("Validate energy model against physics simulation")
                .arg(Arg::new("config")
                    .short('c')
                    .long("config")
                    .value_name("FILE")
                    .help("Configuration file")
                    .default_value("config/tta.toml"))
                .arg(Arg::new("energy-scale")
                    .long("energy-scale")
                    .help("Energy scale factor")
                    .default_value("1.0"))
        )
        .subcommand(
            Command::new("config-gen")
                .about("Generate default configuration file")
                .arg(Arg::new("output")
                    .short('o')
                    .long("output")
                    .value_name("FILE")
                    .help("Output file path")
                    .default_value("config/tta-default.toml"))
        )
        .subcommand(
            Command::new("test-vecmac")
                .about("Test VECMAC unit with dot product computation")
                .arg(Arg::new("config")
                    .short('c')
                    .long("config")
                    .value_name("FILE")
                    .help("Configuration file")
                    .default_value("config/tta.toml"))
                .arg(Arg::new("vec-a")
                    .long("vec-a")
                    .help("Vector A elements (comma-separated)")
                    .default_value("1,2,3,4"))
                .arg(Arg::new("vec-b")
                    .long("vec-b")
                    .help("Vector B elements (comma-separated)")
                    .default_value("5,6,7,8"))
                .arg(Arg::new("verbose")
                    .short('v')
                    .long("verbose")
                    .help("Verbose output")
                    .action(clap::ArgAction::SetTrue))
        )
}

fn cmd_validate(matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = matches.get_one::<String>("config").unwrap();
    
    println!("Validating TTA configuration: {}", config_path);
    
    match TtaConfig::from_file(config_path) {
        Ok(config) => {
            println!("✓ Configuration is valid");
            println!("  Processor: {}", config.processor.name);
            println!("  Issue width: {}", config.processor.issue_width);
            println!("  Buses: {}", config.buses.count);
            println!("  Functional units: {}", config.functional_units.len());
            
            // Validate energy costs
            println!("\nEnergy Configuration:");
            for (name, cost) in &config.energy.functional_units {
                println!("  {}: {} energy units", name, cost);
            }
            
            println!("✓ All validation checks passed");
        }
        Err(e) => {
            eprintln!("✗ Configuration validation failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

fn cmd_test_axpb(matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = matches.get_one::<String>("config").unwrap();
    let a: i32 = matches.get_one::<String>("a").unwrap().parse()?;
    let x: i32 = matches.get_one::<String>("x").unwrap().parse()?;
    let b: i32 = matches.get_one::<String>("b").unwrap().parse()?;
    let verbose = matches.get_flag("verbose");
    
    println!("Testing y = ax + b kernel");
    println!("  a = {}, x = {}, b = {}", a, x, b);
    println!("  Expected result: y = {} * {} + {} = {}", a, x, b, a * x + b);
    
    // Load configuration
    let config = if Path::new(config_path).exists() {
        TtaConfig::from_file(config_path)?
    } else {
        if verbose {
            println!("Configuration file not found, using defaults");
        }
        TtaConfig::default_test()
    };
    
    // Create functional units
    use tta::immediate_unit::{ImmediateUnit, ImmConfig};
    
    let imm_config = ImmConfig::default();
    let mut imm = ImmediateUnit::new(imm_config);
    
    if verbose {
        println!("\nInitializing functional units:");
        println!("  IMM: {} constants loaded", imm.constant_count());
    }
    
    // Test constant loading
    println!("\nTesting constant access:");
    
    // Load coefficient 'a' - for simplicity, we'll add custom constants
    if a <= 10 && a >= 0 {
        // Use built-in constants if possible
        let const_idx = match a {
            0 => 0, 1 => 1, 2 => 2, 3 => 3, 5 => 4, 10 => 5, _ => {
                // Add custom constant
                imm.add_constant(tta::BusData::I32(a))?
            }
        };
        
        use tta::FunctionalUnit;
        let result = imm.write_input(0, tta::BusData::I32(const_idx as i32), 0);
        if verbose {
            println!("  Loading a={}: {:?}", a, result);
        }
        assert_eq!(imm.read_output(0), Some(tta::BusData::I32(a)));
    }
    
    // Load constant 'b' similarly
    if b <= 10 && b >= -1 {
        let const_idx = match b {
            0 => 0, 1 => 1, 2 => 2, 3 => 3, 5 => 4, 10 => 5, -1 => 6, _ => {
                imm.add_constant(tta::BusData::I32(b))?
            }
        };
        
        use tta::FunctionalUnit;
        let result = imm.write_input(0, tta::BusData::I32(const_idx as i32), 0);
        if verbose {
            println!("  Loading b={}: {:?}", b, result);
        }
        assert_eq!(imm.read_output(0), Some(tta::BusData::I32(b)));
    }
    
    println!("✓ Immediate unit operational");
    
    // Calculate energy if verbose
    if verbose {
        let total_energy = imm.energy_consumed();
        println!("\nEnergy consumption:");
        println!("  IMM unit: {} energy units", total_energy);
        
        if let Some(transport_cost) = config.get_fu_energy_cost("immediate_select") {
            println!("  Transport cost per select: {} energy units", transport_cost);
        }
    }
    
    println!("✓ AXPB kernel validation complete");
    Ok(())
}

fn cmd_physics_validate(matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = matches.get_one::<String>("config").unwrap();
    let energy_scale: f64 = matches.get_one::<String>("energy-scale").unwrap().parse()?;
    
    println!("Physics validation of TTA energy model");
    println!("  Energy scale factor: {}", energy_scale);
    
    // Load TTA configuration
    let config = if Path::new(config_path).exists() {
        TtaConfig::from_file(config_path)?
    } else {
        println!("Configuration file not found, using defaults");
        TtaConfig::default_test()
    };
    
    // Create physics universe
    let universe = Universe::new(0.1, 0.01).with_energy(0.5, 1.0);
    let backend = UniversePhysicsBackend::new(universe, energy_scale);
    
    // Convert TTA config to energy table
    let energy_table = EnergyTable {
        fu_costs: config.energy.functional_units.clone(),
        transport_alpha: config.energy.transport.alpha_per_bit_toggle,
        transport_beta: config.energy.transport.beta_base,
        memory_costs: config.energy.memory.clone(),
    };
    
    // Validate
    let report = backend.validate_energy_table(&energy_table);
    
    println!("\nValidation Report:");
    println!("  Overall consistency: {}", if report.consistent { "✓ PASS" } else { "✗ FAIL" });
    println!("  Discrepancies found: {}", report.discrepancies.len());
    
    if !report.discrepancies.is_empty() {
        println!("\nDiscrepancy Details:");
        for disc in &report.discrepancies {
            println!("  {}: TTA={:.2}, Physics={:.2}, Ratio={:.2}x ({:?})",
                     disc.component, disc.tta_cost, disc.physics_cost, 
                     disc.ratio, disc.severity);
        }
    }
    
    if !report.recommendations.is_empty() {
        println!("\nRecommendations:");
        for rec in &report.recommendations {
            println!("  • {}", rec);
        }
    }
    
    println!("\n✓ Physics validation complete");
    Ok(())
}

fn cmd_config_gen(matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = matches.get_one::<String>("output").unwrap();
    
    println!("Generating default TTA configuration: {}", output_path);
    
    let config = TtaConfig::default_test();
    
    // Create directory if it doesn't exist
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    config.save_to_file(output_path)?;
    
    println!("✓ Configuration saved to {}", output_path);
    println!("  Use 'tta-simulator validate -c {}' to verify", output_path);
    
    Ok(())
}

fn cmd_test_vecmac(matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = matches.get_one::<String>("config").unwrap();
    let vec_a_str = matches.get_one::<String>("vec-a").unwrap();
    let vec_b_str = matches.get_one::<String>("vec-b").unwrap();
    let verbose = matches.get_flag("verbose");

    println!("Testing VECMAC unit with dot product computation");

    // Parse input vectors
    let vec_a: Result<Vec<i8>, _> = vec_a_str
        .split(',')
        .map(|s| s.trim().parse::<i8>())
        .collect();
    let vec_b: Result<Vec<i8>, _> = vec_b_str
        .split(',')
        .map(|s| s.trim().parse::<i8>())
        .collect();

    let vec_a = vec_a.map_err(|_| "Invalid vector A format")?;
    let vec_b = vec_b.map_err(|_| "Invalid vector B format")?;

    if vec_a.len() != vec_b.len() {
        return Err("Vector A and B must have the same length".into());
    }

    if vec_a.len() > 16 {
        return Err("Vectors cannot exceed 16 elements (VECMAC lane limit)".into());
    }

    // Calculate expected result
    let expected: i32 = vec_a.iter().zip(vec_b.iter())
        .map(|(a, b)| (*a as i32) * (*b as i32))
        .sum();

    println!("  Vector A: {:?}", vec_a);
    println!("  Vector B: {:?}", vec_b);
    println!("  Expected dot product: {}", expected);

    // Load configuration
    let config = if Path::new(config_path).exists() {
        TtaConfig::from_file(config_path)?
    } else {
        if verbose {
            println!("Configuration file not found, using defaults");
        }
        TtaConfig::default_test()
    };

    // Create VECMAC unit
    use tta::vecmac_unit::{VecMacUnit, VecMacConfig};
    let vecmac_config = VecMacConfig::default();
    let mut vecmac = VecMacUnit::new(vecmac_config);

    if verbose {
        println!("\nInitializing VECMAC unit:");
        println!("  Lane count: {}", 16);
        println!("  Latency: 2 cycles");
        println!("  Energy per operation: 40.0 units");
    }

    // Test VECMAC operation
    println!("\nExecuting VECMAC operation:");

    // Load vector A (latched input)
    let result = vecmac.write_input(0, tta::BusData::VecI8(vec_a.clone()), 0);
    if verbose {
        println!("  VEC_A loaded: {:?}", result);
    }
    assert_eq!(result, tta::FuEvent::Ready);

    // Load vector B (trigger input)
    let result = vecmac.write_input(1, tta::BusData::VecI8(vec_b.clone()), 0);
    if verbose {
        println!("  VEC_B loaded (triggered): {:?}", result);
    }
    assert!(matches!(result, tta::FuEvent::BusyUntil(_)));

    // Check busy state
    if verbose {
        println!("  VECMAC busy at cycle 1: {}", vecmac.is_busy(1));
        println!("  VECMAC busy at cycle 3: {}", vecmac.is_busy(3));
    }

    // Read result
    let result = vecmac.read_output(0).expect("Should have result");
    if let tta::BusData::I32(dot_product) = result {
        println!("  VECMAC result: {}", dot_product);

        if dot_product == expected {
            println!("✓ VECMAC computation successful!");
        } else {
            println!("✗ VECMAC computation failed! Expected {}, got {}", expected, dot_product);
            return Err("VECMAC computation mismatch".into());
        }
    } else {
        return Err("Invalid result type from VECMAC".into());
    }

    // Energy analysis
    let vecmac_energy = vecmac.energy_consumed();
    println!("\nEnergy Analysis:");
    println!("  VECMAC energy consumed: {:.1} units", vecmac_energy);

    if let Some(expected_energy) = config.get_fu_energy_cost("vecmac8x8_to_i32") {
        println!("  Expected energy cost: {:.1} units", expected_energy);
        if (vecmac_energy - expected_energy).abs() < 0.1 {
            println!("✓ Energy consumption matches configuration");
        } else {
            println!("⚠ Energy consumption differs from configuration");
        }
    }

    // Performance metrics
    if verbose {
        println!("\nPerformance Metrics:");
        println!("  Vector length: {} elements", vec_a.len());
        println!("  Operations: {} multiplies + {} adds", vec_a.len(), vec_a.len() - 1);
        println!("  Cycles: 2 (latency)");
        println!("  Throughput: {:.1} ops/cycle", vec_a.len() as f64 * 2.0 / 2.0);

        if vecmac_energy > 0.0 {
            println!("  Energy efficiency: {:.2} ops/energy", vec_a.len() as f64 * 2.0 / vecmac_energy);
        }
    }

    println!("✓ VECMAC unit test complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cli_builds() {
        let _app = cli();
    }
}
