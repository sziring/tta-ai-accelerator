mod physics;
mod substrate;
mod analysis;
mod evolution;

use physics::Universe;
use evolution::Evolution;
use analysis::EmergentAnalysis;

fn main() {
    println!("AI Ground-Up Computing Research");
    println!("================================");
    println!("Using {} CPU cores", rayon::current_num_threads());
    println!();
    
    // Create universe with physical constraints
    let universe = Universe::new(0.1, 0.01);
    
    // Run evolution
    let mut evolution = Evolution::new(100, 0.1); // population_size, mutation_rate
    
    println!("Starting evolution from pure information theory...");
    println!("==================================================");
    
    let best = evolution.run(1000, &mut universe); // 1000 generations
    
    // Analyze what emerged
    let analysis = EmergentAnalysis::new();
    let structure = analysis.detect_discretization(&best);
    let info_preserved = analysis.measure_information_preservation(&best, &mut universe);
    
    println!("\n==================================================");
    println!("=== DISCOVERED COMPUTATION ===");
    println!("==================================================");
    println!("Structure: {:?}", structure);
    println!("Information preservation: {:.3}", info_preserved);
    println!("Energy consumed: {:.3}", best.energy_consumed());
    
    match structure {
        analysis::DiscretizationType::Discrete { n_states, state_values } => {
            println!("\nDiscovered {}-state logic!", n_states);
            println!("States: {:?}", state_values);
            if n_states == 2 {
                println!("Converged on binary - validating our current paradigm");
            } else {
                println!("Found alternative to binary computation!");
            }
        }
        analysis::DiscretizationType::Continuous { range } => {
            println!("\nRemained in continuous/analog computation");
            println!("Range: [{:.3}, {:.3}]", range.0, range.1);
        }
    }
}
