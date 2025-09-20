// Add to the match statement in main.rs
"ternary_extreme" => {
    println!("Testing: EXTREME ternary preference (5x bonus!)");
    (
        Universe::new(0.1, 0.01),
        EvolutionConfig {
            binary_bonus: 1.5,      // Much lower
            ternary_bonus: 5.0,     // HUGE bonus for ternary!
            quaternary_bonus: 2.0,
            other_discrete_bonus: 1.2,
        },
        "Extreme ternary-biased evolution"
    )
},
