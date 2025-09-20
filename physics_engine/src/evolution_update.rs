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
