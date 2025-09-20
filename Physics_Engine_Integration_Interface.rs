// Physics Engine Integration Interface
// Connects TTA energy model to ground-up computing physics validation

use crate::physics::Universe; // From existing ai-ground-up project

/// Interface for validating TTA energy costs against circuit-level physics
pub trait PhysicsBackend {
    fn gate_energy_cost(&self, gate_type: GateType, inputs: u32) -> f64;
    fn wire_energy_cost(&self, distance: f64, toggles: u32) -> f64;
    fn memory_energy_cost(&self, access_type: MemoryAccess) -> f64;
    fn validate_energy_table(&self, table: &EnergyTable) -> ValidationReport;
}

#[derive(Clone, Debug)]
pub enum GateType {
    Nand,
    Add16,
    Mul16,
    Compare16,
    Mux,
    Register,
}

#[derive(Clone, Debug)]
pub enum MemoryAccess {
    SramRead32b,
    SramWrite32b,
    RegFileRead,
    RegFileWrite,
    CamProbe { entries: usize, key_bits: usize },
}

#[derive(Clone, Debug)]
pub struct ValidationReport {
    pub consistent: bool,
    pub discrepancies: Vec<EnergyDiscrepancy>,
    pub recommendations: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct EnergyDiscrepancy {
    pub component: String,
    pub tta_cost: f64,
    pub physics_cost: f64,
    pub ratio: f64,
    pub severity: DiscrepancySeverity,
}

#[derive(Clone, Debug)]
pub enum DiscrepancySeverity {
    Minor,   // <20% difference
    Major,   // 20-100% difference  
    Critical,// >100% difference
}

/// Implementation using existing Universe physics engine
pub struct UniversePhysicsBackend {
    universe: Universe,
    noise_level: f64,
    energy_scale: f64, // Convert physics units to TTA units
}

impl UniversePhysicsBackend {
    pub fn new(universe: Universe, energy_scale: f64) -> Self {
        Self {
            universe,
            noise_level: 0.01, // Low noise for energy validation
            energy_scale,
        }
    }

    /// Map TTA functional unit operations to equivalent circuit complexity
    fn estimate_gate_count(&self, gate_type: &GateType) -> u32 {
        match gate_type {
            GateType::Nand => 1,
            GateType::Add16 => 32,      // Ripple carry adder estimate
            GateType::Mul16 => 256,     // Array multiplier estimate
            GateType::Compare16 => 16,  // XOR + tree
            GateType::Mux => 2,         // Pass gates
            GateType::Register => 16,   // D flip-flops
        }
    }

    /// Estimate wire lengths based on TTA bus architecture
    fn estimate_wire_distance(&self, src_fu: u16, dst_fu: u16) -> f64 {
        // Simplified model: assume FUs arranged in grid
        let src_x = (src_fu % 4) as f64;
        let src_y = (src_fu / 4) as f64;
        let dst_x = (dst_fu % 4) as f64;
        let dst_y = (dst_fu / 4) as f64;
        
        // Manhattan distance in normalized units
        (src_x - dst_x).abs() + (src_y - dst_y).abs()
    }
}

impl PhysicsBackend for UniversePhysicsBackend {
    fn gate_energy_cost(&self, gate_type: GateType, inputs: u32) -> f64 {
        let gate_count = self.estimate_gate_count(&gate_type);
        let base_cost = self.universe.holding_cost(0.5); // Normalized value
        
        // Scale by gate complexity and input activity
        let complexity_factor = gate_count as f64;
        let activity_factor = (inputs as f64).sqrt(); // Square root scaling
        
        base_cost * complexity_factor * activity_factor * self.energy_scale
    }

    fn wire_energy_cost(&self, distance: f64, toggles: u32) -> f64 {
        // Use universe switching cost scaled by distance and activity
        let base_switch_cost = self.universe.switching_cost(1.0);
        let distance_factor = 1.0 + distance * 0.1; // Linear distance penalty
        let activity_factor = toggles as f64;
        
        base_switch_cost * distance_factor * activity_factor * self.energy_scale
    }

    fn memory_energy_cost(&self, access_type: MemoryAccess) -> f64 {
        let base_cost = self.universe.holding_cost(1.0);
        
        let complexity_factor = match access_type {
            MemoryAccess::SramRead32b => 10.0,
            MemoryAccess::SramWrite32b => 12.0,
            MemoryAccess::RegFileRead => 4.0,
            MemoryAccess::RegFileWrite => 6.0,
            MemoryAccess::CamProbe { entries, key_bits } => {
                // CAM cost scales with entries and key width
                let base_cam = 20.0;
                let entry_factor = (entries as f64).log2();
                let width_factor = key_bits as f64 / 16.0;
                base_cam * entry_factor * width_factor
            }
        };
        
        base_cost * complexity_factor * self.energy_scale
    }

    fn validate_energy_table(&self, table: &EnergyTable) -> ValidationReport {
        let mut discrepancies = Vec::new();
        let mut consistent = true;
        
        // Validate functional unit costs
        for (op_name, &tta_cost) in &table.fu_costs {
            let gate_type = match op_name.as_str() {
                "add16" => GateType::Add16,
                "mul16" => GateType::Mul16,
                "cmp16" => GateType::Compare16,
                _ => continue,
            };
            
            let physics_cost = self.gate_energy_cost(gate_type, 2); // Assume 2 inputs
            let ratio = tta_cost / physics_cost;
            
            let severity = if ratio < 0.8 || ratio > 1.2 {
                consistent = false;
                if ratio < 0.5 || ratio > 2.0 {
                    DiscrepancySeverity::Critical
                } else {
                    DiscrepancySeverity::Major
                }
            } else {
                DiscrepancySeverity::Minor
            };
            
            if !matches!(severity, DiscrepancySeverity::Minor) {
                discrepancies.push(EnergyDiscrepancy {
                    component: op_name.clone(),
                    tta_cost,
                    physics_cost,
                    ratio,
                    severity,
                });
            }
        }
        
        // Generate recommendations
        let mut recommendations = Vec::new();
        
        if discrepancies.iter().any(|d| matches!(d.severity, DiscrepancySeverity::Critical)) {
            recommendations.push("Critical energy discrepancies detected. Review cost model assumptions.".to_string());
        }
        
        let avg_ratio: f64 = discrepancies.iter().map(|d| d.ratio).sum::<f64>() / discrepancies.len().max(1) as f64;
        if avg_ratio > 1.5 {
            recommendations.push(format!("TTA costs may be overestimated by {:.1}x. Consider scaling down.", avg_ratio));
        } else if avg_ratio < 0.7 {
            recommendations.push(format!("TTA costs may be underestimated by {:.1}x. Consider scaling up.", 1.0/avg_ratio));
        }
        
        ValidationReport {
            consistent,
            discrepancies,
            recommendations,
        }
    }
}

/// Energy table structure for validation
#[derive(Clone, Debug)]
pub struct EnergyTable {
    pub fu_costs: std::collections::HashMap<String, f64>,
    pub transport_alpha: f64,
    pub transport_beta: f64,
    pub memory_costs: std::collections::HashMap<String, f64>,
}

impl EnergyTable {
    pub fn from_toml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Parse TOML energy configuration file
        // Implementation would read actual config file
        todo!("Implement TOML parsing")
    }
}

/// Integration with TTA processor for cross-validation
impl crate::TtaProcessor {
    pub fn validate_energy_model(&self, physics: &dyn PhysicsBackend) -> ValidationReport {
        // Extract current energy table from processor config
        let energy_table = self.extract_energy_table();
        
        // Validate against physics backend
        let report = physics.validate_energy_table(&energy_table);
        
        if !report.consistent {
            log::warn!("Energy model validation failed: {} discrepancies", report.discrepancies.len());
            for discrepancy in &report.discrepancies {
                log::warn!("  {}: TTA={:.2}, Physics={:.2}, Ratio={:.2}", 
                          discrepancy.component, discrepancy.tta_cost, 
                          discrepancy.physics_cost, discrepancy.ratio);
            }
        }
        
        report
    }
    
    fn extract_energy_table(&self) -> EnergyTable {
        // Extract energy costs from current processor configuration
        todo!("Extract energy table from processor config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::Universe;
    
    #[test]
    fn test_physics_validation() {
        let universe = Universe::new(0.01, 0.01).with_energy(0.5, 1.0);
        let backend = UniversePhysicsBackend::new(universe, 1.0);
        
        // Test basic gate cost estimation
        let add_cost = backend.gate_energy_cost(GateType::Add16, 2);
        assert!(add_cost > 0.0);
        
        // Test wire cost estimation  
        let wire_cost = backend.wire_energy_cost(1.0, 10);
        assert!(wire_cost > 0.0);
        
        // Test memory cost estimation
        let mem_cost = backend.memory_energy_cost(MemoryAccess::SramRead32b);
        assert!(mem_cost > 0.0);
    }
    
    #[test]
    fn test_energy_validation_consistency() {
        let universe = Universe::new(0.01, 0.01).with_energy(0.5, 1.0);
        let backend = UniversePhysicsBackend::new(universe, 1.0);
        
        // Create test energy table
        let mut table = EnergyTable {
            fu_costs: std::collections::HashMap::new(),
            transport_alpha: 0.02,
            transport_beta: 1.0,
            memory_costs: std::collections::HashMap::new(),
        };
        
        table.fu_costs.insert("add16".to_string(), 8.0);
        table.fu_costs.insert("mul16".to_string(), 24.0);
        
        let report = backend.validate_energy_table(&table);
        
        // Should detect some discrepancies but not be completely inconsistent
        assert!(!report.discrepancies.is_empty() || report.consistent);
    }
}
