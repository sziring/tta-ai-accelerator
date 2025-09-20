// src/tta/immediate_unit.rs
use crate::tta::{FunctionalUnit, BusData, PortId, FuEvent};
use std::collections::HashMap;

/// Immediate Value Unit - provides constants from a pre-loaded table
#[derive(Clone, Debug)]
pub struct ImmediateUnit {
    /// Pre-loaded constant table
    constants: Vec<BusData>,
    /// Current output value
    current_output: Option<BusData>,
    /// Configuration
    config: ImmConfig,
}

#[derive(Clone, Debug)]
pub struct ImmConfig {
    /// Maximum number of constants supported
    pub max_constants: usize,
    /// Default constants to load
    pub default_constants: Vec<BusData>,
}

impl Default for ImmConfig {
    fn default() -> Self {
        Self {
            max_constants: 256,
            // Common constants for testing - includes values needed for axpb
            default_constants: vec![
                BusData::I32(0),    // Constants[0] = 0
                BusData::I32(1),    // Constants[1] = 1  
                BusData::I32(2),    // Constants[2] = 2
                BusData::I32(3),    // Constants[3] = 3
                BusData::I32(5),    // Constants[4] = 5
                BusData::I32(10),   // Constants[5] = 10
                BusData::I32(-1),   // Constants[6] = -1
                BusData::I32(42),   // Constants[7] = 42 (for testing)
            ],
        }
    }
}

impl ImmediateUnit {
    pub fn new(config: ImmConfig) -> Self {
        let mut unit = Self {
            constants: Vec::with_capacity(config.max_constants),
            current_output: None,
            config,
        };
        
        // Load default constants
        for constant in &unit.config.default_constants {
            unit.constants.push(constant.clone());
        }
        
        // Initialize output with first constant
        if !unit.constants.is_empty() {
            unit.current_output = Some(unit.constants[0].clone());
        }
        
        unit
    }
    
    /// Add a constant to the table (for dynamic configuration)
    pub fn add_constant(&mut self, value: BusData) -> Result<usize, String> {
        if self.constants.len() >= self.config.max_constants {
            return Err(format!("Constant table full (max: {})", self.config.max_constants));
        }
        
        self.constants.push(value);
        Ok(self.constants.len() - 1)
    }
    
    /// Get constant by index (for debugging/validation)
    pub fn get_constant(&self, index: usize) -> Option<&BusData> {
        self.constants.get(index)
    }
    
    /// Get current constant table size
    pub fn constant_count(&self) -> usize {
        self.constants.len()
    }
}

impl FunctionalUnit for ImmediateUnit {
    fn name(&self) -> &'static str {
        "IMM"
    }
    
    fn input_ports(&self) -> Vec<String> {
        vec!["SELECT_IN".to_string()]
    }
    
    fn output_ports(&self) -> Vec<String> {
        vec!["OUT".to_string()]
    }
    
    fn write_input(&mut self, port: u16, data: BusData, _cycle: u64) -> FuEvent {
        match port {
            0 => { // SELECT_IN port
                match data {
                    BusData::I32(index) => {
                        let idx = index as usize;
                        if idx < self.constants.len() {
                            // Immediate effect - update output right away
                            self.current_output = Some(self.constants[idx].clone());
                            FuEvent::Ready
                        } else {
                            FuEvent::Error(format!("Constant index {} out of bounds (max: {})", 
                                                 idx, self.constants.len() - 1))
                        }
                    }
                    _ => FuEvent::Error("SELECT_IN port requires I32 data type".to_string())
                }
            }
            _ => FuEvent::Error(format!("Invalid input port: {}", port))
        }
    }
    
    fn read_output(&self, port: u16) -> Option<BusData> {
        match port {
            0 => self.current_output.clone(), // OUT port
            _ => None,
        }
    }
    
    fn is_busy(&self, _cycle: u64) -> bool {
        false // Immediate unit is never busy (combinational)
    }
    
    fn energy_consumed(&self) -> f64 {
        0.0 // No energy cost for constant selection (pure combinational)
    }
    
    fn reset(&mut self) {
        // Reset to first constant
        if !self.constants.is_empty() {
            self.current_output = Some(self.constants[0].clone());
        }
    }
    
    fn step(&mut self, _cycle: u64) {
        // No state to update - purely combinational
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_imm_basic_operation() {
        let config = ImmConfig::default();
        let mut imm = ImmediateUnit::new(config);
        
        // Should start with first constant (0)
        assert_eq!(imm.read_output(0), Some(BusData::I32(0)));
        
        // Select constant 1
        let result = imm.write_input(0, BusData::I32(1), 0);
        assert_eq!(result, FuEvent::Ready);
        assert_eq!(imm.read_output(0), Some(BusData::I32(1)));
        
        // Select constant 7 (42)
        let result = imm.write_input(0, BusData::I32(7), 0);
        assert_eq!(result, FuEvent::Ready);
        assert_eq!(imm.read_output(0), Some(BusData::I32(42)));
    }
    
    #[test]
    fn test_imm_out_of_bounds() {
        let config = ImmConfig::default();
        let mut imm = ImmediateUnit::new(config);
        
        // Try to select non-existent constant
        let result = imm.write_input(0, BusData::I32(100), 0);
        assert!(matches!(result, FuEvent::Error(_)));
    }
    
    #[test]
    fn test_imm_wrong_data_type() {
        let config = ImmConfig::default();
        let mut imm = ImmediateUnit::new(config);
        
        // Try to select with wrong data type
        let result = imm.write_input(0, BusData::I16(1), 0);
        assert!(matches!(result, FuEvent::Error(_)));
    }
    
    #[test]
    fn test_imm_energy_cost() {
        let config = ImmConfig::default();
        let imm = ImmediateUnit::new(config);
        
        // Should have zero energy cost
        assert_eq!(imm.energy_consumed(), 0.0);
    }
    
    #[test]
    fn test_imm_custom_constants() {
        let mut config = ImmConfig::default();
        config.default_constants = vec![
            BusData::I32(100),
            BusData::I32(200), 
            BusData::I32(300),
        ];
        
        let mut imm = ImmediateUnit::new(config);
        
        // Should start with first custom constant
        assert_eq!(imm.read_output(0), Some(BusData::I32(100)));
        
        // Select second custom constant
        let result = imm.write_input(0, BusData::I32(1), 0);
        assert_eq!(result, FuEvent::Ready);
        assert_eq!(imm.read_output(0), Some(BusData::I32(200)));
    }
    
    #[test]
    fn test_imm_add_runtime_constant() {
        let config = ImmConfig::default();
        let mut imm = ImmediateUnit::new(config);
        
        let initial_count = imm.constant_count();
        
        // Add a new constant at runtime
        let index = imm.add_constant(BusData::I32(999)).unwrap();
        assert_eq!(imm.constant_count(), initial_count + 1);
        
        // Should be able to select the new constant
        let result = imm.write_input(0, BusData::I32(index as i32), 0);
        assert_eq!(result, FuEvent::Ready);
        assert_eq!(imm.read_output(0), Some(BusData::I32(999)));
    }
}
