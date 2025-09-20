// src/tta/processor.rs
//! TTA Processor core - placeholder for your existing implementation

use crate::tta::{FunctionalUnit, BusData, PortId};
use std::collections::HashMap;

/// TTA Processor - manages functional units, buses, and move execution
pub struct TtaProcessor {
    /// Collection of functional units
    functional_units: Vec<Box<dyn FunctionalUnit>>,
    /// Current cycle counter
    current_cycle: u64,
    /// Total energy consumed
    total_energy: f64,
}

impl TtaProcessor {
    /// Create a new TTA processor
    pub fn new() -> Self {
        Self {
            functional_units: Vec::new(),
            current_cycle: 0,
            total_energy: 0.0,
        }
    }

    /// Add a functional unit to the processor
    pub fn add_functional_unit(&mut self, fu: Box<dyn FunctionalUnit>) {
        self.functional_units.push(fu);
    }

    /// Get current cycle count
    pub fn current_cycle(&self) -> u64 {
        self.current_cycle
    }

    /// Get total energy consumed
    pub fn total_energy(&self) -> f64 {
        self.total_energy
    }

    /// Execute a single cycle
    pub fn step(&mut self) {
        self.current_cycle += 1;
        
        // Update all functional units
        for fu in &mut self.functional_units {
            fu.step(self.current_cycle);
        }
        
        // Update total energy
        self.total_energy = self.functional_units
            .iter()
            .map(|fu| fu.energy_consumed())
            .sum();
    }

    /// Reset processor to initial state
    pub fn reset(&mut self) {
        self.current_cycle = 0;
        self.total_energy = 0.0;
        
        for fu in &mut self.functional_units {
            fu.reset();
        }
    }

    /// Get number of functional units
    pub fn fu_count(&self) -> usize {
        self.functional_units.len()
    }
}

impl Default for TtaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tta::immediate_unit::{ImmediateUnit, ImmConfig};

    #[test]
    fn test_processor_creation() {
        let processor = TtaProcessor::new();
        assert_eq!(processor.current_cycle(), 0);
        assert_eq!(processor.total_energy(), 0.0);
        assert_eq!(processor.fu_count(), 0);
    }

    #[test]
    fn test_processor_with_fu() {
        let mut processor = TtaProcessor::new();
        
        let imm_config = ImmConfig::default();
        let imm = ImmediateUnit::new(imm_config);
        processor.add_functional_unit(Box::new(imm));
        
        assert_eq!(processor.fu_count(), 1);
    }

    #[test]
    fn test_processor_step() {
        let mut processor = TtaProcessor::new();
        
        assert_eq!(processor.current_cycle(), 0);
        processor.step();
        assert_eq!(processor.current_cycle(), 1);
        processor.step();
        assert_eq!(processor.current_cycle(), 2);
    }

    #[test]
    fn test_processor_reset() {
        let mut processor = TtaProcessor::new();
        
        processor.step();
        processor.step();
        assert_eq!(processor.current_cycle(), 2);
        
        processor.reset();
        assert_eq!(processor.current_cycle(), 0);
    }
}
