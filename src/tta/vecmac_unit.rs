// src/tta/vecmac_unit.rs
//! Vector Multiply-Accumulate Unit for AI acceleration
//!
//! Implements element-wise vector multiplication followed by accumulation,
//! which is the core operation for dot products and neural network inference.

use crate::tta::{FunctionalUnit, BusData, FuEvent};

/// Vector Multiply-Accumulate functional unit
#[derive(Clone, Debug)]
pub struct VecMacUnit {
    /// Vector A input (latched)
    vec_a: Option<BusData>,
    /// Vector B input (triggers operation)
    vec_b: Option<BusData>,
    /// Accumulator input (optional)
    acc_in: Option<BusData>,
    /// Operation mode: false=clear, true=accumulate
    accumulate_mode: bool,
    /// Output result
    scalar_out: Option<BusData>,
    /// Configuration
    config: VecMacConfig,
    /// Energy consumed
    energy_consumed: f64,
    /// Busy state
    busy_until: u64,
}

#[derive(Clone, Debug)]
pub struct VecMacConfig {
    /// Number of parallel lanes (8 or 16)
    pub lane_count: usize,
    /// Latency in cycles (2-3 cycles depending on lane count)
    pub latency_cycles: u64,
    /// Energy cost per operation
    pub operation_energy: f64,
}

impl Default for VecMacConfig {
    fn default() -> Self {
        Self {
            lane_count: 16,
            latency_cycles: 2,
            operation_energy: 40.0, // matches vecmac8x8_to_i32 for 16 lanes
        }
    }
}

impl VecMacUnit {
    pub fn new(config: VecMacConfig) -> Self {
        Self {
            vec_a: None,
            vec_b: None,
            acc_in: None,
            accumulate_mode: false,
            scalar_out: None,
            energy_consumed: 0.0,
            busy_until: 0,
            config,
        }
    }

    /// Perform vector multiply-accumulate operation
    fn execute_vecmac(&mut self, cycle: u64) -> FuEvent {
        let vec_a = match &self.vec_a {
            Some(data) => data,
            None => return FuEvent::Error("VEC_A not set".to_string()),
        };

        let vec_b = match &self.vec_b {
            Some(data) => data,
            None => return FuEvent::Error("VEC_B not set".to_string()),
        };

        // Extract vector data and compute
        let result = match (vec_a, vec_b) {
            (BusData::VecI8(a), BusData::VecI8(b)) => {
                self.compute_mac_i8(a, b)
            }
            (BusData::VecI16(a), BusData::VecI16(b)) => {
                self.compute_mac_i16(a, b)
            }
            _ => {
                return FuEvent::Error("Incompatible vector types for VECMAC".to_string());
            }
        };

        match result {
            Ok(output) => {
                self.scalar_out = Some(output);
                self.busy_until = cycle + self.config.latency_cycles;
                self.energy_consumed += self.config.operation_energy;
                FuEvent::BusyUntil(self.busy_until)
            }
            Err(err) => FuEvent::Error(err),
        }
    }

    /// Compute MAC for i8 vectors
    fn compute_mac_i8(&self, vec_a: &[i8], vec_b: &[i8]) -> Result<BusData, String> {
        if vec_a.len() != vec_b.len() {
            return Err(format!("Vector length mismatch: {} vs {}", vec_a.len(), vec_b.len()));
        }

        if vec_a.len() > self.config.lane_count {
            return Err(format!("Vector too long for {} lanes", self.config.lane_count));
        }

        // Compute element-wise multiply and accumulate
        let mut sum: i32 = 0;
        for i in 0..vec_a.len() {
            sum += (vec_a[i] as i32) * (vec_b[i] as i32);
        }

        // Add accumulator input if in accumulate mode
        if self.accumulate_mode {
            if let Some(BusData::I32(acc)) = &self.acc_in {
                sum += acc;
            }
        }

        Ok(BusData::I32(sum))
    }

    /// Compute MAC for i16 vectors
    fn compute_mac_i16(&self, vec_a: &[i16], vec_b: &[i16]) -> Result<BusData, String> {
        if vec_a.len() != vec_b.len() {
            return Err(format!("Vector length mismatch: {} vs {}", vec_a.len(), vec_b.len()));
        }

        if vec_a.len() > self.config.lane_count {
            return Err(format!("Vector too long for {} lanes", self.config.lane_count));
        }

        // Compute element-wise multiply and accumulate
        let mut sum: i32 = 0;
        for i in 0..vec_a.len() {
            sum += (vec_a[i] as i32) * (vec_b[i] as i32);
        }

        // Add accumulator input if in accumulate mode
        if self.accumulate_mode {
            if let Some(BusData::I32(acc)) = &self.acc_in {
                sum += acc;
            }
        }

        Ok(BusData::I32(sum))
    }
}

impl FunctionalUnit for VecMacUnit {
    fn name(&self) -> &'static str {
        "VECMAC"
    }

    fn input_ports(&self) -> Vec<String> {
        vec![
            "VEC_A".to_string(),     // Port 0: Vector A (latched)
            "VEC_B".to_string(),     // Port 1: Vector B (trigger)
            "ACC_IN".to_string(),    // Port 2: Accumulator input (optional)
            "MODE_IN".to_string(),   // Port 3: Mode (accumulate=1/clear=0, latched)
        ]
    }

    fn output_ports(&self) -> Vec<String> {
        vec![
            "SCALAR_OUT".to_string(), // Port 0: Scalar accumulation result
        ]
    }

    fn write_input(&mut self, port: u16, data: BusData, cycle: u64) -> FuEvent {
        // Check if unit is busy
        if self.is_busy(cycle) {
            return FuEvent::BusyUntil(self.busy_until);
        }

        match port {
            0 => { // VEC_A (latched)
                match data {
                    BusData::VecI8(_) | BusData::VecI16(_) => {
                        self.vec_a = Some(data);
                        FuEvent::Ready
                    }
                    _ => FuEvent::Error("VEC_A requires vector data type".to_string()),
                }
            }
            1 => { // VEC_B (trigger)
                match data {
                    BusData::VecI8(_) | BusData::VecI16(_) => {
                        self.vec_b = Some(data);
                        // Trigger operation
                        self.execute_vecmac(cycle)
                    }
                    _ => FuEvent::Error("VEC_B requires vector data type".to_string()),
                }
            }
            2 => { // ACC_IN (optional)
                match data {
                    BusData::I32(_) => {
                        self.acc_in = Some(data);
                        FuEvent::Ready
                    }
                    _ => FuEvent::Error("ACC_IN requires I32 data type".to_string()),
                }
            }
            3 => { // MODE_IN (latched)
                match data {
                    BusData::I32(mode) => {
                        self.accumulate_mode = mode != 0;
                        FuEvent::Ready
                    }
                    _ => FuEvent::Error("MODE_IN requires I32 data type".to_string()),
                }
            }
            _ => FuEvent::Error(format!("Invalid input port: {}", port)),
        }
    }

    fn read_output(&self, port: u16) -> Option<BusData> {
        match port {
            0 => self.scalar_out.clone(), // SCALAR_OUT
            _ => None,
        }
    }

    fn is_busy(&self, cycle: u64) -> bool {
        cycle < self.busy_until
    }

    fn energy_consumed(&self) -> f64 {
        self.energy_consumed
    }

    fn reset(&mut self) {
        self.vec_a = None;
        self.vec_b = None;
        self.acc_in = None;
        self.accumulate_mode = false;
        self.scalar_out = None;
        self.energy_consumed = 0.0;
        self.busy_until = 0;
    }

    fn step(&mut self, _cycle: u64) {
        // No additional state updates needed - operation completes when busy_until is reached
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vecmac_basic_i8() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        // Set up vectors: [1, 2, 3] * [4, 5, 6] = 4 + 10 + 18 = 32
        let vec_a = BusData::VecI8(vec![1, 2, 3]);
        let vec_b = BusData::VecI8(vec![4, 5, 6]);

        // Write vector A (latched)
        let result = vecmac.write_input(0, vec_a, 0);
        assert_eq!(result, FuEvent::Ready);

        // Write vector B (trigger)
        let result = vecmac.write_input(1, vec_b, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        // Check output after operation completes
        assert_eq!(vecmac.read_output(0), Some(BusData::I32(32)));
        assert!(vecmac.energy_consumed() > 0.0);
    }

    #[test]
    fn test_vecmac_accumulate_mode() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        // Set up vectors and accumulator
        let vec_a = BusData::VecI8(vec![1, 2]);
        let vec_b = BusData::VecI8(vec![3, 4]);
        let acc_in = BusData::I32(100);

        // Enable accumulate mode
        let result = vecmac.write_input(3, BusData::I32(1), 0);
        assert_eq!(result, FuEvent::Ready);

        // Set accumulator input
        let result = vecmac.write_input(2, acc_in, 0);
        assert_eq!(result, FuEvent::Ready);

        // Set vector A
        let result = vecmac.write_input(0, vec_a, 0);
        assert_eq!(result, FuEvent::Ready);

        // Trigger with vector B: (1*3 + 2*4) + 100 = 11 + 100 = 111
        let result = vecmac.write_input(1, vec_b, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        assert_eq!(vecmac.read_output(0), Some(BusData::I32(111)));
    }

    #[test]
    fn test_vecmac_clear_mode() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        // Set up vectors and accumulator
        let vec_a = BusData::VecI8(vec![1, 2]);
        let vec_b = BusData::VecI8(vec![3, 4]);
        let acc_in = BusData::I32(100);

        // Clear mode (default, but explicit)
        let result = vecmac.write_input(3, BusData::I32(0), 0);
        assert_eq!(result, FuEvent::Ready);

        // Set accumulator input (should be ignored in clear mode)
        let result = vecmac.write_input(2, acc_in, 0);
        assert_eq!(result, FuEvent::Ready);

        // Set vector A
        let result = vecmac.write_input(0, vec_a, 0);
        assert_eq!(result, FuEvent::Ready);

        // Trigger with vector B: 1*3 + 2*4 = 11 (accumulator ignored)
        let result = vecmac.write_input(1, vec_b, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        assert_eq!(vecmac.read_output(0), Some(BusData::I32(11)));
    }

    #[test]
    fn test_vecmac_i16_vectors() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        // Test with i16 vectors: [10, 20] * [30, 40] = 300 + 800 = 1100
        let vec_a = BusData::VecI16(vec![10, 20]);
        let vec_b = BusData::VecI16(vec![30, 40]);

        let result = vecmac.write_input(0, vec_a, 0);
        assert_eq!(result, FuEvent::Ready);

        let result = vecmac.write_input(1, vec_b, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        assert_eq!(vecmac.read_output(0), Some(BusData::I32(1100)));
    }

    #[test]
    fn test_vecmac_length_mismatch() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        // Mismatched vector lengths
        let vec_a = BusData::VecI8(vec![1, 2, 3]);
        let vec_b = BusData::VecI8(vec![4, 5]);

        let result = vecmac.write_input(0, vec_a, 0);
        assert_eq!(result, FuEvent::Ready);

        let result = vecmac.write_input(1, vec_b, 0);
        assert!(matches!(result, FuEvent::Error(_)));
    }

    #[test]
    fn test_vecmac_type_mismatch() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        // Mixed vector types
        let vec_a = BusData::VecI8(vec![1, 2]);
        let vec_b = BusData::VecI16(vec![3, 4]);

        let result = vecmac.write_input(0, vec_a, 0);
        assert_eq!(result, FuEvent::Ready);

        let result = vecmac.write_input(1, vec_b, 0);
        assert!(matches!(result, FuEvent::Error(_)));
    }

    #[test]
    fn test_vecmac_busy_state() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        // Start operation
        let vec_a = BusData::VecI8(vec![1, 2]);
        let vec_b = BusData::VecI8(vec![3, 4]);

        vecmac.write_input(0, vec_a, 10);
        let result = vecmac.write_input(1, vec_b, 10);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        // Should be busy during latency period
        assert!(vecmac.is_busy(11));

        // Should not be busy after latency completes
        assert!(!vecmac.is_busy(15));
    }

    #[test]
    fn test_vecmac_energy_accounting() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        let initial_energy = vecmac.energy_consumed();

        // Perform operation
        let vec_a = BusData::VecI8(vec![1, 2]);
        let vec_b = BusData::VecI8(vec![3, 4]);

        vecmac.write_input(0, vec_a, 0);
        vecmac.write_input(1, vec_b, 0);

        // Energy should have increased
        assert!(vecmac.energy_consumed() > initial_energy);
        assert_eq!(vecmac.energy_consumed(), initial_energy + 40.0);
    }

    #[test]
    fn test_vecmac_reset() {
        let config = VecMacConfig::default();
        let mut vecmac = VecMacUnit::new(config);

        // Set up state
        let vec_a = BusData::VecI8(vec![1, 2]);
        vecmac.write_input(0, vec_a, 0);
        vecmac.write_input(3, BusData::I32(1), 0); // accumulate mode

        // Reset
        vecmac.reset();

        // State should be cleared
        assert_eq!(vecmac.read_output(0), None);
        assert_eq!(vecmac.energy_consumed(), 0.0);
        assert!(!vecmac.is_busy(100));
    }
}