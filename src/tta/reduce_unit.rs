// src/tta/reduce_unit.rs
//! Vector Reduction Unit for AI acceleration
//!
//! Implements various reduction operations on vectors including sum, max, and argmax.
//! These are fundamental operations for neural network computations and data processing.

use crate::tta::{FunctionalUnit, BusData, FuEvent};

/// Vector Reduction functional unit
#[derive(Clone, Debug)]
pub struct ReduceUnit {
    /// Vector input (triggers operation)
    vec_in: Option<BusData>,
    /// Operation mode: 0=sum, 1=max, 2=argmax
    mode: ReduceMode,
    /// Scalar output result
    scalar_out: Option<BusData>,
    /// Index output (for argmax operations)
    index_out: Option<BusData>,
    /// Configuration
    config: ReduceConfig,
    /// Energy consumed
    energy_consumed: f64,
    /// Busy state
    busy_until: u64,
}

#[derive(Clone, Debug)]
pub struct ReduceConfig {
    /// Energy cost for sum operations
    pub sum_energy: f64,
    /// Energy cost for max operations
    pub max_energy: f64,
    /// Energy cost for argmax operations
    pub argmax_energy: f64,
    /// Latency for sum/max operations
    pub sum_max_latency: u64,
    /// Latency for argmax operations
    pub argmax_latency: u64,
}

#[derive(Clone, Debug, PartialEq)]
enum ReduceMode {
    Sum,    // 0
    Max,    // 1
    ArgMax, // 2
}

impl Default for ReduceConfig {
    fn default() -> Self {
        Self {
            sum_energy: 10.0,      // matches reduce_sum16 from documentation
            max_energy: 10.0,      // same as sum for max operation
            argmax_energy: 16.0,   // matches reduce_argmax16 from documentation
            sum_max_latency: 1,    // 1 cycle for sum/max
            argmax_latency: 2,     // 2 cycles for argmax (needs index tracking)
        }
    }
}

impl ReduceUnit {
    pub fn new(config: ReduceConfig) -> Self {
        Self {
            vec_in: None,
            mode: ReduceMode::Sum,
            scalar_out: None,
            index_out: None,
            energy_consumed: 0.0,
            busy_until: 0,
            config,
        }
    }

    /// Perform vector reduction operation
    fn execute_reduce(&mut self, cycle: u64) -> FuEvent {
        let vec_data = match &self.vec_in {
            Some(data) => data,
            None => return FuEvent::Error("VEC_IN not set".to_string()),
        };

        // Execute reduction based on mode and vector type
        let result = match vec_data {
            BusData::VecI8(vec) => {
                self.compute_reduce_i8(vec)
            }
            BusData::VecI16(vec) => {
                self.compute_reduce_i16(vec)
            }
            _ => {
                return FuEvent::Error("Invalid vector type for REDUCE".to_string());
            }
        };

        match result {
            Ok((scalar, index)) => {
                self.scalar_out = Some(scalar);
                self.index_out = index;

                // Set timing and energy based on operation mode
                let (latency, energy) = match self.mode {
                    ReduceMode::Sum => (self.config.sum_max_latency, self.config.sum_energy),
                    ReduceMode::Max => (self.config.sum_max_latency, self.config.max_energy),
                    ReduceMode::ArgMax => (self.config.argmax_latency, self.config.argmax_energy),
                };

                self.busy_until = cycle + latency;
                self.energy_consumed += energy;
                FuEvent::BusyUntil(self.busy_until)
            }
            Err(err) => FuEvent::Error(err),
        }
    }

    /// Compute reduction for i8 vectors
    fn compute_reduce_i8(&self, vec: &[i8]) -> Result<(BusData, Option<BusData>), String> {
        if vec.is_empty() {
            return Err("Cannot reduce empty vector".to_string());
        }

        match self.mode {
            ReduceMode::Sum => {
                let sum: i32 = vec.iter().map(|&x| x as i32).sum();
                Ok((BusData::I32(sum), None))
            }
            ReduceMode::Max => {
                let max_val = *vec.iter().max().unwrap();
                Ok((BusData::I32(max_val as i32), None))
            }
            ReduceMode::ArgMax => {
                let (max_idx, &max_val) = vec.iter()
                    .enumerate()
                    .max_by_key(|(_, &val)| val)
                    .unwrap();
                Ok((
                    BusData::I32(max_val as i32),
                    Some(BusData::I32(max_idx as i32))
                ))
            }
        }
    }

    /// Compute reduction for i16 vectors
    fn compute_reduce_i16(&self, vec: &[i16]) -> Result<(BusData, Option<BusData>), String> {
        if vec.is_empty() {
            return Err("Cannot reduce empty vector".to_string());
        }

        match self.mode {
            ReduceMode::Sum => {
                let sum: i32 = vec.iter().map(|&x| x as i32).sum();
                Ok((BusData::I32(sum), None))
            }
            ReduceMode::Max => {
                let max_val = *vec.iter().max().unwrap();
                Ok((BusData::I32(max_val as i32), None))
            }
            ReduceMode::ArgMax => {
                let (max_idx, &max_val) = vec.iter()
                    .enumerate()
                    .max_by_key(|(_, &val)| val)
                    .unwrap();
                Ok((
                    BusData::I32(max_val as i32),
                    Some(BusData::I32(max_idx as i32))
                ))
            }
        }
    }
}

impl FunctionalUnit for ReduceUnit {
    fn name(&self) -> &'static str {
        "REDUCE"
    }

    fn input_ports(&self) -> Vec<String> {
        vec![
            "VEC_IN".to_string(),    // Port 0: Vector input (trigger)
            "MODE_IN".to_string(),   // Port 1: Mode selection (latched)
        ]
    }

    fn output_ports(&self) -> Vec<String> {
        vec![
            "SCALAR_OUT".to_string(), // Port 0: Scalar result
            "INDEX_OUT".to_string(),  // Port 1: Index (for argmax)
        ]
    }

    fn write_input(&mut self, port: u16, data: BusData, cycle: u64) -> FuEvent {
        // Check if unit is busy
        if self.is_busy(cycle) {
            return FuEvent::BusyUntil(self.busy_until);
        }

        match port {
            0 => { // VEC_IN (trigger)
                match data {
                    BusData::VecI8(_) | BusData::VecI16(_) => {
                        self.vec_in = Some(data);
                        // Trigger operation
                        self.execute_reduce(cycle)
                    }
                    _ => FuEvent::Error("VEC_IN requires vector data type".to_string()),
                }
            }
            1 => { // MODE_IN (latched)
                match data {
                    BusData::I32(mode) => {
                        self.mode = match mode {
                            0 => ReduceMode::Sum,
                            1 => ReduceMode::Max,
                            2 => ReduceMode::ArgMax,
                            _ => return FuEvent::Error("Invalid mode: use 0=sum, 1=max, 2=argmax".to_string()),
                        };
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
            1 => self.index_out.clone(),  // INDEX_OUT
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
        self.vec_in = None;
        self.mode = ReduceMode::Sum;
        self.scalar_out = None;
        self.index_out = None;
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
    fn test_reduce_sum_i8() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Set sum mode
        let result = reduce.write_input(1, BusData::I32(0), 0);
        assert_eq!(result, FuEvent::Ready);

        // Test sum: [1, 2, 3, 4] -> 10
        let vec_data = BusData::VecI8(vec![1, 2, 3, 4]);
        let result = reduce.write_input(0, vec_data, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        assert_eq!(reduce.read_output(0), Some(BusData::I32(10)));
        assert_eq!(reduce.read_output(1), None); // No index for sum
        assert!(reduce.energy_consumed() > 0.0);
    }

    #[test]
    fn test_reduce_max_i8() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Set max mode
        let result = reduce.write_input(1, BusData::I32(1), 0);
        assert_eq!(result, FuEvent::Ready);

        // Test max: [3, 1, 4, 2] -> 4
        let vec_data = BusData::VecI8(vec![3, 1, 4, 2]);
        let result = reduce.write_input(0, vec_data, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        assert_eq!(reduce.read_output(0), Some(BusData::I32(4)));
        assert_eq!(reduce.read_output(1), None); // No index for max
    }

    #[test]
    fn test_reduce_argmax_i8() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Set argmax mode
        let result = reduce.write_input(1, BusData::I32(2), 0);
        assert_eq!(result, FuEvent::Ready);

        // Test argmax: [3, 1, 4, 2] -> value=4, index=2
        let vec_data = BusData::VecI8(vec![3, 1, 4, 2]);
        let result = reduce.write_input(0, vec_data, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        assert_eq!(reduce.read_output(0), Some(BusData::I32(4))); // max value
        assert_eq!(reduce.read_output(1), Some(BusData::I32(2))); // index of max
    }

    #[test]
    fn test_reduce_sum_i16() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Test with i16 vectors: [100, 200, 300] -> 600
        let vec_data = BusData::VecI16(vec![100, 200, 300]);
        let result = reduce.write_input(0, vec_data, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        assert_eq!(reduce.read_output(0), Some(BusData::I32(600)));
    }

    #[test]
    fn test_reduce_argmax_i16() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Set argmax mode
        reduce.write_input(1, BusData::I32(2), 0);

        // Test argmax with i16: [100, 300, 200] -> value=300, index=1
        let vec_data = BusData::VecI16(vec![100, 300, 200]);
        let result = reduce.write_input(0, vec_data, 0);
        assert!(matches!(result, FuEvent::BusyUntil(_)));

        assert_eq!(reduce.read_output(0), Some(BusData::I32(300)));
        assert_eq!(reduce.read_output(1), Some(BusData::I32(1)));
    }

    #[test]
    fn test_reduce_timing() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Test sum timing (1 cycle)
        reduce.write_input(1, BusData::I32(0), 10); // sum mode at cycle 10
        let result = reduce.write_input(0, BusData::VecI8(vec![1, 2]), 10);
        if let FuEvent::BusyUntil(until) = result {
            assert_eq!(until, 11); // 10 + 1 cycle latency
        } else {
            panic!("Expected BusyUntil");
        }

        // Reset and test argmax timing (2 cycles)
        reduce.reset();
        reduce.write_input(1, BusData::I32(2), 20); // argmax mode at cycle 20
        let result = reduce.write_input(0, BusData::VecI8(vec![1, 2]), 20);
        if let FuEvent::BusyUntil(until) = result {
            assert_eq!(until, 22); // 20 + 2 cycles latency
        } else {
            panic!("Expected BusyUntil");
        }
    }

    #[test]
    fn test_reduce_energy_accounting() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        let initial_energy = reduce.energy_consumed();

        // Test sum energy
        reduce.write_input(1, BusData::I32(0), 0); // sum mode
        reduce.write_input(0, BusData::VecI8(vec![1, 2, 3]), 0);
        assert_eq!(reduce.energy_consumed(), initial_energy + 10.0);

        // Reset and test argmax energy
        reduce.reset();
        reduce.write_input(1, BusData::I32(2), 0); // argmax mode
        reduce.write_input(0, BusData::VecI8(vec![1, 2, 3]), 0);
        assert_eq!(reduce.energy_consumed(), 16.0);
    }

    #[test]
    fn test_reduce_invalid_mode() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Test invalid mode
        let result = reduce.write_input(1, BusData::I32(99), 0);
        assert!(matches!(result, FuEvent::Error(_)));
    }

    #[test]
    fn test_reduce_empty_vector() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Test empty vector
        let result = reduce.write_input(0, BusData::VecI8(vec![]), 0);
        assert!(matches!(result, FuEvent::Error(_)));
    }

    #[test]
    fn test_reduce_invalid_input_type() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Test invalid input type
        let result = reduce.write_input(0, BusData::I32(42), 0);
        assert!(matches!(result, FuEvent::Error(_)));
    }

    #[test]
    fn test_reduce_busy_state() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Start operation
        reduce.write_input(1, BusData::I32(2), 5); // argmax mode
        let result = reduce.write_input(0, BusData::VecI8(vec![1, 2, 3]), 5);
        assert!(matches!(result, FuEvent::BusyUntil(7))); // 5 + 2 cycles

        // Should be busy during latency
        assert!(reduce.is_busy(6));
        assert!(!reduce.is_busy(7));

        // Should reject new inputs while busy
        let result = reduce.write_input(0, BusData::VecI8(vec![4, 5, 6]), 6);
        assert!(matches!(result, FuEvent::BusyUntil(_)));
    }

    #[test]
    fn test_reduce_reset() {
        let config = ReduceConfig::default();
        let mut reduce = ReduceUnit::new(config);

        // Set up state
        reduce.write_input(1, BusData::I32(2), 0); // argmax mode
        reduce.write_input(0, BusData::VecI8(vec![1, 2, 3]), 0);

        // Reset
        reduce.reset();

        // State should be cleared
        assert_eq!(reduce.read_output(0), None);
        assert_eq!(reduce.read_output(1), None);
        assert_eq!(reduce.energy_consumed(), 0.0);
        assert!(!reduce.is_busy(100));
    }
}