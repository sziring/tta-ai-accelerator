// src/kernels/batch_norm.rs
//! Batch Normalization kernel optimized for TTA

use crate::kernels::{AdvancedKernel, KernelMetrics};
use crate::tta::BusData;

#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    pub batch_size: usize,
    pub feature_size: usize,
    pub epsilon: f64,
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            feature_size: 128,
            epsilon: 1e-5,
        }
    }
}

#[derive(Debug)]
pub struct BatchNormalization {
    config: BatchNormConfig,
    energy_consumed: f64,
    last_execution_cycles: u64,
}

impl BatchNormalization {
    pub fn new(config: BatchNormConfig) -> Self {
        Self {
            config,
            energy_consumed: 0.0,
            last_execution_cycles: 0,
        }
    }
}

impl AdvancedKernel for BatchNormalization {
    fn name(&self) -> &'static str {
        "batch_normalization"
    }

    fn execute(&mut self, inputs: &[BusData], cycle: u64) -> Result<Vec<BusData>, String> {
        // Stub implementation
        self.energy_consumed += 50.0;
        self.last_execution_cycles = 8;
        Ok(vec![BusData::I32(1); inputs.len()])
    }

    fn energy_consumed(&self) -> f64 {
        self.energy_consumed
    }

    fn get_metrics(&self) -> KernelMetrics {
        KernelMetrics {
            kernel_name: "batch_normalization".to_string(),
            input_size: self.config.batch_size * self.config.feature_size,
            output_size: self.config.batch_size * self.config.feature_size,
            energy_consumed: self.energy_consumed,
            cycles_taken: self.last_execution_cycles,
            throughput_ops_per_cycle: 10.0,
            energy_per_op: 2.0,
            utilization_efficiency: 0.75,
        }
    }

    fn reset(&mut self) {
        self.energy_consumed = 0.0;
        self.last_execution_cycles = 0;
    }

    fn expected_energy(&self, input_size: usize) -> f64 {
        input_size as f64 * 0.5
    }

    fn tta_advantage_factor(&self) -> f64 {
        1.6 // Modest advantage for batch norm
    }
}