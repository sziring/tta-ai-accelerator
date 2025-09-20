// src/kernels/mod.rs
//! Advanced AI Kernels for TTA Research
//!
//! This module implements novel and advanced AI computation kernels
//! that showcase TTA's unique advantages over traditional architectures.

// Core validated kernels (used for 7x efficiency proof)
pub mod attention;
pub mod optimized_attention;
pub mod batch_norm;
pub mod softmax;
pub mod sparse_matmul;
pub mod winograd;
pub mod quantized_ops;

// Extended kernel suite (optional - preserves existing results)
#[cfg(feature = "extended-kernels")]
pub mod conv2d;
#[cfg(feature = "extended-kernels")]
pub mod gemm;
#[cfg(feature = "extended-kernels")]
pub mod softmax_pipeline;

// Re-export core kernel types (always available)
pub use attention::{MultiHeadAttention, AttentionConfig};
pub use optimized_attention::{OptimizedAttention, OptimizedAttentionConfig, OptimizationAnalysis};
pub use batch_norm::{BatchNormalization, BatchNormConfig};
pub use softmax::{SoftmaxKernel, SoftmaxConfig};
pub use sparse_matmul::{SparseMatMul, SparseConfig};
pub use winograd::{WinogradConv, WinogradConfig};
pub use quantized_ops::{QuantizedOps, QuantConfig};

// Re-export extended kernels (feature-gated to preserve existing functionality)
#[cfg(feature = "extended-kernels")]
pub use conv2d::{Conv2DKernel, Conv2DConfig};
#[cfg(feature = "extended-kernels")]
pub use gemm::{GemmKernel, GemmConfig};
#[cfg(feature = "extended-kernels")]
pub use softmax_pipeline::{SoftmaxPipelineKernel, SoftmaxPipelineConfig};

use crate::tta::{FunctionalUnit, BusData, FuEvent};
use crate::validation::{ReferenceData, ValidationResult};

/// Kernel execution metrics for performance analysis
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    pub kernel_name: String,
    pub input_size: usize,
    pub output_size: usize,
    pub energy_consumed: f64,
    pub cycles_taken: u64,
    pub throughput_ops_per_cycle: f64,
    pub energy_per_op: f64,
    pub utilization_efficiency: f64,
}

/// Trait for advanced AI kernels that can run on TTA
pub trait AdvancedKernel {
    /// Get the kernel name
    fn name(&self) -> &'static str;

    /// Execute the kernel with given inputs
    fn execute(&mut self, inputs: &[BusData], cycle: u64) -> Result<Vec<BusData>, String>;

    /// Get energy consumption
    fn energy_consumed(&self) -> f64;

    /// Get execution metrics
    fn get_metrics(&self) -> KernelMetrics;

    /// Reset kernel state
    fn reset(&mut self);

    /// Get expected energy for performance analysis
    fn expected_energy(&self, input_size: usize) -> f64;

    /// Check if this kernel benefits from TTA vs RISC
    fn tta_advantage_factor(&self) -> f64; // Expected speedup/energy improvement
}

/// Kernel suite for advanced AI operations
#[derive(Debug)]
pub struct AdvancedKernelSuite {
    pub attention: MultiHeadAttention,
    pub batch_norm: BatchNormalization,
    pub softmax: SoftmaxKernel,
    pub sparse_matmul: SparseMatMul,
    pub winograd: WinogradConv,
    pub quantized: QuantizedOps,
}

impl AdvancedKernelSuite {
    pub fn new() -> Self {
        Self {
            attention: MultiHeadAttention::new(AttentionConfig::default()),
            batch_norm: BatchNormalization::new(BatchNormConfig::default()),
            softmax: SoftmaxKernel::new(SoftmaxConfig::default()),
            sparse_matmul: SparseMatMul::new(SparseConfig::default()),
            winograd: WinogradConv::new(WinogradConfig::default()),
            quantized: QuantizedOps::new(QuantConfig::default()),
        }
    }

    /// Run comprehensive benchmarks against RISC baseline
    pub fn benchmark_vs_risc(&mut self) -> KernelBenchmarkReport {
        let mut results = Vec::new();

        // Test each kernel with multiple input sizes
        let test_sizes = vec![8, 16, 32, 64, 128];

        for &size in &test_sizes {
            results.push(self.benchmark_attention(size));
            results.push(self.benchmark_batch_norm(size));
            results.push(self.benchmark_softmax(size));
            results.push(self.benchmark_sparse_matmul(size));
        }

        KernelBenchmarkReport {
            total_kernels_tested: results.len(),
            tta_wins: results.iter().filter(|r| r.tta_advantage > 1.0).count(),
            average_tta_advantage: results.iter().map(|r| r.tta_advantage).sum::<f64>() / results.len() as f64,
            results,
        }
    }

    fn benchmark_attention(&mut self, size: usize) -> KernelBenchmarkResult {
        // Implementation will compare TTA vs RISC for attention operations
        KernelBenchmarkResult {
            kernel_name: "multi_head_attention".to_string(),
            input_size: size,
            tta_energy: 0.0,
            risc_energy: 0.0,
            tta_cycles: 0,
            risc_cycles: 0,
            tta_advantage: 1.0,
        }
    }

    fn benchmark_batch_norm(&mut self, size: usize) -> KernelBenchmarkResult {
        KernelBenchmarkResult {
            kernel_name: "batch_normalization".to_string(),
            input_size: size,
            tta_energy: 0.0,
            risc_energy: 0.0,
            tta_cycles: 0,
            risc_cycles: 0,
            tta_advantage: 1.0,
        }
    }

    fn benchmark_softmax(&mut self, size: usize) -> KernelBenchmarkResult {
        KernelBenchmarkResult {
            kernel_name: "softmax".to_string(),
            input_size: size,
            tta_energy: 0.0,
            risc_energy: 0.0,
            tta_cycles: 0,
            risc_cycles: 0,
            tta_advantage: 1.0,
        }
    }

    fn benchmark_sparse_matmul(&mut self, size: usize) -> KernelBenchmarkResult {
        KernelBenchmarkResult {
            kernel_name: "sparse_matrix_multiply".to_string(),
            input_size: size,
            tta_energy: 0.0,
            risc_energy: 0.0,
            tta_cycles: 0,
            risc_cycles: 0,
            tta_advantage: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelBenchmarkResult {
    pub kernel_name: String,
    pub input_size: usize,
    pub tta_energy: f64,
    pub risc_energy: f64,
    pub tta_cycles: u64,
    pub risc_cycles: u64,
    pub tta_advantage: f64, // tta_energy/risc_energy (lower is better for TTA)
}

#[derive(Debug)]
pub struct KernelBenchmarkReport {
    pub total_kernels_tested: usize,
    pub tta_wins: usize,
    pub average_tta_advantage: f64,
    pub results: Vec<KernelBenchmarkResult>,
}

impl Default for AdvancedKernelSuite {
    fn default() -> Self {
        Self::new()
    }
}