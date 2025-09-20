// src/validation/mod.rs
//! Golden reference validation framework for TTA kernels
//!
//! This module provides comprehensive validation of TTA kernel implementations
//! against golden reference outputs with energy variance analysis.

pub mod golden_reference;
pub mod kernel_suite;
pub mod variance_analysis;
pub mod realistic_workloads;

// Re-export core validation types
pub use golden_reference::{GoldenReference, ReferenceData, ValidationResult};
pub use kernel_suite::{KernelSuite, KernelTest, TestCategory};
pub use variance_analysis::{VarianceAnalyzer, EnergyVariance, EnergyCategory, VarianceReport};
pub use realistic_workloads::{ModelConfig, WorkloadValidationSuite, SparsityPattern};