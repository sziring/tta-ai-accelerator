// src/risc/mod.rs
//! RISC baseline processor for EDP comparison
//!
//! This module implements a simple RISC-style processor baseline
//! to compare Energy-Delay Product (EDP) against the TTA implementation.

pub mod processor;
pub mod instruction_set;
pub mod benchmarks;

// Re-export core RISC types
pub use processor::{RiscProcessor, RiscConfig, ExecutionResult};
pub use instruction_set::{RiscInstruction, InstructionType, Register, ReduceMode};
pub use benchmarks::{BenchmarkSuite, BenchmarkResult, EdpComparison};