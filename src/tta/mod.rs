// src/tta/mod.rs
//! Transport-Triggered Architecture simulator core module

pub mod functional_unit;
pub mod immediate_unit;
pub mod spm_unit;
pub mod vecmac_unit;
pub mod reduce_unit;
pub mod scheduler;
pub mod instruction;
pub mod execution_engine;
pub mod processor;

// Re-export core types for easy access
pub use functional_unit::{FunctionalUnit, BusData, PortId, FuEvent};

// Re-export specific functional units
pub use immediate_unit::{ImmediateUnit, ImmConfig};
pub use spm_unit::{ScratchpadMemory, SpmConfig};
pub use vecmac_unit::{VecMacUnit, VecMacConfig};
pub use reduce_unit::{ReduceUnit, ReduceConfig};

// Re-export scheduler and execution components
pub use scheduler::{TtaScheduler, SchedulerConfig, Move, StallReason};
pub use instruction::{TtaProgram, TtaParser, Instruction, MoveInstruction};
pub use execution_engine::{TtaExecutionEngine, ExecutionStats};

// Re-export processor
pub use processor::TtaProcessor;
