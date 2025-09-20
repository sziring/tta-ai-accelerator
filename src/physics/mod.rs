// src/physics/mod.rs
//! Physics engine integration for TTA energy validation
//! Moved and adapted from physics_engine/ folder

pub mod universe;
pub mod energy_validation;

pub use universe::Universe;
pub use energy_validation::{PhysicsBackend, ValidationReport, EnergyTable};

// Re-export key types for compatibility
pub use universe::PhysicsRegime;
