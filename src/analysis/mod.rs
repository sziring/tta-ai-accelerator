// src/analysis/mod.rs
//! Advanced Analysis Framework for TTA Research
//!
//! This module provides comprehensive analysis tools for end-to-end
//! model evaluation, competitive benchmarking, and publication-ready results.

pub mod transformer_models;
pub mod competitive_benchmarks;
pub mod scaling_analysis;
pub mod publication_metrics;
pub mod performance_energy_analysis;
pub mod performance_summary;

// Re-export key analysis types
pub use transformer_models::{TransformerBlock, ModelAnalysis, ModelMetrics};
pub use competitive_benchmarks::{BenchmarkSuite, CompetitiveAnalysis, AcceleratorComparison};
pub use scaling_analysis::{ScalingStudy, PerformanceProjection, EfficiencyTrends};
pub use publication_metrics::{PublicationReport, MetricsCollector, VisualizationData};
pub use performance_energy_analysis::{PerformanceEnergyAnalysis, PerformanceVerdict, KernelPerfEnergy};
pub use performance_summary::{TtaPerformanceSummary, PerformanceEnergyVerdict};