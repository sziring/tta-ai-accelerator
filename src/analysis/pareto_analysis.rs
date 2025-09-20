// src/analysis/pareto_analysis.rs
//! Pareto front analysis for multi-objective optimization of TTA configurations
//!
//! Provides tools to identify Pareto-optimal configurations considering
//! multiple objectives like energy, performance, area, and utilization.

use crate::analysis::parameter_sweep::{SweepResult, BenchmarkMetrics};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub configuration_index: usize,
    pub objectives: Vec<f64>,
    pub labels: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFront {
    pub points: Vec<ParetoPoint>,
    pub objective_names: Vec<String>,
    pub dominated_points: Vec<ParetoPoint>,
}

#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    Minimize(String),  // e.g., "energy", "cycles", "edp"
    Maximize(String),  // e.g., "throughput", "utilization"
}

pub struct ParetoAnalyzer {
    objectives: Vec<OptimizationObjective>,
}

impl ParetoAnalyzer {
    pub fn new(objectives: Vec<OptimizationObjective>) -> Self {
        Self { objectives }
    }

    /// Create analyzer with common TTA objectives
    pub fn default_tta_objectives() -> Self {
        Self::new(vec![
            OptimizationObjective::Minimize("energy".to_string()),
            OptimizationObjective::Minimize("cycles".to_string()),
            OptimizationObjective::Maximize("throughput".to_string()),
            OptimizationObjective::Maximize("utilization".to_string()),
        ])
    }

    /// Analyze sweep results and extract Pareto front
    pub fn analyze(&self, sweep_results: &[SweepResult]) -> ParetoFront {
        println!("🔍 Performing Pareto front analysis");
        println!("📊 Analyzing {} configurations across {} objectives",
                 sweep_results.len(), self.objectives.len());

        // Extract objective values for all configurations
        let mut candidates = Vec::new();
        for (idx, result) in sweep_results.iter().enumerate() {
            if let Some(point) = self.extract_objectives(idx, result) {
                candidates.push(point);
            }
        }

        println!("✅ Extracted objectives for {} valid configurations", candidates.len());

        // Find Pareto front
        let (pareto_points, dominated) = self.find_pareto_front(candidates);

        println!("🎯 Pareto front contains {} configurations", pareto_points.len());
        println!("📉 {} configurations are dominated", dominated.len());

        ParetoFront {
            points: pareto_points,
            objective_names: self.objectives.iter()
                .map(|obj| match obj {
                    OptimizationObjective::Minimize(name) |
                    OptimizationObjective::Maximize(name) => name.clone()
                })
                .collect(),
            dominated_points: dominated,
        }
    }

    fn extract_objectives(&self, config_idx: usize, result: &SweepResult) -> Option<ParetoPoint> {
        let mut objectives = Vec::new();
        let mut labels = Vec::new();
        let mut metadata = HashMap::new();

        // Get average metrics across all benchmarks
        let tta_metrics = self.average_metrics(&result.tta_results);
        let _risc_metrics = self.average_metrics(&result.risc_results);

        // Add configuration metadata
        metadata.insert("vector_lanes".to_string(), result.configuration.vector_lanes.to_string());
        metadata.insert("bus_count".to_string(), result.configuration.bus_count.to_string());
        metadata.insert("memory_banks".to_string(), result.configuration.memory_banks.to_string());
        metadata.insert("issue_width".to_string(), result.configuration.issue_width.to_string());

        for objective in &self.objectives {
            match objective {
                OptimizationObjective::Minimize(name) => {
                    let value = match name.as_str() {
                        "energy" => tta_metrics.energy,
                        "cycles" => tta_metrics.cycles as f64,
                        "edp" => tta_metrics.edp,
                        "energy_ratio" => result.comparative_metrics.energy_efficiency_ratio,
                        _ => return None,
                    };
                    objectives.push(value);
                    labels.push(format!("min({})", name));
                },
                OptimizationObjective::Maximize(name) => {
                    let value = match name.as_str() {
                        "throughput" => tta_metrics.throughput,
                        "utilization" => tta_metrics.utilization,
                        "edp_improvement" => result.comparative_metrics.edp_improvement,
                        "performance_ratio" => 1.0 / result.comparative_metrics.performance_ratio, // Invert for maximization
                        _ => return None,
                    };
                    objectives.push(-value); // Negate for maximization (treat as minimization)
                    labels.push(format!("max({})", name));
                },
            }
        }

        Some(ParetoPoint {
            configuration_index: config_idx,
            objectives,
            labels,
            metadata,
        })
    }

    fn average_metrics(&self, benchmark_results: &HashMap<String, BenchmarkMetrics>) -> BenchmarkMetrics {
        if benchmark_results.is_empty() {
            return BenchmarkMetrics {
                cycles: 0,
                energy: 0.0,
                edp: 0.0,
                instructions: 0,
                utilization: 0.0,
                throughput: 0.0,
            };
        }

        let count = benchmark_results.len() as f64;
        let sum_cycles: u64 = benchmark_results.values().map(|m| m.cycles).sum();
        let sum_energy: f64 = benchmark_results.values().map(|m| m.energy).sum();
        let sum_edp: f64 = benchmark_results.values().map(|m| m.edp).sum();
        let sum_instructions: u64 = benchmark_results.values().map(|m| m.instructions).sum();
        let sum_utilization: f64 = benchmark_results.values().map(|m| m.utilization).sum();
        let sum_throughput: f64 = benchmark_results.values().map(|m| m.throughput).sum();

        BenchmarkMetrics {
            cycles: (sum_cycles as f64 / count) as u64,
            energy: sum_energy / count,
            edp: sum_edp / count,
            instructions: (sum_instructions as f64 / count) as u64,
            utilization: sum_utilization / count,
            throughput: sum_throughput / count,
        }
    }

    fn find_pareto_front(&self, candidates: Vec<ParetoPoint>) -> (Vec<ParetoPoint>, Vec<ParetoPoint>) {
        let mut pareto_points: Vec<ParetoPoint> = Vec::new();
        let mut dominated_points = Vec::new();

        for candidate in candidates {
            let mut is_dominated = false;

            // Check if this candidate is dominated by any existing Pareto point
            for pareto_point in &pareto_points {
                if self.dominates(&pareto_point.objectives, &candidate.objectives) {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                // Remove any existing Pareto points that are dominated by this candidate
                let mut new_pareto_points = Vec::new();
                for pareto_point in pareto_points {
                    if self.dominates(&candidate.objectives, &pareto_point.objectives) {
                        dominated_points.push(pareto_point);
                    } else {
                        new_pareto_points.push(pareto_point);
                    }
                }

                new_pareto_points.push(candidate);
                pareto_points = new_pareto_points;
            } else {
                dominated_points.push(candidate);
            }
        }

        (pareto_points, dominated_points)
    }

    /// Check if objectives1 dominates objectives2 (all values are better or equal, at least one strictly better)
    fn dominates(&self, objectives1: &[f64], objectives2: &[f64]) -> bool {
        if objectives1.len() != objectives2.len() {
            return false;
        }

        let mut at_least_one_better = false;
        for i in 0..objectives1.len() {
            if objectives1[i] > objectives2[i] {
                return false; // objectives1 is worse in this dimension
            }
            if objectives1[i] < objectives2[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Generate trade-off analysis between two specific objectives
    pub fn analyze_tradeoff(&self, pareto_front: &ParetoFront, obj1_name: &str, obj2_name: &str) -> TradeoffAnalysis {
        let obj1_idx = pareto_front.objective_names.iter().position(|name| name == obj1_name);
        let obj2_idx = pareto_front.objective_names.iter().position(|name| name == obj2_name);

        if obj1_idx.is_none() || obj2_idx.is_none() {
            return TradeoffAnalysis {
                objective1: obj1_name.to_string(),
                objective2: obj2_name.to_string(),
                correlation: 0.0,
                points: Vec::new(),
                best_compromise: None,
            };
        }

        let idx1 = obj1_idx.unwrap();
        let idx2 = obj2_idx.unwrap();

        let mut points = Vec::new();
        for point in &pareto_front.points {
            points.push(TradeoffPoint {
                config_index: point.configuration_index,
                objective1_value: point.objectives[idx1],
                objective2_value: point.objectives[idx2],
                metadata: point.metadata.clone(),
            });
        }

        // Calculate correlation
        let correlation = self.calculate_correlation(&points);

        // Find best compromise (minimize normalized distance from origin)
        let best_compromise = self.find_best_compromise(&points);

        TradeoffAnalysis {
            objective1: obj1_name.to_string(),
            objective2: obj2_name.to_string(),
            correlation,
            points,
            best_compromise,
        }
    }

    fn calculate_correlation(&self, points: &[TradeoffPoint]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }

        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|p| p.objective1_value).sum();
        let sum_y: f64 = points.iter().map(|p| p.objective2_value).sum();
        let sum_xx: f64 = points.iter().map(|p| p.objective1_value * p.objective1_value).sum();
        let sum_yy: f64 = points.iter().map(|p| p.objective2_value * p.objective2_value).sum();
        let sum_xy: f64 = points.iter().map(|p| p.objective1_value * p.objective2_value).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn find_best_compromise(&self, points: &[TradeoffPoint]) -> Option<usize> {
        if points.is_empty() {
            return None;
        }

        // Normalize objectives to [0, 1] range
        let min_obj1 = points.iter().map(|p| p.objective1_value).fold(f64::INFINITY, f64::min);
        let max_obj1 = points.iter().map(|p| p.objective1_value).fold(f64::NEG_INFINITY, f64::max);
        let min_obj2 = points.iter().map(|p| p.objective2_value).fold(f64::INFINITY, f64::min);
        let max_obj2 = points.iter().map(|p| p.objective2_value).fold(f64::NEG_INFINITY, f64::max);

        let range1 = max_obj1 - min_obj1;
        let range2 = max_obj2 - min_obj2;

        if range1.abs() < 1e-10 || range2.abs() < 1e-10 {
            return Some(0);
        }

        // Find point with minimum Euclidean distance from normalized origin
        let mut best_distance = f64::INFINITY;
        let mut best_index = 0;

        for (idx, point) in points.iter().enumerate() {
            let norm1 = (point.objective1_value - min_obj1) / range1;
            let norm2 = (point.objective2_value - min_obj2) / range2;
            let distance = (norm1 * norm1 + norm2 * norm2).sqrt();

            if distance < best_distance {
                best_distance = distance;
                best_index = idx;
            }
        }

        Some(best_index)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeoffAnalysis {
    pub objective1: String,
    pub objective2: String,
    pub correlation: f64,
    pub points: Vec<TradeoffPoint>,
    pub best_compromise: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeoffPoint {
    pub config_index: usize,
    pub objective1_value: f64,
    pub objective2_value: f64,
    pub metadata: HashMap<String, String>,
}

impl ParetoFront {
    pub fn print_summary(&self) {
        println!("\n🎯 Pareto Front Analysis");
        println!("========================");
        println!("Pareto-optimal configurations: {}", self.points.len());
        println!("Dominated configurations: {}", self.dominated_points.len());
        println!("Objectives: {}", self.objective_names.join(", "));

        if !self.points.is_empty() {
            println!("\n🏆 Pareto-Optimal Configurations:");
            for (idx, point) in self.points.iter().enumerate() {
                println!("  Configuration {}:", idx + 1);
                println!("    Vector lanes: {}", point.metadata.get("vector_lanes").unwrap_or(&"?".to_string()));
                println!("    Bus count: {}", point.metadata.get("bus_count").unwrap_or(&"?".to_string()));
                println!("    Memory banks: {}", point.metadata.get("memory_banks").unwrap_or(&"?".to_string()));
                println!("    Objectives: {:?}", point.objectives);
                println!();
            }
        }
    }

    pub fn export_csv(&self, path: &str) -> Result<(), String> {
        let mut csv_content = String::new();

        // Header
        csv_content.push_str("config_index,vector_lanes,bus_count,memory_banks,issue_width");
        for obj_name in &self.objective_names {
            csv_content.push_str(&format!(",{}", obj_name));
        }
        csv_content.push('\n');

        // Data rows
        for point in &self.points {
            csv_content.push_str(&format!("{}", point.configuration_index));
            csv_content.push_str(&format!(",{}", point.metadata.get("vector_lanes").unwrap_or(&"?".to_string())));
            csv_content.push_str(&format!(",{}", point.metadata.get("bus_count").unwrap_or(&"?".to_string())));
            csv_content.push_str(&format!(",{}", point.metadata.get("memory_banks").unwrap_or(&"?".to_string())));
            csv_content.push_str(&format!(",{}", point.metadata.get("issue_width").unwrap_or(&"?".to_string())));

            for objective in &point.objectives {
                csv_content.push_str(&format!(",{:.6}", objective));
            }
            csv_content.push('\n');
        }

        std::fs::write(path, csv_content)
            .map_err(|e| format!("Failed to write CSV: {}", e))?;

        println!("📊 Pareto front exported to: {}", path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_analyzer_creation() {
        let analyzer = ParetoAnalyzer::default_tta_objectives();
        assert_eq!(analyzer.objectives.len(), 4);
    }

    #[test]
    fn test_dominance_check() {
        let analyzer = ParetoAnalyzer::new(vec![
            OptimizationObjective::Minimize("obj1".to_string()),
            OptimizationObjective::Minimize("obj2".to_string()),
        ]);

        // Point A dominates point B if all objectives are better or equal, with at least one strictly better
        assert!(analyzer.dominates(&[1.0, 2.0], &[2.0, 3.0])); // A dominates B
        assert!(!analyzer.dominates(&[2.0, 3.0], &[1.0, 2.0])); // B does not dominate A
        assert!(!analyzer.dominates(&[1.0, 3.0], &[2.0, 2.0])); // No dominance (trade-off)
    }

    #[test]
    fn test_pareto_point_creation() {
        let point = ParetoPoint {
            configuration_index: 0,
            objectives: vec![1.0, 2.0],
            labels: vec!["obj1".to_string(), "obj2".to_string()],
            metadata: HashMap::new(),
        };

        assert_eq!(point.configuration_index, 0);
        assert_eq!(point.objectives.len(), 2);
    }
}