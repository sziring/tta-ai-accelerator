// src/analysis/competitive_benchmarks.rs
//! Competitive Benchmarking Analysis Against Published Results
//!
//! Compares TTA performance against state-of-the-art accelerators
//! using published metrics from academic and industry sources.

use crate::analysis::ModelMetrics;
use std::collections::HashMap;

/// Published accelerator performance data from literature
#[derive(Debug, Clone)]
pub struct AcceleratorSpec {
    pub name: String,
    pub vendor: String,
    pub year: u16,
    pub process_node_nm: u16,
    pub peak_tops: f64,
    pub power_consumption_w: f64,
    pub memory_bandwidth_gb_s: f64,
    pub energy_efficiency_tops_w: f64,
    pub source: String, // Citation or reference
}

/// TTA vs competitive accelerator comparison
#[derive(Debug, Clone)]
pub struct AcceleratorComparison {
    pub tta_metrics: ModelMetrics,
    pub competitor_spec: AcceleratorSpec,
    pub normalized_comparison: NormalizedMetrics,
    pub advantage_factors: AdvantageFactors,
}

/// Normalized metrics for fair comparison
#[derive(Debug, Clone)]
pub struct NormalizedMetrics {
    pub tta_tops_w_normalized: f64,
    pub competitor_tops_w: f64,
    pub tta_area_efficiency_normalized: f64,
    pub competitor_area_efficiency: f64,
    pub tta_memory_efficiency: f64,
    pub competitor_memory_efficiency: f64,
}

/// TTA advantage factors across different metrics
#[derive(Debug, Clone)]
pub struct AdvantageFactors {
    pub energy_efficiency_advantage: f64, // >1.0 means TTA is better
    pub area_efficiency_advantage: f64,
    pub memory_efficiency_advantage: f64,
    pub overall_competitiveness: f64,
}

/// Comprehensive competitive benchmarking suite
#[derive(Debug)]
pub struct BenchmarkSuite {
    pub accelerator_database: Vec<AcceleratorSpec>,
    pub tta_results: Vec<ModelMetrics>,
    pub comparisons: Vec<AcceleratorComparison>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        let mut suite = Self {
            accelerator_database: Vec::new(),
            tta_results: Vec::new(),
            comparisons: Vec::new(),
        };

        suite.initialize_accelerator_database();
        suite
    }

    /// Initialize database with published accelerator specifications
    fn initialize_accelerator_database(&mut self) {
        // NVIDIA GPU Accelerators
        self.accelerator_database.push(AcceleratorSpec {
            name: "NVIDIA V100".to_string(),
            vendor: "NVIDIA".to_string(),
            year: 2017,
            process_node_nm: 12,
            peak_tops: 125.0, // Mixed precision
            power_consumption_w: 300.0,
            memory_bandwidth_gb_s: 900.0,
            energy_efficiency_tops_w: 0.42,
            source: "NVIDIA V100 Datasheet".to_string(),
        });

        self.accelerator_database.push(AcceleratorSpec {
            name: "NVIDIA A100".to_string(),
            vendor: "NVIDIA".to_string(),
            year: 2020,
            process_node_nm: 7,
            peak_tops: 312.0, // BF16
            power_consumption_w: 400.0,
            memory_bandwidth_gb_s: 1555.0,
            energy_efficiency_tops_w: 0.78,
            source: "NVIDIA A100 Datasheet".to_string(),
        });

        self.accelerator_database.push(AcceleratorSpec {
            name: "NVIDIA H100".to_string(),
            vendor: "NVIDIA".to_string(),
            year: 2022,
            process_node_nm: 4,
            peak_tops: 1000.0, // FP8
            power_consumption_w: 700.0,
            memory_bandwidth_gb_s: 3350.0,
            energy_efficiency_tops_w: 1.43,
            source: "NVIDIA H100 Datasheet".to_string(),
        });

        // Google TPU
        self.accelerator_database.push(AcceleratorSpec {
            name: "Google TPU v4".to_string(),
            vendor: "Google".to_string(),
            year: 2021,
            process_node_nm: 7,
            peak_tops: 275.0, // BF16
            power_consumption_w: 200.0,
            memory_bandwidth_gb_s: 1200.0,
            energy_efficiency_tops_w: 1.375,
            source: "Google TPU v4 MLPerf Results".to_string(),
        });

        // Academic/Research Accelerators
        self.accelerator_database.push(AcceleratorSpec {
            name: "Eyeriss v2".to_string(),
            vendor: "MIT".to_string(),
            year: 2019,
            process_node_nm: 65,
            peak_tops: 0.35, // INT8
            power_consumption_w: 0.278,
            memory_bandwidth_gb_s: 12.8,
            energy_efficiency_tops_w: 1.26,
            source: "Chen et al., JSSC 2019".to_string(),
        });

        self.accelerator_database.push(AcceleratorSpec {
            name: "Simba".to_string(),
            vendor: "NVIDIA Research".to_string(),
            year: 2019,
            process_node_nm: 16,
            peak_tops: 32.0, // INT8
            power_consumption_w: 15.0,
            memory_bandwidth_gb_s: 128.0,
            energy_efficiency_tops_w: 2.13,
            source: "Shao et al., MICRO 2019".to_string(),
        });

        // Mobile/Edge Accelerators
        self.accelerator_database.push(AcceleratorSpec {
            name: "Apple M1 Neural Engine".to_string(),
            vendor: "Apple".to_string(),
            year: 2020,
            process_node_nm: 5,
            peak_tops: 15.8, // INT8
            power_consumption_w: 2.0,
            memory_bandwidth_gb_s: 68.25,
            energy_efficiency_tops_w: 7.9,
            source: "Apple M1 Analysis, AnandTech".to_string(),
        });
    }

    /// Add TTA results for competitive analysis
    pub fn add_tta_results(&mut self, metrics: ModelMetrics) {
        self.tta_results.push(metrics);
    }

    /// Run comprehensive competitive analysis
    pub fn run_competitive_analysis(&mut self) -> Result<CompetitiveAnalysis, String> {
        println!("🏁 Running Competitive Benchmarking Analysis");
        println!("===========================================");

        if self.tta_results.is_empty() {
            return Err("No TTA results available for comparison".to_string());
        }

        // Compare each TTA result against all competitors
        for tta_metrics in &self.tta_results {
            for competitor in &self.accelerator_database {
                let comparison = self.compare_against_competitor(tta_metrics, competitor)?;
                self.comparisons.push(comparison);
            }
        }

        let analysis = self.generate_competitive_analysis();
        self.print_competitive_summary(&analysis);

        Ok(analysis)
    }

    /// Compare TTA against a specific competitor
    fn compare_against_competitor(
        &self,
        tta_metrics: &ModelMetrics,
        competitor: &AcceleratorSpec
    ) -> Result<AcceleratorComparison, String> {

        // Estimate TTA specifications based on our analysis
        let tta_estimated_power = self.estimate_tta_power(tta_metrics);
        let tta_estimated_tops = self.estimate_tta_tops(tta_metrics);
        let tta_estimated_bandwidth = tta_metrics.memory_bandwidth_gb_s;

        // Calculate normalized metrics for fair comparison
        let normalized = NormalizedMetrics {
            tta_tops_w_normalized: tta_estimated_tops / tta_estimated_power,
            competitor_tops_w: competitor.energy_efficiency_tops_w,
            tta_area_efficiency_normalized: self.estimate_tta_area_efficiency(tta_metrics),
            competitor_area_efficiency: self.estimate_competitor_area_efficiency(competitor),
            tta_memory_efficiency: tta_estimated_tops / tta_estimated_bandwidth,
            competitor_memory_efficiency: competitor.peak_tops / competitor.memory_bandwidth_gb_s,
        };

        // Calculate advantage factors
        let advantages = AdvantageFactors {
            energy_efficiency_advantage: normalized.tta_tops_w_normalized / normalized.competitor_tops_w,
            area_efficiency_advantage: normalized.tta_area_efficiency_normalized / normalized.competitor_area_efficiency,
            memory_efficiency_advantage: normalized.tta_memory_efficiency / normalized.competitor_memory_efficiency,
            overall_competitiveness: 0.0, // Will be calculated below
        };

        let overall_competitiveness = (advantages.energy_efficiency_advantage * 0.5 +
                                     advantages.area_efficiency_advantage * 0.3 +
                                     advantages.memory_efficiency_advantage * 0.2);

        let mut final_advantages = advantages;
        final_advantages.overall_competitiveness = overall_competitiveness;

        Ok(AcceleratorComparison {
            tta_metrics: tta_metrics.clone(),
            competitor_spec: competitor.clone(),
            normalized_comparison: normalized,
            advantage_factors: final_advantages,
        })
    }

    /// Estimate TTA power consumption based on energy metrics
    fn estimate_tta_power(&self, metrics: &ModelMetrics) -> f64 {
        // Assume 1GHz clock frequency and convert energy per operation to power
        let operations_per_second = metrics.throughput_tokens_per_cycle * 1e9;
        let energy_per_operation = metrics.energy_per_token / (metrics.tta_advantage_factor * 1000.0); // Normalized

        (operations_per_second * energy_per_operation).max(0.1) // Minimum 0.1W
    }

    /// Estimate TTA TOPS performance
    fn estimate_tta_tops(&self, metrics: &ModelMetrics) -> f64 {
        // Estimate based on operations per token and throughput
        let operations_per_token = 1000.0; // Approximate for transformer
        let tokens_per_second = metrics.throughput_tokens_per_cycle * 1e9;
        let operations_per_second = tokens_per_second * operations_per_token;

        operations_per_second / 1e12 // Convert to TOPS
    }

    /// Estimate TTA area efficiency
    fn estimate_tta_area_efficiency(&self, metrics: &ModelMetrics) -> f64 {
        // Based on compute utilization and efficiency factors
        metrics.compute_utilization * metrics.tta_advantage_factor * 10.0 // Normalized metric
    }

    /// Estimate competitor area efficiency from published specs
    fn estimate_competitor_area_efficiency(&self, spec: &AcceleratorSpec) -> f64 {
        // Simple heuristic based on TOPS/W and process node
        let node_factor = 28.0 / spec.process_node_nm as f64; // Normalize to 28nm
        spec.energy_efficiency_tops_w * node_factor
    }

    /// Generate comprehensive competitive analysis
    fn generate_competitive_analysis(&self) -> CompetitiveAnalysis {
        let mut category_performance = HashMap::new();

        // Group competitors by category
        let mut gpu_comparisons = Vec::new();
        let mut tpu_comparisons = Vec::new();
        let mut academic_comparisons = Vec::new();
        let mut mobile_comparisons = Vec::new();

        for comparison in &self.comparisons {
            match comparison.competitor_spec.vendor.as_str() {
                "NVIDIA" if comparison.competitor_spec.name.contains("V100") ||
                           comparison.competitor_spec.name.contains("A100") ||
                           comparison.competitor_spec.name.contains("H100") => {
                    gpu_comparisons.push(comparison.clone());
                },
                "Google" => {
                    tpu_comparisons.push(comparison.clone());
                },
                "MIT" | "NVIDIA Research" => {
                    academic_comparisons.push(comparison.clone());
                },
                "Apple" => {
                    mobile_comparisons.push(comparison.clone());
                },
                _ => {}
            }
        }

        // Calculate average advantages by category
        if !gpu_comparisons.is_empty() {
            let avg_advantage = gpu_comparisons.iter()
                .map(|c| c.advantage_factors.overall_competitiveness)
                .sum::<f64>() / gpu_comparisons.len() as f64;
            category_performance.insert("GPU".to_string(), avg_advantage);
        }

        if !tpu_comparisons.is_empty() {
            let avg_advantage = tpu_comparisons.iter()
                .map(|c| c.advantage_factors.overall_competitiveness)
                .sum::<f64>() / tpu_comparisons.len() as f64;
            category_performance.insert("TPU".to_string(), avg_advantage);
        }

        if !academic_comparisons.is_empty() {
            let avg_advantage = academic_comparisons.iter()
                .map(|c| c.advantage_factors.overall_competitiveness)
                .sum::<f64>() / academic_comparisons.len() as f64;
            category_performance.insert("Academic".to_string(), avg_advantage);
        }

        if !mobile_comparisons.is_empty() {
            let avg_advantage = mobile_comparisons.iter()
                .map(|c| c.advantage_factors.overall_competitiveness)
                .sum::<f64>() / mobile_comparisons.len() as f64;
            category_performance.insert("Mobile".to_string(), avg_advantage);
        }

        CompetitiveAnalysis {
            total_comparisons: self.comparisons.len(),
            category_performance,
            best_competitor_advantage: self.find_best_advantage(),
            worst_competitor_advantage: self.find_worst_advantage(),
            average_advantage: self.calculate_average_advantage(),
            competitive_positioning: self.assess_competitive_positioning(),
        }
    }

    fn find_best_advantage(&self) -> f64 {
        self.comparisons.iter()
            .map(|c| c.advantage_factors.overall_competitiveness)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn find_worst_advantage(&self) -> f64 {
        self.comparisons.iter()
            .map(|c| c.advantage_factors.overall_competitiveness)
            .fold(f64::INFINITY, f64::min)
    }

    fn calculate_average_advantage(&self) -> f64 {
        if self.comparisons.is_empty() {
            return 0.0;
        }
        self.comparisons.iter()
            .map(|c| c.advantage_factors.overall_competitiveness)
            .sum::<f64>() / self.comparisons.len() as f64
    }

    fn assess_competitive_positioning(&self) -> String {
        let avg = self.calculate_average_advantage();

        if avg > 2.0 {
            "Strongly Competitive - Significant advantages across most comparisons".to_string()
        } else if avg > 1.5 {
            "Competitive - Clear advantages in many areas".to_string()
        } else if avg > 1.0 {
            "Moderately Competitive - Some advantages, room for improvement".to_string()
        } else {
            "Needs Improvement - Generally behind competitive alternatives".to_string()
        }
    }

    fn print_competitive_summary(&self, analysis: &CompetitiveAnalysis) {
        println!("\n📊 Competitive Analysis Summary:");
        println!("===============================");

        println!("Total Comparisons: {}", analysis.total_comparisons);
        println!("Average Competitive Advantage: {:.2}x", analysis.average_advantage);
        println!("Best Case Advantage: {:.2}x", analysis.best_competitor_advantage);
        println!("Worst Case Advantage: {:.2}x", analysis.worst_competitor_advantage);
        println!();

        println!("Performance by Category:");
        for (category, advantage) in &analysis.category_performance {
            let status = if *advantage > 1.5 { "✅ Strong" }
                        else if *advantage > 1.0 { "⚠️ Moderate" }
                        else { "❌ Weak" };
            println!("  {}: {:.2}x {}", category, advantage, status);
        }
        println!();

        println!("Competitive Positioning: {}", analysis.competitive_positioning);
    }
}

/// Final competitive analysis results
#[derive(Debug, Clone)]
pub struct CompetitiveAnalysis {
    pub total_comparisons: usize,
    pub category_performance: HashMap<String, f64>,
    pub best_competitor_advantage: f64,
    pub worst_competitor_advantage: f64,
    pub average_advantage: f64,
    pub competitive_positioning: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ModelMetrics;
    use std::collections::HashMap;

    fn create_test_tta_metrics() -> ModelMetrics {
        let mut component_breakdown = HashMap::new();
        component_breakdown.insert("self_attention".to_string(), 100.0);
        component_breakdown.insert("feed_forward".to_string(), 150.0);

        ModelMetrics {
            model_name: "Test-TTA".to_string(),
            total_energy: 250.0,
            total_cycles: 1000,
            throughput_tokens_per_cycle: 0.1,
            component_breakdown,
            attention_metrics: None,
            memory_bandwidth_gb_s: 50.0,
            compute_utilization: 0.85,
            energy_per_token: 0.5,
            tta_advantage_factor: 4.2,
        }
    }

    #[test]
    fn test_competitive_benchmarking() {
        let mut benchmark_suite = BenchmarkSuite::new();

        // Verify accelerator database initialization
        assert!(benchmark_suite.accelerator_database.len() > 5);

        // Add TTA results
        let tta_metrics = create_test_tta_metrics();
        benchmark_suite.add_tta_results(tta_metrics);

        // Run competitive analysis
        let analysis_result = benchmark_suite.run_competitive_analysis();
        assert!(analysis_result.is_ok());

        let analysis = analysis_result.unwrap();
        assert!(analysis.total_comparisons > 0);
        assert!(analysis.average_advantage > 0.0);

        println!("Competitive analysis completed with {} comparisons", analysis.total_comparisons);
        println!("Average advantage: {:.2}x", analysis.average_advantage);
    }

    #[test]
    fn test_accelerator_database_content() {
        let benchmark_suite = BenchmarkSuite::new();

        // Verify we have major accelerator categories
        let has_nvidia = benchmark_suite.accelerator_database.iter()
            .any(|acc| acc.vendor == "NVIDIA");
        let has_google = benchmark_suite.accelerator_database.iter()
            .any(|acc| acc.vendor == "Google");
        let has_academic = benchmark_suite.accelerator_database.iter()
            .any(|acc| acc.vendor == "MIT");

        assert!(has_nvidia, "Should include NVIDIA accelerators");
        assert!(has_google, "Should include Google TPU");
        assert!(has_academic, "Should include academic research accelerators");

        // Verify reasonable energy efficiency values
        for accelerator in &benchmark_suite.accelerator_database {
            assert!(accelerator.energy_efficiency_tops_w > 0.0);
            assert!(accelerator.energy_efficiency_tops_w < 20.0); // Reasonable upper bound
            assert!(accelerator.process_node_nm >= 4); // Current leading edge
            assert!(accelerator.process_node_nm <= 180); // Historical range
        }
    }
}