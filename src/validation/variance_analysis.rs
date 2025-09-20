// src/validation/variance_analysis.rs
//! Energy variance analysis for TTA kernel validation
//!
//! Provides detailed analysis of energy consumption variance
//! to ensure TTA implementations meet the <5% requirement.

use crate::validation::golden_reference::ValidationResult;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyVariance {
    pub test_name: String,
    pub expected_energy: f64,
    pub actual_energy: f64,
    pub absolute_variance: f64,
    pub relative_variance: f64,
    pub meets_requirement: bool,
    pub energy_category: EnergyCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnergyCategory {
    VeryLow,    // < 50 units
    Low,        // 50-100 units
    Medium,     // 100-200 units
    High,       // 200-500 units
    VeryHigh,   // > 500 units
}

impl EnergyCategory {
    pub fn from_energy(energy: f64) -> Self {
        if energy < 50.0 {
            EnergyCategory::VeryLow
        } else if energy < 100.0 {
            EnergyCategory::Low
        } else if energy < 200.0 {
            EnergyCategory::Medium
        } else if energy < 500.0 {
            EnergyCategory::High
        } else {
            EnergyCategory::VeryHigh
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceReport {
    pub total_tests: usize,
    pub tests_meeting_requirement: usize,
    pub average_variance: f64,
    pub maximum_variance: f64,
    pub minimum_variance: f64,
    pub variance_by_category: HashMap<String, CategoryVariance>,
    pub problematic_tests: Vec<EnergyVariance>,
    pub statistical_summary: StatisticalSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryVariance {
    pub category: String,
    pub test_count: usize,
    pub average_variance: f64,
    pub tests_passing: usize,
    pub worst_variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub mean: f64,
    pub median: f64,
    pub std_deviation: f64,
    pub q1: f64,
    pub q3: f64,
    pub outliers: Vec<String>,
}

pub struct VarianceAnalyzer {
    pub variance_threshold: f64,
}

impl VarianceAnalyzer {
    pub fn new() -> Self {
        Self {
            variance_threshold: 5.0, // 5% requirement for A8
        }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            variance_threshold: threshold,
        }
    }

    pub fn analyze_results(&self, results: &[ValidationResult]) -> VarianceReport {
        println!("🔍 Analyzing energy variance for {} tests", results.len());

        let mut energy_variances = Vec::new();
        let mut variance_by_category = HashMap::new();

        // Extract energy variance data
        for result in results {
            let variance = EnergyVariance {
                test_name: result.test_name.clone(),
                expected_energy: self.estimate_expected_energy(&result.test_name),
                actual_energy: result.actual_energy,
                absolute_variance: (result.actual_energy - self.estimate_expected_energy(&result.test_name)).abs(),
                relative_variance: result.energy_variance,
                meets_requirement: result.energy_variance < self.variance_threshold,
                energy_category: EnergyCategory::from_energy(result.actual_energy),
            };

            energy_variances.push(variance);
        }

        // Group by energy category
        for variance in &energy_variances {
            let category_name = format!("{:?}", variance.energy_category);
            let entry = variance_by_category.entry(category_name.clone()).or_insert(CategoryVariance {
                category: category_name,
                test_count: 0,
                average_variance: 0.0,
                tests_passing: 0,
                worst_variance: 0.0,
            });

            entry.test_count += 1;
            entry.average_variance += variance.relative_variance;
            if variance.meets_requirement {
                entry.tests_passing += 1;
            }
            entry.worst_variance = entry.worst_variance.max(variance.relative_variance);
        }

        // Finalize category averages
        for category in variance_by_category.values_mut() {
            if category.test_count > 0 {
                category.average_variance /= category.test_count as f64;
            }
        }

        // Calculate overall statistics
        let variances: Vec<f64> = energy_variances.iter().map(|v| v.relative_variance).collect();
        let statistical_summary = self.calculate_statistics(&variances, &energy_variances);

        let tests_meeting_requirement = energy_variances.iter().filter(|v| v.meets_requirement).count();
        let average_variance = variances.iter().sum::<f64>() / variances.len() as f64;
        let maximum_variance = variances.iter().fold(0.0f64, |a, &b| a.max(b));
        let minimum_variance = variances.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let problematic_tests = energy_variances.iter()
            .filter(|v| !v.meets_requirement)
            .cloned()
            .collect();

        VarianceReport {
            total_tests: energy_variances.len(),
            tests_meeting_requirement,
            average_variance,
            maximum_variance,
            minimum_variance,
            variance_by_category,
            problematic_tests,
            statistical_summary,
        }
    }

    fn estimate_expected_energy(&self, test_name: &str) -> f64 {
        // Hardcoded expected energies based on golden reference
        match test_name {
            "dot_product_16" => 120.0,
            "vector_add_8" => 45.0,
            "vector_sum_8" => 25.0,
            "vector_max_8" => 28.0,
            "vector_argmax_8" => 30.0,
            "matrix_vector_4x4" => 180.0,
            "convolution_3x3" => 250.0,
            "fir_filter_4tap" => 150.0,
            _ => 100.0, // Default estimate
        }
    }

    fn calculate_statistics(&self, variances: &[f64], energy_variances: &[EnergyVariance]) -> StatisticalSummary {
        if variances.is_empty() {
            return StatisticalSummary {
                mean: 0.0,
                median: 0.0,
                std_deviation: 0.0,
                q1: 0.0,
                q3: 0.0,
                outliers: Vec::new(),
            };
        }

        let mut sorted_variances = variances.to_vec();
        sorted_variances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = variances.iter().sum::<f64>() / variances.len() as f64;

        let median = if sorted_variances.len() % 2 == 0 {
            (sorted_variances[sorted_variances.len() / 2 - 1] + sorted_variances[sorted_variances.len() / 2]) / 2.0
        } else {
            sorted_variances[sorted_variances.len() / 2]
        };

        let variance = variances.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / variances.len() as f64;
        let std_deviation = variance.sqrt();

        let q1_idx = sorted_variances.len() / 4;
        let q3_idx = 3 * sorted_variances.len() / 4;
        let q1 = sorted_variances[q1_idx];
        let q3 = sorted_variances[q3_idx.min(sorted_variances.len() - 1)];

        // Identify outliers (variance > Q3 + 1.5 * IQR)
        let iqr = q3 - q1;
        let outlier_threshold = q3 + 1.5 * iqr;
        let outliers = energy_variances.iter()
            .filter(|v| v.relative_variance > outlier_threshold)
            .map(|v| v.test_name.clone())
            .collect();

        StatisticalSummary {
            mean,
            median,
            std_deviation,
            q1,
            q3,
            outliers,
        }
    }

    pub fn print_detailed_report(&self, report: &VarianceReport) {
        println!("\n📊 Energy Variance Analysis Report");
        println!("===================================");

        println!("\n🎯 A8 Requirement Status: {}",
                 if report.tests_meeting_requirement == report.total_tests {
                     "✅ ALL TESTS PASS (<5% variance)"
                 } else {
                     "❌ SOME TESTS FAIL (≥5% variance)"
                 });

        println!("\n📈 Overall Statistics:");
        println!("  Total tests: {}", report.total_tests);
        println!("  Tests meeting <5% requirement: {} ({:.1}%)",
                 report.tests_meeting_requirement,
                 report.tests_meeting_requirement as f64 / report.total_tests as f64 * 100.0);
        println!("  Average variance: {:.2}%", report.average_variance);
        println!("  Maximum variance: {:.2}%", report.maximum_variance);
        println!("  Minimum variance: {:.2}%", report.minimum_variance);

        println!("\n📊 Statistical Summary:");
        println!("  Mean: {:.2}%", report.statistical_summary.mean);
        println!("  Median: {:.2}%", report.statistical_summary.median);
        println!("  Standard deviation: {:.2}%", report.statistical_summary.std_deviation);
        println!("  Q1: {:.2}%", report.statistical_summary.q1);
        println!("  Q3: {:.2}%", report.statistical_summary.q3);

        if !report.statistical_summary.outliers.is_empty() {
            println!("  Outliers: {:?}", report.statistical_summary.outliers);
        }

        println!("\n🏷️  Variance by Energy Category:");
        for (category, stats) in &report.variance_by_category {
            println!("  {}: {} tests, {:.1}% avg variance, {}/{} passing, {:.1}% worst",
                     category, stats.test_count, stats.average_variance,
                     stats.tests_passing, stats.test_count, stats.worst_variance);
        }

        if !report.problematic_tests.is_empty() {
            println!("\n⚠️  Problematic Tests (≥{}% variance):", self.variance_threshold);
            for test in &report.problematic_tests {
                println!("  {}: {:.1}% variance (expected: {:.1}, actual: {:.1})",
                         test.test_name, test.relative_variance,
                         test.expected_energy, test.actual_energy);
            }

            println!("\n💡 Recommendations for problematic tests:");
            self.provide_recommendations(&report.problematic_tests);
        }
    }

    fn provide_recommendations(&self, problematic_tests: &[EnergyVariance]) {
        for test in problematic_tests {
            println!("  📝 {}: ", test.test_name);

            if test.actual_energy > test.expected_energy {
                println!("    - Energy consumption too high. Check for:");
                println!("      * Inefficient instruction sequences");
                println!("      * Excessive bus utilization");
                println!("      * Suboptimal functional unit usage");
            } else {
                println!("    - Energy consumption too low. Check for:");
                println!("      * Missing energy accounting");
                println!("      * Underestimated operation costs");
                println!("      * Incomplete energy model");
            }

            match test.energy_category {
                EnergyCategory::VeryHigh => {
                    println!("    - High energy test: Focus on major algorithmic optimizations");
                },
                EnergyCategory::VeryLow => {
                    println!("    - Low energy test: Verify energy model completeness");
                },
                _ => {
                    println!("    - Medium energy test: Fine-tune energy parameters");
                }
            }
        }
    }

    pub fn export_report(&self, report: &VarianceReport, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        std::fs::write(path, json)
            .map_err(|e| format!("File write failed: {}", e))?;

        println!("📁 Variance analysis report exported to: {}", path);
        Ok(())
    }

    pub fn generate_variance_plot(&self, report: &VarianceReport) -> String {
        let mut plot = String::new();
        plot.push_str("\n📊 Energy Variance Distribution\n");
        plot.push_str("================================\n");

        // Create histogram of variance ranges
        let mut histogram = HashMap::new();
        for test in &report.problematic_tests {
            let range = if test.relative_variance < 10.0 {
                "5-10%"
            } else if test.relative_variance < 20.0 {
                "10-20%"
            } else {
                ">20%"
            };
            *histogram.entry(range).or_insert(0) += 1;
        }

        plot.push_str("Variance Distribution (problematic tests only):\n");
        for (range, count) in &histogram {
            let bar = "█".repeat(*count);
            plot.push_str(&format!("{:>6}: {} ({})\n", range, bar, count));
        }

        plot.push_str("\nLegend: Each █ represents one test\n");
        plot
    }
}

impl Default for VarianceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::golden_reference::ValidationResult;

    #[test]
    fn test_variance_analyzer_creation() {
        let analyzer = VarianceAnalyzer::new();
        assert_eq!(analyzer.variance_threshold, 5.0);

        let custom_analyzer = VarianceAnalyzer::with_threshold(3.0);
        assert_eq!(custom_analyzer.variance_threshold, 3.0);
    }

    #[test]
    fn test_energy_category_classification() {
        assert!(matches!(EnergyCategory::from_energy(30.0), EnergyCategory::VeryLow));
        assert!(matches!(EnergyCategory::from_energy(75.0), EnergyCategory::Low));
        assert!(matches!(EnergyCategory::from_energy(150.0), EnergyCategory::Medium));
        assert!(matches!(EnergyCategory::from_energy(300.0), EnergyCategory::High));
        assert!(matches!(EnergyCategory::from_energy(600.0), EnergyCategory::VeryHigh));
    }

    #[test]
    fn test_variance_analysis() {
        let analyzer = VarianceAnalyzer::new();

        let test_results = vec![
            ValidationResult::success("test1".to_string(), vec![1], 100.0, 5, 95.0, 5),
            ValidationResult::success("test2".to_string(), vec![2], 110.0, 6, 100.0, 6),
        ];

        let report = analyzer.analyze_results(&test_results);

        assert_eq!(report.total_tests, 2);
        assert!(report.average_variance > 0.0);
        assert!(!report.variance_by_category.is_empty());
    }

    #[test]
    fn test_statistical_calculations() {
        let analyzer = VarianceAnalyzer::new();
        let variances = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let energy_variances = vec![
            EnergyVariance {
                test_name: "test1".to_string(),
                expected_energy: 100.0,
                actual_energy: 101.0,
                absolute_variance: 1.0,
                relative_variance: 1.0,
                meets_requirement: true,
                energy_category: EnergyCategory::Medium,
            }
        ];

        let stats = analyzer.calculate_statistics(&variances, &energy_variances);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert!(stats.std_deviation > 0.0);
    }
}