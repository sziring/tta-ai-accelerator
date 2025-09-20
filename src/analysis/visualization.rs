// src/analysis/visualization.rs
//! Visualization tools for parameter sweep and Pareto analysis results
//!
//! Provides ASCII-based plotting and data export for visualization tools

use crate::analysis::parameter_sweep::SweepResult;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub width: usize,
    pub height: usize,
    pub show_pareto_front: bool,
    pub show_dominated_points: bool,
    pub export_format: ExportFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Csv,
    Json,
    Gnuplot,
    Python,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Scatter2D,
    Pareto3D,
    Heatmap,
    BoxPlot,
    Correlation,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 24,
            show_pareto_front: true,
            show_dominated_points: true,
            export_format: ExportFormat::Csv,
        }
    }
}

pub struct PlotGenerator {
    config: VisualizationConfig,
}

impl PlotGenerator {
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate ASCII scatter plot for 2D data
    pub fn scatter_plot_2d(&self, data: &[(f64, f64, String)], x_label: &str, y_label: &str) -> String {
        if data.is_empty() {
            return "No data to plot".to_string();
        }

        let mut plot = String::new();

        // Find data bounds
        let min_x = data.iter().map(|(x, _, _)| *x).fold(f64::INFINITY, f64::min);
        let max_x = data.iter().map(|(x, _, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let min_y = data.iter().map(|(_, y, _)| *y).fold(f64::INFINITY, f64::min);
        let max_y = data.iter().map(|(_, y, _)| *y).fold(f64::NEG_INFINITY, f64::max);

        let x_range = max_x - min_x;
        let y_range = max_y - min_y;

        if x_range.abs() < 1e-10 || y_range.abs() < 1e-10 {
            return "Data range too small for plotting".to_string();
        }

        // Create plot grid
        let mut grid = vec![vec![' '; self.config.width]; self.config.height];

        // Plot points
        for (x, y, label) in data {
            let plot_x = ((x - min_x) / x_range * (self.config.width - 1) as f64) as usize;
            let plot_y = self.config.height - 1 - ((y - min_y) / y_range * (self.config.height - 1) as f64) as usize;

            if plot_x < self.config.width && plot_y < self.config.height {
                let symbol = if label.contains("pareto") { '*' } else { '•' };
                grid[plot_y][plot_x] = symbol;
            }
        }

        // Add axes
        for y in 0..self.config.height {
            grid[y][0] = '|';
        }
        for x in 0..self.config.width {
            grid[self.config.height - 1][x] = '-';
        }
        grid[self.config.height - 1][0] = '+';

        // Convert grid to string
        plot.push_str(&format!("\n{} vs {}\n", y_label, x_label));
        plot.push_str(&"=".repeat(self.config.width));
        plot.push('\n');

        for row in &grid {
            plot.push_str(&row.iter().collect::<String>());
            plot.push('\n');
        }

        // Add axis labels
        plot.push_str(&format!("{:<width$}\n", x_label, width = self.config.width));
        plot.push_str(&format!("{:.2e} {:<width$} {:.2e}\n",
                              min_x, "", max_x, width = self.config.width - 20));

        plot.push_str(&format!("Y: {} | Range: {:.2e} to {:.2e}\n", y_label, min_y, max_y));
        plot.push_str("Legend: • = dominated, * = Pareto optimal\n");

        plot
    }

    /// Generate parameter heatmap showing EDP improvement across configurations
    pub fn parameter_heatmap(&self, results: &[SweepResult]) -> String {
        if results.is_empty() {
            return "No data for heatmap".to_string();
        }

        let mut heatmap = String::new();
        heatmap.push_str("\n🔥 EDP Improvement Heatmap (TTA vs RISC)\n");
        heatmap.push_str("==========================================\n");

        // Group by bus count and vector lanes
        let mut grid_data: HashMap<(u16, usize), f64> = HashMap::new();
        let mut bus_counts = std::collections::BTreeSet::new();
        let mut lane_counts = std::collections::BTreeSet::new();

        for result in results {
            let bus_count = result.configuration.bus_count;
            let lanes = result.configuration.vector_lanes;
            let edp_improvement = result.comparative_metrics.edp_improvement;

            grid_data.insert((bus_count, lanes), edp_improvement);
            bus_counts.insert(bus_count);
            lane_counts.insert(lanes);
        }

        // Create header
        heatmap.push_str("Lanes\\Buses ");
        for &bus_count in &bus_counts {
            heatmap.push_str(&format!("{:>8}", bus_count));
        }
        heatmap.push('\n');

        // Create grid
        for &lanes in &lane_counts {
            heatmap.push_str(&format!("{:>10} ", lanes));
            for &bus_count in &bus_counts {
                if let Some(&improvement) = grid_data.get(&(bus_count, lanes)) {
                    let symbol = self.improvement_to_symbol(improvement);
                    heatmap.push_str(&format!("{:>7.1}{}", improvement, symbol));
                } else {
                    heatmap.push_str("    -   ");
                }
            }
            heatmap.push('\n');
        }

        heatmap.push_str("\nLegend: ++ (>10%), + (>5%), ~ (>0%), - (<0%)\n");
        heatmap
    }

    fn improvement_to_symbol(&self, improvement: f64) -> char {
        if improvement > 10.0 { '+' }
        else if improvement > 5.0 { '+' }
        else if improvement > 0.0 { '~' }
        else { '-' }
    }

    /// Generate box plot showing distribution of metrics across configurations
    pub fn box_plot(&self, results: &[SweepResult], metric: &str) -> String {
        if results.is_empty() {
            return "No data for box plot".to_string();
        }

        let mut plot = String::new();
        plot.push_str(&format!("\n📦 {} Distribution Box Plot\n", metric));
        plot.push_str(&"=".repeat(40));
        plot.push('\n');

        // Extract values
        let mut values: Vec<f64> = results.iter()
            .filter_map(|r| self.extract_metric_value(r, metric))
            .collect();

        if values.is_empty() {
            return format!("No valid {} values found", metric);
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate statistics
        let min = values[0];
        let max = values[values.len() - 1];
        let median = if values.len() % 2 == 0 {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };

        let q1_idx = values.len() / 4;
        let q3_idx = 3 * values.len() / 4;
        let q1 = values[q1_idx];
        let q3 = values[q3_idx];

        let mean = values.iter().sum::<f64>() / values.len() as f64;

        // ASCII box plot
        plot.push_str(&format!("Min:    {:>10.2}\n", min));
        plot.push_str(&format!("Q1:     {:>10.2}\n", q1));
        plot.push_str(&format!("Median: {:>10.2}\n", median));
        plot.push_str(&format!("Mean:   {:>10.2}\n", mean));
        plot.push_str(&format!("Q3:     {:>10.2}\n", q3));
        plot.push_str(&format!("Max:    {:>10.2}\n", max));

        // Visual representation
        let range = max - min;
        if range > 1e-10 {
            let width = 30;
            let q1_pos = ((q1 - min) / range * width as f64) as usize;
            let median_pos = ((median - min) / range * width as f64) as usize;
            let q3_pos = ((q3 - min) / range * width as f64) as usize;

            plot.push('\n');
            plot.push_str("Visual: ");
            for i in 0..width {
                if i == q1_pos || i == q3_pos {
                    plot.push('|');
                } else if i == median_pos {
                    plot.push('█');
                } else if i > q1_pos && i < q3_pos {
                    plot.push('▬');
                } else {
                    plot.push('─');
                }
            }
            plot.push('\n');
        }

        plot.push_str(&format!("Count: {} configurations\n", values.len()));
        plot
    }

    fn extract_metric_value(&self, result: &SweepResult, metric: &str) -> Option<f64> {
        match metric {
            "edp_improvement" => Some(result.comparative_metrics.edp_improvement),
            "energy_ratio" => Some(result.comparative_metrics.energy_efficiency_ratio),
            "performance_ratio" => Some(result.comparative_metrics.performance_ratio),
            "tta_energy" => {
                let total: f64 = result.tta_results.values().map(|m| m.energy).sum();
                if result.tta_results.is_empty() { None } else { Some(total / result.tta_results.len() as f64) }
            },
            "tta_cycles" => {
                let total: u64 = result.tta_results.values().map(|m| m.cycles).sum();
                if result.tta_results.is_empty() { None } else { Some(total as f64 / result.tta_results.len() as f64) }
            },
            _ => None,
        }
    }

    /// Generate correlation matrix for multiple metrics
    pub fn correlation_matrix(&self, results: &[SweepResult], metrics: &[&str]) -> String {
        if results.is_empty() || metrics.is_empty() {
            return "No data for correlation analysis".to_string();
        }

        let mut matrix = String::new();
        matrix.push_str("\n📊 Correlation Matrix\n");
        matrix.push_str("====================\n");

        // Extract all metric values
        let mut data: Vec<Vec<f64>> = vec![Vec::new(); metrics.len()];
        for result in results {
            let mut valid_row = true;
            let mut row_values = Vec::new();

            for metric in metrics {
                if let Some(value) = self.extract_metric_value(result, metric) {
                    row_values.push(value);
                } else {
                    valid_row = false;
                    break;
                }
            }

            if valid_row {
                for (i, value) in row_values.into_iter().enumerate() {
                    data[i].push(value);
                }
            }
        }

        // Calculate correlation matrix
        matrix.push_str("        ");
        for metric in metrics {
            matrix.push_str(&format!("{:>8.8}", metric));
        }
        matrix.push('\n');

        for (i, metric1) in metrics.iter().enumerate() {
            matrix.push_str(&format!("{:>8.8}", metric1));
            for (j, _metric2) in metrics.iter().enumerate() {
                let correlation = if i == j {
                    1.0
                } else {
                    self.calculate_correlation(&data[i], &data[j])
                };
                matrix.push_str(&format!("{:>8.2}", correlation));
            }
            matrix.push('\n');
        }

        matrix.push_str("\nRange: -1.0 (negative) to +1.0 (positive correlation)\n");
        matrix
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xx: f64 = x.iter().map(|&val| val * val).sum();
        let sum_yy: f64 = y.iter().map(|&val| val * val).sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Export data for external visualization tools
    pub fn export_for_plotting(&self, results: &[SweepResult], path: &str) -> Result<(), String> {
        match self.config.export_format {
            ExportFormat::Csv => self.export_csv(results, path),
            ExportFormat::Json => self.export_json(results, path),
            ExportFormat::Gnuplot => self.export_gnuplot(results, path),
            ExportFormat::Python => self.export_python(results, path),
        }
    }

    fn export_csv(&self, results: &[SweepResult], path: &str) -> Result<(), String> {
        let mut csv = String::new();
        csv.push_str("vector_lanes,bus_count,memory_banks,issue_width,");
        csv.push_str("edp_improvement,energy_ratio,performance_ratio,area_efficiency\n");

        for result in results {
            csv.push_str(&format!("{},{},{},{},",
                result.configuration.vector_lanes,
                result.configuration.bus_count,
                result.configuration.memory_banks,
                result.configuration.issue_width,
            ));
            csv.push_str(&format!("{:.4},{:.4},{:.4},{:.4}\n",
                result.comparative_metrics.edp_improvement,
                result.comparative_metrics.energy_efficiency_ratio,
                result.comparative_metrics.performance_ratio,
                result.comparative_metrics.area_efficiency,
            ));
        }

        std::fs::write(path, csv)
            .map_err(|e| format!("Failed to write CSV: {}", e))?;

        println!("📊 Data exported to CSV: {}", path);
        Ok(())
    }

    fn export_json(&self, results: &[SweepResult], path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(results)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;

        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write JSON: {}", e))?;

        println!("📊 Data exported to JSON: {}", path);
        Ok(())
    }

    fn export_gnuplot(&self, results: &[SweepResult], path: &str) -> Result<(), String> {
        let mut gnuplot_script = String::new();
        gnuplot_script.push_str("# Gnuplot script for TTA parameter sweep visualization\n");
        gnuplot_script.push_str("set terminal pngcairo size 800,600\n");
        gnuplot_script.push_str("set output 'tta_pareto.png'\n");
        gnuplot_script.push_str("set xlabel 'Energy Efficiency Ratio (TTA/RISC)'\n");
        gnuplot_script.push_str("set ylabel 'EDP Improvement (%)'\n");
        gnuplot_script.push_str("set title 'TTA Parameter Sweep: Energy vs Performance'\n");
        gnuplot_script.push_str("plot '-' using 1:2 with points pointtype 7 title 'Configurations'\n");

        for result in results {
            gnuplot_script.push_str(&format!("{:.4} {:.4}\n",
                result.comparative_metrics.energy_efficiency_ratio,
                result.comparative_metrics.edp_improvement,
            ));
        }
        gnuplot_script.push_str("e\n");

        std::fs::write(path, gnuplot_script)
            .map_err(|e| format!("Failed to write Gnuplot script: {}", e))?;

        println!("📊 Gnuplot script exported: {}", path);
        Ok(())
    }

    fn export_python(&self, results: &[SweepResult], path: &str) -> Result<(), String> {
        let mut python_script = String::new();
        python_script.push_str("#!/usr/bin/env python3\n");
        python_script.push_str("# Python visualization script for TTA parameter sweep\n");
        python_script.push_str("import matplotlib.pyplot as plt\n");
        python_script.push_str("import numpy as np\n\n");

        python_script.push_str("# Data\n");
        python_script.push_str("energy_ratios = [");
        for (i, result) in results.iter().enumerate() {
            if i > 0 { python_script.push_str(", "); }
            python_script.push_str(&format!("{:.4}", result.comparative_metrics.energy_efficiency_ratio));
        }
        python_script.push_str("]\n\n");

        python_script.push_str("edp_improvements = [");
        for (i, result) in results.iter().enumerate() {
            if i > 0 { python_script.push_str(", "); }
            python_script.push_str(&format!("{:.4}", result.comparative_metrics.edp_improvement));
        }
        python_script.push_str("]\n\n");

        python_script.push_str("# Create plot\n");
        python_script.push_str("plt.figure(figsize=(10, 6))\n");
        python_script.push_str("plt.scatter(energy_ratios, edp_improvements, alpha=0.7)\n");
        python_script.push_str("plt.xlabel('Energy Efficiency Ratio (TTA/RISC)')\n");
        python_script.push_str("plt.ylabel('EDP Improvement (%)')\n");
        python_script.push_str("plt.title('TTA Parameter Sweep: Energy vs Performance')\n");
        python_script.push_str("plt.grid(True, alpha=0.3)\n");
        python_script.push_str("plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)\n");
        python_script.push_str("plt.tight_layout()\n");
        python_script.push_str("plt.savefig('tta_parameter_sweep.png', dpi=300, bbox_inches='tight')\n");
        python_script.push_str("plt.show()\n");

        std::fs::write(path, python_script)
            .map_err(|e| format!("Failed to write Python script: {}", e))?;

        println!("📊 Python visualization script exported: {}", path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_generator_creation() {
        let config = VisualizationConfig::default();
        let generator = PlotGenerator::new(config);
        assert!(generator.config.width > 0);
        assert!(generator.config.height > 0);
    }

    #[test]
    fn test_scatter_plot_empty_data() {
        let generator = PlotGenerator::new(VisualizationConfig::default());
        let plot = generator.scatter_plot_2d(&[], "X", "Y");
        assert_eq!(plot, "No data to plot");
    }

    #[test]
    fn test_correlation_calculation() {
        let generator = PlotGenerator::new(VisualizationConfig::default());

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let corr = generator.calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);

        // Perfect negative correlation
        let y_neg = vec![8.0, 6.0, 4.0, 2.0];
        let corr_neg = generator.calculate_correlation(&x, &y_neg);
        assert!((corr_neg + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_improvement_symbol() {
        let generator = PlotGenerator::new(VisualizationConfig::default());
        assert_eq!(generator.improvement_to_symbol(15.0), '+');
        assert_eq!(generator.improvement_to_symbol(7.0), '+');
        assert_eq!(generator.improvement_to_symbol(2.0), '~');
        assert_eq!(generator.improvement_to_symbol(-5.0), '-');
    }
}