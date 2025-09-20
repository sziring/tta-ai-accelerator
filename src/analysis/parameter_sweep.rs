// src/analysis/parameter_sweep.rs
//! Parameter sweep framework for TTA design space exploration
//!
//! Provides systematic exploration of design parameters including:
//! - Vector lane counts (8, 16, 32)
//! - Bus configurations (1, 2, 4)
//! - Memory bank counts
//! - Functional unit configurations

use crate::tta::{SchedulerConfig, TtaExecutionEngine, ExecutionStats};
use crate::risc::{RiscConfig, RiscProcessor, ExecutionResult};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    pub vector_lanes: Vec<usize>,
    pub bus_counts: Vec<u16>,
    pub memory_banks: Vec<usize>,
    pub issue_widths: Vec<u16>,
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            vector_lanes: vec![8, 16, 32],
            bus_counts: vec![1, 2, 4],
            memory_banks: vec![1, 2, 4],
            issue_widths: vec![1, 2, 4],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfiguration {
    pub parameter_space: ParameterSpace,
    pub benchmarks: Vec<String>,
    pub max_cycles: u64,
    pub iterations: usize,
}

impl Default for SweepConfiguration {
    fn default() -> Self {
        Self {
            parameter_space: ParameterSpace::default(),
            benchmarks: vec![
                "dot_product_16".to_string(),
                "vector_add".to_string(),
                "matrix_mult_4x4".to_string(),
                "reduction_sum".to_string(),
                "convolution_3x3".to_string(),
            ],
            max_cycles: 1000,
            iterations: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub configuration: ConfigurationPoint,
    pub tta_results: HashMap<String, BenchmarkMetrics>,
    pub risc_results: HashMap<String, BenchmarkMetrics>,
    pub comparative_metrics: ComparativeMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationPoint {
    pub vector_lanes: usize,
    pub bus_count: u16,
    pub memory_banks: usize,
    pub issue_width: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub cycles: u64,
    pub energy: f64,
    pub edp: f64,
    pub instructions: u64,
    pub utilization: f64,
    pub throughput: f64, // Operations per cycle
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeMetrics {
    pub energy_efficiency_ratio: f64, // TTA/RISC energy
    pub performance_ratio: f64,       // TTA/RISC cycles
    pub edp_improvement: f64,         // Percentage improvement
    pub area_efficiency: f64,         // Performance per area estimate
}

pub struct ParameterSweep {
    config: SweepConfiguration,
    results: Vec<SweepResult>,
}

impl ParameterSweep {
    pub fn new(config: SweepConfiguration) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    pub fn run_sweep(&mut self) -> Result<&Vec<SweepResult>, String> {
        println!("🔬 Starting parameter sweep analysis");
        println!("📊 Configuration space: {} points", self.total_configurations());

        let mut completed = 0;
        let total = self.total_configurations();

        for &lanes in &self.config.parameter_space.vector_lanes {
            for &buses in &self.config.parameter_space.bus_counts {
                for &memory_banks in &self.config.parameter_space.memory_banks {
                    for &issue_width in &self.config.parameter_space.issue_widths {
                        completed += 1;
                        println!("⚙️  Configuration {}/{}: lanes={}, buses={}, mem_banks={}, issue_width={}",
                                completed, total, lanes, buses, memory_banks, issue_width);

                        let config_point = ConfigurationPoint {
                            vector_lanes: lanes,
                            bus_count: buses,
                            memory_banks,
                            issue_width,
                        };

                        match self.evaluate_configuration(&config_point) {
                            Ok(result) => {
                                self.results.push(result);
                                println!("✅ Configuration completed");
                            },
                            Err(e) => {
                                println!("❌ Configuration failed: {}", e);
                                continue;
                            }
                        }
                    }
                }
            }
        }

        println!("🎯 Parameter sweep completed: {}/{} configurations successful",
                 self.results.len(), total);
        Ok(&self.results)
    }

    fn evaluate_configuration(&self, config: &ConfigurationPoint) -> Result<SweepResult, String> {
        // Create TTA configuration
        let tta_config = SchedulerConfig {
            bus_count: config.bus_count,
            issue_width: config.issue_width,
            transport_alpha: 0.02,
            transport_beta: 1.2,
            memory_banks: config.memory_banks,
        };

        // Create RISC configuration
        let risc_config = RiscConfig {
            register_count: 32,
            memory_size: 4096,
            vector_lanes: config.vector_lanes,
            pipeline_stages: 5,
            fetch_energy: 2.0,
            decode_energy: 1.5,
            register_file_energy: 1.0,
        };

        let mut tta_results = HashMap::new();
        let mut risc_results = HashMap::new();

        // Run benchmarks for this configuration
        for benchmark in &self.config.benchmarks {
            // Run TTA benchmark
            if let Ok(tta_metrics) = self.run_tta_benchmark(benchmark, &tta_config) {
                tta_results.insert(benchmark.clone(), tta_metrics);
            }

            // Run RISC benchmark
            if let Ok(risc_metrics) = self.run_risc_benchmark(benchmark, &risc_config) {
                risc_results.insert(benchmark.clone(), risc_metrics);
            }
        }

        // Calculate comparative metrics
        let comparative_metrics = self.calculate_comparative_metrics(&tta_results, &risc_results);

        Ok(SweepResult {
            configuration: config.clone(),
            tta_results,
            risc_results,
            comparative_metrics,
        })
    }

    fn run_tta_benchmark(&self, benchmark: &str, config: &SchedulerConfig) -> Result<BenchmarkMetrics, String> {
        let mut engine = TtaExecutionEngine::new(config.clone());

        let program = match benchmark {
            "dot_product_16" => {
                r#"
# 16-element dot product using TTA moves
imm.const_1 -> vecmac.vec_a
imm.const_2 -> vecmac.vec_b
imm.const_0 -> vecmac.acc_in
vecmac.mac_out -> reduce.vec_in
reduce.scalar_out -> imm.result
"#
            },
            "vector_add" => {
                r#"
# Vector addition
imm.const_1 -> vecmac.vec_a
imm.const_2 -> vecmac.vec_b
vecmac.add_out -> imm.result
"#
            },
            "reduction_sum" => {
                r#"
# Vector reduction (sum)
imm.const_1 -> reduce.vec_in
reduce.scalar_out -> imm.result
"#
            },
            _ => return Err(format!("Unknown benchmark: {}", benchmark)),
        };

        engine.load_program(program)?;
        let stats = engine.execute(self.config.max_cycles)?;

        Ok(self.stats_to_metrics(&stats))
    }

    fn run_risc_benchmark(&self, benchmark: &str, config: &RiscConfig) -> Result<BenchmarkMetrics, String> {
        use crate::risc::{RiscInstruction, InstructionType, Register, ReduceMode};

        let mut processor = RiscProcessor::new(config.clone());

        // Load test data for vector operations
        processor.load_vector_data(Register::R1,
            (1..=16).map(|x| x as i8).collect());
        processor.load_vector_data(Register::R2,
            (1..=16).rev().map(|x| x as i8).collect());

        let instructions = match benchmark {
            "dot_product_16" => vec![
                RiscInstruction::new(InstructionType::Addi {
                    rd: Register::R10, rs1: Register::R0, imm: 0
                }),
                RiscInstruction::new(InstructionType::VecMac {
                    rd: Register::R3, rs1: Register::R1, rs2: Register::R2, acc: Register::R10
                }),
            ],
            "vector_add" => vec![
                RiscInstruction::new(InstructionType::VecAdd {
                    rd: Register::R3, rs1: Register::R1, rs2: Register::R2
                }),
            ],
            "reduction_sum" => vec![
                RiscInstruction::new(InstructionType::VecReduce {
                    rd: Register::R3, rs1: Register::R1, mode: ReduceMode::Sum
                }),
            ],
            _ => return Err(format!("Unknown benchmark: {}", benchmark)),
        };

        // Execute manually to maintain vector data
        processor.reset();
        processor.load_vector_data(Register::R1,
            (1..=16).map(|x| x as i8).collect());
        processor.load_vector_data(Register::R2,
            (1..=16).rev().map(|x| x as i8).collect());

        for _ in 0..instructions.len() {
            if processor.program_counter() as usize >= instructions.len() {
                break;
            }
            processor.step(&instructions);
        }

        let result = ExecutionResult {
            cycles_executed: processor.current_cycle(),
            total_energy: processor.total_energy(),
            instructions_executed: instructions.len() as u64,
            pipeline_stalls: 0,
            memory_accesses: 0,
            vector_operations: instructions.len() as u64,
        };

        Ok(self.execution_result_to_metrics(&result))
    }

    fn stats_to_metrics(&self, stats: &ExecutionStats) -> BenchmarkMetrics {
        let total_energy: f64 = stats.energy_breakdown.values().sum();
        BenchmarkMetrics {
            cycles: stats.total_cycles,
            energy: total_energy,
            edp: total_energy * stats.total_cycles as f64,
            instructions: stats.successful_moves,
            utilization: stats.bus_utilization,
            throughput: stats.successful_moves as f64 / stats.total_cycles as f64,
        }
    }

    fn execution_result_to_metrics(&self, result: &ExecutionResult) -> BenchmarkMetrics {
        BenchmarkMetrics {
            cycles: result.cycles_executed,
            energy: result.total_energy,
            edp: result.energy_delay_product(),
            instructions: result.instructions_executed,
            utilization: 0.8, // Simplified estimate
            throughput: result.instructions_executed as f64 / result.cycles_executed as f64,
        }
    }

    fn calculate_comparative_metrics(&self, tta: &HashMap<String, BenchmarkMetrics>,
                                   risc: &HashMap<String, BenchmarkMetrics>) -> ComparativeMetrics {
        if tta.is_empty() || risc.is_empty() {
            return ComparativeMetrics {
                energy_efficiency_ratio: 1.0,
                performance_ratio: 1.0,
                edp_improvement: 0.0,
                area_efficiency: 1.0,
            };
        }

        // Average across all benchmarks
        let tta_avg_energy: f64 = tta.values().map(|m| m.energy).sum::<f64>() / tta.len() as f64;
        let risc_avg_energy: f64 = risc.values().map(|m| m.energy).sum::<f64>() / risc.len() as f64;

        let tta_avg_cycles: f64 = tta.values().map(|m| m.cycles as f64).sum::<f64>() / tta.len() as f64;
        let risc_avg_cycles: f64 = risc.values().map(|m| m.cycles as f64).sum::<f64>() / risc.len() as f64;

        let tta_avg_edp: f64 = tta.values().map(|m| m.edp).sum::<f64>() / tta.len() as f64;
        let risc_avg_edp: f64 = risc.values().map(|m| m.edp).sum::<f64>() / risc.len() as f64;

        ComparativeMetrics {
            energy_efficiency_ratio: tta_avg_energy / risc_avg_energy,
            performance_ratio: tta_avg_cycles / risc_avg_cycles,
            edp_improvement: (risc_avg_edp - tta_avg_edp) / risc_avg_edp * 100.0,
            area_efficiency: 1.0, // Placeholder - would need area models
        }
    }

    fn total_configurations(&self) -> usize {
        self.config.parameter_space.vector_lanes.len() *
        self.config.parameter_space.bus_counts.len() *
        self.config.parameter_space.memory_banks.len() *
        self.config.parameter_space.issue_widths.len()
    }

    pub fn results(&self) -> &Vec<SweepResult> {
        &self.results
    }

    pub fn best_edp_configuration(&self) -> Option<&SweepResult> {
        self.results.iter()
            .max_by(|a, b| a.comparative_metrics.edp_improvement
                          .partial_cmp(&b.comparative_metrics.edp_improvement)
                          .unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn most_energy_efficient(&self) -> Option<&SweepResult> {
        self.results.iter()
            .min_by(|a, b| a.comparative_metrics.energy_efficiency_ratio
                          .partial_cmp(&b.comparative_metrics.energy_efficiency_ratio)
                          .unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn export_results(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.results)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        std::fs::write(path, json)
            .map_err(|e| format!("File write failed: {}", e))?;

        println!("📁 Results exported to: {}", path);
        Ok(())
    }

    pub fn print_summary(&self) {
        println!("\n📈 Parameter Sweep Summary");
        println!("==========================");
        println!("Total configurations tested: {}", self.results.len());

        if let Some(best_edp) = self.best_edp_configuration() {
            println!("\n🏆 Best EDP Configuration:");
            println!("  Vector lanes: {}", best_edp.configuration.vector_lanes);
            println!("  Bus count: {}", best_edp.configuration.bus_count);
            println!("  Memory banks: {}", best_edp.configuration.memory_banks);
            println!("  Issue width: {}", best_edp.configuration.issue_width);
            println!("  EDP improvement: {:.1}%", best_edp.comparative_metrics.edp_improvement);
        }

        if let Some(most_efficient) = self.most_energy_efficient() {
            println!("\n⚡ Most Energy Efficient:");
            println!("  Vector lanes: {}", most_efficient.configuration.vector_lanes);
            println!("  Bus count: {}", most_efficient.configuration.bus_count);
            println!("  Energy ratio: {:.2}x", most_efficient.comparative_metrics.energy_efficiency_ratio);
        }

        // Configuration statistics
        let avg_edp_improvement: f64 = self.results.iter()
            .map(|r| r.comparative_metrics.edp_improvement)
            .sum::<f64>() / self.results.len() as f64;

        let avg_energy_ratio: f64 = self.results.iter()
            .map(|r| r.comparative_metrics.energy_efficiency_ratio)
            .sum::<f64>() / self.results.len() as f64;

        println!("\n📊 Overall Statistics:");
        println!("  Average EDP improvement: {:.1}%", avg_edp_improvement);
        println!("  Average energy efficiency: {:.2}x", avg_energy_ratio);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_space_creation() {
        let space = ParameterSpace::default();
        assert_eq!(space.vector_lanes, vec![8, 16, 32]);
        assert_eq!(space.bus_counts, vec![1, 2, 4]);
    }

    #[test]
    fn test_sweep_configuration() {
        let config = SweepConfiguration::default();
        assert!(!config.benchmarks.is_empty());
        assert!(config.max_cycles > 0);
    }

    #[test]
    fn test_configuration_point() {
        let point = ConfigurationPoint {
            vector_lanes: 16,
            bus_count: 2,
            memory_banks: 2,
            issue_width: 2,
        };
        assert_eq!(point.vector_lanes, 16);
        assert_eq!(point.bus_count, 2);
    }

    #[test]
    fn test_parameter_sweep_creation() {
        let config = SweepConfiguration::default();
        let sweep = ParameterSweep::new(config);
        assert_eq!(sweep.results.len(), 0);
    }
}