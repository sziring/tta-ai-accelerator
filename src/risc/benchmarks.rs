// src/risc/benchmarks.rs
//! Benchmarking infrastructure for TTA vs RISC EDP comparison

use super::processor::{RiscProcessor, RiscConfig, ExecutionResult};
use super::instruction_set::{RiscInstruction, InstructionType, Register, ReduceMode};
use crate::tta::{TtaExecutionEngine, SchedulerConfig, ExecutionStats};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub risc_result: ExecutionResult,
    pub tta_result: Option<ExecutionStats>,
    pub risc_edp: f64,
    pub tta_edp: f64,
    pub edp_improvement: f64, // TTA improvement over RISC (negative = worse)
    pub energy_efficiency: f64, // TTA energy / RISC energy
    pub performance_ratio: f64, // TTA cycles / RISC cycles
}

impl BenchmarkResult {
    pub fn new(
        name: String,
        risc_result: ExecutionResult,
        tta_result: Option<ExecutionStats>,
    ) -> Self {
        let risc_edp = risc_result.energy_delay_product();
        let (tta_edp, edp_improvement, energy_efficiency, performance_ratio) =
            if let Some(tta) = &tta_result {
                let tta_total_energy: f64 = tta.energy_breakdown.values().sum();
                let tta_edp = tta_total_energy * tta.total_cycles as f64;
                let edp_improvement = (risc_edp - tta_edp) / risc_edp * 100.0;
                let energy_efficiency = tta_total_energy / risc_result.total_energy;
                let performance_ratio = tta.total_cycles as f64 / risc_result.cycles_executed as f64;
                (tta_edp, edp_improvement, energy_efficiency, performance_ratio)
            } else {
                (0.0, 0.0, 1.0, 1.0)
            };

        Self {
            benchmark_name: name,
            risc_result,
            tta_result,
            risc_edp,
            tta_edp,
            edp_improvement,
            energy_efficiency,
            performance_ratio,
        }
    }
}

pub struct EdpComparison {
    pub risc_processor: RiscProcessor,
    pub tta_engine: Option<TtaExecutionEngine>,
    pub results: Vec<BenchmarkResult>,
}

impl EdpComparison {
    pub fn new(risc_config: RiscConfig, tta_config: Option<SchedulerConfig>) -> Self {
        let risc_processor = RiscProcessor::new(risc_config);
        let tta_engine = tta_config.map(|config| TtaExecutionEngine::new(config));

        Self {
            risc_processor,
            tta_engine,
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    pub fn summary_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if self.results.is_empty() {
            return stats;
        }

        let edp_improvements: Vec<f64> = self.results.iter().map(|r| r.edp_improvement).collect();
        let energy_ratios: Vec<f64> = self.results.iter().map(|r| r.energy_efficiency).collect();
        let performance_ratios: Vec<f64> = self.results.iter().map(|r| r.performance_ratio).collect();

        stats.insert("avg_edp_improvement".to_string(),
                     edp_improvements.iter().sum::<f64>() / edp_improvements.len() as f64);
        stats.insert("min_edp_improvement".to_string(),
                     edp_improvements.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        stats.insert("max_edp_improvement".to_string(),
                     edp_improvements.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

        stats.insert("avg_energy_efficiency".to_string(),
                     energy_ratios.iter().sum::<f64>() / energy_ratios.len() as f64);
        stats.insert("avg_performance_ratio".to_string(),
                     performance_ratios.iter().sum::<f64>() / performance_ratios.len() as f64);

        stats
    }

    pub fn print_summary(&self) {
        println!("\n🔬 EDP Comparison Summary");
        println!("========================");

        for result in &self.results {
            println!("\n📊 Benchmark: {}", result.benchmark_name);
            println!("   RISC EDP: {:.2} (energy: {:.2}, cycles: {})",
                     result.risc_edp, result.risc_result.total_energy, result.risc_result.cycles_executed);

            if let Some(tta) = &result.tta_result {
                let tta_total_energy: f64 = tta.energy_breakdown.values().sum();
                println!("   TTA EDP:  {:.2} (energy: {:.2}, cycles: {})",
                         result.tta_edp, tta_total_energy, tta.total_cycles);
                println!("   EDP Improvement: {:.1}% (TTA better: {})",
                         result.edp_improvement, result.edp_improvement > 0.0);
                println!("   Energy Efficiency: {:.2}x", result.energy_efficiency);
                println!("   Performance Ratio: {:.2}x", result.performance_ratio);
            } else {
                println!("   TTA: Not available");
            }
        }

        let stats = self.summary_statistics();
        if !stats.is_empty() {
            println!("\n📈 Overall Statistics");
            println!("   Average EDP Improvement: {:.1}%",
                     stats.get("avg_edp_improvement").unwrap_or(&0.0));
            println!("   Best EDP Improvement: {:.1}%",
                     stats.get("max_edp_improvement").unwrap_or(&0.0));
            println!("   Worst EDP Improvement: {:.1}%",
                     stats.get("min_edp_improvement").unwrap_or(&0.0));
            println!("   Average Energy Efficiency: {:.2}x",
                     stats.get("avg_energy_efficiency").unwrap_or(&1.0));
            println!("   Average Performance Ratio: {:.2}x",
                     stats.get("avg_performance_ratio").unwrap_or(&1.0));
        }
    }
}

pub struct BenchmarkSuite {
    comparison: EdpComparison,
}

impl BenchmarkSuite {
    pub fn new(risc_config: RiscConfig, tta_config: Option<SchedulerConfig>) -> Self {
        Self {
            comparison: EdpComparison::new(risc_config, tta_config),
        }
    }

    pub fn run_all_benchmarks(&mut self) -> &EdpComparison {
        println!("🚀 Running A6 EDP Comparison Benchmarks");

        // Run dot product benchmark
        self.run_dot_product_benchmark();

        // Run basic arithmetic benchmark
        self.run_arithmetic_benchmark();

        // Run vector reduction benchmark
        self.run_vector_reduce_benchmark();

        &self.comparison
    }

    fn run_dot_product_benchmark(&mut self) {
        println!("🔢 Running dot product benchmark (16 elements)...");

        // RISC implementation of dot product
        let risc_instructions = vec![
            // Load vector data (simulated by immediate values)
            RiscInstruction::new(InstructionType::Addi { rd: Register::R10, rs1: Register::R0, imm: 0 }), // acc = 0

            // Use VecMac for the entire dot product in one operation
            RiscInstruction::new(InstructionType::VecMac {
                rd: Register::R10, rs1: Register::R1, rs2: Register::R2, acc: Register::R10
            }),
        ];

        // Set up vector data
        self.comparison.risc_processor.load_vector_data(Register::R1,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        self.comparison.risc_processor.load_vector_data(Register::R2,
            vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);

        let risc_result = self.comparison.risc_processor.execute_program(&risc_instructions, 100);

        // TTA comparison (if available)
        let tta_result = if let Some(ref mut tta_engine) = self.comparison.tta_engine {
            // Simple TTA program equivalent
            let tta_program = r#"
# Dot product using TTA moves
imm.const -> vecmac.vec_a
imm.const -> vecmac.vec_b
vecmac.mac_out -> reduce.vec_in
reduce.scalar_out -> imm.result
"#;

            match tta_engine.load_program(tta_program) {
                Ok(_) => {
                    match tta_engine.execute(100) {
                        Ok(stats) => Some(stats),
                        Err(_) => None,
                    }
                },
                Err(_) => None,
            }
        } else {
            None
        };

        let benchmark_result = BenchmarkResult::new(
            "Dot Product (16 elements)".to_string(),
            risc_result,
            tta_result,
        );

        self.comparison.add_result(benchmark_result);
    }

    fn run_arithmetic_benchmark(&mut self) {
        println!("➕ Running basic arithmetic benchmark...");

        let risc_instructions = vec![
            RiscInstruction::new(InstructionType::Addi { rd: Register::R1, rs1: Register::R0, imm: 5 }),
            RiscInstruction::new(InstructionType::Addi { rd: Register::R2, rs1: Register::R0, imm: 3 }),
            RiscInstruction::new(InstructionType::Add { rd: Register::R3, rs1: Register::R1, rs2: Register::R2 }),
            RiscInstruction::new(InstructionType::Mul { rd: Register::R4, rs1: Register::R3, rs2: Register::R1 }),
            RiscInstruction::new(InstructionType::Sub { rd: Register::R5, rs1: Register::R4, rs2: Register::R2 }),
        ];

        self.comparison.risc_processor.reset();
        let risc_result = self.comparison.risc_processor.execute_program(&risc_instructions, 100);

        // Basic arithmetic doesn't have a direct TTA equivalent in this simplified comparison
        let benchmark_result = BenchmarkResult::new(
            "Basic Arithmetic".to_string(),
            risc_result,
            None,
        );

        self.comparison.add_result(benchmark_result);
    }

    fn run_vector_reduce_benchmark(&mut self) {
        println!("🔽 Running vector reduction benchmark...");

        let risc_instructions = vec![
            RiscInstruction::new(InstructionType::VecReduce {
                rd: Register::R10, rs1: Register::R1, mode: ReduceMode::Sum
            }),
            RiscInstruction::new(InstructionType::VecReduce {
                rd: Register::R11, rs1: Register::R1, mode: ReduceMode::Max
            }),
            RiscInstruction::new(InstructionType::VecReduce {
                rd: Register::R12, rs1: Register::R1, mode: ReduceMode::ArgMax
            }),
        ];

        self.comparison.risc_processor.reset();
        self.comparison.risc_processor.load_vector_data(Register::R1,
            vec![3, 7, 2, 9, 1, 8, 5, 6, 4, 0, 0, 0, 0, 0, 0, 0]);

        let risc_result = self.comparison.risc_processor.execute_program(&risc_instructions, 100);

        let benchmark_result = BenchmarkResult::new(
            "Vector Reduction".to_string(),
            risc_result,
            None,
        );

        self.comparison.add_result(benchmark_result);
    }

    pub fn get_comparison(&self) -> &EdpComparison {
        &self.comparison
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_creation() {
        let risc_result = ExecutionResult {
            cycles_executed: 10,
            total_energy: 100.0,
            instructions_executed: 5,
            pipeline_stalls: 1,
            memory_accesses: 2,
            vector_operations: 1,
        };

        let benchmark = BenchmarkResult::new(
            "Test".to_string(),
            risc_result,
            None,
        );

        assert_eq!(benchmark.benchmark_name, "Test");
        assert_eq!(benchmark.risc_edp, 1000.0); // 10 * 100.0
        assert_eq!(benchmark.tta_edp, 0.0);
    }

    #[test]
    fn test_edp_comparison_creation() {
        let risc_config = RiscConfig::default();
        let comparison = EdpComparison::new(risc_config, None);

        assert!(comparison.results.is_empty());
        assert!(comparison.tta_engine.is_none());
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let risc_config = RiscConfig::default();
        let suite = BenchmarkSuite::new(risc_config, None);

        assert!(suite.comparison.results.is_empty());
    }
}