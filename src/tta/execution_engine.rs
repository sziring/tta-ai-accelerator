// src/tta/execution_engine.rs
//! TTA Execution Engine
//!
//! Provides cycle-accurate execution of TTA programs with comprehensive
//! resource conflict detection, stall handling, and performance tracking.

use crate::tta::{
    scheduler::{TtaScheduler, SchedulerConfig, StallReason},
    instruction::{TtaProgram, TtaParser, MoveInstruction, MoveSource, MoveDestination},
    immediate_unit::{ImmediateUnit, ImmConfig},
    vecmac_unit::{VecMacUnit, VecMacConfig},
    reduce_unit::{ReduceUnit, ReduceConfig},
    spm_unit::{ScratchpadMemory, SpmConfig},
    BusData, PortId,
};
use std::collections::HashMap;

/// Execution state for a TTA program
#[derive(Clone, Debug)]
pub struct ExecutionState {
    /// Program counter
    pub pc: u32,
    /// Execution complete flag
    pub halted: bool,
    /// Current cycle count
    pub cycle: u64,
    /// Total instructions executed
    pub instructions_executed: u64,
    /// Branch taken flag
    pub branch_taken: bool,
}

/// Advanced resource conflict detector
#[derive(Clone, Debug)]
pub struct ConflictDetector {
    /// Resource usage tracking per cycle
    resource_usage: HashMap<u64, ResourceUsage>,
    /// Memory bank usage tracking
    memory_banks: HashMap<usize, u64>, // bank -> last_access_cycle
    /// Port access tracking
    port_access: HashMap<PortId, u64>, // port -> last_write_cycle
}

/// Resource usage for a specific cycle
#[derive(Clone, Debug)]
struct ResourceUsage {
    /// Buses in use
    buses_used: Vec<u16>,
    /// FUs being written to
    fu_writes: Vec<u16>,
    /// Memory banks being accessed
    memory_accesses: Vec<usize>,
    /// Ports being accessed
    port_writes: Vec<PortId>,
}

impl ConflictDetector {
    /// Create a new conflict detector
    pub fn new() -> Self {
        Self {
            resource_usage: HashMap::new(),
            memory_banks: HashMap::new(),
            port_access: HashMap::new(),
        }
    }

    /// Check for resource conflicts for a move at given cycle
    pub fn check_conflicts(&mut self, move_instr: &MoveInstruction, cycle: u64, scheduler: &TtaScheduler) -> Vec<StallReason> {
        let mut conflicts = Vec::new();

        // Resolve destination information first
        let dst_port = self.resolve_destination_port(&move_instr.destination).ok();
        let memory_bank = self.get_memory_bank(&move_instr.destination);

        // Get or create resource usage for this cycle
        let usage = self.resource_usage.entry(cycle).or_insert_with(|| ResourceUsage {
            buses_used: Vec::new(),
            fu_writes: Vec::new(),
            memory_accesses: Vec::new(),
            port_writes: Vec::new(),
        });

        // Check bus availability
        if usage.buses_used.len() >= scheduler.config.bus_count as usize {
            conflicts.push(StallReason::BusFull);
        }

        // Check destination port conflicts
        if let Some(dst_port) = dst_port {
            if usage.port_writes.contains(&dst_port) {
                conflicts.push(StallReason::PortConflict);
            }

            // Check if destination FU is busy
            if let Some(last_write) = self.port_access.get(&dst_port) {
                if cycle <= *last_write + 1 { // Simplified latency check
                    conflicts.push(StallReason::DstBusy { busy_until: *last_write + 2 });
                }
            }
        }

        // Check memory conflicts
        if let Some(bank) = memory_bank {
            if usage.memory_accesses.contains(&bank) {
                conflicts.push(StallReason::MemConflict {
                    bank,
                    queue_depth: usage.memory_accesses.iter().filter(|&&b| b == bank).count()
                });
            }
        }

        conflicts
    }

    /// Record successful move execution
    pub fn record_move(&mut self, move_instr: &MoveInstruction, cycle: u64, bus_id: u16) {
        // Resolve information first
        let dst_port = self.resolve_destination_port(&move_instr.destination).ok();
        let memory_bank = self.get_memory_bank(&move_instr.destination);

        // Record port access first
        if let Some(dst_port) = dst_port {
            self.port_access.insert(dst_port, cycle);
        }

        // Record memory access
        if let Some(bank) = memory_bank {
            self.memory_banks.insert(bank, cycle);
        }

        // Record usage
        let usage = self.resource_usage.entry(cycle).or_insert_with(|| ResourceUsage {
            buses_used: Vec::new(),
            fu_writes: Vec::new(),
            memory_accesses: Vec::new(),
            port_writes: Vec::new(),
        });

        // Record bus usage
        usage.buses_used.push(bus_id);

        // Record port access
        if let Some(dst_port) = dst_port {
            usage.port_writes.push(dst_port);
        }

        // Record memory access
        if let Some(bank) = memory_bank {
            usage.memory_accesses.push(bank);
        }
    }

    /// Get memory bank for destination (simplified)
    fn get_memory_bank(&self, destination: &MoveDestination) -> Option<usize> {
        match destination {
            MoveDestination::Memory { base: _, offset } => {
                Some((offset.abs() as usize) % 2) // 2-bank system
            }
            MoveDestination::FuInput { fu_name, port_name: _ } => {
                if fu_name.starts_with("SPM") {
                    Some(0) // Default to bank 0 for SPM accesses
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Resolve destination to PortId (simplified)
    fn resolve_destination_port(&self, destination: &MoveDestination) -> Result<PortId, String> {
        match destination {
            MoveDestination::FuInput { fu_name, port_name: _ } => {
                // Simplified mapping
                let fu_id = match fu_name.as_str() {
                    "ALU0" => 0,
                    "RF0" => 1,
                    "SPM0" => 2,
                    "IMM0" => 3,
                    "VECMAC0" => 4,
                    "REDUCE0" => 5,
                    _ => return Err("Unknown FU".to_string()),
                };
                Ok(PortId { fu: fu_id, port: 0 })
            }
            MoveDestination::Register { rf_name: _, address: _ } => {
                Ok(PortId { fu: 1, port: 2 }) // RF0.WRITE_DATA
            }
            MoveDestination::Memory { base: _, offset: _ } => {
                Ok(PortId { fu: 2, port: 3 }) // SPM0.WRITE_TRIG
            }
        }
    }

    /// Clear old resource usage data
    pub fn cleanup_old_cycles(&mut self, current_cycle: u64, keep_cycles: u64) {
        if current_cycle > keep_cycles {
            let cutoff = current_cycle - keep_cycles;
            self.resource_usage.retain(|&cycle, _| cycle >= cutoff);
        }
    }
}

/// TTA Execution Engine with advanced conflict detection
pub struct TtaExecutionEngine {
    /// Move scheduler
    scheduler: TtaScheduler,
    /// Instruction parser
    parser: TtaParser,
    /// Conflict detector
    conflict_detector: ConflictDetector,
    /// Current execution state
    state: ExecutionState,
    /// Loaded program
    program: Option<TtaProgram>,
    /// Execution statistics
    stats: ExecutionStats,
}

/// Execution statistics
#[derive(Clone, Debug, Default)]
pub struct ExecutionStats {
    /// Total cycles executed
    pub total_cycles: u64,
    /// Total instructions executed
    pub total_instructions: u64,
    /// Total moves attempted
    pub total_moves: u64,
    /// Total moves successful
    pub successful_moves: u64,
    /// Stall breakdown
    pub stall_counts: HashMap<String, u64>,
    /// Energy consumption breakdown
    pub energy_breakdown: HashMap<String, f64>,
    /// Average IPC (Instructions Per Cycle)
    pub average_ipc: f64,
    /// Bus utilization percentage
    pub bus_utilization: f64,
}

impl TtaExecutionEngine {
    /// Create a new execution engine
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            scheduler: TtaScheduler::new(config),
            parser: TtaParser::new(),
            conflict_detector: ConflictDetector::new(),
            state: ExecutionState {
                pc: 0,
                halted: false,
                cycle: 0,
                instructions_executed: 0,
                branch_taken: false,
            },
            program: None,
            stats: ExecutionStats::default(),
        }
    }

    /// Load a program from assembly text
    pub fn load_program(&mut self, assembly: &str) -> Result<(), String> {
        let program = self.parser.parse_program(assembly)?;

        // Initialize functional units based on program requirements
        self.setup_functional_units(&program)?;

        self.program = Some(program);
        self.reset_execution_state();
        Ok(())
    }

    /// Setup functional units based on program requirements
    fn setup_functional_units(&mut self, _program: &TtaProgram) -> Result<(), String> {
        // Add standard functional units
        let imm_config = ImmConfig::default();
        self.scheduler.add_functional_unit(3, Box::new(ImmediateUnit::new(imm_config)));

        let vecmac_config = VecMacConfig::default();
        self.scheduler.add_functional_unit(4, Box::new(VecMacUnit::new(vecmac_config)));

        let reduce_config = ReduceConfig::default();
        self.scheduler.add_functional_unit(5, Box::new(ReduceUnit::new(reduce_config)));

        let spm_config = SpmConfig::default();
        self.scheduler.add_functional_unit(2, Box::new(ScratchpadMemory::new(spm_config)));

        // Note: ALU and RF would need to be implemented similarly
        // For now, we focus on the units we have

        Ok(())
    }

    /// Execute the loaded program
    pub fn execute(&mut self, max_cycles: u64) -> Result<ExecutionStats, String> {
        if self.program.is_none() {
            return Err("No program loaded".to_string());
        }

        while !self.state.halted && self.state.cycle < max_cycles {
            // Fetch instruction
            let instruction = {
                let program = self.program.as_ref().unwrap();
                if self.state.pc as usize >= program.instructions.len() {
                    self.state.halted = true;
                    break;
                }
                program.instructions[self.state.pc as usize].clone()
            };

            // Execute instruction
            self.execute_instruction(&instruction)?;

            // Update statistics
            self.update_stats();

            // Advance to next instruction (unless branch occurred)
            if !self.state.branch_taken {
                self.state.pc += 1;
            }
            self.state.branch_taken = false;
            self.state.instructions_executed += 1;

            // Cleanup old conflict detection data
            self.conflict_detector.cleanup_old_cycles(self.state.cycle, 10);
        }

        // Finalize statistics
        self.finalize_stats();

        Ok(self.stats.clone())
    }

    /// Execute a single instruction
    fn execute_instruction(&mut self, instruction: &crate::tta::instruction::Instruction) -> Result<(), String> {
        for move_instr in &instruction.moves {
            // Check for conflicts
            let conflicts = self.conflict_detector.check_conflicts(
                move_instr,
                self.state.cycle,
                &self.scheduler
            );

            if !conflicts.is_empty() {
                // Handle stalls
                for conflict in conflicts {
                    self.handle_stall(move_instr, conflict);
                }
                continue;
            }

            // Execute move
            self.execute_move(move_instr)?;
        }

        // Step scheduler
        self.scheduler.step()?;
        self.state.cycle += 1;

        Ok(())
    }

    /// Execute a single move
    fn execute_move(&mut self, move_instr: &MoveInstruction) -> Result<(), String> {
        // Resolve source and destination ports
        let src_port = self.resolve_source_port(&move_instr.source)?;
        let dst_port = self.resolve_destination_port(&move_instr.destination)?;

        // Handle immediate values
        if let MoveSource::Immediate(ref data) = move_instr.source {
            // For immediate values, we need to set up the IMM unit first
            self.setup_immediate_value(data.clone())?;
        }

        // Schedule the move
        self.scheduler.schedule_move(src_port, dst_port)?;

        // Record successful move
        self.conflict_detector.record_move(move_instr, self.state.cycle, 0); // Simplified bus assignment

        self.stats.total_moves += 1;
        self.stats.successful_moves += 1;

        Ok(())
    }

    /// Setup immediate value in IMM unit
    fn setup_immediate_value(&mut self, _data: BusData) -> Result<(), String> {
        // This is a simplified implementation
        // Real implementation would manage IMM unit constants properly
        Ok(())
    }

    /// Handle move stall
    fn handle_stall(&mut self, _move_instr: &MoveInstruction, reason: StallReason) {
        let reason_name = format!("{:?}", reason).split('(').next().unwrap_or("Unknown").to_string();
        *self.stats.stall_counts.entry(reason_name).or_insert(0) += 1;
        self.stats.total_moves += 1;
        // Note: successful_moves is not incremented for stalled moves
    }

    /// Resolve source port (simplified)
    fn resolve_source_port(&self, source: &MoveSource) -> Result<PortId, String> {
        self.parser.resolve_source_port(source)
    }

    /// Resolve destination port (simplified)
    fn resolve_destination_port(&self, destination: &MoveDestination) -> Result<PortId, String> {
        self.parser.resolve_destination_port(destination)
    }

    /// Update execution statistics
    fn update_stats(&mut self) {
        self.stats.total_cycles = self.state.cycle;
        self.stats.total_instructions = self.state.instructions_executed;

        // Update energy breakdown
        self.stats.energy_breakdown.insert(
            "transport".to_string(),
            self.scheduler.total_energy()
        );
    }

    /// Finalize statistics after execution
    fn finalize_stats(&mut self) {
        // Calculate IPC
        if self.stats.total_cycles > 0 {
            self.stats.average_ipc = self.stats.total_instructions as f64 / self.stats.total_cycles as f64;
        }

        // Calculate bus utilization
        self.stats.bus_utilization = self.scheduler.bus_utilization();
    }

    /// Reset execution state
    fn reset_execution_state(&mut self) {
        self.state = ExecutionState {
            pc: 0,
            halted: false,
            cycle: 0,
            instructions_executed: 0,
            branch_taken: false,
        };
        self.stats = ExecutionStats::default();
        self.scheduler.reset();
    }

    /// Get current execution state
    pub fn execution_state(&self) -> &ExecutionState {
        &self.state
    }

    /// Get execution statistics
    pub fn statistics(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Get scheduler report
    pub fn scheduler_report(&self) -> crate::tta::scheduler::SchedulerReport {
        self.scheduler.execution_report()
    }
}

impl Default for ConflictDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_engine_creation() {
        let config = SchedulerConfig::default();
        let engine = TtaExecutionEngine::new(config);

        assert_eq!(engine.state.pc, 0);
        assert!(!engine.state.halted);
        assert!(engine.program.is_none());
    }

    #[test]
    fn test_conflict_detector() {
        let mut detector = ConflictDetector::new();

        // Create a simple move instruction
        let move_instr = MoveInstruction {
            source: MoveSource::Immediate(BusData::I32(42)),
            destination: MoveDestination::FuInput {
                fu_name: "ALU0".to_string(),
                port_name: "IN_A".to_string(),
            },
            guard: None,
        };

        let config = SchedulerConfig::default();
        let scheduler = TtaScheduler::new(config);

        // Should have no conflicts initially
        let conflicts = detector.check_conflicts(&move_instr, 0, &scheduler);
        assert!(conflicts.is_empty());

        // Record the move
        detector.record_move(&move_instr, 0, 0);

        // Same move should now conflict
        let conflicts = detector.check_conflicts(&move_instr, 0, &scheduler);
        assert!(!conflicts.is_empty());
    }

    #[test]
    fn test_program_loading() {
        let config = SchedulerConfig::default();
        let mut engine = TtaExecutionEngine::new(config);

        let assembly = r#"
            main:
                42 -> ALU0.IN_A
                10 -> ALU0.IN_B
        "#;

        let result = engine.load_program(assembly);
        assert!(result.is_ok());
        assert!(engine.program.is_some());
    }

    #[test]
    fn test_execution_stats() {
        let stats = ExecutionStats::default();

        assert_eq!(stats.total_cycles, 0);
        assert_eq!(stats.total_instructions, 0);
        assert_eq!(stats.average_ipc, 0.0);
    }

    #[test]
    fn test_resource_usage_tracking() {
        let usage = ResourceUsage {
            buses_used: vec![0, 1],
            fu_writes: vec![0],
            memory_accesses: vec![0],
            port_writes: vec![PortId { fu: 0, port: 0 }],
        };

        assert_eq!(usage.buses_used.len(), 2);
        assert_eq!(usage.fu_writes.len(), 1);
    }
}