// src/tta/scheduler.rs
//! TTA Move Scheduler
//!
//! Implements the core scheduling engine for Transport-Triggered Architecture.
//! Manages data movement between functional unit ports, handles resource conflicts,
//! and provides cycle-accurate execution with comprehensive stall tracking.

use crate::tta::{FunctionalUnit, BusData, PortId, FuEvent};
use std::collections::{HashMap, VecDeque};

/// A single data movement operation in TTA
#[derive(Clone, Debug, PartialEq)]
pub struct Move {
    /// Source port (FU.port)
    pub src: PortId,
    /// Destination port (FU.port)
    pub dst: PortId,
    /// Data being moved (optional - may be resolved at runtime)
    pub data: Option<BusData>,
    /// Cycle when this move was scheduled
    pub cycle_scheduled: u64,
    /// Bus assignment (assigned by scheduler)
    pub bus_id: Option<u16>,
}

/// Reasons why a move might stall
#[derive(Clone, Debug, PartialEq)]
pub enum StallReason {
    /// Destination FU is busy processing previous operation
    DstBusy { busy_until: u64 },
    /// All buses are occupied this cycle
    BusFull,
    /// Multiple moves target same destination port
    PortConflict,
    /// Memory bank conflict in SPM
    MemConflict { bank: usize, queue_depth: usize },
    /// Source port does not have valid data
    SrcNotReady,
    /// Data dependency violation
    DataHazard,
    /// FU operation latency
    FuLatency { cycles_remaining: u64 },
}

/// A stalled move with tracking information
#[derive(Clone, Debug)]
pub struct StalledMove {
    pub move_op: Move,
    pub reason: StallReason,
    pub stall_cycle: u64,
    pub energy_saved: f64,
}

/// Bus state for resource tracking
#[derive(Clone, Debug)]
pub struct BusState {
    pub id: u16,
    pub occupied: bool,
    pub current_move: Option<Move>,
    pub energy_consumed: f64,
}

/// Scheduler configuration
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Number of buses available for data transport
    pub bus_count: u16,
    /// Maximum issue width (moves per cycle)
    pub issue_width: u16,
    /// Transport energy costs
    pub transport_alpha: f64, // Per-bit toggle cost
    pub transport_beta: f64,  // Base transport cost
    /// Memory bank configuration
    pub memory_banks: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            bus_count: 2,
            issue_width: 2,
            transport_alpha: 0.02,
            transport_beta: 1.0,
            memory_banks: 2,
        }
    }
}

/// Core TTA Move Scheduler
pub struct TtaScheduler {
    /// Configuration
    pub config: SchedulerConfig,
    /// Current simulation cycle
    current_cycle: u64,
    /// Bus state tracking
    buses: Vec<BusState>,
    /// Functional units (indexed by FU ID)
    functional_units: HashMap<u16, Box<dyn FunctionalUnit>>,
    /// Pending moves queue
    pending_moves: VecDeque<Move>,
    /// Stalled moves with reasons
    stalled_moves: Vec<StalledMove>,
    /// Move history for debugging
    executed_moves: Vec<(u64, Move, f64)>, // (cycle, move, energy)
    /// Stall statistics
    stall_stats: HashMap<String, u64>,
    /// Total energy consumed
    total_energy: f64,
}

impl TtaScheduler {
    /// Create a new TTA scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        let buses: Vec<BusState> = (0..config.bus_count)
            .map(|id| BusState {
                id,
                occupied: false,
                current_move: None,
                energy_consumed: 0.0,
            })
            .collect();

        Self {
            config,
            current_cycle: 0,
            buses,
            functional_units: HashMap::new(),
            pending_moves: VecDeque::new(),
            stalled_moves: Vec::new(),
            executed_moves: Vec::new(),
            stall_stats: HashMap::new(),
            total_energy: 0.0,
        }
    }

    /// Add a functional unit to the scheduler
    pub fn add_functional_unit(&mut self, fu_id: u16, fu: Box<dyn FunctionalUnit>) {
        self.functional_units.insert(fu_id, fu);
    }

    /// Schedule a move for execution
    pub fn schedule_move(&mut self, src: PortId, dst: PortId) -> Result<(), String> {
        let move_op = Move {
            src,
            dst,
            data: None, // Will be resolved when executed
            cycle_scheduled: self.current_cycle,
            bus_id: None, // Will be assigned by scheduler
        };

        self.pending_moves.push_back(move_op);
        Ok(())
    }

    /// Execute one simulation cycle
    pub fn step(&mut self) -> Result<(), String> {
        // Step 1: Update FU states
        self.step_functional_units();

        // Step 2: Release buses from previous cycle
        self.release_buses();

        // Step 3: Retry stalled moves
        self.retry_stalled_moves();

        // Step 4: Schedule new moves
        self.schedule_pending_moves();

        // Step 5: Execute moves on assigned buses
        self.execute_moves();

        // Step 6: Advance cycle
        self.current_cycle += 1;

        Ok(())
    }

    /// Step all functional units
    fn step_functional_units(&mut self) {
        for fu in self.functional_units.values_mut() {
            fu.step(self.current_cycle);
        }
    }

    /// Release buses that completed their transport
    fn release_buses(&mut self) {
        for bus in &mut self.buses {
            bus.occupied = false;
            bus.current_move = None;
        }
    }

    /// Retry previously stalled moves
    fn retry_stalled_moves(&mut self) {
        let stalled_moves = std::mem::take(&mut self.stalled_moves);
        let mut remaining_stalled = Vec::new();

        for stalled in stalled_moves {
            match self.try_execute_move(stalled.move_op.clone()) {
                Ok(_) => {
                    // Move succeeded, add to executed list
                    continue;
                }
                Err(reason) => {
                    // Still stalled, keep tracking
                    remaining_stalled.push(StalledMove {
                        move_op: stalled.move_op,
                        reason,
                        stall_cycle: stalled.stall_cycle,
                        energy_saved: stalled.energy_saved,
                    });
                }
            }
        }

        self.stalled_moves = remaining_stalled;
    }

    /// Schedule moves from pending queue
    fn schedule_pending_moves(&mut self) {
        let mut moves_scheduled = 0;
        let max_moves = self.config.issue_width as usize;

        while moves_scheduled < max_moves && !self.pending_moves.is_empty() {
            let move_op = self.pending_moves.pop_front().unwrap();

            match self.try_execute_move(move_op.clone()) {
                Ok(_) => {
                    moves_scheduled += 1;
                }
                Err(reason) => {
                    // Move stalled, track it
                    let stall_reason_str = format!("{:?}", reason).split('(').next().unwrap_or("Unknown").to_string();
                    *self.stall_stats.entry(stall_reason_str).or_insert(0) += 1;

                    self.stalled_moves.push(StalledMove {
                        move_op,
                        reason,
                        stall_cycle: self.current_cycle,
                        energy_saved: 0.0, // Calculate based on operation
                    });
                }
            }
        }
    }

    /// Try to execute a move, returning stall reason if failed
    fn try_execute_move(&mut self, mut move_op: Move) -> Result<(), StallReason> {
        // Step 1: Check if destination FU is busy
        if let Some(dst_fu) = self.functional_units.get(&move_op.dst.fu) {
            if dst_fu.is_busy(self.current_cycle) {
                return Err(StallReason::DstBusy {
                    busy_until: self.current_cycle + 1 // Simplified - real implementation would track exact cycles
                });
            }
        } else {
            // FU doesn't exist - this would be a configuration error
            return Err(StallReason::SrcNotReady);
        }

        // Step 2: Check for available bus
        let available_bus = self.buses.iter().position(|bus| !bus.occupied);
        if available_bus.is_none() {
            return Err(StallReason::BusFull);
        }
        let bus_id = available_bus.unwrap() as u16;

        // Step 3: Check for port conflicts (same destination)
        for bus in &self.buses {
            if let Some(ref current_move) = bus.current_move {
                if current_move.dst == move_op.dst {
                    return Err(StallReason::PortConflict);
                }
            }
        }

        // Step 4: Resolve source data
        let data = self.resolve_source_data(&move_op.src)?;
        move_op.data = Some(data.clone());

        // Step 5: Check for memory conflicts (if destination is SPM)
        if let Some(memory_conflict) = self.check_memory_conflict(&move_op) {
            return Err(memory_conflict);
        }

        // Step 6: Assign bus and execute
        move_op.bus_id = Some(bus_id);
        self.buses[bus_id as usize].occupied = true;
        self.buses[bus_id as usize].current_move = Some(move_op.clone());

        // Calculate and consume transport energy
        let transport_energy = self.calculate_transport_energy(&data);
        self.total_energy += transport_energy;
        self.buses[bus_id as usize].energy_consumed += transport_energy;

        // Record successful move
        self.executed_moves.push((self.current_cycle, move_op, transport_energy));

        Ok(())
    }

    /// Execute moves that have been assigned to buses
    fn execute_moves(&mut self) {
        for bus in &mut self.buses {
            if let Some(ref move_op) = bus.current_move.clone() {
                if let Some(ref data) = move_op.data {
                    // Write data to destination port
                    if let Some(dst_fu) = self.functional_units.get_mut(&move_op.dst.fu) {
                        let result = dst_fu.write_input(move_op.dst.port, data.clone(), self.current_cycle);

                        // Track FU energy consumption
                        match result {
                            FuEvent::Ready | FuEvent::BusyUntil(_) => {
                                self.total_energy += dst_fu.energy_consumed();
                            }
                            FuEvent::Error(_) => {
                                // Error occurred, but transport energy was already consumed
                            }
                            FuEvent::Stalled(_) => {
                                // Stalled, no additional energy consumed
                            }
                        }
                    }
                }
            }
        }
    }

    /// Resolve data from source port
    fn resolve_source_data(&self, src: &PortId) -> Result<BusData, StallReason> {
        if let Some(src_fu) = self.functional_units.get(&src.fu) {
            if let Some(data) = src_fu.read_output(src.port) {
                Ok(data)
            } else {
                Err(StallReason::SrcNotReady)
            }
        } else {
            Err(StallReason::SrcNotReady)
        }
    }

    /// Check for memory bank conflicts
    fn check_memory_conflict(&self, move_op: &Move) -> Option<StallReason> {
        // Simplified memory conflict detection
        // Real implementation would track bank usage per cycle

        // Check if this is a memory operation by FU type
        // This is a simplified check - real implementation would be more sophisticated
        if move_op.dst.fu >= 100 { // Assume SPM FUs have IDs >= 100
            // Calculate bank from address (simplified)
            if let Some(BusData::I32(addr)) = &move_op.data {
                let bank = (addr % self.config.memory_banks as i32) as usize;

                // Check if bank is already being accessed this cycle
                for bus in &self.buses {
                    if let Some(ref other_move) = bus.current_move {
                        if other_move.dst.fu >= 100 { // Another memory operation
                            // Simplified conflict detection
                            return Some(StallReason::MemConflict { bank, queue_depth: 1 });
                        }
                    }
                }
            }
        }

        None
    }

    /// Calculate energy cost for transporting data
    fn calculate_transport_energy(&self, data: &BusData) -> f64 {
        let bit_width = data.bit_width();
        let toggles = bit_width / 2; // Simplified toggle estimation

        self.config.transport_alpha * (toggles as f64) + self.config.transport_beta
    }

    /// Get current cycle
    pub fn current_cycle(&self) -> u64 {
        self.current_cycle
    }

    /// Get total energy consumed
    pub fn total_energy(&self) -> f64 {
        self.total_energy
    }

    /// Get stall statistics
    pub fn stall_statistics(&self) -> &HashMap<String, u64> {
        &self.stall_stats
    }

    /// Get bus utilization percentage
    pub fn bus_utilization(&self) -> f64 {
        if self.current_cycle == 0 {
            return 0.0;
        }

        let total_bus_cycles = self.current_cycle * (self.config.bus_count as u64);
        let used_bus_cycles = self.executed_moves.len() as u64;

        (used_bus_cycles as f64 / total_bus_cycles as f64) * 100.0
    }

    /// Get detailed execution report
    pub fn execution_report(&self) -> SchedulerReport {
        SchedulerReport {
            total_cycles: self.current_cycle,
            total_moves: self.executed_moves.len(),
            stalled_moves: self.stalled_moves.len(),
            total_energy: self.total_energy,
            bus_utilization: self.bus_utilization(),
            stall_breakdown: self.stall_stats.clone(),
        }
    }

    /// Reset scheduler state
    pub fn reset(&mut self) {
        self.current_cycle = 0;
        self.pending_moves.clear();
        self.stalled_moves.clear();
        self.executed_moves.clear();
        self.stall_stats.clear();
        self.total_energy = 0.0;

        for bus in &mut self.buses {
            bus.occupied = false;
            bus.current_move = None;
            bus.energy_consumed = 0.0;
        }

        for fu in self.functional_units.values_mut() {
            fu.reset();
        }
    }
}

/// Scheduler execution report
#[derive(Clone, Debug)]
pub struct SchedulerReport {
    pub total_cycles: u64,
    pub total_moves: usize,
    pub stalled_moves: usize,
    pub total_energy: f64,
    pub bus_utilization: f64,
    pub stall_breakdown: HashMap<String, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tta::immediate_unit::{ImmediateUnit, ImmConfig};

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = TtaScheduler::new(config);

        assert_eq!(scheduler.current_cycle(), 0);
        assert_eq!(scheduler.total_energy(), 0.0);
        assert_eq!(scheduler.buses.len(), 2);
    }

    #[test]
    fn test_add_functional_unit() {
        let config = SchedulerConfig::default();
        let mut scheduler = TtaScheduler::new(config);

        let imm_config = ImmConfig::default();
        let imm = ImmediateUnit::new(imm_config);

        scheduler.add_functional_unit(0, Box::new(imm));
        assert!(scheduler.functional_units.contains_key(&0));
    }

    #[test]
    fn test_move_scheduling() {
        let config = SchedulerConfig::default();
        let mut scheduler = TtaScheduler::new(config);

        let src = PortId { fu: 0, port: 0 };
        let dst = PortId { fu: 1, port: 0 };

        let result = scheduler.schedule_move(src, dst);
        assert!(result.is_ok());
        assert_eq!(scheduler.pending_moves.len(), 1);
    }

    #[test]
    fn test_bus_assignment() {
        let config = SchedulerConfig::default();
        let scheduler = TtaScheduler::new(config);

        // All buses should start unoccupied
        for bus in &scheduler.buses {
            assert!(!bus.occupied);
            assert!(bus.current_move.is_none());
        }
    }

    #[test]
    fn test_energy_calculation() {
        let config = SchedulerConfig::default();
        let scheduler = TtaScheduler::new(config.clone());

        let data = BusData::I32(42);
        let energy = scheduler.calculate_transport_energy(&data);

        // Should be alpha * toggles + beta
        let expected = config.transport_alpha * 16.0 + config.transport_beta; // 32-bit / 2 = 16 toggles
        assert!((energy - expected).abs() < 0.001);
    }

    #[test]
    fn test_stall_tracking() {
        let config = SchedulerConfig::default();
        let scheduler = TtaScheduler::new(config);

        // Initially no stalls
        assert_eq!(scheduler.stall_statistics().len(), 0);

        // After some operations, stall stats should be tracked
        // (Detailed testing would require more complex setup)
    }

    #[test]
    fn test_scheduler_reset() {
        let config = SchedulerConfig::default();
        let mut scheduler = TtaScheduler::new(config);

        // Add some state
        let src = PortId { fu: 0, port: 0 };
        let dst = PortId { fu: 1, port: 0 };
        scheduler.schedule_move(src, dst).unwrap();

        // Reset should clear everything
        scheduler.reset();
        assert_eq!(scheduler.current_cycle(), 0);
        assert_eq!(scheduler.total_energy(), 0.0);
        assert_eq!(scheduler.pending_moves.len(), 0);
    }
}