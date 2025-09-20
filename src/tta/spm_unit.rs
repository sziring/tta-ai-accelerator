// src/tta/spm_unit.rs
use crate::tta::{FunctionalUnit, BusData, PortId, FuEvent};
use std::collections::{HashMap, VecDeque};

/// Scratchpad Memory Unit with bank conflict detection
#[derive(Clone, Debug)]
pub struct ScratchpadMemory {
    /// Memory banks (each bank is separate to enable parallel access)
    banks: Vec<HashMap<u32, BusData>>,
    /// Current address (latched)
    current_addr: Option<u32>,
    /// Current write data (latched)
    current_data: Option<BusData>,
    /// Last read result
    last_read: Option<BusData>,
    /// Configuration
    config: SpmConfig,
    /// Bank access queues for conflict resolution
    bank_queues: Vec<VecDeque<QueuedOperation>>,
    /// Energy consumed
    energy_consumed: f64,
    /// Busy state per bank
    bank_busy_until: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct SpmConfig {
    /// Number of memory banks
    pub bank_count: usize,
    /// Size per bank in bytes
    pub bank_size_bytes: usize,
    /// Energy cost per read operation
    pub read_energy: f64,
    /// Energy cost per write operation  
    pub write_energy: f64,
    /// Access latency in cycles
    pub latency_cycles: u64,
}

#[derive(Clone, Debug)]
struct QueuedOperation {
    op_type: MemoryOp,
    address: u32,
    data: Option<BusData>,
    cycle_issued: u64,
}

#[derive(Clone, Debug, PartialEq)]
enum MemoryOp {
    Read,
    Write,
}

impl Default for SpmConfig {
    fn default() -> Self {
        Self {
            bank_count: 2,
            bank_size_bytes: 16 * 1024, // 16KiB per bank
            read_energy: 10.0,  // matches Functional_Unit_Port_Contracts.md
            write_energy: 12.0, // matches Functional_Unit_Port_Contracts.md
            latency_cycles: 1,
        }
    }
}

impl ScratchpadMemory {
    pub fn new(config: SpmConfig) -> Self {
        let bank_count = config.bank_count;
        Self {
            banks: vec![HashMap::new(); bank_count],
            current_addr: None,
            current_data: None,
            last_read: None,
            bank_queues: vec![VecDeque::new(); bank_count],
            bank_busy_until: vec![0; bank_count],
            energy_consumed: 0.0,
            config,
        }
    }
    
    /// Determine which bank an address maps to
    fn address_to_bank(&self, address: u32) -> usize {
        // Bank selection: use LSB for even/odd distribution
        (address & 1) as usize % self.config.bank_count
    }
    
    /// Check if a bank is currently busy
    fn is_bank_busy(&self, bank: usize, cycle: u64) -> bool {
        bank < self.bank_busy_until.len() && self.bank_busy_until[bank] > cycle
    }
    
    /// Execute a queued operation
    fn execute_operation(&mut self, op: QueuedOperation, cycle: u64) -> FuEvent {
        let bank = self.address_to_bank(op.address);
        
        match op.op_type {
            MemoryOp::Read => {
                let data = self.banks[bank].get(&op.address).cloned()
                    .unwrap_or(BusData::I32(0)); // Default to 0 for uninitialized
                self.last_read = Some(data);
                self.energy_consumed += self.config.read_energy;
                self.bank_busy_until[bank] = cycle + self.config.latency_cycles;
                FuEvent::Ready
            }
            MemoryOp::Write => {
                if let Some(data) = op.data {
                    self.banks[bank].insert(op.address, data);
                    self.energy_consumed += self.config.write_energy;
                    self.bank_busy_until[bank] = cycle + self.config.latency_cycles;
                    FuEvent::Ready
                } else {
                    FuEvent::Error("Write operation missing data".to_string())
                }
            }
        }
    }
    
    /// Process pending operations in bank queues
    fn process_queues(&mut self, cycle: u64) {
        for bank in 0..self.config.bank_count {
            if !self.is_bank_busy(bank, cycle) && !self.bank_queues[bank].is_empty() {
                if let Some(op) = self.bank_queues[bank].pop_front() {
                    self.execute_operation(op, cycle);
                }
            }
        }
    }
}

impl FunctionalUnit for ScratchpadMemory {
    fn name(&self) -> &'static str {
        "SPM"
    }
    
    fn input_ports(&self) -> Vec<String> {
        vec![
            "ADDR_IN".to_string(),    // Port 0: Address input (latched)
            "DATA_IN".to_string(),    // Port 1: Data input (latched) 
            "READ_TRIG".to_string(),  // Port 2: Read trigger
            "WRITE_TRIG".to_string(), // Port 3: Write trigger
        ]
    }
    
    fn output_ports(&self) -> Vec<String> {
        vec!["DATA_OUT".to_string()] // Port 0: Data output
    }
    
    fn write_input(&mut self, port: u16, data: BusData, cycle: u64) -> FuEvent {
        match port {
            0 => { // ADDR_IN - latch address
                match data {
                    BusData::I32(addr) => {
                        self.current_addr = Some(addr as u32);
                        FuEvent::Ready
                    }
                    _ => FuEvent::Error("ADDR_IN requires I32 data type".to_string())
                }
            }
            1 => { // DATA_IN - latch data for writes
                self.current_data = Some(data);
                FuEvent::Ready
            }
            2 => { // READ_TRIG - trigger read operation
                if let Some(addr) = self.current_addr {
                    let bank = self.address_to_bank(addr);
                    
                    if self.is_bank_busy(bank, cycle) {
                        // Bank conflict - queue the operation
                        let op = QueuedOperation {
                            op_type: MemoryOp::Read,
                            address: addr,
                            data: None,
                            cycle_issued: cycle,
                        };
                        self.bank_queues[bank].push_back(op);
                        FuEvent::Stalled("mem_conflict".to_string())
                    } else {
                        // Execute immediately
                        let op = QueuedOperation {
                            op_type: MemoryOp::Read,
                            address: addr,
                            data: None,
                            cycle_issued: cycle,
                        };
                        self.execute_operation(op, cycle)
                    }
                } else {
                    FuEvent::Error("No address latched for read operation".to_string())
                }
            }
            3 => { // WRITE_TRIG - trigger write operation
                if let (Some(addr), Some(data)) = (self.current_addr, &self.current_data) {
                    let bank = self.address_to_bank(addr);
                    
                    if self.is_bank_busy(bank, cycle) {
                        // Bank conflict - queue the operation
                        let op = QueuedOperation {
                            op_type: MemoryOp::Write,
                            address: addr,
                            data: Some(data.clone()),
                            cycle_issued: cycle,
                        };
                        self.bank_queues[bank].push_back(op);
                        FuEvent::Stalled("mem_conflict".to_string())
                    } else {
                        // Execute immediately
                        let op = QueuedOperation {
                            op_type: MemoryOp::Write,
                            address: addr,
                            data: Some(data.clone()),
                            cycle_issued: cycle,
                        };
                        self.execute_operation(op, cycle)
                    }
                } else {
                    FuEvent::Error("Missing address or data for write operation".to_string())
                }
            }
            _ => FuEvent::Error(format!("Invalid input port: {}", port))
        }
    }
    
    fn read_output(&self, port: u16) -> Option<BusData> {
        match port {
            0 => self.last_read.clone(), // DATA_OUT
            _ => None,
        }
    }
    
    fn is_busy(&self, cycle: u64) -> bool {
        // Check if any bank is busy
        self.bank_busy_until.iter().any(|&busy_until| busy_until > cycle)
    }
    
    fn energy_consumed(&self) -> f64 {
        self.energy_consumed
    }
    
    fn reset(&mut self) {
        self.current_addr = None;
        self.current_data = None;
        self.last_read = None;
        for queue in &mut self.bank_queues {
            queue.clear();
        }
        self.bank_busy_until.fill(0);
        self.energy_consumed = 0.0;
    }
    
    fn step(&mut self, cycle: u64) {
        // Process any queued operations that can now execute
        self.process_queues(cycle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spm_basic_read_write() {
        let config = SpmConfig::default();
        let mut spm = ScratchpadMemory::new(config);
        
        // Write address and data
        assert_eq!(spm.write_input(0, BusData::I32(0x1000), 0), FuEvent::Ready); // ADDR_IN
        assert_eq!(spm.write_input(1, BusData::I32(42), 0), FuEvent::Ready);     // DATA_IN
        
        // Trigger write
        assert_eq!(spm.write_input(3, BusData::I32(0), 0), FuEvent::Ready); // WRITE_TRIG
        
        // Read back
        assert_eq!(spm.write_input(0, BusData::I32(0x1000), 1), FuEvent::Ready); // ADDR_IN  
        assert_eq!(spm.write_input(2, BusData::I32(0), 1), FuEvent::Ready);      // READ_TRIG
        
        assert_eq!(spm.read_output(0), Some(BusData::I32(42))); // DATA_OUT
    }
    
    #[test]
    fn test_spm_bank_selection() {
        let config = SpmConfig::default();
        let spm = ScratchpadMemory::new(config);
        
        // Even addresses go to bank 0
        assert_eq!(spm.address_to_bank(0x1000), 0);
        assert_eq!(spm.address_to_bank(0x1002), 0);
        
        // Odd addresses go to bank 1  
        assert_eq!(spm.address_to_bank(0x1001), 1);
        assert_eq!(spm.address_to_bank(0x1003), 1);
    }
    
    #[test]
    fn test_spm_bank_conflict() {
        let config = SpmConfig::default();
        let mut spm = ScratchpadMemory::new(config);
        
        // First operation to bank 0
        assert_eq!(spm.write_input(0, BusData::I32(0x1000), 0), FuEvent::Ready); // ADDR_IN (even = bank 0)
        assert_eq!(spm.write_input(2, BusData::I32(0), 0), FuEvent::Ready);      // READ_TRIG
        
        // Second operation to same bank should conflict
        assert_eq!(spm.write_input(0, BusData::I32(0x1002), 0), FuEvent::Ready); // ADDR_IN (even = bank 0)
        let result = spm.write_input(2, BusData::I32(0), 0); // READ_TRIG
        assert!(matches!(result, FuEvent::Stalled(_)));
    }
    
    #[test]
    fn test_spm_no_conflict_different_banks() {
        let config = SpmConfig::default();
        let mut spm = ScratchpadMemory::new(config);
        
        // Operation to bank 0
        assert_eq!(spm.write_input(0, BusData::I32(0x1000), 0), FuEvent::Ready); // ADDR_IN (even = bank 0)
        assert_eq!(spm.write_input(2, BusData::I32(0), 0), FuEvent::Ready);      // READ_TRIG
        
        // Operation to bank 1 should not conflict
        assert_eq!(spm.write_input(0, BusData::I32(0x1001), 0), FuEvent::Ready); // ADDR_IN (odd = bank 1)
        assert_eq!(spm.write_input(2, BusData::I32(0), 0), FuEvent::Ready);      // READ_TRIG
    }
    
    #[test]
    fn test_spm_energy_accounting() {
        let config = SpmConfig::default();
        let mut spm = ScratchpadMemory::new(config.clone());
        
        let initial_energy = spm.energy_consumed();
        
        // Write operation
        spm.write_input(0, BusData::I32(0x1000), 0);
        spm.write_input(1, BusData::I32(42), 0);
        spm.write_input(3, BusData::I32(0), 0); // WRITE_TRIG
        
        assert_eq!(spm.energy_consumed(), initial_energy + config.write_energy);
        
        // Read operation
        spm.write_input(2, BusData::I32(0), 1); // READ_TRIG
        
        assert_eq!(spm.energy_consumed(), initial_energy + config.write_energy + config.read_energy);
    }
}
