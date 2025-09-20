// src/risc/processor.rs
//! RISC processor implementation for EDP baseline comparison

use super::instruction_set::{RiscInstruction, InstructionType, Register, ReduceMode};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RiscConfig {
    pub register_count: usize,
    pub memory_size: usize,
    pub vector_lanes: usize,
    pub pipeline_stages: usize,
    pub fetch_energy: f64,
    pub decode_energy: f64,
    pub register_file_energy: f64,
}

impl Default for RiscConfig {
    fn default() -> Self {
        Self {
            register_count: 32,
            memory_size: 4096,
            vector_lanes: 16,
            pipeline_stages: 5,
            fetch_energy: 2.0,
            decode_energy: 1.5,
            register_file_energy: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub cycles_executed: u64,
    pub total_energy: f64,
    pub instructions_executed: u64,
    pub pipeline_stalls: u64,
    pub memory_accesses: u64,
    pub vector_operations: u64,
}

impl ExecutionResult {
    pub fn energy_delay_product(&self) -> f64 {
        self.total_energy * self.cycles_executed as f64
    }

    pub fn energy_per_instruction(&self) -> f64 {
        if self.instructions_executed > 0 {
            self.total_energy / self.instructions_executed as f64
        } else {
            0.0
        }
    }

    pub fn cycles_per_instruction(&self) -> f64 {
        if self.instructions_executed > 0 {
            self.cycles_executed as f64 / self.instructions_executed as f64
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct RiscProcessor {
    config: RiscConfig,
    registers: Vec<i32>,
    memory: Vec<u8>,
    pc: u32,
    cycle: u64,
    total_energy: f64,

    // Execution statistics
    instructions_executed: u64,
    pipeline_stalls: u64,
    memory_accesses: u64,
    vector_operations: u64,

    // Pipeline state
    pipeline_busy_until: u64,
    memory_busy_until: u64,

    // Vector register file (simplified)
    vector_registers: HashMap<Register, Vec<i8>>,
}

impl RiscProcessor {
    pub fn new(config: RiscConfig) -> Self {
        let mut registers = vec![0i32; config.register_count];
        let memory = vec![0u8; config.memory_size];
        let mut vector_registers = HashMap::new();

        // Initialize vector registers with default size
        for i in 0..32 {
            if let Some(reg) = Register::from_u8(i) {
                vector_registers.insert(reg, vec![0i8; config.vector_lanes]);
            }
        }

        Self {
            config,
            registers,
            memory,
            pc: 0,
            cycle: 0,
            total_energy: 0.0,
            instructions_executed: 0,
            pipeline_stalls: 0,
            memory_accesses: 0,
            vector_operations: 0,
            pipeline_busy_until: 0,
            memory_busy_until: 0,
            vector_registers,
        }
    }

    pub fn execute_program(&mut self, instructions: &[RiscInstruction], max_cycles: u64) -> ExecutionResult {
        // Reset computational state but preserve register and vector values set by tests
        let saved_registers = self.registers.clone();
        let saved_vector_registers = self.vector_registers.clone();
        self.reset();
        self.registers = saved_registers;
        self.vector_registers = saved_vector_registers;

        while self.cycle < max_cycles && (self.pc as usize) < instructions.len() {
            self.step(&instructions);
        }

        ExecutionResult {
            cycles_executed: self.cycle,
            total_energy: self.total_energy,
            instructions_executed: self.instructions_executed,
            pipeline_stalls: self.pipeline_stalls,
            memory_accesses: self.memory_accesses,
            vector_operations: self.vector_operations,
        }
    }

    pub fn step(&mut self, instructions: &[RiscInstruction]) {
        self.cycle += 1;

        // Add baseline energy for fetch, decode, register file access
        self.total_energy += self.config.fetch_energy + self.config.decode_energy + self.config.register_file_energy;

        // Check if pipeline is stalled
        if self.cycle < self.pipeline_busy_until {
            self.pipeline_stalls += 1;
            return;
        }

        // Check if memory is busy
        if self.cycle < self.memory_busy_until {
            self.pipeline_stalls += 1;
            return;
        }

        // Fetch and execute instruction
        if let Some(instruction) = instructions.get(self.pc as usize) {
            self.execute_instruction(instruction);
            self.instructions_executed += 1;
            self.pc += 1;
        }
    }

    fn execute_instruction(&mut self, instruction: &RiscInstruction) {
        // Add instruction-specific energy cost
        self.total_energy += instruction.energy_cost;

        // Set pipeline busy for multi-cycle instructions
        if instruction.cycles > 1 {
            self.pipeline_busy_until = self.cycle + instruction.cycles - 1;
        }

        match &instruction.opcode {
            InstructionType::Add { rd, rs1, rs2 } => {
                let val1 = self.read_register(*rs1);
                let val2 = self.read_register(*rs2);
                self.write_register(*rd, val1.wrapping_add(val2));
            },

            InstructionType::Sub { rd, rs1, rs2 } => {
                let val1 = self.read_register(*rs1);
                let val2 = self.read_register(*rs2);
                self.write_register(*rd, val1.wrapping_sub(val2));
            },

            InstructionType::Mul { rd, rs1, rs2 } => {
                let val1 = self.read_register(*rs1);
                let val2 = self.read_register(*rs2);
                self.write_register(*rd, val1.wrapping_mul(val2));
            },

            InstructionType::Div { rd, rs1, rs2 } => {
                let val1 = self.read_register(*rs1);
                let val2 = self.read_register(*rs2);
                if val2 != 0 {
                    self.write_register(*rd, val1 / val2);
                } else {
                    self.write_register(*rd, 0); // Handle division by zero
                }
            },

            InstructionType::Addi { rd, rs1, imm } => {
                let val1 = self.read_register(*rs1);
                self.write_register(*rd, val1.wrapping_add(*imm as i32));
            },

            InstructionType::Muli { rd, rs1, imm } => {
                let val1 = self.read_register(*rs1);
                self.write_register(*rd, val1.wrapping_mul(*imm as i32));
            },

            InstructionType::Load { rd, rs1, offset } => {
                let addr = (self.read_register(*rs1) + *offset as i32) as usize;
                if addr + 4 <= self.memory.len() {
                    let val = i32::from_le_bytes([
                        self.memory[addr],
                        self.memory[addr + 1],
                        self.memory[addr + 2],
                        self.memory[addr + 3],
                    ]);
                    self.write_register(*rd, val);
                    self.memory_accesses += 1;
                    self.memory_busy_until = self.cycle + 1;
                }
            },

            InstructionType::Store { rs1, rs2, offset } => {
                let addr = (self.read_register(*rs1) + *offset as i32) as usize;
                let val = self.read_register(*rs2);
                if addr + 4 <= self.memory.len() {
                    let bytes = val.to_le_bytes();
                    self.memory[addr..addr + 4].copy_from_slice(&bytes);
                    self.memory_accesses += 1;
                    self.memory_busy_until = self.cycle + 1;
                }
            },

            InstructionType::VecMac { rd, rs1, rs2, acc } => {
                // Vector multiply-accumulate: simulate 16-lane int8 MAC
                let vec_a = self.read_vector_register(*rs1);
                let vec_b = self.read_vector_register(*rs2);
                let acc_val = self.read_register(*acc);

                let mut result = acc_val;
                for i in 0..self.config.vector_lanes.min(vec_a.len()).min(vec_b.len()) {
                    result = result.wrapping_add((vec_a[i] as i32) * (vec_b[i] as i32));
                }
                self.write_register(*rd, result);
                self.vector_operations += 1;
            },

            InstructionType::VecAdd { rd, rs1, rs2 } => {
                let vec_a = self.read_vector_register(*rs1);
                let vec_b = self.read_vector_register(*rs2);
                let mut result = vec![0i8; self.config.vector_lanes];

                for i in 0..self.config.vector_lanes.min(vec_a.len()).min(vec_b.len()) {
                    result[i] = vec_a[i].wrapping_add(vec_b[i]);
                }

                self.write_vector_register(*rd, result);
                self.vector_operations += 1;
            },

            InstructionType::VecReduce { rd, rs1, mode } => {
                let vec_data = self.read_vector_register(*rs1);
                let result = match mode {
                    ReduceMode::Sum => {
                        vec_data.iter().map(|&x| x as i32).sum::<i32>()
                    },
                    ReduceMode::Max => {
                        vec_data.iter().map(|&x| x as i32).max().unwrap_or(0)
                    },
                    ReduceMode::ArgMax => {
                        vec_data.iter()
                            .enumerate()
                            .max_by_key(|(_, &val)| val)
                            .map(|(idx, _)| idx as i32)
                            .unwrap_or(0)
                    },
                };

                self.write_register(*rd, result);
                self.vector_operations += 1;
            },

            InstructionType::Branch { rs1, rs2, offset } => {
                let val1 = self.read_register(*rs1);
                let val2 = self.read_register(*rs2);
                if val1 == val2 {
                    self.pc = (self.pc as i32 + *offset as i32) as u32;
                }
            },

            InstructionType::Jump { offset } => {
                self.pc = (self.pc as i32 + *offset as i32) as u32;
            },

            InstructionType::Nop => {
                // Do nothing
            },
        }
    }

    fn read_register(&self, reg: Register) -> i32 {
        self.registers[reg.to_u8() as usize]
    }

    fn write_register(&mut self, reg: Register, value: i32) {
        self.registers[reg.to_u8() as usize] = value;
    }

    fn read_vector_register(&self, reg: Register) -> Vec<i8> {
        self.vector_registers.get(&reg).cloned().unwrap_or_else(|| vec![0i8; self.config.vector_lanes])
    }

    fn write_vector_register(&mut self, reg: Register, value: Vec<i8>) {
        self.vector_registers.insert(reg, value);
    }

    pub fn reset(&mut self) {
        self.registers.fill(0);
        self.memory.fill(0);
        self.pc = 0;
        self.cycle = 0;
        self.total_energy = 0.0;
        self.instructions_executed = 0;
        self.pipeline_stalls = 0;
        self.memory_accesses = 0;
        self.vector_operations = 0;
        self.pipeline_busy_until = 0;
        self.memory_busy_until = 0;

        // Reset vector registers
        for value in self.vector_registers.values_mut() {
            value.fill(0);
        }
    }

    // Debug and analysis functions
    pub fn current_cycle(&self) -> u64 {
        self.cycle
    }

    pub fn total_energy(&self) -> f64 {
        self.total_energy
    }

    pub fn register_value(&self, reg: Register) -> i32 {
        self.read_register(reg)
    }

    pub fn memory_at(&self, addr: usize) -> Option<u8> {
        self.memory.get(addr).copied()
    }

    pub fn load_vector_data(&mut self, reg: Register, data: Vec<i8>) {
        self.write_vector_register(reg, data);
    }

    pub fn execution_stats(&self) -> (u64, f64, u64, u64) {
        (self.instructions_executed, self.total_energy, self.pipeline_stalls, self.vector_operations)
    }

    pub fn program_counter(&self) -> u32 {
        self.pc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risc::instruction_set::*;

    #[test]
    fn test_risc_processor_creation() {
        let config = RiscConfig::default();
        let processor = RiscProcessor::new(config);

        assert_eq!(processor.current_cycle(), 0);
        assert_eq!(processor.total_energy(), 0.0);
        assert_eq!(processor.register_value(Register::R0), 0);
    }

    #[test]
    fn test_basic_arithmetic() {
        let config = RiscConfig::default();
        let mut processor = RiscProcessor::new(config);

        let instructions = vec![
            RiscInstruction::new(InstructionType::Add {
                rd: Register::R3,
                rs1: Register::R1,
                rs2: Register::R2,
            }),
        ];

        // Set up initial register values AFTER creating instructions but BEFORE execution
        processor.write_register(Register::R1, 5);
        processor.write_register(Register::R2, 3);

        let result = processor.execute_program(&instructions, 10);

        assert_eq!(processor.register_value(Register::R3), 8);
        assert_eq!(result.instructions_executed, 1);
        assert!(result.total_energy > 0.0);
    }

    #[test]
    fn test_vector_operations() {
        let config = RiscConfig::default();
        let mut processor = RiscProcessor::new(config);

        // Load vector data
        processor.load_vector_data(Register::R1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        processor.load_vector_data(Register::R2, vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        processor.write_register(Register::R4, 0);

        let instructions = vec![
            RiscInstruction::new(InstructionType::VecMac {
                rd: Register::R3,
                rs1: Register::R1,
                rs2: Register::R2,
                acc: Register::R4,
            }),
        ];

        let result = processor.execute_program(&instructions, 10);

        // Should compute dot product: 1*16 + 2*15 + ... + 16*1 = 816
        assert_eq!(processor.register_value(Register::R3), 816);
        assert_eq!(result.vector_operations, 1);
        assert!(result.total_energy > 0.0);
    }

    #[test]
    fn test_vector_reduce() {
        let config = RiscConfig::default();
        let mut processor = RiscProcessor::new(config);

        processor.load_vector_data(Register::R1, vec![1, 5, 3, 8, 2, 7, 4, 6, 9, 1, 0, 0, 0, 0, 0, 0]);

        let instructions = vec![
            RiscInstruction::new(InstructionType::VecReduce {
                rd: Register::R2,
                rs1: Register::R1,
                mode: ReduceMode::Sum,
            }),
            RiscInstruction::new(InstructionType::VecReduce {
                rd: Register::R3,
                rs1: Register::R1,
                mode: ReduceMode::Max,
            }),
            RiscInstruction::new(InstructionType::VecReduce {
                rd: Register::R4,
                rs1: Register::R1,
                mode: ReduceMode::ArgMax,
            }),
        ];

        let result = processor.execute_program(&instructions, 10);

        assert_eq!(processor.register_value(Register::R2), 46); // Sum
        assert_eq!(processor.register_value(Register::R3), 9);  // Max value
        assert_eq!(processor.register_value(Register::R4), 8);  // ArgMax index
        assert_eq!(result.vector_operations, 3);
    }
}