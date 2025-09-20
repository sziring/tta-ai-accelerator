// src/risc/instruction_set.rs
//! RISC instruction set definition for baseline comparison

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Register {
    R0 = 0, R1, R2, R3, R4, R5, R6, R7,
    R8, R9, R10, R11, R12, R13, R14, R15,
    R16, R17, R18, R19, R20, R21, R22, R23,
    R24, R25, R26, R27, R28, R29, R30, R31,
}

impl Register {
    pub fn from_u8(val: u8) -> Option<Self> {
        if val < 32 {
            unsafe { Some(std::mem::transmute(val)) }
        } else {
            None
        }
    }

    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InstructionType {
    // Arithmetic operations
    Add { rd: Register, rs1: Register, rs2: Register },
    Sub { rd: Register, rs1: Register, rs2: Register },
    Mul { rd: Register, rs1: Register, rs2: Register },
    Div { rd: Register, rs1: Register, rs2: Register },

    // Immediate operations
    Addi { rd: Register, rs1: Register, imm: i16 },
    Muli { rd: Register, rs1: Register, imm: i16 },

    // Memory operations
    Load { rd: Register, rs1: Register, offset: i16 },
    Store { rs1: Register, rs2: Register, offset: i16 },

    // Vector operations (for fair comparison with TTA)
    VecMac { rd: Register, rs1: Register, rs2: Register, acc: Register },
    VecAdd { rd: Register, rs1: Register, rs2: Register },
    VecReduce { rd: Register, rs1: Register, mode: ReduceMode },

    // Control flow
    Branch { rs1: Register, rs2: Register, offset: i16 },
    Jump { offset: i16 },
    Nop,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReduceMode {
    Sum,
    Max,
    ArgMax,
}

#[derive(Debug, Clone)]
pub struct RiscInstruction {
    pub opcode: InstructionType,
    pub energy_cost: f64,
    pub cycles: u64,
}

impl RiscInstruction {
    pub fn new(opcode: InstructionType) -> Self {
        let (energy_cost, cycles) = Self::get_instruction_characteristics(&opcode);
        Self {
            opcode,
            energy_cost,
            cycles,
        }
    }

    /// Get energy cost and cycle count for instruction type
    /// These values are calibrated to be comparable with TTA implementation
    fn get_instruction_characteristics(opcode: &InstructionType) -> (f64, u64) {
        match opcode {
            // Basic arithmetic: 1 cycle, moderate energy
            InstructionType::Add { .. } => (12.0, 1),
            InstructionType::Sub { .. } => (12.0, 1),
            InstructionType::Addi { .. } => (10.0, 1),

            // Multiplication: 3 cycles, high energy
            InstructionType::Mul { .. } => (36.0, 3),
            InstructionType::Muli { .. } => (32.0, 3),
            InstructionType::Div { .. } => (48.0, 5),

            // Memory operations: 2-3 cycles, moderate energy
            InstructionType::Load { .. } => (18.0, 2),
            InstructionType::Store { .. } => (20.0, 2),

            // Vector operations: high energy due to parallelism
            InstructionType::VecMac { .. } => (45.0, 1), // Slightly higher than TTA
            InstructionType::VecAdd { .. } => (25.0, 1),
            InstructionType::VecReduce { .. } => (18.0, 1), // Higher than TTA reduce

            // Control flow: low energy
            InstructionType::Branch { .. } => (8.0, 1),
            InstructionType::Jump { .. } => (6.0, 1),
            InstructionType::Nop => (2.0, 1),
        }
    }

    pub fn encode(&self) -> u32 {
        // Simple encoding for simulation purposes
        match &self.opcode {
            InstructionType::Add { rd, rs1, rs2 } => {
                0x00000000 | ((rd.to_u8() as u32) << 20) | ((rs1.to_u8() as u32) << 15) | ((rs2.to_u8() as u32) << 10)
            },
            InstructionType::Addi { rd, rs1, imm } => {
                0x10000000 | ((rd.to_u8() as u32) << 20) | ((rs1.to_u8() as u32) << 15) | ((*imm as u32) & 0x7FFF)
            },
            InstructionType::VecMac { rd, rs1, rs2, acc } => {
                0x70000000 | ((rd.to_u8() as u32) << 20) | ((rs1.to_u8() as u32) << 15) |
                ((rs2.to_u8() as u32) << 10) | ((acc.to_u8() as u32) << 5)
            },
            InstructionType::Nop => 0x00000001,
            _ => 0x00000000, // Simplified for now
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_conversion() {
        assert_eq!(Register::from_u8(0), Some(Register::R0));
        assert_eq!(Register::from_u8(31), Some(Register::R31));
        assert_eq!(Register::from_u8(32), None);

        assert_eq!(Register::R0.to_u8(), 0);
        assert_eq!(Register::R31.to_u8(), 31);
    }

    #[test]
    fn test_instruction_characteristics() {
        let add_instr = RiscInstruction::new(InstructionType::Add {
            rd: Register::R1,
            rs1: Register::R2,
            rs2: Register::R3,
        });

        assert_eq!(add_instr.energy_cost, 12.0);
        assert_eq!(add_instr.cycles, 1);

        let vecmac_instr = RiscInstruction::new(InstructionType::VecMac {
            rd: Register::R1,
            rs1: Register::R2,
            rs2: Register::R3,
            acc: Register::R4,
        });

        assert_eq!(vecmac_instr.energy_cost, 45.0);
        assert_eq!(vecmac_instr.cycles, 1);
    }

    #[test]
    fn test_instruction_encoding() {
        let add_instr = RiscInstruction::new(InstructionType::Add {
            rd: Register::R1,
            rs1: Register::R2,
            rs2: Register::R3,
        });

        let encoded = add_instr.encode();
        assert_ne!(encoded, 0);

        let nop_instr = RiscInstruction::new(InstructionType::Nop);
        assert_eq!(nop_instr.encode(), 0x00000001);
    }
}