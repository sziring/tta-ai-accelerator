// src/tta/instruction.rs
//! TTA Instruction Format and Parsing
//!
//! Defines the instruction format for Transport-Triggered Architecture
//! and provides parsing capabilities for TTA assembly programs.

use crate::tta::{PortId, BusData};
use std::collections::HashMap;
use std::str::FromStr;

/// A TTA instruction containing multiple parallel moves
#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    /// Parallel moves executed in this instruction
    pub moves: Vec<MoveInstruction>,
    /// Instruction address/line number
    pub address: u32,
    /// Optional label for this instruction
    pub label: Option<String>,
}

/// A single move within an instruction
#[derive(Clone, Debug, PartialEq)]
pub struct MoveInstruction {
    /// Source port specification
    pub source: MoveSource,
    /// Destination port specification
    pub destination: MoveDestination,
    /// Optional guard condition
    pub guard: Option<GuardCondition>,
}

/// Source of a move operation
#[derive(Clone, Debug, PartialEq)]
pub enum MoveSource {
    /// Read from functional unit output port
    FuOutput { fu_name: String, port_name: String },
    /// Immediate value
    Immediate(BusData),
    /// Register file read
    Register { rf_name: String, address: u32 },
}

/// Destination of a move operation
#[derive(Clone, Debug, PartialEq)]
pub enum MoveDestination {
    /// Write to functional unit input port
    FuInput { fu_name: String, port_name: String },
    /// Register file write
    Register { rf_name: String, address: u32 },
    /// Memory write
    Memory { base: String, offset: i32 },
}

/// Guard condition for conditional moves
#[derive(Clone, Debug, PartialEq)]
pub enum GuardCondition {
    /// Move if register/port is true (non-zero)
    IfTrue(String),
    /// Move if register/port is false (zero)
    IfFalse(String),
    /// Move if equal to value
    IfEqual(String, i32),
    /// Move if not equal to value
    IfNotEqual(String, i32),
}

/// TTA Program representation
#[derive(Clone, Debug)]
pub struct TtaProgram {
    /// Instructions in execution order
    pub instructions: Vec<Instruction>,
    /// Label to instruction address mapping
    pub labels: HashMap<String, u32>,
    /// Program metadata
    pub metadata: ProgramMetadata,
}

/// Program metadata and configuration
#[derive(Clone, Debug)]
pub struct ProgramMetadata {
    /// Target processor configuration
    pub processor_config: String,
    /// Expected functional units
    pub required_fus: Vec<String>,
    /// Entry point label
    pub entry_point: Option<String>,
    /// Program description
    pub description: String,
}

impl Default for ProgramMetadata {
    fn default() -> Self {
        Self {
            processor_config: "default".to_string(),
            required_fus: vec!["ALU".to_string(), "RF".to_string(), "SPM".to_string()],
            entry_point: Some("main".to_string()),
            description: "TTA Program".to_string(),
        }
    }
}

/// TTA Assembly Parser
pub struct TtaParser {
    /// Functional unit name to ID mapping
    fu_mapping: HashMap<String, u16>,
    /// Port name to port ID mapping per FU
    port_mapping: HashMap<String, HashMap<String, u16>>,
}

impl TtaParser {
    /// Create a new TTA parser with default mappings
    pub fn new() -> Self {
        let mut parser = Self {
            fu_mapping: HashMap::new(),
            port_mapping: HashMap::new(),
        };

        // Set up default FU mappings
        parser.add_default_mappings();
        parser
    }

    /// Add default functional unit and port mappings
    fn add_default_mappings(&mut self) {
        // Functional unit mappings
        self.fu_mapping.insert("ALU0".to_string(), 0);
        self.fu_mapping.insert("RF0".to_string(), 1);
        self.fu_mapping.insert("SPM0".to_string(), 2);
        self.fu_mapping.insert("IMM0".to_string(), 3);
        self.fu_mapping.insert("VECMAC0".to_string(), 4);
        self.fu_mapping.insert("REDUCE0".to_string(), 5);

        // ALU port mappings
        let mut alu_ports = HashMap::new();
        alu_ports.insert("IN_A".to_string(), 0);
        alu_ports.insert("IN_B".to_string(), 1);
        alu_ports.insert("OUT".to_string(), 0);
        self.port_mapping.insert("ALU0".to_string(), alu_ports);

        // Register file port mappings
        let mut rf_ports = HashMap::new();
        rf_ports.insert("READ_ADDR".to_string(), 0);
        rf_ports.insert("WRITE_ADDR".to_string(), 1);
        rf_ports.insert("WRITE_DATA".to_string(), 2);
        rf_ports.insert("READ_OUT".to_string(), 0);
        self.port_mapping.insert("RF0".to_string(), rf_ports);

        // SPM port mappings
        let mut spm_ports = HashMap::new();
        spm_ports.insert("ADDR_IN".to_string(), 0);
        spm_ports.insert("DATA_IN".to_string(), 1);
        spm_ports.insert("READ_TRIG".to_string(), 2);
        spm_ports.insert("WRITE_TRIG".to_string(), 3);
        spm_ports.insert("DATA_OUT".to_string(), 0);
        self.port_mapping.insert("SPM0".to_string(), spm_ports);

        // IMM port mappings
        let mut imm_ports = HashMap::new();
        imm_ports.insert("SELECT_IN".to_string(), 0);
        imm_ports.insert("OUT".to_string(), 0);
        self.port_mapping.insert("IMM0".to_string(), imm_ports);

        // VECMAC port mappings
        let mut vecmac_ports = HashMap::new();
        vecmac_ports.insert("VEC_A".to_string(), 0);
        vecmac_ports.insert("VEC_B".to_string(), 1);
        vecmac_ports.insert("ACC_IN".to_string(), 2);
        vecmac_ports.insert("MODE_IN".to_string(), 3);
        vecmac_ports.insert("SCALAR_OUT".to_string(), 0);
        self.port_mapping.insert("VECMAC0".to_string(), vecmac_ports);

        // REDUCE port mappings
        let mut reduce_ports = HashMap::new();
        reduce_ports.insert("VEC_IN".to_string(), 0);
        reduce_ports.insert("MODE_IN".to_string(), 1);
        reduce_ports.insert("SCALAR_OUT".to_string(), 0);
        reduce_ports.insert("INDEX_OUT".to_string(), 1);
        self.port_mapping.insert("REDUCE0".to_string(), reduce_ports);
    }

    /// Parse a complete TTA program from assembly text
    pub fn parse_program(&self, assembly: &str) -> Result<TtaProgram, String> {
        let lines: Vec<&str> = assembly.lines().collect();
        let mut instructions = Vec::new();
        let mut labels = HashMap::new();
        let mut current_address = 0u32;

        for line in lines {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }

            // Handle labels
            if trimmed.ends_with(':') {
                let label = trimmed.trim_end_matches(':').to_string();
                labels.insert(label.clone(), current_address);
                continue;
            }

            // Parse instruction
            match self.parse_instruction(trimmed, current_address) {
                Ok(instruction) => {
                    instructions.push(instruction);
                    current_address += 1;
                }
                Err(e) => {
                    return Err(format!("Line {}: {}", current_address + 1, e));
                }
            }
        }

        Ok(TtaProgram {
            instructions,
            labels,
            metadata: ProgramMetadata::default(),
        })
    }

    /// Parse a single instruction from assembly line
    pub fn parse_instruction(&self, line: &str, address: u32) -> Result<Instruction, String> {
        // Split parallel moves by ';' or '||'
        let move_parts: Vec<&str> = if line.contains("||") {
            line.split("||").collect()
        } else {
            line.split(';').collect()
        };

        let mut moves = Vec::new();

        for move_part in move_parts {
            let trimmed = move_part.trim();
            if !trimmed.is_empty() {
                moves.push(self.parse_move(trimmed)?);
            }
        }

        if moves.is_empty() {
            return Err("No valid moves in instruction".to_string());
        }

        Ok(Instruction {
            moves,
            address,
            label: None,
        })
    }

    /// Parse a single move instruction
    pub fn parse_move(&self, move_str: &str) -> Result<MoveInstruction, String> {
        // Handle guarded moves: ?condition source -> destination
        let (guard, move_part) = if move_str.starts_with('?') {
            let parts: Vec<&str> = move_str[1..].splitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err("Invalid guard syntax".to_string());
            }
            (Some(self.parse_guard(parts[0])?), parts[1])
        } else {
            (None, move_str)
        };

        // Parse move: source -> destination
        let arrow_parts: Vec<&str> = move_part.split("->").collect();
        if arrow_parts.len() != 2 {
            return Err("Move must have format 'source -> destination'".to_string());
        }

        let source = self.parse_source(arrow_parts[0].trim())?;
        let destination = self.parse_destination(arrow_parts[1].trim())?;

        Ok(MoveInstruction {
            source,
            destination,
            guard,
        })
    }

    /// Parse guard condition
    fn parse_guard(&self, guard_str: &str) -> Result<GuardCondition, String> {
        if guard_str.starts_with('!') {
            Ok(GuardCondition::IfFalse(guard_str[1..].to_string()))
        } else {
            Ok(GuardCondition::IfTrue(guard_str.to_string()))
        }
    }

    /// Parse move source
    fn parse_source(&self, source_str: &str) -> Result<MoveSource, String> {
        // Check for immediate values
        if let Ok(value) = source_str.parse::<i32>() {
            return Ok(MoveSource::Immediate(BusData::I32(value)));
        }

        // Check for FU.port format
        if source_str.contains('.') {
            let parts: Vec<&str> = source_str.split('.').collect();
            if parts.len() == 2 {
                return Ok(MoveSource::FuOutput {
                    fu_name: parts[0].to_string(),
                    port_name: parts[1].to_string(),
                });
            }
        }

        // Default to register read
        Ok(MoveSource::Register {
            rf_name: "RF0".to_string(),
            address: source_str.parse().map_err(|_| "Invalid source format")?,
        })
    }

    /// Parse move destination
    fn parse_destination(&self, dest_str: &str) -> Result<MoveDestination, String> {
        // Check for FU.port format
        if dest_str.contains('.') {
            let parts: Vec<&str> = dest_str.split('.').collect();
            if parts.len() == 2 {
                return Ok(MoveDestination::FuInput {
                    fu_name: parts[0].to_string(),
                    port_name: parts[1].to_string(),
                });
            }
        }

        // Default to register write
        Ok(MoveDestination::Register {
            rf_name: "RF0".to_string(),
            address: dest_str.parse().map_err(|_| "Invalid destination format")?,
        })
    }

    /// Resolve source to PortId
    pub fn resolve_source_port(&self, source: &MoveSource) -> Result<PortId, String> {
        match source {
            MoveSource::FuOutput { fu_name, port_name } => {
                let fu_id = self.fu_mapping.get(fu_name)
                    .ok_or_else(|| format!("Unknown FU: {}", fu_name))?;

                let port_map = self.port_mapping.get(fu_name)
                    .ok_or_else(|| format!("No port mapping for FU: {}", fu_name))?;

                let port_id = port_map.get(port_name)
                    .ok_or_else(|| format!("Unknown port: {}.{}", fu_name, port_name))?;

                Ok(PortId { fu: *fu_id, port: *port_id })
            }
            MoveSource::Register { rf_name, address: _ } => {
                let fu_id = self.fu_mapping.get(rf_name)
                    .ok_or_else(|| format!("Unknown RF: {}", rf_name))?;
                Ok(PortId { fu: *fu_id, port: 0 }) // READ_OUT port
            }
            MoveSource::Immediate(_) => {
                // Immediate values come from IMM unit
                Ok(PortId { fu: 3, port: 0 }) // IMM0.OUT
            }
        }
    }

    /// Resolve destination to PortId
    pub fn resolve_destination_port(&self, destination: &MoveDestination) -> Result<PortId, String> {
        match destination {
            MoveDestination::FuInput { fu_name, port_name } => {
                let fu_id = self.fu_mapping.get(fu_name)
                    .ok_or_else(|| format!("Unknown FU: {}", fu_name))?;

                let port_map = self.port_mapping.get(fu_name)
                    .ok_or_else(|| format!("No port mapping for FU: {}", fu_name))?;

                let port_id = port_map.get(port_name)
                    .ok_or_else(|| format!("Unknown port: {}.{}", fu_name, port_name))?;

                Ok(PortId { fu: *fu_id, port: *port_id })
            }
            MoveDestination::Register { rf_name, address: _ } => {
                let fu_id = self.fu_mapping.get(rf_name)
                    .ok_or_else(|| format!("Unknown RF: {}", rf_name))?;
                Ok(PortId { fu: *fu_id, port: 2 }) // WRITE_DATA port
            }
            MoveDestination::Memory { base: _, offset: _ } => {
                // Memory writes go to SPM
                Ok(PortId { fu: 2, port: 3 }) // SPM0.WRITE_TRIG
            }
        }
    }
}

impl Default for TtaParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = TtaParser::new();
        assert!(parser.fu_mapping.contains_key("ALU0"));
        assert!(parser.fu_mapping.contains_key("RF0"));
    }

    #[test]
    fn test_simple_move_parsing() {
        let parser = TtaParser::new();
        let move_str = "ALU0.OUT -> RF0.WRITE_DATA";

        let result = parser.parse_move(move_str);
        assert!(result.is_ok());

        let move_instr = result.unwrap();
        assert!(matches!(move_instr.source, MoveSource::FuOutput { .. }));
        assert!(matches!(move_instr.destination, MoveDestination::FuInput { .. }));
    }

    #[test]
    fn test_immediate_move_parsing() {
        let parser = TtaParser::new();
        let move_str = "42 -> ALU0.IN_A";

        let result = parser.parse_move(move_str);
        assert!(result.is_ok());

        let move_instr = result.unwrap();
        assert!(matches!(move_instr.source, MoveSource::Immediate(BusData::I32(42))));
    }

    #[test]
    fn test_guarded_move_parsing() {
        let parser = TtaParser::new();
        let move_str = "?flag ALU0.OUT -> RF0.WRITE_DATA";

        let result = parser.parse_move(move_str);
        assert!(result.is_ok());

        let move_instr = result.unwrap();
        assert!(matches!(move_instr.guard, Some(GuardCondition::IfTrue(_))));
    }

    #[test]
    fn test_parallel_moves_parsing() {
        let parser = TtaParser::new();
        let instruction_str = "ALU0.OUT -> RF0.WRITE_DATA || 42 -> ALU0.IN_A";

        let result = parser.parse_instruction(instruction_str, 0);
        assert!(result.is_ok());

        let instruction = result.unwrap();
        assert_eq!(instruction.moves.len(), 2);
    }

    #[test]
    fn test_program_parsing() {
        let parser = TtaParser::new();
        let assembly = r#"
            # Simple TTA program
            main:
                42 -> ALU0.IN_A
                10 -> ALU0.IN_B
                ALU0.OUT -> RF0.WRITE_DATA
        "#;

        let result = parser.parse_program(assembly);
        assert!(result.is_ok());

        let program = result.unwrap();
        assert_eq!(program.instructions.len(), 3);
        assert!(program.labels.contains_key("main"));
    }

    #[test]
    fn test_port_resolution() {
        let parser = TtaParser::new();
        let source = MoveSource::FuOutput {
            fu_name: "ALU0".to_string(),
            port_name: "OUT".to_string(),
        };

        let result = parser.resolve_source_port(&source);
        assert!(result.is_ok());

        let port_id = result.unwrap();
        assert_eq!(port_id.fu, 0);
        assert_eq!(port_id.port, 0);
    }

    #[test]
    fn test_invalid_move_parsing() {
        let parser = TtaParser::new();
        let invalid_move = "invalid_syntax";

        let result = parser.parse_move(invalid_move);
        assert!(result.is_err());
    }
}