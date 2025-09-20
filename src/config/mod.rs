// src/config/mod.rs
//! TTA processor configuration system
//! Handles TOML parsing and validation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtaConfig {
    pub processor: ProcessorConfig,
    pub functional_units: Vec<FunctionalUnitConfig>,
    pub buses: BusConfig,
    pub energy: EnergyConfig,
    pub memory: MemoryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    pub name: String,
    pub issue_width: usize,
    pub pipeline_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionalUnitConfig {
    pub fu_type: String,
    pub count: usize,
    pub latency_cycles: u64,
    pub config: Option<toml::Value>, // Unit-specific configuration
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusConfig {
    pub count: usize,
    pub width_bits: usize,
    pub transport_energy_per_bit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyConfig {
    pub functional_units: HashMap<String, f64>,
    pub transport: TransportEnergyConfig,
    pub memory: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportEnergyConfig {
    pub alpha_per_bit_toggle: f64,
    pub beta_base: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub scratchpad: ScratchpadConfig,
    pub register_file: RegisterFileConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchpadConfig {
    pub bank_count: usize,
    pub bank_size_bytes: usize,
    pub latency_cycles: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterFileConfig {
    pub size: usize,
    pub read_ports: usize,
    pub write_ports: usize,
}

impl TtaConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::Io(e))?;
        
        let config: TtaConfig = toml::from_str(&content)
            .map_err(|e| ConfigError::Parse(e.to_string()))?;
        
        config.validate()?;
        Ok(config)
    }

    /// Create default configuration for testing
    pub fn default_test() -> Self {
        Self {
            processor: ProcessorConfig {
                name: "TTA-Test".to_string(),
                issue_width: 2,
                pipeline_depth: 1,
            },
            functional_units: vec![
                FunctionalUnitConfig {
                    fu_type: "ALU".to_string(),
                    count: 1,
                    latency_cycles: 1,
                    config: None,
                },
                FunctionalUnitConfig {
                    fu_type: "IMM".to_string(),
                    count: 1,
                    latency_cycles: 0,
                    config: None,
                },
                FunctionalUnitConfig {
                    fu_type: "SPM".to_string(),
                    count: 1,
                    latency_cycles: 1,
                    config: Some(toml::Value::Table({
                        let mut table = toml::value::Table::new();
                        table.insert("bank_count".to_string(), toml::Value::Integer(2));
                        table.insert("bank_size_bytes".to_string(), toml::Value::Integer(16384));
                        table
                    })),
                },
            ],
            buses: BusConfig {
                count: 2,
                width_bits: 32,
                transport_energy_per_bit: 0.02,
            },
            energy: EnergyConfig {
                functional_units: {
                    let mut map = HashMap::new();
                    map.insert("add16".to_string(), 8.0);
                    map.insert("sub16".to_string(), 8.0);
                    map.insert("mul16".to_string(), 24.0);
                    map.insert("vecmac8x8_to_i32".to_string(), 40.0);
                    map.insert("reduce_sum16".to_string(), 10.0);
                    map.insert("reduce_argmax16".to_string(), 16.0);
                    map
                },
                transport: TransportEnergyConfig {
                    alpha_per_bit_toggle: 0.02,
                    beta_base: 1.0,
                },
                memory: {
                    let mut map = HashMap::new();
                    map.insert("regfile_read".to_string(), 4.0);
                    map.insert("regfile_write".to_string(), 6.0);
                    map.insert("spm_read_32b".to_string(), 10.0);
                    map.insert("spm_write_32b".to_string(), 12.0);
                    map
                },
            },
            memory: MemoryConfig {
                scratchpad: ScratchpadConfig {
                    bank_count: 2,
                    bank_size_bytes: 16384,
                    latency_cycles: 1,
                },
                register_file: RegisterFileConfig {
                    size: 32,
                    read_ports: 2,
                    write_ports: 1,
                },
            },
        }
    }

    /// Validate configuration for consistency
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Check processor constraints
        if self.processor.issue_width == 0 {
            return Err(ConfigError::Validation("Issue width must be > 0".to_string()));
        }

        if self.buses.count == 0 {
            return Err(ConfigError::Validation("Bus count must be > 0".to_string()));
        }

        // Check that we have required functional units
        let required_units = ["ALU", "IMM"];
        for required in &required_units {
            if !self.functional_units.iter().any(|fu| fu.fu_type == *required) {
                return Err(ConfigError::Validation(
                    format!("Missing required functional unit: {}", required)
                ));
            }
        }

        // Validate energy costs are positive
        for (name, &cost) in &self.energy.functional_units {
            if cost < 0.0 {
                return Err(ConfigError::Validation(
                    format!("Negative energy cost for FU {}: {}", name, cost)
                ));
            }
        }

        for (name, &cost) in &self.energy.memory {
            if cost < 0.0 {
                return Err(ConfigError::Validation(
                    format!("Negative energy cost for memory {}: {}", name, cost)
                ));
            }
        }

        // Check memory configuration
        if self.memory.scratchpad.bank_count == 0 {
            return Err(ConfigError::Validation("SPM bank count must be > 0".to_string()));
        }

        if self.memory.register_file.size == 0 {
            return Err(ConfigError::Validation("Register file size must be > 0".to_string()));
        }

        Ok(())
    }

    /// Get energy cost for a functional unit operation
    pub fn get_fu_energy_cost(&self, operation: &str) -> Option<f64> {
        self.energy.functional_units.get(operation).copied()
    }

    /// Get energy cost for a memory operation
    pub fn get_memory_energy_cost(&self, operation: &str) -> Option<f64> {
        self.energy.memory.get(operation).copied()
    }

    /// Get transport energy cost
    pub fn calculate_transport_energy(&self, bits_transferred: usize, toggles: usize) -> f64 {
        let alpha = self.energy.transport.alpha_per_bit_toggle;
        let beta = self.energy.transport.beta_base;
        alpha * (toggles as f64) + beta * (bits_transferred as f64)
    }

    /// Export configuration to TOML string
    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        toml::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialize(e.to_string()))
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let toml_str = self.to_toml_string()?;
        std::fs::write(path.as_ref(), toml_str)
            .map_err(|e| ConfigError::Io(e))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error: {0}")]
    Parse(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Serialization error: {0}")]
    Serialize(String),
}

// Add to Cargo.toml:
// [dependencies]
// serde = { version = "1.0", features = ["derive"] }
// toml = "0.8"
// thiserror = "1.0"

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let config = TtaConfig::default_test();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_energy_lookup() {
        let config = TtaConfig::default_test();
        
        assert_eq!(config.get_fu_energy_cost("add16"), Some(8.0));
        assert_eq!(config.get_fu_energy_cost("mul16"), Some(24.0));
        assert_eq!(config.get_fu_energy_cost("nonexistent"), None);
        
        assert_eq!(config.get_memory_energy_cost("spm_read_32b"), Some(10.0));
        assert_eq!(config.get_memory_energy_cost("spm_write_32b"), Some(12.0));
    }

    #[test]
    fn test_transport_energy_calculation() {
        let config = TtaConfig::default_test();
        
        // 32-bit transfer with 10 bit toggles
        let energy = config.calculate_transport_energy(32, 10);
        let expected = 0.02 * 10.0 + 1.0 * 32.0; // alpha * toggles + beta * bits
        assert_eq!(energy, expected);
    }

    #[test]
    fn test_config_validation_errors() {
        let mut config = TtaConfig::default_test();
        
        // Test invalid issue width
        config.processor.issue_width = 0;
        assert!(config.validate().is_err());
        
        // Fix and test missing FU
        config.processor.issue_width = 1;
        config.functional_units.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_toml_serialization() {
        let config = TtaConfig::default_test();
        let toml_str = config.to_toml_string().unwrap();
        
        // Should be able to parse it back
        let parsed: TtaConfig = toml::from_str(&toml_str).unwrap();
        assert!(parsed.validate().is_ok());
        
        // Basic checks
        assert_eq!(parsed.processor.name, config.processor.name);
        assert_eq!(parsed.buses.count, config.buses.count);
    }
}
