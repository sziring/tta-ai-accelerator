// src/tta/functional_unit.rs
//! Core functional unit trait and supporting types for TTA architecture

/// Unique identifier for a port (FU ID + port number)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortId {
    pub fu: u16,
    pub port: u16,
}

impl PortId {
    pub fn new(fu: u16, port: u16) -> Self {
        Self { fu, port }
    }
}

/// Data types that can be transported on TTA buses
#[derive(Clone, Debug, PartialEq)]
pub enum BusData {
    I32(i32),
    I16(i16),
    I8(i8),
    VecI8(Vec<i8>),
    VecI16(Vec<i16>),
}

impl BusData {
    /// Get the bit width of this data type
    pub fn bit_width(&self) -> usize {
        match self {
            BusData::I32(_) => 32,
            BusData::I16(_) => 16,
            BusData::I8(_) => 8,
            BusData::VecI8(v) => v.len() * 8,
            BusData::VecI16(v) => v.len() * 16,
        }
    }

    /// Check if this data type is compatible with another
    pub fn is_compatible(&self, other: &BusData) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// Events returned by functional unit operations
#[derive(Clone, Debug, PartialEq)]
pub enum FuEvent {
    /// Operation completed successfully
    Ready,
    /// Operation is stalled until the specified cycle
    BusyUntil(u64),
    /// Operation is stalled with reason
    Stalled(String),
    /// Operation failed with error message
    Error(String),
}

/// Core trait that all functional units must implement
pub trait FunctionalUnit {
    /// Get the name/type of this functional unit
    fn name(&self) -> &'static str;
    
    /// Get list of input port names
    fn input_ports(&self) -> Vec<String>;
    
    /// Get list of output port names  
    fn output_ports(&self) -> Vec<String>;
    
    /// Write data to an input port
    /// Returns FuEvent indicating success/failure/stall
    fn write_input(&mut self, port: u16, data: BusData, cycle: u64) -> FuEvent;
    
    /// Read data from an output port
    /// Returns None if no data available or invalid port
    fn read_output(&self, port: u16) -> Option<BusData>;
    
    /// Check if the functional unit is busy at the given cycle
    fn is_busy(&self, cycle: u64) -> bool;
    
    /// Get total energy consumed by this unit
    fn energy_consumed(&self) -> f64;
    
    /// Reset the functional unit to initial state
    fn reset(&mut self);
    
    /// Advance the functional unit by one cycle
    /// Used for multi-cycle operations and state updates
    fn step(&mut self, cycle: u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_id() {
        let port = PortId::new(5, 3);
        assert_eq!(port.fu, 5);
        assert_eq!(port.port, 3);
    }

    #[test]
    fn test_bus_data_bit_width() {
        assert_eq!(BusData::I32(42).bit_width(), 32);
        assert_eq!(BusData::I16(42).bit_width(), 16);
        assert_eq!(BusData::I8(42).bit_width(), 8);
        assert_eq!(BusData::VecI8(vec![1, 2, 3]).bit_width(), 24);
        assert_eq!(BusData::VecI16(vec![1, 2]).bit_width(), 32);
    }

    #[test]
    fn test_bus_data_compatibility() {
        let i32_a = BusData::I32(42);
        let i32_b = BusData::I32(100);
        let i16_a = BusData::I16(42);
        
        assert!(i32_a.is_compatible(&i32_b));
        assert!(!i32_a.is_compatible(&i16_a));
    }

    #[test]
    fn test_fu_event_equality() {
        assert_eq!(FuEvent::Ready, FuEvent::Ready);
        assert_eq!(FuEvent::BusyUntil(5), FuEvent::BusyUntil(5));
        assert_ne!(FuEvent::BusyUntil(5), FuEvent::BusyUntil(6));
        assert_eq!(FuEvent::Error("test".to_string()), FuEvent::Error("test".to_string()));
    }
}
