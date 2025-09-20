// src/validation/golden_reference.rs
//! Golden reference validation framework
//!
//! Provides infrastructure to validate TTA kernel outputs against
//! known-good golden references with configurable tolerance.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceData {
    pub input: Vec<i32>,
    pub expected_output: Vec<i32>,
    pub expected_energy: f64,
    pub expected_cycles: u64,
    pub tolerance: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub test_name: String,
    pub passed: bool,
    pub output_match: bool,
    pub energy_match: bool,
    pub cycle_match: bool,
    pub actual_output: Vec<i32>,
    pub actual_energy: f64,
    pub actual_cycles: u64,
    pub energy_variance: f64,
    pub cycle_variance: f64,
    pub error_message: Option<String>,
}

impl ValidationResult {
    pub fn success(test_name: String, actual_output: Vec<i32>, actual_energy: f64, actual_cycles: u64, expected_energy: f64, expected_cycles: u64) -> Self {
        let energy_variance = if expected_energy > 0.0 {
            ((actual_energy - expected_energy) / expected_energy * 100.0).abs()
        } else {
            0.0
        };

        let cycle_variance = if expected_cycles > 0 {
            ((actual_cycles as f64 - expected_cycles as f64) / expected_cycles as f64 * 100.0).abs()
        } else {
            0.0
        };

        Self {
            test_name,
            passed: true,
            output_match: true,
            energy_match: energy_variance < 5.0,
            cycle_match: cycle_variance < 10.0, // More lenient for cycles
            actual_output,
            actual_energy,
            actual_cycles,
            energy_variance,
            cycle_variance,
            error_message: None,
        }
    }

    pub fn failure(test_name: String, error: String) -> Self {
        Self {
            test_name,
            passed: false,
            output_match: false,
            energy_match: false,
            cycle_match: false,
            actual_output: Vec::new(),
            actual_energy: 0.0,
            actual_cycles: 0,
            energy_variance: 0.0,
            cycle_variance: 0.0,
            error_message: Some(error),
        }
    }
}

pub struct GoldenReference {
    references: HashMap<String, ReferenceData>,
}

impl GoldenReference {
    pub fn new() -> Self {
        Self {
            references: HashMap::new(),
        }
    }

    pub fn with_default_kernels() -> Self {
        let mut reference = Self::new();
        reference.add_default_references();
        reference
    }

    fn add_default_references(&mut self) {
        // Dot product 16 elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] · [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
        // Expected: 1*16 + 2*15 + ... + 16*1 = 1360
        self.add_reference("dot_product_16", ReferenceData {
            input: (1..=16).chain((1..=16).rev()).collect(),
            expected_output: vec![1360],
            expected_energy: 120.0, // Based on TTA VECMAC + REDUCE energy
            expected_cycles: 3,
            tolerance: 0.1,
            description: "16-element dot product using VECMAC and REDUCE".to_string(),
        });

        // Vector addition: [1,2,3,4,5,6,7,8] + [8,7,6,5,4,3,2,1] = [9,9,9,9,9,9,9,9]
        self.add_reference("vector_add_8", ReferenceData {
            input: (1..=8).chain((1..=8).rev()).collect(),
            expected_output: vec![9; 8],
            expected_energy: 45.0, // VECMAC configured for addition
            expected_cycles: 2,
            tolerance: 0.1,
            description: "8-element vector addition".to_string(),
        });

        // Vector sum reduction: sum([1,2,3,4,5,6,7,8]) = 36
        self.add_reference("vector_sum_8", ReferenceData {
            input: (1..=8).collect(),
            expected_output: vec![36],
            expected_energy: 25.0, // REDUCE sum operation
            expected_cycles: 2,
            tolerance: 0.1,
            description: "8-element vector sum reduction".to_string(),
        });

        // Vector max: max([1,5,3,8,2,7,4,6]) = 8
        self.add_reference("vector_max_8", ReferenceData {
            input: vec![1, 5, 3, 8, 2, 7, 4, 6],
            expected_output: vec![8],
            expected_energy: 28.0, // REDUCE max operation
            expected_cycles: 2,
            tolerance: 0.1,
            description: "8-element vector max reduction".to_string(),
        });

        // Vector argmax: argmax([1,5,3,8,2,7,4,6]) = 3 (0-indexed position of max value 8)
        self.add_reference("vector_argmax_8", ReferenceData {
            input: vec![1, 5, 3, 8, 2, 7, 4, 6],
            expected_output: vec![3],
            expected_energy: 30.0, // REDUCE argmax operation
            expected_cycles: 2,
            tolerance: 0.1,
            description: "8-element vector argmax reduction".to_string(),
        });

        // Matrix-vector multiply 4x4: Simple 4x4 identity matrix times [1,2,3,4] = [1,2,3,4]
        self.add_reference("matrix_vector_4x4", ReferenceData {
            input: vec![
                1, 0, 0, 0,  // Matrix row 1
                0, 1, 0, 0,  // Matrix row 2
                0, 0, 1, 0,  // Matrix row 3
                0, 0, 0, 1,  // Matrix row 4
                1, 2, 3, 4,  // Vector
            ],
            expected_output: vec![1, 2, 3, 4],
            expected_energy: 180.0, // 4 dot products
            expected_cycles: 8,
            tolerance: 0.1,
            description: "4x4 matrix-vector multiplication".to_string(),
        });

        // Convolution 3x3: Simple edge detection filter
        self.add_reference("convolution_3x3", ReferenceData {
            input: vec![
                // 5x5 input image (flattened)
                1, 2, 3, 2, 1,
                2, 4, 6, 4, 2,
                3, 6, 9, 6, 3,
                2, 4, 6, 4, 2,
                1, 2, 3, 2, 1,
                // 3x3 kernel (edge detection)
                -1, -1, -1,
                -1,  8, -1,
                -1, -1, -1,
            ],
            expected_output: vec![
                // 3x3 output (center 3x3 of convolution)
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
            ],
            expected_energy: 250.0, // 9 multiply-accumulates per output
            expected_cycles: 12,
            tolerance: 0.2,
            description: "3x3 convolution with edge detection kernel".to_string(),
        });

        // FIR filter: Simple 4-tap filter
        self.add_reference("fir_filter_4tap", ReferenceData {
            input: vec![
                // Signal: [1, 2, 3, 4, 5, 6, 7, 8]
                1, 2, 3, 4, 5, 6, 7, 8,
                // Coefficients: [0.25, 0.25, 0.25, 0.25] (moving average)
                1, 1, 1, 1, // Will be scaled by 0.25 in implementation
            ],
            expected_output: vec![
                // First valid output starts at position 3 (4 samples needed)
                10, 14, 18, 22, 26  // (1+2+3+4)*0.25 = 2.5, but using integer math
            ],
            expected_energy: 150.0,
            expected_cycles: 10,
            tolerance: 0.15,
            description: "4-tap FIR filter with moving average".to_string(),
        });
    }

    pub fn add_reference(&mut self, name: &str, reference: ReferenceData) {
        self.references.insert(name.to_string(), reference);
    }

    pub fn get_reference(&self, name: &str) -> Option<&ReferenceData> {
        self.references.get(name)
    }

    pub fn validate_output(&self, test_name: &str, actual_output: &[i32], actual_energy: f64, actual_cycles: u64) -> ValidationResult {
        let reference = match self.get_reference(test_name) {
            Some(ref_data) => ref_data,
            None => return ValidationResult::failure(test_name.to_string(), format!("No reference data found for test: {}", test_name)),
        };

        // Check output correctness
        let output_match = self.compare_outputs(&reference.expected_output, actual_output, reference.tolerance);
        if !output_match {
            return ValidationResult {
                test_name: test_name.to_string(),
                passed: false,
                output_match: false,
                energy_match: false,
                cycle_match: false,
                actual_output: actual_output.to_vec(),
                actual_energy,
                actual_cycles,
                energy_variance: 0.0,
                cycle_variance: 0.0,
                error_message: Some(format!("Output mismatch. Expected: {:?}, Got: {:?}", reference.expected_output, actual_output)),
            };
        }

        // Check energy variance
        let energy_variance = if reference.expected_energy > 0.0 {
            ((actual_energy - reference.expected_energy) / reference.expected_energy * 100.0).abs()
        } else {
            0.0
        };

        let energy_match = energy_variance < 5.0; // <5% energy variance requirement

        // Check cycle variance (more lenient)
        let cycle_variance = if reference.expected_cycles > 0 {
            ((actual_cycles as f64 - reference.expected_cycles as f64) / reference.expected_cycles as f64 * 100.0).abs()
        } else {
            0.0
        };

        let cycle_match = cycle_variance < 10.0; // <10% cycle variance (more lenient)

        ValidationResult {
            test_name: test_name.to_string(),
            passed: output_match && energy_match && cycle_match,
            output_match,
            energy_match,
            cycle_match,
            actual_output: actual_output.to_vec(),
            actual_energy,
            actual_cycles,
            energy_variance,
            cycle_variance,
            error_message: if !energy_match || !cycle_match {
                Some(format!("Performance variance too high. Energy: {:.1}%, Cycles: {:.1}%", energy_variance, cycle_variance))
            } else {
                None
            },
        }
    }

    fn compare_outputs(&self, expected: &[i32], actual: &[i32], tolerance: f64) -> bool {
        if expected.len() != actual.len() {
            return false;
        }

        for (exp, act) in expected.iter().zip(actual.iter()) {
            let diff = (*exp - *act).abs() as f64;
            let max_val = (*exp).abs().max((*act).abs()) as f64;
            let relative_error = if max_val > 0.0 { diff / max_val } else { diff };

            if relative_error > tolerance {
                return false;
            }
        }

        true
    }

    pub fn list_available_tests(&self) -> Vec<String> {
        self.references.keys().cloned().collect()
    }

    pub fn export_references(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.references)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        std::fs::write(path, json)
            .map_err(|e| format!("File write failed: {}", e))?;

        println!("📁 Golden references exported to: {}", path);
        Ok(())
    }

    pub fn import_references(&mut self, path: &str) -> Result<(), String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("File read failed: {}", e))?;

        let references: HashMap<String, ReferenceData> = serde_json::from_str(&content)
            .map_err(|e| format!("Deserialization failed: {}", e))?;

        self.references.extend(references);
        println!("📁 Golden references imported from: {}", path);
        Ok(())
    }

    pub fn print_summary(&self) {
        println!("\n📚 Golden Reference Summary");
        println!("===========================");
        println!("Available tests: {}", self.references.len());

        for (name, reference) in &self.references {
            println!("\n🔬 Test: {}", name);
            println!("  Description: {}", reference.description);
            println!("  Input size: {}", reference.input.len());
            println!("  Expected output: {:?}", reference.expected_output);
            println!("  Expected energy: {:.1} units", reference.expected_energy);
            println!("  Expected cycles: {}", reference.expected_cycles);
            println!("  Tolerance: {:.1}%", reference.tolerance * 100.0);
        }
    }
}

impl Default for GoldenReference {
    fn default() -> Self {
        Self::with_default_kernels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_reference_creation() {
        let reference = GoldenReference::with_default_kernels();
        assert!(!reference.references.is_empty());
        assert!(reference.get_reference("dot_product_16").is_some());
    }

    #[test]
    fn test_output_validation_success() {
        let reference = GoldenReference::with_default_kernels();

        // Test exact match
        let result = reference.validate_output(
            "dot_product_16",
            &[1360],
            120.0,
            3
        );

        assert!(result.passed);
        assert!(result.output_match);
        assert!(result.energy_match);
        assert!(result.cycle_match);
    }

    #[test]
    fn test_output_validation_failure() {
        let reference = GoldenReference::with_default_kernels();

        // Test output mismatch
        let result = reference.validate_output(
            "dot_product_16",
            &[1000], // Wrong output
            120.0,
            3
        );

        assert!(!result.passed);
        assert!(!result.output_match);
    }

    #[test]
    fn test_energy_variance() {
        let reference = GoldenReference::with_default_kernels();

        // Test energy variance too high (>5%)
        let result = reference.validate_output(
            "dot_product_16",
            &[1360],
            130.0, // 8.3% higher than expected 120.0
            3
        );

        assert!(!result.passed);
        assert!(result.output_match);
        assert!(!result.energy_match);
        assert!(result.energy_variance > 5.0);
    }

    #[test]
    fn test_compare_outputs_tolerance() {
        let reference = GoldenReference::new();

        // Test within tolerance
        assert!(reference.compare_outputs(&[100], &[101], 0.02)); // 1% difference, 2% tolerance

        // Test outside tolerance
        assert!(!reference.compare_outputs(&[100], &[105], 0.02)); // 5% difference, 2% tolerance
    }

    #[test]
    fn test_nonexistent_reference() {
        let reference = GoldenReference::with_default_kernels();

        let result = reference.validate_output(
            "nonexistent_test",
            &[1],
            10.0,
            1
        );

        assert!(!result.passed);
        assert!(result.error_message.is_some());
    }
}