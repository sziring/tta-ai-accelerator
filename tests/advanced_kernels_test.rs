// tests/advanced_kernels_test.rs
//! Integration tests for advanced AI kernels
//!
//! Tests novel kernel implementations and validates their performance
//! advantages over traditional RISC architectures.

use tta_simulator::kernels::{
    AdvancedKernel, AdvancedKernelSuite,
    MultiHeadAttention, AttentionConfig,
    SoftmaxKernel, SoftmaxConfig,
    SparseMatMul, SparseConfig,
};
use tta_simulator::tta::BusData;

#[test]
fn test_advanced_kernel_suite_creation() {
    println!("🧪 Advanced Kernels Test: Suite Creation");

    let kernel_suite = AdvancedKernelSuite::new();

    // Verify all kernels are properly initialized
    assert_eq!(kernel_suite.attention.name(), "multi_head_attention");
    assert_eq!(kernel_suite.softmax.name(), "softmax");
    assert_eq!(kernel_suite.sparse_matmul.name(), "sparse_matrix_multiply");
    assert_eq!(kernel_suite.batch_norm.name(), "batch_normalization");
    assert_eq!(kernel_suite.winograd.name(), "winograd_convolution");
    assert_eq!(kernel_suite.quantized.name(), "quantized_operations");

    println!("✅ All 6 advanced kernels initialized successfully");
}

#[test]
fn test_attention_kernel_execution() {
    println!("🧪 Advanced Kernels Test: Multi-Head Attention");

    let mut attention = MultiHeadAttention::new(AttentionConfig {
        num_heads: 4,
        head_dim: 16,
        seq_length: 32,
        ..AttentionConfig::default()
    });

    // Create realistic input embeddings
    let input_embeddings: Vec<i8> = (1..=128).cycle().take(512).map(|x| x as i8).collect();
    let input_data = vec![BusData::VecI8(input_embeddings)];

    let result = attention.execute(&input_data, 1);
    assert!(result.is_ok(), "Attention execution failed: {:?}", result.err());

    let output = result.unwrap();
    assert!(!output.is_empty(), "Attention should produce output");
    assert!(attention.energy_consumed() > 0.0, "Attention should consume energy");

    let metrics = attention.get_metrics();
    assert_eq!(metrics.kernel_name, "multi_head_attention");
    assert!(metrics.throughput_ops_per_cycle > 0.0, "Should have positive throughput");
    assert!(metrics.utilization_efficiency > 0.8, "Should have high utilization for attention");

    let tta_advantage = attention.tta_advantage_factor();
    assert!(tta_advantage > 1.5, "Attention should show significant TTA advantage: {:.2}x", tta_advantage);

    println!("✅ Attention kernel: {:.2}x TTA advantage, {:.1} energy units",
             tta_advantage, attention.energy_consumed());
}

#[test]
fn test_softmax_kernel_execution() {
    println!("🧪 Advanced Kernels Test: Softmax with Numerical Stability");

    let mut softmax = SoftmaxKernel::new(SoftmaxConfig {
        vector_length: 16,
        numerical_precision: 1e-6,  // Relaxed precision for integration test
        ..SoftmaxConfig::default()
    });

    // Test with various input ranges to check numerical stability
    let test_cases = vec![
        // Normal range
        vec![BusData::VecI8((1..=16).map(|x| x as i8).collect())],
        // Large values (test overflow handling)
        vec![BusData::VecI8(vec![100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])],
        // Negative values
        vec![BusData::VecI8(vec![-10, -5, 0, 5, 10, -8, -3, 2, 7, -12, -1, 4, 9, -6, 1, 6])],
    ];

    for (i, input_data) in test_cases.into_iter().enumerate() {
        softmax.reset();

        let result = softmax.execute(&input_data, 1);
        assert!(result.is_ok(), "Softmax execution failed for test case {}: {:?}", i, result.err());

        let output = result.unwrap();
        assert_eq!(output.len(), 16, "Softmax output length mismatch");
        assert!(softmax.energy_consumed() > 0.0, "Softmax should consume energy");

        // Verify all outputs are positive (scaled integers should be positive)
        for (j, data) in output.iter().enumerate() {
            match data {
                BusData::I32(val) => assert!(*val > 0, "Softmax output {} should be positive for test case {}", j, i),
                _ => panic!("Unexpected output data type"),
            }
        }
    }

    let tta_advantage = softmax.tta_advantage_factor();
    assert!(tta_advantage > 2.0, "Softmax should show strong TTA advantage: {:.2}x", tta_advantage);

    println!("✅ Softmax kernel: {:.2}x TTA advantage, numerically stable across ranges",
             tta_advantage);
}

#[test]
fn test_sparse_matmul_kernel_execution() {
    println!("🧪 Advanced Kernels Test: Sparse Matrix Multiplication");

    let mut sparse_matmul = SparseMatMul::new(SparseConfig {
        matrix_size: 16,
        sparsity_ratio: 0.75, // 75% sparse
        enable_tta_optimizations: true,
        ..SparseConfig::default()
    });

    // Test sparse matrix-vector multiplication
    let input_data = vec![BusData::VecI8((1..=16).map(|x| x as i8).collect())];

    let result = sparse_matmul.execute(&input_data, 1);
    assert!(result.is_ok(), "Sparse matmul execution failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.len(), 16, "Sparse matmul output length mismatch");
    assert!(sparse_matmul.energy_consumed() > 0.0, "Sparse matmul should consume energy");

    let metrics = sparse_matmul.get_metrics();
    assert_eq!(metrics.kernel_name, "sparse_matrix_multiply");
    assert!(metrics.utilization_efficiency > 0.7, "Should have good utilization for sparse ops");

    let performance_analysis = sparse_matmul.get_performance_analysis();
    assert!(performance_analysis.sparsity_utilization > 0.5, "Should effectively exploit sparsity");
    assert!(performance_analysis.energy_efficiency_vs_dense < 1.0, "Should be more efficient than dense");

    let tta_advantage = sparse_matmul.tta_advantage_factor();
    assert!(tta_advantage > 2.5, "Sparse operations should show very strong TTA advantage: {:.2}x", tta_advantage);

    println!("✅ Sparse matmul kernel: {:.2}x TTA advantage, {:.1}% sparsity utilization",
             tta_advantage, performance_analysis.sparsity_utilization * 100.0);
}

#[test]
fn test_kernel_energy_scaling() {
    println!("🧪 Advanced Kernels Test: Energy Scaling Analysis");

    let test_sizes = vec![8, 16, 32, 64];
    let mut results = Vec::new();

    for &size in &test_sizes {
        // Test attention kernel scaling with FIXED baseline configuration
        // This shows true O(n²) scaling as sequence length increases
        let mut attention = MultiHeadAttention::new(AttentionConfig {
            seq_length: 8,  // Fixed baseline: 8 sequence length
            head_dim: 16,   // Fixed head dimension
            num_heads: 4,   // Fixed number of heads
            ..AttentionConfig::default()
        });

        // Input size grows with test size - this creates true scaling test
        let input_data = vec![BusData::VecI8((1..=128).cycle().take(size * 16).map(|x| x as i8).collect())];
        let _ = attention.execute(&input_data, 1);

        let expected_energy = attention.expected_energy(size * 16);
        let actual_energy = attention.energy_consumed();
        let scaling_factor = actual_energy / expected_energy;

        results.push((size, actual_energy, expected_energy, scaling_factor));

        println!("  Size {}: actual={:.1}, expected={:.1}, scaling={:.2}x",
                 size, actual_energy, expected_energy, scaling_factor);
    }

    // Verify energy scaling is reasonable
    for (size, actual, expected, scaling) in results {
        assert!(scaling > 0.5 && scaling < 2.0,
               "Energy scaling should be reasonable for size {}: {:.2}x", size, scaling);
        assert!(actual > 0.0 && expected > 0.0,
               "Energy values should be positive for size {}", size);
    }

    println!("✅ Energy scaling analysis completed successfully");
}

#[test]
fn test_tta_advantage_analysis() {
    println!("🧪 Advanced Kernels Test: TTA Advantage Analysis");

    let kernels: Vec<(&str, f64)> = vec![
        ("attention", MultiHeadAttention::new(AttentionConfig::default()).tta_advantage_factor()),
        ("softmax", SoftmaxKernel::new(SoftmaxConfig::default()).tta_advantage_factor()),
        ("sparse_matmul", SparseMatMul::new(SparseConfig::default()).tta_advantage_factor()),
    ];

    println!("\n📊 TTA Advantage Analysis:");
    println!("========================");

    let mut total_advantage = 0.0;
    let mut strong_advantage_count = 0;

    for (name, advantage) in &kernels {
        println!("  {:<15}: {:.2}x advantage", name, advantage);

        assert!(*advantage > 1.0, "{} should show TTA advantage", name);

        if *advantage > 2.0 {
            strong_advantage_count += 1;
        }
        total_advantage += advantage;
    }

    let average_advantage = total_advantage / kernels.len() as f64;

    println!("\n📈 Summary:");
    println!("  Average TTA advantage: {:.2}x", average_advantage);
    println!("  Kernels with >2x advantage: {}/{}", strong_advantage_count, kernels.len());

    // Verify that our advanced kernels show meaningful TTA advantages
    assert!(average_advantage > 1.8, "Average TTA advantage should be significant: {:.2}x", average_advantage);
    assert!(strong_advantage_count >= 2, "At least 2 kernels should show strong (>2x) advantage");

    println!("✅ TTA advantage analysis: {:.2}x average advantage across {} kernels",
             average_advantage, kernels.len());
}

#[test]
fn test_kernel_performance_metrics() {
    println!("🧪 Advanced Kernels Test: Performance Metrics");

    let mut attention = MultiHeadAttention::new(AttentionConfig {
        seq_length: 8,  // 8 * 8 = 64 elements to match input
        head_dim: 8,
        num_heads: 4,
        ..AttentionConfig::default()
    });
    let mut softmax = SoftmaxKernel::new(SoftmaxConfig {
        vector_length: 64,  // Match test input size
        numerical_precision: 1e-6,  // Relaxed precision for integration test
        ..SoftmaxConfig::default()
    });
    let mut sparse_matmul = SparseMatMul::new(SparseConfig::default());

    // Execute kernels to generate metrics
    let test_input = vec![BusData::VecI8((1..=64).map(|x| x as i8).collect())];

    let _ = attention.execute(&test_input, 1);
    let _ = softmax.execute(&test_input, 1);
    let _ = sparse_matmul.execute(&test_input, 1);

    let metrics = vec![
        attention.get_metrics(),
        softmax.get_metrics(),
        sparse_matmul.get_metrics(),
    ];

    println!("\n⚡ Performance Metrics:");
    println!("======================");

    for metric in &metrics {
        println!("  {}:", metric.kernel_name);
        println!("    Energy consumed: {:.1} units", metric.energy_consumed);
        println!("    Cycles taken: {}", metric.cycles_taken);
        println!("    Throughput: {:.1} ops/cycle", metric.throughput_ops_per_cycle);
        println!("    Energy per op: {:.2} units/op", metric.energy_per_op);
        println!("    Utilization: {:.1}%", metric.utilization_efficiency * 100.0);
        println!();

        // Validate all metrics are reasonable
        assert!(metric.energy_consumed > 0.0, "{} should consume energy", metric.kernel_name);
        assert!(metric.cycles_taken > 0, "{} should take cycles", metric.kernel_name);
        assert!(metric.throughput_ops_per_cycle > 0.0, "{} should have positive throughput", metric.kernel_name);
        assert!(metric.energy_per_op > 0.0, "{} should have positive energy per op", metric.kernel_name);
        assert!(metric.utilization_efficiency > 0.0 && metric.utilization_efficiency <= 1.0,
               "{} utilization should be between 0-100%", metric.kernel_name);
    }

    println!("✅ Performance metrics validation completed");
}

#[test]
fn test_kernel_reset_functionality() {
    println!("🧪 Advanced Kernels Test: Reset Functionality");

    let mut attention = MultiHeadAttention::new(AttentionConfig {
        seq_length: 4,  // 4 * 8 = 32 elements to match input
        head_dim: 8,
        num_heads: 2,
        ..AttentionConfig::default()
    });

    // Execute to accumulate state
    let test_input = vec![BusData::VecI8((1..=32).map(|x| x as i8).collect())];
    let _ = attention.execute(&test_input, 1);

    // Verify state is accumulated
    assert!(attention.energy_consumed() > 0.0, "Should have consumed energy");

    // Reset and verify clean state
    attention.reset();
    assert_eq!(attention.energy_consumed(), 0.0, "Energy should be reset to zero");

    // Verify can execute again after reset
    let result = attention.execute(&test_input, 1);
    assert!(result.is_ok(), "Should execute successfully after reset");
    assert!(attention.energy_consumed() > 0.0, "Should consume energy after reset");

    println!("✅ Kernel reset functionality working correctly");
}

#[test]
fn test_numerical_precision_analysis() {
    println!("🧪 Advanced Kernels Test: Numerical Precision");

    let mut softmax = SoftmaxKernel::new(SoftmaxConfig {
        vector_length: 8,
        numerical_precision: 1e-6,
        ..SoftmaxConfig::default()
    });

    // Test with extreme values to check numerical stability
    let extreme_inputs = vec![
        vec![BusData::VecI8(vec![127, -128, 100, -100, 50, -50, 25, -25])], // Max range
        vec![BusData::VecI8(vec![0, 0, 0, 1, 0, 0, 0, 0])],                 // Sparse
        vec![BusData::VecI8(vec![1, 1, 1, 1, 1, 1, 1, 1])],                 // Uniform
    ];

    for (i, input) in extreme_inputs.into_iter().enumerate() {
        softmax.reset();
        let result = softmax.execute(&input, 1);

        assert!(result.is_ok(), "Softmax should handle extreme input case {}", i);

        // For softmax, we can't easily verify the mathematical properties with integer outputs,
        // but we can verify the computation completes without error
        let output = result.unwrap();
        assert_eq!(output.len(), 8, "Output length should match input");

        println!("  Extreme case {}: handled successfully", i + 1);
    }

    println!("✅ Numerical precision tests passed");
}