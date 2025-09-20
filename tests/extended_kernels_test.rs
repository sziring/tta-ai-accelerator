// tests/extended_kernels_test.rs
//! Extended Kernels Test Suite
//!
//! Feature-gated tests for new kernel implementations that preserve
//! existing validated functionality while extending capabilities.

#![cfg(feature = "extended-kernels")]

use tta_simulator::kernels::{
    Conv2DKernel, Conv2DConfig,
    GemmKernel, GemmConfig,
    SoftmaxPipelineKernel, SoftmaxPipelineConfig,
    AdvancedKernel,
};
use tta_simulator::tta::BusData;

#[test]
fn test_extended_kernels_integration() {
    println!("🧪 Extended Kernels Integration Test");
    println!("====================================");

    // Test Conv2D kernel
    let mut conv2d = Conv2DKernel::new(Conv2DConfig {
        input_height: 8,
        input_width: 8,
        input_channels: 3,
        output_channels: 16,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        ..Conv2DConfig::default()
    });

    // Create conv2d test input (8x8x3 = 192 elements)
    let conv_input = vec![BusData::VecI8((1..=192).map(|x| (x % 50) as i8).collect())];
    let conv_result = conv2d.execute(&conv_input, 1);
    assert!(conv_result.is_ok(), "Conv2D execution should succeed");

    let conv_metrics = conv2d.get_metrics();
    println!("Conv2D metrics: {:.2} energy, {:.2} ops/cycle",
             conv_metrics.energy_consumed, conv_metrics.throughput_ops_per_cycle);

    // Test GEMM kernel
    let mut gemm = GemmKernel::new(GemmConfig {
        m: 16,
        n: 16,
        k: 16,
        alpha: 1.0,
        beta: 0.0,
        ..GemmConfig::default()
    });

    // Create GEMM test input (A: 16x16, B: 16x16, C: 16x16 = 768 elements)
    let gemm_input = vec![BusData::VecI8((1..=768).map(|x| (x % 30) as i8).collect())];
    let gemm_result = gemm.execute(&gemm_input, 1);
    assert!(gemm_result.is_ok(), "GEMM execution should succeed");

    let gemm_metrics = gemm.get_metrics();
    println!("GEMM metrics: {:.2} energy, {:.2} ops/cycle",
             gemm_metrics.energy_consumed, gemm_metrics.throughput_ops_per_cycle);

    // Test Softmax Pipeline kernel
    let mut pipeline = SoftmaxPipelineKernel::new(SoftmaxPipelineConfig {
        batch_size: 4,
        sequence_length: 8,
        num_heads: 4,
        temperature: 1.0,
        ..SoftmaxPipelineConfig::default()
    });

    // Create pipeline test input (4 batches × 4 heads × 8×8 = 1024 elements)
    let pipeline_input = vec![BusData::VecI8((1..=127).cycle().take(1024).map(|x| x as i8).collect())];
    let pipeline_result = pipeline.execute(&pipeline_input, 1);
    assert!(pipeline_result.is_ok(), "Softmax pipeline execution should succeed");

    let pipeline_metrics = pipeline.get_metrics();
    println!("Pipeline metrics: {:.2} energy, {:.2} ops/cycle",
             pipeline_metrics.energy_consumed, pipeline_metrics.throughput_ops_per_cycle);

    // Verify TTA advantages
    assert!(conv2d.tta_advantage_factor() > 2.0, "Conv2D should show TTA advantage");
    assert!(gemm.tta_advantage_factor() > 2.5, "GEMM should show strong TTA advantage");
    assert!(pipeline.tta_advantage_factor() > 3.0, "Pipeline should show very strong TTA advantage");

    println!("✅ All extended kernels demonstrate TTA advantages and execute successfully");
}

#[test]
fn test_extended_kernels_energy_efficiency() {
    println!("🧪 Extended Kernels Energy Efficiency Test");
    println!("==========================================");

    let test_size = 32; // Moderate size for efficiency testing

    // Test Conv2D energy scaling
    let conv2d_small = Conv2DKernel::new(Conv2DConfig {
        input_height: test_size / 2,
        input_width: test_size / 2,
        input_channels: 4,
        output_channels: 8,
        kernel_size: 3,
        ..Conv2DConfig::default()
    });

    let conv2d_large = Conv2DKernel::new(Conv2DConfig {
        input_height: test_size,
        input_width: test_size,
        input_channels: 4,
        output_channels: 8,
        kernel_size: 3,
        ..Conv2DConfig::default()
    });

    let small_expected = conv2d_small.expected_energy(256);
    let large_expected = conv2d_large.expected_energy(4096);

    // Energy should scale reasonably (not exponentially)
    let energy_ratio = large_expected / small_expected;
    assert!(energy_ratio > 10.0 && energy_ratio < 50.0,
            "Conv2D energy scaling should be reasonable: {:.2}x", energy_ratio);

    // Test GEMM energy scaling
    let gemm_small = GemmKernel::new(GemmConfig {
        m: test_size / 2, n: test_size / 2, k: test_size / 2,
        ..GemmConfig::default()
    });

    let gemm_large = GemmKernel::new(GemmConfig {
        m: test_size, n: test_size, k: test_size,
        ..GemmConfig::default()
    });

    let gemm_small_expected = gemm_small.expected_energy(1024);
    let gemm_large_expected = gemm_large.expected_energy(8192);

    let gemm_ratio = gemm_large_expected / gemm_small_expected;
    assert!(gemm_ratio > 5.0 && gemm_ratio < 30.0,
            "GEMM energy scaling should be reasonable: {:.2}x", gemm_ratio);

    // Test Pipeline energy scaling
    let pipeline_small = SoftmaxPipelineKernel::new(SoftmaxPipelineConfig {
        batch_size: 2, sequence_length: test_size / 4, num_heads: 4,
        ..SoftmaxPipelineConfig::default()
    });

    let pipeline_large = SoftmaxPipelineKernel::new(SoftmaxPipelineConfig {
        batch_size: 4, sequence_length: test_size / 2, num_heads: 8,
        ..SoftmaxPipelineConfig::default()
    });

    let pipeline_small_expected = pipeline_small.expected_energy(512);
    let pipeline_large_expected = pipeline_large.expected_energy(2048);

    let pipeline_ratio = pipeline_large_expected / pipeline_small_expected;
    assert!(pipeline_ratio > 3.0 && pipeline_ratio < 15.0,
            "Pipeline energy scaling should be reasonable: {:.2}x", pipeline_ratio);

    println!("Energy scaling ratios:");
    println!("  Conv2D: {:.2}x", energy_ratio);
    println!("  GEMM: {:.2}x", gemm_ratio);
    println!("  Pipeline: {:.2}x", pipeline_ratio);

    println!("✅ All extended kernels show reasonable energy scaling");
}

#[test]
fn test_extended_kernels_tta_advantage_consistency() {
    println!("🧪 Extended Kernels TTA Advantage Consistency");
    println!("==============================================");

    // Test multiple configurations for each kernel type
    let conv_configs = vec![
        Conv2DConfig { input_height: 16, input_width: 16, input_channels: 3, output_channels: 8, ..Conv2DConfig::default() },
        Conv2DConfig { input_height: 32, input_width: 32, input_channels: 8, output_channels: 16, ..Conv2DConfig::default() },
        Conv2DConfig { input_height: 24, input_width: 24, input_channels: 6, output_channels: 12, ..Conv2DConfig::default() },
    ];

    let mut conv_advantages = Vec::new();
    for config in conv_configs {
        let kernel = Conv2DKernel::new(config);
        conv_advantages.push(kernel.tta_advantage_factor());
    }

    let gemm_configs = vec![
        GemmConfig { m: 16, n: 16, k: 16, ..GemmConfig::default() },
        GemmConfig { m: 32, n: 32, k: 32, ..GemmConfig::default() },
        GemmConfig { m: 24, n: 24, k: 24, ..GemmConfig::default() },
    ];

    let mut gemm_advantages = Vec::new();
    for config in gemm_configs {
        let kernel = GemmKernel::new(config);
        gemm_advantages.push(kernel.tta_advantage_factor());
    }

    let pipeline_configs = vec![
        SoftmaxPipelineConfig { batch_size: 2, sequence_length: 16, num_heads: 4, ..SoftmaxPipelineConfig::default() },
        SoftmaxPipelineConfig { batch_size: 4, sequence_length: 32, num_heads: 8, ..SoftmaxPipelineConfig::default() },
        SoftmaxPipelineConfig { batch_size: 3, sequence_length: 24, num_heads: 6, ..SoftmaxPipelineConfig::default() },
    ];

    let mut pipeline_advantages = Vec::new();
    for config in pipeline_configs {
        let kernel = SoftmaxPipelineKernel::new(config);
        pipeline_advantages.push(kernel.tta_advantage_factor());
    }

    // Check consistency within each kernel type
    let conv_min = conv_advantages.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let conv_max = conv_advantages.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let conv_range = conv_max - conv_min;

    let gemm_min = gemm_advantages.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let gemm_max = gemm_advantages.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let gemm_range = gemm_max - gemm_min;

    let pipeline_min = pipeline_advantages.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let pipeline_max = pipeline_advantages.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let pipeline_range = pipeline_max - pipeline_min;

    println!("TTA Advantage ranges:");
    println!("  Conv2D: {:.2}x - {:.2}x (range: {:.2}x)", conv_min, conv_max, conv_range);
    println!("  GEMM: {:.2}x - {:.2}x (range: {:.2}x)", gemm_min, gemm_max, gemm_range);
    println!("  Pipeline: {:.2}x - {:.2}x (range: {:.2}x)", pipeline_min, pipeline_max, pipeline_range);

    // All advantages should be > 2x and reasonably consistent
    assert!(conv_min > 2.0, "Conv2D advantages should be > 2x");
    assert!(gemm_min > 2.5, "GEMM advantages should be > 2.5x");
    assert!(pipeline_min > 3.0, "Pipeline advantages should be > 3x");

    // Ranges should be reasonable (not too wide)
    assert!(conv_range / conv_min < 0.5, "Conv2D advantage range should be reasonable");
    assert!(gemm_range / gemm_min < 0.5, "GEMM advantage range should be reasonable");
    assert!(pipeline_range / pipeline_min < 0.5, "Pipeline advantage range should be reasonable");

    println!("✅ All extended kernels show consistent TTA advantages across configurations");
}

#[test]
fn test_extended_kernels_preserve_existing_functionality() {
    println!("🧪 Verifying Extended Kernels Don't Break Core Functionality");
    println!("============================================================");

    // This test ensures that adding extended kernels doesn't break existing code
    // We test that core kernels still work when extended kernels are enabled

    use tta_simulator::kernels::{
        MultiHeadAttention, AttentionConfig,
        SoftmaxKernel, SoftmaxConfig,
        SparseMatMul, SparseConfig,
    };

    // Test core attention kernel still works
    let mut attention = MultiHeadAttention::new(AttentionConfig {
        seq_length: 16,
        head_dim: 8,
        num_heads: 4,
        ..AttentionConfig::default()
    });

    let attention_input = vec![BusData::VecI8((1..=128).map(|x| (x % 100) as i8).collect())];
    let attention_result = attention.execute(&attention_input, 1);
    assert!(attention_result.is_ok(), "Core attention should still work with extended kernels");

    // Test core softmax kernel still works
    let mut softmax = SoftmaxKernel::new(SoftmaxConfig {
        vector_length: 16,
        numerical_precision: 1e-6, // Use relaxed precision like other tests
        ..SoftmaxConfig::default()
    });

    let softmax_input = vec![BusData::VecI8((1..=16).map(|x| (x % 20) as i8).collect())];
    let softmax_result = softmax.execute(&softmax_input, 1);
    if let Err(e) = &softmax_result {
        println!("Softmax error: {}", e);
    }
    assert!(softmax_result.is_ok(), "Core softmax should still work with extended kernels");

    // Test core sparse matrix kernel still works
    let mut sparse = SparseMatMul::new(SparseConfig {
        matrix_size: 8,
        sparsity_ratio: 0.3,
        ..SparseConfig::default()
    });

    let sparse_input = vec![BusData::VecI8((1..=64).map(|x| (x % 30) as i8).collect())];
    let sparse_result = sparse.execute(&sparse_input, 1);
    assert!(sparse_result.is_ok(), "Core sparse matmul should still work with extended kernels");

    println!("✅ Core functionality preserved - extended kernels are properly isolated");
}