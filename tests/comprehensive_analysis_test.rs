// tests/comprehensive_analysis_test.rs
//! Comprehensive Analysis Framework Integration Test
//!
//! Tests the complete end-to-end analysis pipeline including
//! transformer models, competitive benchmarking, and scaling analysis.

use tta_simulator::analysis::{
    ModelAnalysis,
    transformer_models::TransformerConfig,
    BenchmarkSuite,
    EfficiencyTrends,
    MetricsCollector,
};

#[test]
fn test_complete_research_pipeline() {
    println!("🧪 Testing Complete Research Analysis Pipeline");
    println!("==============================================");

    // Step 1: End-to-End Transformer Model Analysis
    println!("\n📊 Step 1: Transformer Model Analysis");
    let mut model_analysis = ModelAnalysis::new();

    // Add various transformer configurations for comprehensive analysis
    model_analysis.add_model(TransformerConfig::mobile_efficient());
    model_analysis.add_model(TransformerConfig::bert_base_detailed());
    model_analysis.add_model(TransformerConfig::gpt2_medium());

    let model_result = model_analysis.run_comprehensive_analysis();
    if let Err(e) = &model_result {
        println!("Error in model analysis: {}", e);
    }
    assert!(model_result.is_ok(), "Transformer analysis should succeed");
    assert_eq!(model_analysis.analysis_results.len(), 3);

    // Step 2: Competitive Benchmarking Analysis
    println!("\n🏁 Step 2: Competitive Benchmarking");
    let mut benchmark_suite = BenchmarkSuite::new();

    // Add our TTA results to the benchmarking suite
    for metrics in &model_analysis.analysis_results {
        benchmark_suite.add_tta_results(metrics.clone());
    }

    let competitive_result = benchmark_suite.run_competitive_analysis();
    assert!(competitive_result.is_ok(), "Competitive analysis should succeed");

    let competitive_analysis = competitive_result.unwrap();
    assert!(competitive_analysis.total_comparisons > 0);
    assert!(competitive_analysis.average_advantage > 0.0);

    // Step 3: Scaling Analysis
    println!("\n📈 Step 3: Scaling Trends Analysis");
    let mut scaling_analysis = EfficiencyTrends::new();
    let scaling_result = scaling_analysis.analyze_scaling_trends();
    assert!(scaling_result.is_ok(), "Scaling analysis should succeed");
    assert_eq!(scaling_analysis.projections.len(), 3);

    // Step 4: Publication Metrics Generation
    println!("\n📝 Step 4: Publication Report Generation");
    let mut metrics_collector = MetricsCollector::new();

    // Collect all analysis results
    for metrics in &model_analysis.analysis_results {
        metrics_collector.model_results.push(metrics.clone());
    }
    metrics_collector.competitive_results = Some(competitive_analysis);
    metrics_collector.scaling_results = Some(scaling_analysis);

    let publication_report = metrics_collector.generate_publication_report();

    // Validate publication report content
    assert!(!publication_report.executive_summary.key_achievements.is_empty());
    assert!(!publication_report.technical_results.kernel_performance.is_empty());
    assert!(publication_report.technical_results.end_to_end_efficiency > 2.0);

    println!("\n🎉 Complete Research Pipeline Summary:");
    println!("====================================");
    println!("Models Analyzed: {}", model_analysis.analysis_results.len());
    println!("Competitive Comparisons: {}", publication_report.competitive_positioning.total_comparisons);
    println!("Scaling Projections: {}", publication_report.scaling_projections.projections.len());
    println!("Average TTA Advantage: {:.2}x", publication_report.technical_results.end_to_end_efficiency);

    // Key assertions for research validation
    assert!(publication_report.technical_results.end_to_end_efficiency > 3.0,
            "Should demonstrate significant efficiency improvements");
    assert!(publication_report.technical_results.precision_preservation >= 0.99,
            "Should preserve computational precision");
    assert!(publication_report.technical_results.robustness_validation.success_rate > 0.8,
            "Should show high robustness success rate");

    println!("✅ Complete research pipeline validation successful!");
}

#[test]
fn test_publication_ready_metrics() {
    println!("📊 Testing Publication-Ready Metrics Generation");
    println!("===============================================");

    let mut metrics_collector = MetricsCollector::new();
    let report = metrics_collector.generate_publication_report();

    // Validate executive summary
    assert!(!report.executive_summary.key_achievements.is_empty());
    assert!(!report.executive_summary.significance_statement.is_empty());
    assert!(!report.executive_summary.competitive_advantages.is_empty());

    // Validate technical results
    assert!(!report.technical_results.kernel_performance.is_empty());
    assert!(report.technical_results.end_to_end_efficiency > 0.0);
    assert!(report.technical_results.precision_preservation >= 0.99);

    // Validate methodology summary
    assert!(!report.methodology_summary.simulation_framework.is_empty());
    assert!(!report.methodology_summary.validation_approach.is_empty());
    assert!(!report.methodology_summary.limitations.is_empty());

    // Validate visualization data
    assert!(!report.visualization_data.efficiency_comparison_chart.data_series.is_empty());
    assert!(!report.visualization_data.energy_breakdown_pie.data_series.is_empty());

    println!("📈 Key Publication Metrics:");
    println!("  End-to-end efficiency: {:.2}x", report.technical_results.end_to_end_efficiency);
    println!("  Precision preservation: {:.1}%", report.technical_results.precision_preservation * 100.0);
    println!("  Robustness success rate: {:.1}%", report.technical_results.robustness_validation.success_rate * 100.0);
    println!("  Competitive advantage: {:.2}x", report.competitive_positioning.average_advantage);

    println!("✅ Publication metrics validation successful!");
}

#[test]
fn test_scaling_projections_validity() {
    println!("📈 Testing Scaling Projections Validity");
    println!("======================================");

    let mut scaling_analysis = EfficiencyTrends::new();
    let result = scaling_analysis.analyze_scaling_trends();
    assert!(result.is_ok());

    // Validate scaling studies have reasonable results
    assert!(scaling_analysis.model_size_scaling.scaling_law.r_squared >= 0.0);
    assert!(scaling_analysis.batch_size_scaling.scaling_law.r_squared >= 0.0);
    assert!(scaling_analysis.sequence_length_scaling.scaling_law.r_squared >= 0.0);
    assert!(scaling_analysis.technology_scaling.scaling_law.r_squared >= 0.0);

    // Validate projections are reasonable
    for projection in &scaling_analysis.projections {
        assert!(projection.projected_metrics.tta_advantage_factor > 1.0);
        assert!(projection.projected_metrics.tta_advantage_factor < 20.0); // Sanity check
        assert!(projection.projected_metrics.energy_per_token > 0.0);
        assert!(projection.confidence_interval.0 < projection.confidence_interval.1);

        println!("📊 {}: {:.2}x advantage",
                 projection.scenario_name,
                 projection.projected_metrics.tta_advantage_factor);
    }

    println!("✅ Scaling projections validation successful!");
}

#[test]
fn test_competitive_database_coverage() {
    println!("🏁 Testing Competitive Database Coverage");
    println!("=======================================");

    let benchmark_suite = BenchmarkSuite::new();

    // Verify comprehensive accelerator coverage
    let accelerators = &benchmark_suite.accelerator_database;
    assert!(accelerators.len() >= 6, "Should have comprehensive accelerator database");

    // Check for major vendor coverage
    let has_nvidia = accelerators.iter().any(|acc| acc.vendor == "NVIDIA");
    let has_google = accelerators.iter().any(|acc| acc.vendor == "Google");
    let has_academic = accelerators.iter().any(|acc| acc.vendor == "MIT" || acc.vendor == "NVIDIA Research");
    let has_mobile = accelerators.iter().any(|acc| acc.vendor == "Apple");

    assert!(has_nvidia, "Should include NVIDIA GPU accelerators");
    assert!(has_google, "Should include Google TPU");
    assert!(has_academic, "Should include academic research accelerators");
    assert!(has_mobile, "Should include mobile/edge accelerators");

    // Verify reasonable specifications
    for accelerator in accelerators {
        assert!(accelerator.peak_tops > 0.0);
        assert!(accelerator.power_consumption_w > 0.0);
        assert!(accelerator.energy_efficiency_tops_w > 0.0);
        assert!(accelerator.process_node_nm >= 4 && accelerator.process_node_nm <= 180);
        assert!(accelerator.year >= 2015 && accelerator.year <= 2025);

        println!("🔍 {}: {:.1} TOPS, {:.2} TOPS/W, {}nm",
                 accelerator.name,
                 accelerator.peak_tops,
                 accelerator.energy_efficiency_tops_w,
                 accelerator.process_node_nm);
    }

    println!("✅ Competitive database coverage validation successful!");
}

#[test]
fn test_research_reproducibility() {
    println!("🔄 Testing Research Reproducibility");
    println!("==================================");

    // Run the same analysis multiple times to ensure consistent results
    let mut results = Vec::new();

    for run in 1..=3 {
        println!("  Run {}/3", run);

        let mut model_analysis = ModelAnalysis::new();
        model_analysis.add_model(TransformerConfig::mobile_efficient());

        let analysis_result = model_analysis.run_comprehensive_analysis();
        assert!(analysis_result.is_ok());

        if !model_analysis.analysis_results.is_empty() {
            results.push(model_analysis.analysis_results[0].tta_advantage_factor);
        }
    }

    // Verify results are consistent (deterministic due to fixed seed)
    assert_eq!(results.len(), 3);
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "Results should be reproducible");
    }

    println!("📊 Consistent TTA advantage across runs: {:.2}x", results[0]);
    println!("✅ Research reproducibility validation successful!");
}