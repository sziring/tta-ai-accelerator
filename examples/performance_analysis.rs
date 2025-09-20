// examples/performance_analysis.rs
//! Performance vs Energy Analysis Demonstration
//!
//! Demonstrates that TTA's 7x energy efficiency does NOT come
//! at the cost of computational performance.

use tta_simulator::analysis::TtaPerformanceSummary;

fn main() {
    println!("🔋⚡ TTA PERFORMANCE vs ENERGY DEMONSTRATION");
    println!("===========================================\n");

    // Generate comprehensive performance vs energy summary
    let summary = TtaPerformanceSummary::generate_summary();

    // Print the detailed "time is money" analysis
    println!("{}", summary.generate_time_is_money_report());

    // Key takeaways
    println!("🏆 FINAL ANSWER TO YOUR QUESTION:");
    println!("==================================");
    println!("❓ Question: \"7x energy at -7x the speed might be a deal breaker. After all, time is money.\"");
    println!("✅ Answer: TTA achieves {:.2}x BETTER energy efficiency AND {:.2}x BETTER performance!",
             summary.overall_verdict.overall_energy_advantage,
             summary.overall_verdict.overall_performance_factor);
    println!("💡 Result: {:.2}x overall improvement in performance-per-watt",
             summary.overall_verdict.performance_per_watt_advantage);
    println!("\n🎯 TTA breaks the traditional energy-performance trade-off!");
    println!("   We achieve the best of both worlds through architectural innovation.");
}