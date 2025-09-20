// Debug test for energy scaling
use tta_simulator::kernels::{MultiHeadAttention, AttentionConfig};
use tta_simulator::tta::BusData;

fn main() {
    println!("🔍 Energy Scaling Debug Test");

    for &size in &[8, 16, 32] {
        println!("\n--- SIZE {} ---", size);

        let mut attention = MultiHeadAttention::new(AttentionConfig {
            seq_length: size,
            head_dim: 16,  // Fixed head dim
            num_heads: 4,  // Fixed heads
            energy_per_qkv_projection: 10.0,    // Simple values for debugging
            energy_per_attention_compute: 5.0,
            energy_per_output_projection: 8.0,
        });

        let input_size = size * 16;  // size sequences * 16 head_dim
        let input_data = vec![BusData::VecI8((1..=input_size).map(|x| x as i8).collect())];

        println!("Input size: {}", input_size);

        // Before execution
        println!("Energy before: {}", attention.energy_consumed());

        // Calculate expected manually
        let actual_seq_length = input_size / 16;  // Should equal size
        let seq_length_ratio = actual_seq_length as f64 / size as f64;  // Should be 1.0

        println!("Actual seq length: {}, ratio: {:.2}", actual_seq_length, seq_length_ratio);

        let expected_qkv = 10.0 * 4.0 * seq_length_ratio;  // 40.0 * 1.0 = 40.0
        let expected_attn = 5.0 * 4.0 * seq_length_ratio * seq_length_ratio;  // 20.0 * 1.0 = 20.0
        let expected_out = 8.0 * seq_length_ratio;  // 8.0 * 1.0 = 8.0
        let expected_total = expected_qkv + expected_attn + expected_out;  // 68.0

        println!("Expected: QKV={:.1}, Attn={:.1}, Out={:.1}, Total={:.1}",
                expected_qkv, expected_attn, expected_out, expected_total);

        // Execute
        let result = attention.execute(&input_data, 1);
        println!("Execution result: {:?}", result.is_ok());

        // After execution
        println!("Energy after: {}", attention.energy_consumed());
        println!("Expected by method: {}", attention.expected_energy(input_size));
    }
}