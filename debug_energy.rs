use tta_simulator::kernels::{MultiHeadAttention, AttentionConfig};
use tta_simulator::tta::BusData;

fn main() {
    println!("🔍 Direct Energy Scaling Debug");
    
    let sizes = vec![8, 16, 32, 64];
    
    for &size in &sizes {
        let mut attention = MultiHeadAttention::new(AttentionConfig {
            seq_length: size,
            head_dim: 16,
            num_heads: 4,
            ..AttentionConfig::default()
        });

        // Create input data: size * 16 elements (size sequences of 16 dims each)
        let input_data = vec![BusData::VecI8((1..=128).cycle().take(size * 16).map(|x| x as i8).collect())];
        
        println!("Size {}: input length = {}", size, size * 16);
        
        let _ = attention.execute(&input_data, 1);
        
        let actual = attention.energy_consumed();
        let expected = attention.expected_energy(size * 16);
        
        println!("  Actual: {:.1}, Expected: {:.1}, Ratio: {:.2}", actual, expected, actual/expected);
    }
}
