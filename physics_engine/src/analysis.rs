use rand::Rng;
use crate::substrate::ComputationalSubstrate;
use crate::physics::Universe;

#[derive(Debug, Clone)]
pub enum DiscretizationType {
    Discrete { n_states: usize, state_values: Vec<f64>, hysteresis: bool },
    Continuous { range: (f64, f64) },
}

pub struct EmergentAnalysis;

impl EmergentAnalysis {
    pub fn new() -> Self { Self }

/// Crude info-preservation score in [0,1]
pub fn measure_information_preservation(
    &self,
    substrate: &ComputationalSubstrate,
    universe: &mut Universe,
) -> f64 {
    let mut rng = rand::thread_rng();
    let trials = 64usize;
    let mut s = substrate.clone();

    let mut se = 0.0f64;
    let mut cnt = 0usize;

    for _ in 0..trials {
        let input = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        let out = s.transform(&input, universe);

        for (o, i) in out.iter().zip(input.iter()) {
            let err = o - i;
            se += err * err;
            cnt += 1;
        }
    }

    if cnt == 0 { return 0.0; }
    let mse = se / (cnt as f64);
    let score = (1.0 - mse).clamp(0.0, 1.0);
    score
}

/// Stability/dwell metric
pub fn measure_stability(
    &self,
    substrate: &ComputationalSubstrate,
    universe: &mut Universe,
) -> f64 {
    let mut rng = rand::thread_rng();
    let mut s = substrate.clone();
    let mut u = universe.forked(0xC0FFEE);

    let steps = 100;
    let mut states: Vec<i32> = Vec::new();
    
    for _ in 0..steps {
        let input = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        let out = s.transform(&input, &mut u);
        
        let y = out.iter().copied().sum::<f64>() / (out.len().max(1) as f64);
        let state = self.map_to_discrete_state(y);
        states.push(state);
    }

    if states.len() < 2 {
        return 1.0;
    }

    let mut dwells = Vec::new();
    let mut current_state = states[0];
    let mut current_dwell = 1;

    for &state in &states[1..] {
        if state == current_state {
            current_dwell += 1;
        } else {
            dwells.push(current_dwell);
            current_state = state;
            current_dwell = 1;
        }
    }
    dwells.push(current_dwell);

    let avg_dwell = dwells.iter().sum::<usize>() as f64 / dwells.len() as f64;
    (avg_dwell / 10.0).min(1.0)
}

fn map_to_discrete_state(&self, value: f64) -> i32 {
    if value < -0.33 {
        -1
    } else if value > 0.33 {
        1
    } else {
        0
    }
}



    pub fn detect_hysteresis_artifact(&self, substrate: &ComputationalSubstrate) -> bool {
    let mut s = substrate.clone();
    let mut u = Universe::new(0.0, 0.0).with_energy(0.5, 1.0);

    // Test rising and falling responses
    let rising_inputs = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let falling_inputs = vec![1.0, 0.5, 0.0, -0.5, -1.0];

    let mut rising_outputs = Vec::new();
    let mut falling_outputs = Vec::new();

    // Collect outputs for rising sequence
    for &x in &rising_inputs {
        let out = s.transform(&[x, x, x], &mut u);
        rising_outputs.push(out[0]);
    }

    // Reset and collect for falling sequence
    s = substrate.clone();
    for &x in &falling_inputs {
        let out = s.transform(&[x, x, x], &mut u);
        falling_outputs.push(out[0]);
    }

    // Check for hysteresis signature
    for i in 0..rising_inputs.len() {
        let rising_in = rising_inputs[i];
        for j in 0..falling_inputs.len() {
            let falling_in = falling_inputs[j];
            if (rising_in - falling_in).abs() < 0.01 {
                let out_diff = (rising_outputs[i] - falling_outputs[j]).abs();
                if out_diff > 0.1 {
                    return true;
                }
            }
        }
    }
    false
}

    /// Robust discretization detector with artifact filtering
    pub fn detect_discretization(&self, substrate: &ComputationalSubstrate) -> DiscretizationType {
        let mut s = substrate.clone();
        let mut u = Universe::new(0.0, 0.0).with_energy(0.5, 1.0); // noiseless probe

        // Dense input grid for comprehensive sampling
        let xs_up: Vec<f64> = (-82..=82).map(|k| k as f64 / 55.0).collect();   // ~165 pts
        let mut xs_down = xs_up.clone();
        xs_down.reverse();

        // Collect outputs from up and down sweeps
        let ys_up = self.eval_grid(&mut s, &xs_up, &mut u);
        s = substrate.clone(); // Reset state
        let ys_down = self.eval_grid(&mut s, &xs_down, &mut u);

        // Cluster up-sweep outputs
        let mut centers = self.cluster_values(&ys_up);

        // Snap to nice values and merge close ones
        self.snap_round(&mut centers);
        self.merge_close(&mut centers, 0.10);

        // Filter by occupancy (must have at least 5% of samples)
        let min_occ = (ys_up.len() as f64 * 0.05).ceil() as usize;
        let kept = self.filter_by_occupancy(&ys_up, &centers, min_occ);
        let mut final_centers: Vec<f64> = centers.into_iter().enumerate()
            .filter(|(i, _)| kept[*i])
            .map(|(_, c)| c)
            .collect();
        final_centers.sort_by(|a,b| a.partial_cmp(b).unwrap());

        // Check for hysteresis artifacts
        let hyst = self.detect_hysteresis(&xs_up, &ys_up, &xs_down, &ys_down, &final_centers);

        // Classify as discrete or continuous
        if final_centers.len() >= 2 && final_centers.len() <= 10 {
            DiscretizationType::Discrete { 
                n_states: final_centers.len(), 
                state_values: final_centers, 
                hysteresis: hyst 
            }
        } else if final_centers.len() == 1 {
            DiscretizationType::Discrete { 
                n_states: 1, 
                state_values: final_centers, 
                hysteresis: hyst 
            }
        } else {
            let min = ys_up.iter().copied().fold(f64::INFINITY, f64::min);
            let max = ys_up.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            DiscretizationType::Continuous { range: (min, max) }
        }
    }

    fn eval_grid(&self, s: &mut ComputationalSubstrate, xs: &[f64], u: &mut Universe) -> Vec<f64> {
        let mut out = Vec::with_capacity(xs.len());
        for &x in xs {
            let output = s.transform(&[x, x, x], u);
            // Clamp to reasonable range and take first output
            let y = output[0].clamp(-2.0, 2.0);
            out.push(y);
        }
        out
    }

    fn cluster_values(&self, ys: &[f64]) -> Vec<f64> {
        // 1D agglomerative clustering with minimum separation
        let min_sep = 0.15;
        let mut vals = ys.to_vec();
        vals.sort_by(|a,b| a.partial_cmp(b).unwrap());
        
        let mut clusters: Vec<Vec<f64>> = Vec::new();
        for v in vals {
            if clusters.is_empty() {
                clusters.push(vec![v]);
            } else {
                let last = clusters.last_mut().unwrap();
                if (v - last[last.len()-1]).abs() < min_sep {
                    last.push(v);
                } else {
                    clusters.push(vec![v]);
                }
            }
        }
        clusters.into_iter().map(|c| mean(&c)).collect()
    }

    fn snap_round(&self, centers: &mut [f64]) {
        for c in centers.iter_mut() {
            // Snap to nice symmetric values
            let nice = [-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0];
            let mut best = *c;
            let mut bestd = f64::INFINITY;
            for n in nice {
                let d = (n - *c).abs();
                if d < bestd && d < 0.025 { 
                    best = n; 
                    bestd = d; 
                }
            }
            *c = best;
        }
    }

    fn merge_close(&self, centers: &mut Vec<f64>, tol: f64) {
        centers.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let mut merged: Vec<f64> = Vec::new();
        for c in centers.iter().cloned() {
            if let Some(last) = merged.last_mut() {
                if (c - *last).abs() <= tol {
                    *last = (*last + c) * 0.5;
                } else {
                    merged.push(c);
                }
            } else {
                merged.push(c);
            }
        }
        *centers = merged;
    }

    fn filter_by_occupancy(&self, ys: &[f64], centers: &[f64], min_occ: usize) -> Vec<bool> {
        let mut counts = vec![0usize; centers.len()];
        for &y in ys {
            if let Some((idx, _)) = self.nearest_center(y, centers) {
                counts[idx] += 1;
            }
        }
        counts.into_iter().map(|c| c >= min_occ).collect()
    }

    fn detect_hysteresis(&self, xs_up: &[f64], ys_up: &[f64], xs_down: &[f64], ys_down: &[f64], centers: &[f64]) -> bool {
        // Map outputs to nearest center indices
        let up_idx: Vec<usize> = ys_up.iter()
            .map(|&y| self.nearest_center(y, centers).map(|(i,_)| i).unwrap_or(usize::MAX))
            .collect();
        let down_idx: Vec<usize> = ys_down.iter()
            .map(|&y| self.nearest_center(y, centers).map(|(i,_)| i).unwrap_or(usize::MAX))
            .collect();

        // Count agreement between up and down sweeps
        let mut agree = 0usize;
        let mut total = 0usize;
        for (u, d) in up_idx.iter().zip(down_idx.iter()) {
            if *u != usize::MAX && *d != usize::MAX {
                total += 1;
                if u == d { agree += 1; }
            }
        }
        
        // Flag hysteresis if less than 92% agreement
        total > 0 && (agree as f64 / total as f64) < 0.92
    }

    fn nearest_center(&self, y: f64, centers: &[f64]) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, &c) in centers.iter().enumerate() {
            let d = (y - c).abs();
            match best {
                None => best = Some((i, d)),
                Some((_, bd)) if d < bd => best = Some((i, d)),
                _ => {}
            }
        }
        best
    }
}

fn mean(xs: &[f64]) -> f64 { 
    xs.iter().copied().sum::<f64>() / (xs.len().max(1) as f64) 
}