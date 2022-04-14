use gaugemc::NDDualGraph;
use ndarray_rand::rand::rngs::SmallRng;

fn main() -> Result<(), String> {
    let mut graph = NDDualGraph::new(4, 4, 4, 4, (1..100).map(|i| 0.5 * (i as f64).powi(2)))?;

    let mut rng: Option<SmallRng> = None;

    for _ in 0..100 {
        for _ in 0..10 {
            graph.local_update_sweep(rng.as_mut());
        }
        graph.global_update_sweep(rng.as_mut());
    }
    Ok(())
}
