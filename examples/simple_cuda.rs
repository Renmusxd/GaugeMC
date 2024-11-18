use gaugemc::{CudaBackend, CudaError, DualState, SiteIndex};
use ndarray::{Array2, Array6, Axis};

fn main() -> Result<(), CudaError> {
    env_logger::init();

    let num_replicas = 5;
    let steps_per_sample = 16;
    let num_updates = 16;
    let nvns = 3;

    let (t, x, y, z) = (8, 8, 8, 8);

    let init_state = Array6::zeros((num_replicas, t, x, y, z, 6));
    let mut vns = Array2::zeros((num_replicas, nvns));
    vns.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut v)| {
            v.iter_mut().enumerate().for_each(|(j, v)| {
                // *v = ((i + 1) * (j.pow(2))) as f32 / 2.0;
                *v = match j {
                    0 => 0.0,
                    1 => 0.5,
                    _ => 1000.0
                }
            })
        });

    let mut state = CudaBackend::new(
        SiteIndex::new(t, x, y, z),
        vns,
        Some(DualState::new_plaquettes(init_state)),
        None,
        None,
        None,
    )?;

    for _ in 0..steps_per_sample {
        for _ in 0..num_updates {
            state.run_local_update_sweep()?;
            state.run_global_update_sweep()?;
        }
        let x = state.get_plaquette_counts()?;
        println!("{:?}", x);
    }

    Ok(())
}
