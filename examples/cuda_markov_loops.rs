use std::fs::File;
use log::info;
use gaugemc::{CudaBackend, CudaError, DualState, SiteIndex};
use ndarray::{Array1, Array2, Array3, Array6, Axis, s};
use num_traits::Zero;
use ndarray_npy::NpzWriter;

fn main() -> Result<(), CudaError> {
    env_logger::init();

    let d: usize = 8;
    let num_replicas = d * d.pow(2) + 1;
    let (t, x, y, z) = (d, d, d, d);
    let mut vns = Array2::zeros((num_replicas, 32));
    let k = 0.6;
    vns.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut v)| {
            v.iter_mut().enumerate().for_each(|(j, v)| {
                *v = k * (j.pow(2) as f32);
            })
        });
    let mut state = CudaBackend::new(
        SiteIndex::new(t, x, y, z),
        vns,
        None,
        None,
        None,
        None,
    )?;

    state.initialize_wilson_loops_for_probs_incremental_square((0..num_replicas).map(|x| x).collect(), 0)?;

    let num_counts = 1024;
    let num_steps = 32;
    for i in 0..num_counts {
        info!("Computing count {}/{}", i, num_counts);
        for _ in 0..num_steps {
            state.run_local_update_sweep()?;
            state.run_plane_shift(0)?;
        }

        state.calculate_wilson_loop_transition_probs()?;
    }
    let transition_probs = state.get_wilson_loop_transition_probs()?;

    let mut distribution = Array1::zeros((num_replicas, ));
    let mut free_energies = Array1::zeros((num_replicas, ));
    let mut acc = 1.0;
    free_energies[0] = 0.0;
    distribution[0] = 1.0;
    for i in 1..num_replicas {
        let new_logp = -free_energies[i - 1] + (transition_probs[[i - 1, 1]] as f64).ln()
            - (transition_probs[[i, 0]] as f64).ln();
        free_energies[i] = -new_logp;
        distribution[i] = new_logp.exp();
        acc += distribution[i];
    }
    distribution.iter_mut().for_each(|x| *x /= acc);

    let mut npz = NpzWriter::new(File::create("output_distribution.npz").expect("Could not create file."));
    npz.add_array("transition_probs", &transition_probs).expect("Could not add array to file.");
    npz.add_array("free_energies", &free_energies).expect("Could not add array to file.");
    npz.add_array("sample_probs", &distribution).expect("Could not add array to file.");
    npz.finish().expect("Could not write to file.");

    Ok(())
}