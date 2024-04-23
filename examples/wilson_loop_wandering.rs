use std::fs::File;
use gaugemc::{CudaBackend, CudaError, DualState, SiteIndex};
use ndarray::{Array1, Array2, Array6, Axis};
use ndarray_npy::NpzWriter;
use ndarray_rand::rand;
use ndarray_rand::rand::prelude::SliceRandom;

fn main() -> Result<(), CudaError> {
    env_logger::init();

    let num_replicas = 64;
    let klow = 0.25;
    let khigh = 5.0;
    let ks = if num_replicas > 1 {
        (0..num_replicas).map(|i| (khigh - klow) * (i as f32 / (num_replicas - 1) as f32) + klow).collect()
    } else { vec![(khigh + klow) / 2.0] };
    let ks = Array1::from(ks);

    let (t, x, y, z) = (6, 6, 6, 6);
    let mut init_state = Array6::zeros((num_replicas, t, x, y, z, 6));
    let mut vns = Array2::zeros((num_replicas, 32));
    vns.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut v)| {
            v.iter_mut().enumerate().for_each(|(j, v)| {
                *v = (j.pow(2) as f32) * ks[i];
            })
        });

    let p = 0;

    for r in 0..num_replicas {
        init_state[[r, 0, 0, 0, 0, p]] = 1;
    }

    let mut state = CudaBackend::new(
        SiteIndex::new(t, x, y, z),
        vns.clone(),
        Some(DualState::new_plaquettes(init_state.clone())),
        None,
        None,
        None,
    )?;

    let n = 1024;
    let mn = 32;
    let ln = 0;

    let mut rng = rand::thread_rng();
    let mut updates = (0..mn + ln).map(|i| i < mn).collect::<Vec<_>>();

    let mut output = Array2::zeros((n, num_replicas));
    output.axis_iter_mut(Axis(0)).try_for_each(|mut arr| -> Result<(), CudaError>{
        updates.shuffle(&mut rng);
        for update in &updates {
            if *update {
                for p in 0..6 {
                    for offset in 0..4 {
                        state.run_single_matter_update_single(p as u16, offset)?;
                    }
                }
            } else {
                state.run_local_update_sweep()?;
            }
        }

        let edge_violations = state.get_edge_violations()?;
        let total_per_replica = edge_violations.axis_iter(Axis(0)).map(|x| x.into_iter().map(|x| x.abs()).sum::<i32>());
        arr.iter_mut().zip(total_per_replica).for_each(|(a, b)| {
            *a = b;
        });
        Ok(())
    })?;

    let mut npz = NpzWriter::new(File::create("output_loops.npz").expect("Could not create file."));

    npz.add_array("perimeters", &output).expect("Could not add array to file.");
    npz.add_array("ks", &ks).expect("Could not add array to file.");
    npz.finish().expect("Could not write to file.");


    Ok(())
}