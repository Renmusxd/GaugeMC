use std::fs::File;
use log::info;
use gaugemc::{CudaBackend, CudaError, DualState, SiteIndex};
use ndarray::{Array1, Array2, Array3, Array6, Axis, s};
use num_traits::Zero;
use ndarray_npy::NpzWriter;

fn main() -> Result<(), CudaError> {
    env_logger::init();

    let d: usize = 16;
    let num_replicas = 512;
    let (t, x, y, z) = (d, d, d, d);
    let mut vns = Array2::zeros((num_replicas, 32));
    let k = 1.0 / 1.28746;

    vns.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut v)| {
            v.iter_mut().enumerate().for_each(|(j, v)| {
                *v = k * (j.pow(2) as f32);
            })
        });
    let mut plaquettes = Array6::zeros((num_replicas, t, x, y, z, 6));
    plaquettes.slice_mut(s![..,0,2..d-2,2..d-2,d/2,3]).iter_mut().for_each(|x| {
        *x = 1;
    });
    let plaquettes = Some(plaquettes);

    let mut state = CudaBackend::new(
        SiteIndex::new(t, x, y, z),
        vns,
        plaquettes.map(|plaquettes| DualState::new_plaquettes(plaquettes)),
        None,
        None,
        None,
    )?;

    let num_counts = 128;
    let num_steps = 16;
    for i in 0..num_counts {
        info!("Computing count {}/{}", i, num_counts);
        for _ in 0..num_steps {
            state.run_local_update_sweep()?;
        }
    }
    let foam = state.get_plaquettes()?;
    let edges = state.get_edge_violations()?;

    let mut npz = NpzWriter::new(File::create("foam.npz").expect("Could not create file."));
    npz.add_array("plaquettes", &foam).expect("Could not add array to file.");
    npz.add_array("edges", &edges).expect("Could not add array to file.");
    npz.finish().expect("Could not write to file.");

    Ok(())
}