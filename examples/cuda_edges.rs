use std::fs::File;
use gaugemc::{CudaBackend, CudaError, DualState, SiteIndex};
use ndarray::{Array2, Array3, Array6, Axis, s};
use num_traits::Zero;
use ndarray_npy::NpzWriter;

fn main() -> Result<(), CudaError> {
    env_logger::init();

    let num_replicas = 1;

    let (t, x, y, z) = (128, 128, 2, 2);
    let mut init_state = Array6::zeros((num_replicas, t, x, y, z, 6));
    let mut vns = Array2::zeros((num_replicas, 32));
    vns.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut v)| {
            v.iter_mut().enumerate().for_each(|(j, v)| {
                *v = ((i + 1) * (j.pow(2))) as f32 / 0.75;
            })
        });

    let p = 0;
    println!("=======");
    init_state[[0, t / 2, x / 2, 0, 0, p]] = 1;

    let mut state = CudaBackend::new(
        SiteIndex::new(t, x, y, z),
        vns.clone(),
        Some(DualState::new_plaquettes(init_state.clone())),
        None,
        None,
        None,
    )?;

    let violations = state.get_edge_violations()?;

    ndarray::Zip::indexed(&violations).for_each(|indx, v| {
        if !v.is_zero() {
            println!("{:?}\t{}", indx, v);
        }
    });

    println!("=======");
    let n = 100;
    let nn = 10;
    let mut output = Array3::zeros((n, t, x));
    output.axis_iter_mut(Axis(0)).try_for_each(|mut arr| -> Result<(), CudaError>{
        for _ in 0..nn {
            for offset in 0..4 {
                state.run_single_matter_update_single(p as u16, offset)?;
            }
        }
        let slice = state.get_plaquettes()?;
        let subslice = slice.slice(s![0, .., .., 0, 0, 0]);
        arr.iter_mut().zip(subslice.iter()).for_each(|(a, b)| {
            *a = *b;
        });
        Ok(())
    })?;

    let mut npz = NpzWriter::new(File::create("output_slices.npz").expect("Could not create file."));

    npz.add_array("slices", &output).expect("Could not add array to file.");
    npz.finish().expect("Could not write to file.");

    Ok(())
}
