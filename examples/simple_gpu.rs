use env_logger;
use gaugemc::{NDDualGraph, SiteIndex};
use ndarray::{Array2, Array6, Axis};
use std::iter::repeat;

fn main() -> Result<(), String> {
    env_logger::init();

    let num_replicas = 5;
    let steps_per_sample = 1;
    let num_updates = 1;

    let (t, x, y, z) = (4, 4, 4, 4);

    let init_state = Array6::zeros((num_replicas, t, x, y, z, 6));
    let mut vns = Array2::zeros((num_replicas, 50));
    vns.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut v)| {
            v.iter_mut().enumerate().for_each(|(j, v)| {
                *v = ((i + 1) * (j.pow(2))) as f32 / 8.0;
            })
        });

    let mut state = pollster::block_on(gaugemc::GPUBackend::new_async(
        SiteIndex::new(t, x, y, z),
        vns,
        Some(init_state),
        None,
        None,
    ))?;

    for _ in 0..steps_per_sample {
        for _ in 0..num_updates {
            NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                state.run_local_sweep(&dims, leftover, offset);
            })
        }
        state.run_global_sweep(None);
        state.run_pcg_rotate();
        state.get_edges_with_violations()?;
    }

    // TODO
    let windings = state.get_winding_nums_gpu(None)?;
    println!("{:?}", windings);
    println!("{:?}", state.get_winding_nums(None)?);

    assert_eq!(windings, state.get_winding_nums(None)?);

    // for _ in 0..4 {
    //     let energies_gpu = state.get_energy(Some(false))?;
    //     let energies_state = state.get_energy(Some(true))?;
    //
    //     println!(
    //         "{:?}\t{:?}",
    //         energies_state.as_slice(),
    //         energies_gpu.as_slice()
    //     );
    //     assert_eq!(energies_gpu, energies_state);
    //
    //     state.swap_replica_potentials(false, repeat(true));
    // }

    //
    // let winding_nums = state.get_winding_nums()?;
    // println!("{:?}", winding_nums);
    //
    // for _ in 0..steps_per_sample {
    //     for _ in 0..num_updates {
    //         NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
    //             let leftover = NDDualGraph::get_leftover_dim(&dims);
    //             state.run_local_sweep(&dims, leftover, offset);
    //         })
    //     }
    //     state.run_global_sweep();
    //     state.run_pcg_rotate();
    // }
    //
    // let energies = state.get_energy()?;
    // println!("{:?}", energies);
    //
    // let winding_nums = state.get_winding_nums()?;
    // println!("{:?}", winding_nums);
    //
    // assert_eq!(state.get_edges_with_violations()?.len(), 0);

    // // state.run_global_sweep();
    // state.run_local_sweep(
    //     &[Dimension::T, Dimension::X, Dimension::Y],
    //     Dimension::Z,
    //     false,
    // );
    // let read_state = state.get_state();
    //
    // let max_int = read_state.iter().cloned().max().unwrap_or(0) as usize;
    // let mut entries = HashMap::new();
    //
    // for (i, s) in read_state.iter().cloned().enumerate() {
    //     let original_index = i;
    //     let r_index = i / (t * x * y * z * 6);
    //     let i = i % (t * x * y * z * 6);
    //     let t_index = i / (x * y * z * 6);
    //     let i = i % (x * y * z * 6);
    //     let x_index = i / (y * z * 6);
    //     let i = i % (y * z * 6);
    //     let y_index = i / (z * 6);
    //     let i = i % (z * 6);
    //     let z_index = i / (6);
    //     let i = i % 6;
    //     let p_index = i;
    //     println!(
    //         "{} -> {:?}:\t{}",
    //         original_index,
    //         (r_index, t_index, x_index, y_index, z_index, p_index),
    //         s
    //     );
    //     if !entries.contains_key(&s) {
    //         entries.insert(s, vec![]);
    //     }
    //     entries
    //         .get_mut(&s)
    //         .unwrap()
    //         .push((r_index, t_index, x_index, y_index, z_index, p_index));
    // }
    // // let arr = state.get_state_array();
    // // arr.indexed_iter()
    // //     .filter(|(_, s)| **s > 0)
    // //     .for_each(|(indx, s)| println!("{:?}", indx));
    //
    // println!(
    //     "{}/{}",
    //     read_state.iter().cloned().filter(|c| !c.is_zero()).count(),
    //     read_state.len()
    // );
    //
    // for (i, v) in entries.iter() {
    //     println!("{}\t{}", i, v.len());
    // }

    Ok(())
}
