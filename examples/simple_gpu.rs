use gaugemc::NDDualGraph;
use ndarray::Array6;

fn main() -> Result<(), String> {
    env_logger::init();

    let num_replicas = 1;
    let steps_per_sample = 10;
    let num_updates = 10;

    let (t, x, y, z) = (8, 8, 8, 8);

    let mut init_state = Array6::zeros((num_replicas, t, x, y, z, 6));
    *init_state.get_mut([0, 0, 0, 0, 0, 1]).unwrap() = 1;

    // let vns = VNS.to_vec();
    let vns = (0..50).map(|n| n as f32).collect();
    let mut state = pollster::block_on(gaugemc::GPUBackend::new_async(
        t,
        x,
        y,
        z,
        vns,
        Some(num_replicas),
        Some(init_state),
        None,
    ))?;

    for _ in 0..steps_per_sample {
        for _ in 0..num_updates {
            NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                state.run_local_sweep(&dims, leftover, offset);
            })
        }
        state.run_global_sweep();
        state.run_pcg_rotate();
    }

    let energies_gpu = state.get_energy(Some(false))?;

    let energies_state = state.get_energy(Some(true))?;

    println!(
        "{:?}\t{:?}",
        energies_state.as_slice(),
        energies_gpu.as_slice()
    );
    assert_eq!(energies_gpu, energies_state);

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
