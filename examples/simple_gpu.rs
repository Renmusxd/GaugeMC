use gaugemc;
use gaugemc::{Dimension, NDDualGraph};
use num_traits::Zero;

fn main() {
    env_logger::init();
    let (t, x, y, z) = (4, 4, 4, 4);
    let mut state = pollster::block_on(gaugemc::GPUBackend::new_async(
        t,
        x,
        y,
        z,
        (0..20u32)
            .map(|i| f32::from(u16::try_from(i.pow(2)).unwrap()) / 4.0)
            .collect(),
        None,
    ));
    // for _ in 0..100 {
    //     NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
    //         let leftover = NDDualGraph::get_leftover_dim(&dims);
    //         state.run_local_sweep(&dims, leftover, offset);
    //     })
    // }

    state.run_global_sweep();

    let read_state = state.get_state();
    for (i, s) in read_state
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
    {
        let original_index = i;
        let t_index = i / (x * y * z * 6);
        let i = i % (x * y * z * 6);
        let x_index = i / (y * z * 6);
        let i = i % (y * z * 6);
        let y_index = i / (z * 6);
        let i = i % (z * 6);
        let z_index = i / (6);
        let i = i % 6;
        let p_index = i;
        println!(
            "{} -> {:?}:\t{}",
            original_index,
            (t_index, x_index, y_index, z_index, p_index),
            s
        );
    }
    println!(
        "{}/{}",
        read_state.iter().cloned().filter(|c| !c.is_zero()).count(),
        read_state.len()
    );
}
