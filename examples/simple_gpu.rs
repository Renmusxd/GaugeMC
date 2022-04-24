use env_logger;
use gaugemc;
use gaugemc::{Dimension, NDDualGraph};
use num_traits::Zero;
use pollster;
use std::collections::HashMap;

fn main() -> Result<(), String> {
    env_logger::init();

    let (t, x, y, z) = (20, 20, 20, 20);
    let mut state = pollster::block_on(gaugemc::GPUBackend::new_async(
        t,
        x,
        y,
        z,
        VNS.to_vec(),
        Some(1),
        None,
        None,
    ))?;

    for _ in 0..1000 {
        for _ in 0..10 {
            NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                state.run_local_sweep(&dims, leftover, offset);
            })
        }
        state.run_global_sweep();
    }

    assert_eq!(state.get_edges_with_violations().len(), 0);

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

const VNS: [f32; 100] = [
    0.,
    1.04033274,
    2.81322273,
    5.0010925,
    7.48040728,
    10.18473702,
    13.07245449,
    16.11498776,
    19.29149527,
    22.58609556,
    25.98628147,
    29.48194633,
    33.06475233,
    36.72770296,
    40.4648431,
    44.27104233,
    48.14183461,
    52.07329695,
    56.06195596,
    60.10471471,
    64.19879462,
    68.34168877,
    72.53112385,
    76.76502901,
    81.0415099,
    85.35882703,
    89.71537754,
    94.10967969,
    98.54035963,
    103.00614001,
    107.50583019,
    112.03831766,
    116.60256061,
    121.19758142,
    125.82246085,
    130.476333,
    135.1583808,
    139.86783197,
    144.60395544,
    149.36605814,
    154.15348212,
    158.96560195,
    163.80182233,
    168.661576,
    173.5443218,
    178.4495429,
    183.37674519,
    188.32545582,
    193.29522188,
    198.28560911,
    203.29620083,
    208.32659685,
    213.37641254,
    218.44527792,
    223.53283684,
    228.63874623,
    233.7626754,
    238.90430535,
    244.06332818,
    249.23944652,
    254.43237301,
    259.64182978,
    264.86754801,
    270.10926747,
    275.36673613,
    280.63970979,
    285.92795169,
    291.23123221,
    296.54932852,
    301.88202431,
    307.22910948,
    312.59037991,
    317.96563718,
    323.35468834,
    328.7573457,
    334.17342661,
    339.60275325,
    345.04515246,
    350.50045554,
    355.9684981,
    361.44911988,
    366.9421646,
    372.44747981,
    377.96491676,
    383.49433029,
    389.03557863,
    394.58852337,
    400.15302926,
    405.72896418,
    411.31619897,
    416.91460736,
    422.52406588,
    428.14445376,
    433.77565283,
    439.41754746,
    445.07002446,
    450.732973,
    456.40628458,
    462.0898529,
    467.78357381,
];
