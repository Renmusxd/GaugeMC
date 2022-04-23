use env_logger;
use gaugemc;
use num_traits::Zero;
use pollster;
use std::collections::HashMap;

fn main() -> Result<(), String> {
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
        Some(2),
        None,
        None,
    ))?;

    state.run_global_sweep();
    let read_state = state.get_state();

    let max_int = read_state.iter().cloned().max().unwrap_or(0) as usize;
    let mut entries = HashMap::new();

    for (i, s) in read_state.iter().cloned().enumerate() {
        let original_index = i;
        let r_index = i / (t * x * y * z * 6);
        let i = i % (t * x * y * z * 6);
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
            (r_index, t_index, x_index, y_index, z_index, p_index),
            s
        );
        if !entries.contains_key(&s) {
            entries.insert(s, vec![]);
        }
        entries
            .get_mut(&s)
            .unwrap()
            .push((r_index, t_index, x_index, y_index, z_index, p_index));
    }
    // let arr = state.get_state_array();
    // arr.indexed_iter()
    //     .filter(|(_, s)| **s > 0)
    //     .for_each(|(indx, s)| println!("{:?}", indx));

    println!(
        "{}/{}",
        read_state.iter().cloned().filter(|c| !c.is_zero()).count(),
        read_state.len()
    );

    for (i, v) in entries.iter() {
        println!("{}\t{}", i, v.len());
    }

    Ok(())
}
