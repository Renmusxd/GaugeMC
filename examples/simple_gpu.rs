use gaugemc;
use gaugemc::Dimension;
use log;
use num_traits::Zero;

fn main() {
    env_logger::init();
    let (t, x, y, z) = (2, 2, 2, 2);
    let mut state = pollster::block_on(gaugemc::GPUBackend::new_async(
        t,
        x,
        y,
        z,
        vec![0.0, 1.0, 4.0, 9.0],
    ));
    state.run_local_sweep(
        &[Dimension::T, Dimension::X, Dimension::Y],
        Dimension::Z,
        false,
    );
    let read_state = state.get_state();
    for (i, s) in read_state
        .into_iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
    {
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
            "{:?}:\t{}",
            (t_index, x_index, y_index, z_index, p_index),
            s
        );
    }
}
