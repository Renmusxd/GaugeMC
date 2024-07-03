use std::fs::File;
use log::info;
use ndarray::{Array, Array0, Array1};
use ndarray_npy::NpzWriter;
use ndarray_rand::rand::thread_rng;
use num_complex::Complex;
use gaugemc::NDDualGraph;

fn make_cosine_potentials(npots: usize, k: f64) -> Vec<f64> {
    let f = |n: usize| {
        let k = Complex::from(k);
        let t = scilib::math::bessel::i_nu(n as f64, 1. / k);
        let b = scilib::math::bessel::i_nu(0., 1. / k);
        assert!(t.im < f64::EPSILON);
        assert!(b.im < f64::EPSILON);
        let res = -(t.re / b.re).ln();
        res
    };

    let mut pots = vec![0.0; npots];
    pots.iter_mut()
        .enumerate()
        .for_each(|(p, x)| *x = f(p));

    pots
}


fn main() {
    env_logger::init();

    let warmup = 128;
    let steps_per_sample = 16;
    let nsamples = 256;
    let d = 16;

    for k in Array::linspace(1.01, 1.02, 10) {
        let mut result = Array1::zeros((nsamples,));

        let mut state = NDDualGraph::new(d, d, d, d, make_cosine_potentials(32, k)).expect("Could not create graph");
        let mut rng = thread_rng();

        for i in 0..warmup {
            info!("Warmup {}/{}", i, warmup);
            for _ in 0..steps_per_sample {
                state.local_update_sweep(Some(&mut rng));
            }
        }
        result.iter_mut().enumerate().for_each(|(i, x)| {
            info!("Sample {}/{}", i, nsamples);
            for _ in 0..steps_per_sample {
                state.local_update_sweep(Some(&mut rng));
            }
            *x = state.get_energy();
        });

        let mut npz = NpzWriter::new(File::create(format!("energies-{:.5}.npz", k)).expect("Could not create file."));
        npz.add_array("energy", &result).expect("Could not add array to file.");
        npz.add_array(
            "k",
            &Array0::from_elem((), k),
        );
        npz.finish().expect("Could not write to file.");
    }
}