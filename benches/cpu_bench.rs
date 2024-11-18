#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use gaugemc::NDDualGraph;

    use ndarray_rand::rand::thread_rng;
    use test::Bencher;

    fn make_simple_potentials(npots: usize) -> Vec<f64> {
        let mut pots = vec![0.0; npots];
        pots.iter_mut()
            .enumerate()
            .for_each(|(p, x)| *x = 0.5 * p.pow(2) as f64);

        pots
    }

    #[bench]
    fn bench_local_update_single_l16(b: &mut Bencher) -> Result<(), String> {
        let d = 16;
        let mut state = NDDualGraph::new(d, d, d, d, make_simple_potentials(32))?;

        let mut rng = thread_rng();
        b.iter(|| state.local_update_sweep(Some(&mut rng)));
        Ok(())
    }

    #[bench]
    fn bench_global_update_single_l16(b: &mut Bencher) -> Result<(), String> {
        let d = 16;
        let mut state = NDDualGraph::new(d, d, d, d, make_simple_potentials(32))?;

        let mut rng = thread_rng();
        b.iter(|| state.global_update_sweep(Some(&mut rng)));
        Ok(())
    }
}
