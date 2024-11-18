#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use gaugemc::{CudaBackend, CudaError, SiteIndex};
    use ndarray::Array2;
    use test::Bencher;

    fn make_simple_potentials(nreplicas: usize, npots: usize) -> Array2<f32> {
        let mut pots = Array2::zeros((nreplicas, npots));
        ndarray::Zip::indexed(&mut pots).for_each(|(_, p), x| *x = 0.5 * p.pow(2) as f32);

        pots
    }

    #[bench]
    fn bench_local_update_single_l16(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 1;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;

        b.iter(|| {
            state.run_single_local_update_single(0, false).unwrap();

            state.wait_for_gpu().unwrap();
        });
        Ok(())
    }

    #[bench]
    fn bench_local_update_sweep_single_l16(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 1;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;

        b.iter(|| {
            state.run_local_update_sweep().unwrap();
            state.wait_for_gpu().unwrap();
        });
        Ok(())
    }

    #[bench]
    fn bench_local_update_sweep_many_l16(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 1;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;
        let nupdates = 100;

        b.iter(|| {
            // Inner closure, the actual test
            for _ in 0..nupdates {
                state.run_local_update_sweep().unwrap();
            }

            state.wait_for_gpu().unwrap();
        });
        Ok(())
    }

    #[bench]
    fn bench_global_update_single_l16(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 1;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;

        b.iter(|| {
            state.run_global_update_sweep().unwrap();
            state.wait_for_gpu().unwrap();
        });
        Ok(())
    }

    #[bench]
    fn bench_global_update_many_l16(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 1;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;
        let nupdates = 100;

        b.iter(|| {
            for _ in 0..nupdates {
                state.run_global_update_sweep().unwrap()
            }
            state.wait_for_gpu().unwrap();
        });
        Ok(())
    }

    #[bench]
    fn bench_get_action_l16(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 1;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;

        b.iter(|| {
            let _ = state.get_action_per_replica().unwrap();
        });
        Ok(())
    }

    #[bench]
    fn bench_get_winding_l16(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 1;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;

        b.iter(|| {
            let _ = state.get_winding_per_replica().unwrap();
        });
        Ok(())
    }


    #[bench]
    fn bench_get_action_call(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 128;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;

        // Set up nontrivial configuration.
        for _ in 0..128 {
            state.run_local_update_sweep()?;
        }

        b.iter(|| {
            let action = state.get_action_per_replica().unwrap();
        });
        Ok(())
    }

    #[bench]
    fn bench_count_plaquettes_call(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 128;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;

        // Set up nontrivial configuration.
        for _ in 0..128 {
            state.run_local_update_sweep()?;
        }

        b.iter(|| {
            let counts = state.get_plaquette_counts().unwrap();
        });
        Ok(())
    }


    #[bench]
    fn bench_count_plaquettes_full_graph(b: &mut Bencher) -> Result<(), CudaError> {
        let r = 128;
        let d = 16;
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
            None,
        )?;

        // Set up nontrivial configuration.
        for _ in 0..128 {
            state.run_local_update_sweep()?;
        }

        b.iter(|| {
            let graph = state.get_plaquettes().unwrap();
        });
        Ok(())
    }
}
