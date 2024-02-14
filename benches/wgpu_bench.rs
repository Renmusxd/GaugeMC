#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use gaugemc::{GPUBackend, NDDualGraph, SiteIndex};
    use ndarray::Array2;
    use test::Bencher;

    fn make_simple_potentials(nreplicas: usize, npots: usize) -> Array2<f32> {
        let mut pots = Array2::zeros((nreplicas, npots));
        ndarray::Zip::indexed(&mut pots).for_each(|(_, p), x| *x = 0.5 * p.pow(2) as f32);

        pots
    }

    #[bench]
    fn bench_local_update_single_l16(b: &mut Bencher) -> Result<(), String> {
        let r = 1;
        let d = 16;
        let mut state = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
        ))?;

        let (dims, leftover, offset) = NDDualGraph::get_cube_dim_and_offset_iterator()
            .map(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                (dims, leftover, offset)
            })
            .next()
            .unwrap();

        b.iter(|| {
            state.run_local_sweep(&dims, leftover, offset);
            state.wait_for_gpu();
        });
        Ok(())
    }

    #[bench]
    fn bench_local_update_sweep_single_l16(b: &mut Bencher) -> Result<(), String> {
        let r = 1;
        let d = 16;
        let mut state = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
        ))?;

        b.iter(|| {
            NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                state.run_local_sweep(&dims, leftover, offset);
            });
            state.wait_for_gpu();
        });
        Ok(())
    }

    #[bench]
    fn bench_local_update_sweep_many_l16(b: &mut Bencher) -> Result<(), String> {
        let r = 1;
        let d = 16;
        let mut state = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
        ))?;
        let nupdates = 100;

        b.iter(|| {
            for _ in 0..nupdates {
                NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
                    let leftover = NDDualGraph::get_leftover_dim(&dims);
                    state.run_local_sweep(&dims, leftover, offset);
                })
            }

            state.wait_for_gpu();
        });
        Ok(())
    }

    #[bench]
    fn bench_global_update_single_l16(b: &mut Bencher) -> Result<(), String> {
        let r = 1;
        let d = 16;
        let mut state = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
        ))?;

        b.iter(|| {
            state.run_global_sweep(None);
            state.wait_for_gpu();
        });
        Ok(())
    }

    #[bench]
    fn bench_global_update_many_l16(b: &mut Bencher) -> Result<(), String> {
        let r = 1;
        let d = 16;
        let mut state = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
        ))?;
        let nupdates = 100;

        b.iter(|| {
            for _ in 0..nupdates {
                state.run_global_sweep(None);
            }
            state.wait_for_gpu();
        });
        Ok(())
    }

    #[bench]
    fn bench_get_action_l16(b: &mut Bencher) -> Result<(), String> {
        let r = 1;
        let d = 16;
        let mut state = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            None,
            None,
            None,
        ))?;

        b.iter(|| {
            let _ = state.get_energy_from_gpu(None).unwrap();
        });
        Ok(())
    }
}
