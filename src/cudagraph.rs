use crate::util::MeshIterator;
use crate::{Dimension, NDDualGraph, SiteIndex};
use cudarc::curand::result::CurandError;
use cudarc::curand::CudaRng;
use cudarc::driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx, CompileError};
use ndarray::{Array1, Array2, Array6};
use ndarray_rand::rand::prelude::SliceRandom;
use ndarray_rand::rand::{random, thread_rng};
use rayon::prelude::*;
use std::sync::Arc;

pub struct CudaBackend {
    nreplicas: usize,
    bounds: SiteIndex,
    potential_size: usize,
    device: Arc<CudaDevice>,
    state: CudaSlice<i32>,
    potential_redirect_buffer: CudaSlice<u32>,
    potential_buffer: CudaSlice<f32>,
    rng_buffer: CudaSlice<f32>,
    cuda_rng: CudaRng,
    local_update_types: Option<Vec<(u16, bool)>>,
}

/// A state in the language of integer plaquettes
pub struct DualState {
    // Shape (replicas, T, X, Y, Z, 5)
    plaquettes: Array6<i32>,
}

impl DualState {
    pub fn new_plaquettes(plaquettes: Array6<i32>) -> Self {
        assert_eq!(plaquettes.shape()[5], 6);
        Self { plaquettes }
    }

    pub fn new_volumes(volumes: Array6<i32>) -> Self {
        assert_eq!(volumes.shape()[5], 4);
        let mut shape = volumes.shape().to_vec();
        shape[5] = 6;
        let new_shape: [usize; 6] = shape.clone().try_into().unwrap();
        let mut plaquettes = Array6::<i32>::zeros(new_shape);

        const CUBE_TYPES_FOR_PLANE_TYPE: [[usize; 2]; 6] = [
            [0, 1], // tx -- txy, txz
            [0, 2], // ty -- txy, tyz
            [1, 2], // tz -- txz, tyz
            [0, 3], // xy -- txy, xyz
            [1, 3], // xz -- txz, xyz
            [2, 3], // yz -- tyz, xyz
        ];
        const NORMAL_DIM_FOR_CUBE_PLANEINDEX: [[usize; 2]; 6] = [
            [2, 3], // tx: y, z
            [1, 3], // ty: x, z
            [1, 2], // tz: x, y
            [0, 3], // xy: t, z
            [0, 2], // xz: t, y
            [0, 1], // yz: t, x
        ];
        const SIGN_CONVENTION: [[i32; 6]; 4] = [
            [1, -1, 0, 1, 0, 0],
            [1, 0, -1, 0, 1, 0],
            [0, 1, -1, 0, 0, 1],
            [0, 0, 0, 1, -1, 1],
        ];

        let bounds = &shape[1..5];
        ndarray::Zip::indexed(&mut plaquettes).for_each(
            |(r, t, x, y, z, p): (usize, usize, usize, usize, usize, usize), np: &mut i32| {
                let cubes = &CUBE_TYPES_FOR_PLANE_TYPE[p];
                let normals = &NORMAL_DIM_FOR_CUBE_PLANEINDEX[p];
                for (cube_type, normal) in cubes.iter().zip(normals) {
                    let mut coords = [r, t, x, y, z, *cube_type];
                    let cube_val = volumes[coords];
                    coords[1 + *normal] =
                        (coords[1 + *normal] + bounds[*normal] - 1) % bounds[*normal];
                    let lower_cube_val = volumes[coords];

                    assert_ne!(SIGN_CONVENTION[*cube_type][p], 0);
                    *np += SIGN_CONVENTION[*cube_type][p] * (cube_val - lower_cube_val);
                }
            },
        );
        Self::new_plaquettes(plaquettes)
    }
}

impl CudaBackend {
    pub fn new(
        bounds: SiteIndex,
        vn: Array2<f32>,
        dual_initial_state: Option<DualState>,
        seed: Option<u64>,
        device_id: Option<usize>,
    ) -> Result<Self, CudaError> {
        let local_updates_ptx = compile_ptx(include_str!("kernels/cuda_kernel.cu"));
        let local_updates_ptx = match local_updates_ptx {
            Ok(x) => x,
            Err(cerr) => {
                if let CompileError::CompileError { log, .. } = &cerr {
                    let s = log.to_str().unwrap();
                    eprintln!("{}", s);
                }
                return Err(CudaError::from(cerr));
            }
        };

        let device = CudaDevice::new(device_id.unwrap_or(0)).map_err(CudaError::from)?;
        device
            .load_ptx(
                local_updates_ptx,
                "gauge_kernel",
                &[
                    "single_local_update_plaquettes",
                    "calculate_edge_sums",
                    "partial_sum_energies",
                    "sum_buffer",
                    "global_update_sweep",
                ],
            )
            .map_err(CudaError::from)?;

        let (t, x, y, z) = (bounds.t, bounds.x, bounds.y, bounds.z);
        for d in [t, x, y, z] {
            if d % 2 == 1 {
                return CudaError::value_error(format!(
                    "Expected all dims to be even, found: {:?}",
                    [t, x, y, z]
                ));
            }
        }

        let nreplicas = vn.shape()[0];
        let potential_size = vn.shape()[1];
        if nreplicas == 0 {
            return CudaError::value_error("Replica number must be larger than 0");
        }

        let state = match dual_initial_state {
            None => {
                let num_plaquettes = nreplicas * t * x * y * z * 6;

                device
                    .alloc_zeros::<i32>(num_plaquettes)
                    .map_err(CudaError::from)?
            }
            Some(DualState { plaquettes }) => {
                let num_plaquettes = nreplicas * t * x * y * z * 6;
                debug_assert_eq!(plaquettes.len(), num_plaquettes);
                let mut plaquette_buffer = device
                    .alloc_zeros::<i32>(num_plaquettes)
                    .map_err(CudaError::from)?;
                let plaquettes = plaquettes.into_iter().collect::<Vec<_>>();
                device
                    .htod_copy_into(plaquettes, &mut plaquette_buffer)
                    .map_err(CudaError::from)?;
                plaquette_buffer
            }
        };

        // Set up the potentials
        let vn = vn.iter().copied().collect::<Vec<_>>();
        let mut potential_buffer = device
            .alloc_zeros::<f32>(vn.len())
            .map_err(CudaError::from)?;
        device
            .htod_copy_into(vn, &mut potential_buffer)
            .map_err(CudaError::from)?;

        // Set up the potentials redirect
        let vn_redirect = (0..nreplicas as u32).collect::<Vec<_>>();
        let mut potential_redirect_buffer = device
            .alloc_zeros::<u32>(vn_redirect.len())
            .map_err(CudaError::from)?;
        device
            .htod_copy_into(vn_redirect, &mut potential_redirect_buffer)
            .map_err(CudaError::from)?;

        // Set up the rng
        let local_updates = Self::calculate_simultaneous_local_updates(nreplicas, &bounds);
        let num_spanning_planes = Self::calculate_global_updates_planes(nreplicas, &bounds);
        let num_rng_consumers = local_updates.max(num_spanning_planes);

        let rng_slice = vec![0.0; num_rng_consumers];
        let mut rng_buffer = device
            .alloc_zeros::<f32>(rng_slice.len())
            .map_err(CudaError::from)?;
        device
            .htod_copy_into(rng_slice, &mut rng_buffer)
            .map_err(CudaError::from)?;

        let cuda_rng =
            CudaRng::new(seed.unwrap_or(random()), device.clone()).map_err(CudaError::from)?;

        let local_update_types = (0..4)
            .flat_map(|volume_type| [false, true].map(|offset| (volume_type, offset)))
            .collect();

        Ok(Self {
            nreplicas,
            bounds,
            potential_size,
            device,
            state,
            potential_redirect_buffer,
            potential_buffer,
            rng_buffer,
            cuda_rng,
            local_update_types: Some(local_update_types),
        })
    }

    fn get_plaquettes(&mut self) -> Result<Array6<i32>, CudaError> {
        let (t, x, y, z) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);
        let output = self
            .device
            .dtoh_sync_copy(&self.state)
            .map_err(CudaError::from)?;
        let plaquettes = Array6::from_shape_vec((self.nreplicas, t, x, y, z, 6), output).unwrap();
        Ok(plaquettes)
    }

    fn get_edge_violations(&mut self) -> Result<Array6<i32>, CudaError> {
        let (t, x, y, z) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);

        let num_edges = self.nreplicas * t * x * y * z * 4;
        let mut edge_buffer = self
            .device
            .alloc_zeros::<i32>(num_edges)
            .map_err(CudaError::from)?;
        Self::calculate_edge_violations_static(
            self.device.clone(),
            &mut edge_buffer,
            &self.state,
            self.bounds.clone(),
            self.nreplicas,
        )?;

        let output = self
            .device
            .dtoh_sync_copy(&edge_buffer)
            .map_err(CudaError::from)?;

        let edges = Array6::from_shape_vec((self.nreplicas, t, x, y, z, 4), output).unwrap();
        Ok(edges)
    }

    fn global_update_sweep(&mut self) -> Result<(), CudaError> {
        let update_planes = self.global_updates_planes();

        // Can we not subslice it?
        self.cuda_rng
            .fill_with_uniform(&mut self.rng_buffer)
            .map_err(CudaError::from)?;

        let cfg = LaunchConfig::for_num_elems(update_planes as u32);
        let global_update = self
            .device
            .get_func("gauge_kernel", "global_update_sweep")
            .unwrap();
        unsafe {
            global_update
                .launch(
                    cfg,
                    (
                        &self.state,
                        &mut self.potential_buffer,
                        &self.potential_redirect_buffer,
                        self.potential_size,
                        &mut self.rng_buffer,
                        self.nreplicas,
                        self.bounds.t,
                        self.bounds.x,
                        self.bounds.y,
                        self.bounds.z,
                    ),
                )
                .map_err(CudaError::from)
        }
    }

    fn get_action_per_replica(&mut self) -> Result<Array1<f32>, CudaError> {
        let (t, x, y, _) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);
        let mut threads_to_sum = self.nreplicas * t * x * y; // each thread starts with z*6

        let mut sum_buffer = self
            .device
            .alloc_zeros::<f32>(threads_to_sum)
            .map_err(CudaError::from)?;

        // Initial copying
        let cfg = LaunchConfig::for_num_elems(threads_to_sum as u32);
        let partial_sum_energies = self
            .device
            .get_func("gauge_kernel", "partial_sum_energies")
            .unwrap();
        unsafe {
            partial_sum_energies
                .launch(
                    cfg,
                    (
                        &self.state,
                        &mut sum_buffer,
                        &mut self.potential_buffer,
                        &self.potential_redirect_buffer,
                        self.potential_size,
                        self.nreplicas,
                        self.bounds.t,
                        self.bounds.x,
                        self.bounds.y,
                        self.bounds.z,
                    ),
                )
                .map_err(CudaError::from)?
        };

        // Now for each of y, x, t, and r: sum the potentials
        for n in [y, x, t] {
            threads_to_sum /= n;

            let cfg = LaunchConfig::for_num_elems(threads_to_sum as u32);
            let partial_sum_energies = self.device.get_func("gauge_kernel", "sum_buffer").unwrap();
            unsafe {
                partial_sum_energies
                    .launch(cfg, (&mut sum_buffer, threads_to_sum, n))
                    .map_err(CudaError::from)?
            };
        }

        // Copy out the subslice.
        let subslice = sum_buffer.slice(0..self.nreplicas);
        self.device
            .dtoh_sync_copy(&subslice)
            .map(Array1::from_vec)
            .map_err(CudaError::from)
    }

    pub fn run_local_update_sweep(&mut self) -> Result<(), CudaError> {
        let mut rng = thread_rng();
        let mut local_update_types = self.local_update_types.take().unwrap();
        local_update_types.shuffle(&mut rng);
        let res = local_update_types
            .iter()
            .copied()
            .try_for_each(|(volume, offset)| self.run_single_local_update_sweep(volume, offset));
        self.local_update_types = Some(local_update_types);
        res
    }

    fn run_single_local_update_sweep(
        &mut self,
        volume_type: u16,
        offset: bool,
    ) -> Result<(), CudaError> {
        // Get some rng
        self.cuda_rng
            .fill_with_uniform(&mut self.rng_buffer)
            .map_err(CudaError::from)?;

        let n = self.simultaneous_local_updates();
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let single_local_update = self
            .device
            .get_func("gauge_kernel", "single_local_update_plaquettes")
            .unwrap();
        unsafe {
            single_local_update
                .launch(
                    cfg,
                    (
                        &self.state,
                        &mut self.potential_buffer,
                        &self.potential_redirect_buffer,
                        self.potential_size,
                        &mut self.rng_buffer,
                        volume_type,
                        offset,
                        self.nreplicas,
                        self.bounds.t,
                        self.bounds.x,
                        self.bounds.y,
                        self.bounds.z,
                    ),
                )
                .map_err(CudaError::from)
        }
    }
}

impl CudaBackend {
    fn simultaneous_local_updates(&self) -> usize {
        Self::calculate_simultaneous_local_updates(self.nreplicas, &self.bounds)
    }

    fn calculate_simultaneous_local_updates(nreplicas: usize, bounds: &SiteIndex) -> usize {
        nreplicas * Self::calculate_simultaneous_local_updates_per_replica(bounds)
    }

    fn calculate_simultaneous_local_updates_per_replica(bounds: &SiteIndex) -> usize {
        bounds.t * bounds.x * bounds.y * bounds.z / 2
    }

    fn global_updates_planes(&self) -> usize {
        Self::calculate_global_updates_planes(self.nreplicas, &self.bounds)
    }

    fn calculate_global_updates_planes(nreplicas: usize, bounds: &SiteIndex) -> usize {
        nreplicas * Self::calculate_global_updates_planes_per_replica(bounds)
    }

    fn calculate_global_updates_planes_per_replica(bounds: &SiteIndex) -> usize {
        bounds.t * (bounds.x + bounds.y + bounds.z)
            + bounds.x * (bounds.y + bounds.z)
            + bounds.y * bounds.z
    }

    fn calculate_edge_violations_static(
        device: Arc<CudaDevice>,
        edge_buffer: &mut CudaSlice<i32>,
        plaquette_buffer: &CudaSlice<i32>,
        bounds: SiteIndex,
        nreplicas: usize,
    ) -> Result<(), CudaError> {
        let (t, x, y, z) = (bounds.t, bounds.x, bounds.y, bounds.z);
        let num_edges = nreplicas * t * x * y * z * 4;

        let calculate_edge_sums = device
            .get_func("gauge_kernel", "calculate_edge_sums")
            .unwrap();

        let cfg = LaunchConfig::for_num_elems(num_edges as u32);
        unsafe {
            calculate_edge_sums
                .launch(cfg, (plaquette_buffer, edge_buffer, nreplicas, t, x, y, z))
                .map_err(CudaError::from)
        }
    }

    fn calculate_edge_violations(
        &mut self,
        edge_buffer: &mut CudaSlice<i32>,
        plaquette_buffer: &CudaSlice<i32>,
    ) -> Result<(), CudaError> {
        Self::calculate_edge_violations_static(
            self.device.clone(),
            edge_buffer,
            plaquette_buffer,
            self.bounds.clone(),
            self.nreplicas,
        )
    }

    pub fn get_edges_with_violations(
        &mut self,
    ) -> Result<Vec<((usize, SiteIndex, Dimension), Vec<(SiteIndex, usize)>)>, CudaError> {
        let shape = self.bounds.clone();
        let edge_iterator =
            MeshIterator::new([self.nreplicas, shape.t, shape.x, shape.y, shape.z, 4])
                .build_parallel_iterator()
                .map(|[r, t, x, y, z, d]| (r, SiteIndex { t, x, y, z }, Dimension::from(d)));
        let state = self.get_plaquettes()?;
        let res = edge_iterator
            .filter_map(|(r, s, d)| {
                let (poss, negs) = NDDualGraph::plaquettes_next_to_edge(&s, d, &self.bounds);
                let sum = poss
                    .iter()
                    .cloned()
                    .map(|p| (p, 1))
                    .chain(negs.iter().cloned().map(|n| (n, -1)))
                    .map(|((site, p), mult)| {
                        state.get([r, site.t, site.x, site.y, site.z, p]).unwrap() * mult
                    })
                    .sum::<i32>();
                if sum != 0 {
                    Some(((r, s, d), poss.into_iter().chain(negs).collect::<Vec<_>>()))
                } else {
                    None
                }
            })
            .collect();
        Ok(res)
    }
}

#[derive(Debug)]
pub enum CudaError {
    Value(String),
    Compile(CompileError),
    Driver(DriverError),
    Rand(CurandError),
}

impl CudaError {
    fn value_error<T, Str: Into<String>>(msg: Str) -> Result<T, Self> {
        Err(Self::Value(msg.into()))
    }
}

impl From<CompileError> for CudaError {
    fn from(value: CompileError) -> Self {
        Self::Compile(value)
    }
}

impl From<DriverError> for CudaError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<CurandError> for CudaError {
    fn from(value: CurandError) -> Self {
        Self::Rand(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Axis};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use num_traits::Zero;

    fn make_simple_potentials(nreplicas: usize, npots: usize) -> Array2<f32> {
        let mut pots = Array2::zeros((nreplicas, npots));
        ndarray::Zip::indexed(&mut pots).for_each(|(_, p), x| *x = 0.5 * p.pow(2) as f32);

        pots
    }

    fn make_custom_simple_potentials<F>(nreplicas: usize, npots: usize, f: F) -> Array2<f32>
    where
        F: Fn(usize, usize) -> f32,
    {
        let mut pots = Array2::zeros((nreplicas, npots));
        ndarray::Zip::indexed(&mut pots).for_each(|(r, np), x| *x = f(r, np));

        pots
    }

    #[test]
    fn test_construction() -> Result<(), CudaError> {
        let _state = CudaBackend::new(
            SiteIndex::new(6, 8, 10, 12),
            make_simple_potentials(4, 32),
            None,
            Some(31415),
            None,
        )?;
        Ok(())
    }
    #[test]
    fn test_simple_launch() -> Result<(), CudaError> {
        let mut state = CudaBackend::new(
            SiteIndex::new(4, 4, 4, 4),
            make_simple_potentials(1, 2),
            None,
            Some(31415),
            None,
        )?;
        state.run_single_local_update_sweep(0, false)
    }

    #[test]
    fn test_get_plaquettes() -> Result<(), CudaError> {
        let mut state = CudaBackend::new(
            SiteIndex::new(4, 4, 4, 4),
            make_simple_potentials(1, 2),
            None,
            Some(31415),
            None,
        )?;
        let plaquettes = state.get_plaquettes()?;
        assert!(plaquettes.iter().all(|x| x.is_zero()));
        Ok(())
    }

    #[test]
    fn test_get_plaquettes_single_inc() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 6, 6, 6, 6);
        let mut state = Array::zeros((r, t, x, y, z, 4));
        state[[0, 0, 0, 0, 0, 0]] = 1;
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 2),
            Some(DualState::new_volumes(state)),
            Some(31415),
            None,
        )?;
        let plaquettes = state.get_plaquettes()?;
        let nonzero = plaquettes.iter().filter(|x| !x.is_zero()).count();
        let violations = state.get_edges_with_violations()?;

        assert_eq!(nonzero, 6);
        assert!(violations.is_empty());
        Ok(())
    }

    #[test]
    fn test_get_plaquettes_single_inc_compare_gpu() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 6, 6, 6, 6);
        let mut state = Array::zeros((r, t, x, y, z, 4));
        state[[0, 0, 0, 0, 0, 0]] = 1;
        let state = DualState::new_volumes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 2),
            Some(state),
            Some(31415),
            None,
        )?;
        let _plaquettes = state.get_plaquettes()?;
        let edge_sums = state.get_edge_violations()?;
        let nonzero_edges = edge_sums.iter().filter(|x| !x.is_zero()).count();
        let violations = state.get_edges_with_violations()?;

        assert_eq!(violations.len(), nonzero_edges);
        assert!(violations.is_empty());
        Ok(())
    }

    #[test]
    fn test_get_plaquettes_single_inc_sweep() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 6, 6, 6, 6);
        for i in 0..4 {
            let mut state = Array::zeros((r, t, x, y, z, 4));
            state[[0, 0, 0, 0, 0, i]] = 1;
            let state = DualState::new_volumes(state);
            let mut state = CudaBackend::new(
                SiteIndex::new(t, x, y, z),
                make_simple_potentials(r, 2),
                Some(state),
                None,
                None,
            )?;
            let plaquettes = state.get_plaquettes()?;
            let nonzero = plaquettes.iter().filter(|x| !x.is_zero()).count();
            let violations = state.get_edges_with_violations()?;
            assert_eq!(nonzero, 6);
            assert!(violations.is_empty());
        }
        Ok(())
    }

    #[test]
    fn test_get_plaquettes_single_inc_random_fill() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (16, 6, 6, 6, 6);
        for _ in 0..10 {
            let state = Array6::random((r, t, x, y, z, 4), Uniform::new(-32, 32));
            let state = DualState::new_volumes(state);
            let mut state = CudaBackend::new(
                SiteIndex::new(t, x, y, z),
                make_simple_potentials(r, 2),
                Some(state),
                None,
                None,
            )?;
            let violations = state
                .get_edge_violations()?
                .into_iter()
                .filter(|x| !x.is_zero())
                .count();
            assert_eq!(violations, 0);
        }
        Ok(())
    }

    #[test]
    fn test_get_edges_constructed() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 4, 4, 4, 4);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));
        plaquette_state[[0, 0, 0, 0, 0, 0]] = 1;
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(DualState::new_plaquettes(plaquette_state)),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();

        assert_eq!(nonzero, 4);
        Ok(())
    }
    #[test]
    fn test_get_edges_constructed_pair() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 4, 4, 4, 4);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));
        plaquette_state[[0, 0, 0, 0, 0, 0]] = 1;
        plaquette_state[[0, 1, 0, 0, 0, 0]] = 1;
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(DualState::new_plaquettes(plaquette_state)),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();

        assert_eq!(nonzero, 6);
        Ok(())
    }
    #[test]
    fn test_get_edges_constructed_half_cube() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 4, 4, 4, 4);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));
        plaquette_state[[0, 0, 0, 0, 0, 0]] = 1; // tx
        plaquette_state[[0, 0, 0, 0, 0, 1]] = -1; // ty
        plaquette_state[[0, 0, 0, 0, 0, 3]] = 1; // xy
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(DualState::new_plaquettes(plaquette_state)),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();

        assert_eq!(nonzero, 6);
        Ok(())
    }

    #[test]
    fn test_get_edges_constructed_full_cube_txy() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 4, 4, 4, 4);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));

        plaquette_state[[0, 0, 0, 0, 0, 0]] = 1; // tx
        plaquette_state[[0, 0, 0, 1, 0, 0]] = -1; // tx + y

        plaquette_state[[0, 0, 0, 0, 0, 1]] = -1; // ty
        plaquette_state[[0, 0, 1, 0, 0, 1]] = 1; // ty + x

        plaquette_state[[0, 0, 0, 0, 0, 3]] = 1; // xy
        plaquette_state[[0, 1, 0, 0, 0, 3]] = -1; // xy + t

        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(DualState::new_plaquettes(plaquette_state)),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();

        assert_eq!(nonzero, 0);
        Ok(())
    }

    #[test]
    fn test_get_edges_constructed_full_cube_txy_failed() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 4, 4, 4, 4);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));

        plaquette_state[[0, 0, 0, 0, 0, 0]] = 1; // tx
        plaquette_state[[0, 0, 0, 1, 0, 0]] = -1; // tx + y
        plaquette_state[[0, 0, 0, 0, 0, 1]] = -1; // ty
        plaquette_state[[0, 0, 1, 0, 0, 1]] = 1; // ty + x
        plaquette_state[[0, 0, 0, 0, 0, 3]] = 1; // xy

        // Insert failure, expect 4 edges to contain defects.
        plaquette_state[[0, 1, 0, 0, 0, 3]] = 1; // xy + t

        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(DualState::new_plaquettes(plaquette_state)),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();

        assert_eq!(nonzero, 4);
        Ok(())
    }

    #[test]
    fn test_get_edges_constructed_full_cube_xyz() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 4, 4, 4, 4);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));

        plaquette_state[[0, 0, 0, 0, 0, 3]] = 1; // xy
        plaquette_state[[0, 0, 0, 0, 1, 3]] = -1; // xy + z

        plaquette_state[[0, 0, 0, 0, 0, 4]] = -1; // xz
        plaquette_state[[0, 0, 0, 1, 0, 4]] = 1; // xz + y

        plaquette_state[[0, 0, 0, 0, 0, 5]] = 1; // yz
        plaquette_state[[0, 0, 1, 0, 0, 5]] = -1; // yz + x

        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(DualState::new_plaquettes(plaquette_state)),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();
        assert_eq!(nonzero, 0);
        Ok(())
    }
    #[test]
    fn test_get_edges_constructed_full_cube_txz() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 4, 4, 4, 4);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));

        plaquette_state[[0, 0, 0, 0, 0, 0]] = 1; // tx
        plaquette_state[[0, 0, 0, 0, 1, 0]] = -1; // tx + z

        plaquette_state[[0, 0, 0, 0, 0, 2]] = -1; // tz
        plaquette_state[[0, 0, 1, 0, 0, 2]] = 1; // tz + x

        plaquette_state[[0, 0, 0, 0, 0, 4]] = 1; // xz
        plaquette_state[[0, 1, 0, 0, 0, 4]] = -1; // xz + t

        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(DualState::new_plaquettes(plaquette_state)),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();

        assert_eq!(nonzero, 0);
        Ok(())
    }

    #[test]
    fn test_get_edges_single_inc() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 6, 6, 6, 6);
        let mut state = Array::zeros((r, t, x, y, z, 4));
        state[[0, 0, 0, 0, 0, 0]] = 1;
        let state = DualState::new_volumes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(state),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;

        println!("{:?}", edges);

        Ok(())
    }

    #[test]
    fn test_get_plaquettes_double_inc() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 6, 6, 6, 6);
        let mut state = Array::zeros((r, t, x, y, z, 4));
        state[[0, 0, 0, 0, 0, 0]] = 1;
        state[[0, 1, 0, 0, 0, 0]] = 1;
        let state = DualState::new_volumes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(state),
            Some(31415),
            None,
        )?;
        let plaquettes = state.get_plaquettes()?;
        let nonzero = plaquettes.iter().filter(|x| !x.is_zero()).count();
        assert_eq!(nonzero, 10);
        // Should never have edge violations.
        let nnonzero = state.get_edges_with_violations()?.len();
        assert_eq!(nnonzero, 0);
        let nnonzero = state
            .get_edge_violations()?
            .into_iter()
            .filter(|x| !x.is_zero())
            .count();
        assert_eq!(nnonzero, 0);
        Ok(())
    }
    #[test]
    fn test_simple_launch_replicas() -> Result<(), CudaError> {
        let nreplicas = 1;
        let (t, x, y, z) = (8, 8, 8, 8);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(nreplicas, 32),
            Some(DualState::new_volumes(Array::zeros((
                nreplicas, t, x, y, z, 4,
            )))),
            Some(31415),
            None,
        )?;
        state.run_single_local_update_sweep(0, false)?;

        // Exceedingly unlikely to get 0.
        let plaquettes = state.get_plaquettes()?;
        let nnnonzero = plaquettes.iter().copied().filter(|x| !x.is_zero()).count();
        assert_ne!(nnnonzero, 0);

        // Should never have edge violations.
        let nnonzero = state.get_edges_with_violations()?.len();
        assert_eq!(nnonzero, 0);
        let nnonzero = state
            .get_edge_violations()?
            .into_iter()
            .filter(|x| !x.is_zero())
            .count();
        assert_eq!(nnonzero, 0);
        Ok(())
    }

    #[test]
    fn test_simple_launch_replicas_plaquettes() -> Result<(), CudaError> {
        let nreplicas = 1;
        let (t, x, y, z) = (8, 8, 8, 8);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(nreplicas, 32),
            Some(DualState::new_plaquettes(Array6::zeros((
                nreplicas, t, x, y, z, 6,
            )))),
            Some(31415),
            None,
        )?;
        state.run_single_local_update_sweep(0, false)?;

        // Exceedingly unlikely to get 0.
        let plaquettes = state.get_plaquettes()?;
        let nnnonzero = plaquettes.iter().copied().filter(|x| !x.is_zero()).count();
        assert_ne!(nnnonzero, 0);

        // Should never have edge violations.
        let nnonzero = state.get_edges_with_violations()?.len();
        assert_eq!(nnonzero, 0);
        let nnonzero = state
            .get_edge_violations()?
            .into_iter()
            .filter(|x| !x.is_zero())
            .count();
        assert_eq!(nnonzero, 0);
        Ok(())
    }

    #[test]
    fn test_sweep_launch_replicas_plaquettes() -> Result<(), CudaError> {
        let nreplicas = 1;
        let (t, x, y, z) = (8, 8, 8, 8);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(nreplicas, 32),
            Some(DualState::new_plaquettes(Array6::zeros((
                nreplicas, t, x, y, z, 6,
            )))),
            Some(31415),
            None,
        )?;
        state.run_local_update_sweep()?;

        // Exceedingly unlikely to get 0.
        let plaquettes = state.get_plaquettes()?;
        let nnnonzero = plaquettes.iter().copied().filter(|x| !x.is_zero()).count();
        assert_ne!(nnnonzero, 0);

        // Should never have edge violations.
        let nnonzero = state.get_edges_with_violations()?.len();
        assert_eq!(nnonzero, 0);
        let nnonzero = state
            .get_edge_violations()?
            .into_iter()
            .filter(|x| !x.is_zero())
            .count();
        assert_eq!(nnonzero, 0);

        Ok(())
    }

    #[test]
    fn test_get_edges_constructed_full_cube_txy_energy() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (4, 8, 8, 8, 8);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));

        for i in 0..r {
            plaquette_state[[i, 0, 0, 0, 0, 0]] = 1; // tx
            plaquette_state[[i, 0, 0, 1, 0, 0]] = -1; // tx + y
            plaquette_state[[i, 0, 0, 0, 0, 1]] = -1; // ty
            plaquette_state[[i, 0, 1, 0, 0, 1]] = 1; // ty + x
            plaquette_state[[i, 0, 0, 0, 0, 3]] = 1; // xy
            plaquette_state[[i, 1, 0, 0, 0, 3]] = -1; // xy + t
        }

        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_custom_simple_potentials(r, 32, |r, np| ((r + 1) * np.pow(2)) as f32),
            Some(DualState::new_plaquettes(plaquette_state)),
            Some(31415),
            None,
        )?;
        let edges = state.get_edge_violations()?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();
        assert_eq!(nonzero, 0);

        let actions = state.get_action_per_replica()?;
        assert_eq!(actions, Array1::from_vec(vec![6.0, 12.0, 18.0, 24.0]));

        Ok(())
    }

    #[test]
    fn test_simple_global_launch() -> Result<(), CudaError> {
        let (r, d) = (3, 4);
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_custom_simple_potentials(r, 32, |r, n| (r * n.pow(2)) as f32),
            None,
            Some(31415),
            None,
        )?;
        state.global_update_sweep()?;

        let plaqs = state.get_plaquettes()?;
        // We should get global planes in replica 0 and nowhere else.
        let nonzero = plaqs
            .axis_iter(Axis(0))
            .map(|x| x.iter().copied().filter(|x| !x.is_zero()).count())
            .collect::<Vec<_>>();

        // Check planes
        let planes_sliding = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let planes_indexing = [(2, 3), (1, 3), (1, 2), (0, 2), (0, 2), (0, 1)];

        for p in 0..6 {
            let (sliding_a, sliding_b) = planes_sliding[p];
            let (indexing_a, indexing_b) = planes_indexing[p];
            for rr in 0..r {
                let mut coords = [rr, 0, 0, 0, 0, p];

                for i in 0..d {
                    for j in 0..d {
                        coords[indexing_a + 1] = i;
                        coords[indexing_b + 1] = j;
                        let comp_val = plaqs[coords];
                        for k in 0..d {
                            for l in 0..d {
                                coords[sliding_a + 1] = k;
                                coords[sliding_b + 1] = l;
                                assert_eq!(
                                    plaqs[coords], comp_val,
                                    "Value at {:?} is {} not {}",
                                    coords, plaqs[coords], comp_val
                                );
                            }
                        }
                    }
                }
            }
        }

        assert_ne!(nonzero[0], 0);
        assert_eq!(nonzero[0] % (d * d), 0);
        for rr in 1..r {
            assert_eq!(nonzero[rr], 0);
        }
        Ok(())
    }

    #[test]
    fn test_global_undo_planes() -> Result<(), CudaError> {
        let (r, d) = (3, 4);

        let mut state = Array6::zeros((r, d, d, d, d, 6));
        ndarray::Zip::indexed(&mut state).for_each(|(_, _, _, _, _, p), x| {
            if p % 2 == 0 {
                *x = 1;
            } else {
                *x = -1;
            }
        });

        let state = DualState::new_plaquettes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_custom_simple_potentials(r, 32, |_, n| n as f32 * 10.0),
            Some(state),
            Some(31415),
            None,
        )?;
        state.global_update_sweep()?;

        let plaqs = state.get_plaquettes()?;
        // We should get global planes in replica 0 and nowhere else.
        let nonzero = plaqs
            .axis_iter(Axis(0))
            .map(|x| x.iter().copied().filter(|x| !x.is_zero()).count())
            .collect::<Vec<_>>();

        for rr in 0..r {
            assert_eq!(nonzero[rr], 0);
        }
        Ok(())
    }
}
