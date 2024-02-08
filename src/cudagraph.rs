use crate::util::MeshIterator;
use crate::{Dimension, NDDualGraph, SiteIndex};
use cudarc::curand::result::CurandError;
use cudarc::curand::CudaRng;
use cudarc::driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx, CompileError};
use ndarray::{Array2, Array4, Array6};
use ndarray_rand::rand::{random, Rng};
use rayon::prelude::*;
use std::sync::Arc;

struct CudaBackend {
    nreplicas: usize,
    bounds: SiteIndex,
    device: Arc<CudaDevice>,
    state_buffer: CudaSlice<i32>,
    winding_buffer: CudaSlice<i32>,
    potential_redirect_buffer: CudaSlice<u32>,
    potential_buffer: CudaSlice<f32>,
    rng_buffer: CudaSlice<f32>,
    cuda_rng: CudaRng,
}

/// A state in the language of integer volumes
///
struct DualVolumesState {
    // Shape (replicas, T, X, Y, Z, 4)
    mv: Array6<i32>,
    // Spanning planes (replicas, 4, max(T,X,Y,Z), max(T,X,Y,Z))
    spanning_planes: Array4<i32>,
}

impl CudaBackend {
    pub fn new(
        bounds: SiteIndex,
        vn: Array2<f32>,
        dual_initial_state: Option<DualVolumesState>,
        seed: Option<u64>,
        device_id: Option<usize>,
    ) -> Result<Self, CudaError> {
        let local_updates_ptx = compile_ptx(include_str!("kernels/local_updates.cu"));
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
                "local_updates",
                &[
                    "single_local_update",
                    "calculate_plaquettes",
                    "calculate_edge_sums",
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

        let num_replicas = vn.shape()[0];
        if num_replicas == 0 {
            return CudaError::value_error("Replica number must be larger than 0");
        }

        // Set up the initial state
        let state_buffer_size = Self::calculate_state_volumes(num_replicas, &bounds);
        let mut state_buffer = device
            .alloc_zeros::<i32>(state_buffer_size)
            .map_err(CudaError::from)?;
        if let Some(initial_state) = dual_initial_state.as_ref() {
            let initial_state = initial_state.mv.iter().copied().collect::<Vec<_>>();
            device
                .htod_copy_into(initial_state, &mut state_buffer)
                .map_err(CudaError::from)?;
        };

        // Set up the winding state
        let num_spanning_planes = Self::calculate_global_updates_planes(num_replicas, &bounds);
        let winding_state = dual_initial_state
            .map(|x| x.spanning_planes.iter().copied().collect::<Vec<_>>())
            .unwrap_or(vec![0; num_spanning_planes]);
        let mut winding_buffer = device
            .alloc_zeros::<i32>(winding_state.len())
            .map_err(CudaError::from)?;
        device
            .htod_copy_into(winding_state, &mut winding_buffer)
            .map_err(CudaError::from)?;

        // Set up the potentials
        let vn = vn.iter().copied().collect::<Vec<_>>();
        let mut potential_buffer = device
            .alloc_zeros::<f32>(vn.len())
            .map_err(CudaError::from)?;
        device
            .htod_copy_into(vn, &mut potential_buffer)
            .map_err(CudaError::from)?;

        // Set up the potentials redirect
        let vn_redirect = (0..num_replicas as u32).collect::<Vec<_>>();
        let mut potential_redirect_buffer = device
            .alloc_zeros::<u32>(vn_redirect.len())
            .map_err(CudaError::from)?;
        device
            .htod_copy_into(vn_redirect, &mut potential_redirect_buffer)
            .map_err(CudaError::from)?;

        // Set up the rng
        let local_updates = Self::calculate_simultaneous_local_updates(num_replicas, &bounds);
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

        Ok(Self {
            nreplicas: num_replicas,
            bounds,
            device,
            state_buffer,
            winding_buffer,
            potential_redirect_buffer,
            potential_buffer,
            rng_buffer,
            cuda_rng,
        })
    }

    fn get_plaquettes(&mut self) -> Result<Array6<i32>, CudaError> {
        let (t, x, y, z) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);
        let num_plaquettes = self.nreplicas * t * x * y * z * 6;

        let mut plaquette_buffer = self
            .device
            .alloc_zeros::<i32>(num_plaquettes)
            .map_err(CudaError::from)?;

        self.calculate_plaquettes(&mut plaquette_buffer)?;

        let output = self
            .device
            .dtoh_sync_copy(&plaquette_buffer)
            .map_err(CudaError::from)?;

        let plaquettes = Array6::from_shape_vec((self.nreplicas, t, x, y, z, 6), output).unwrap();
        Ok(plaquettes)
    }

    fn get_edge_violations(&mut self) -> Result<Array6<i32>, CudaError> {
        self.get_edge_violations_for_plaquettes(None)
    }

    fn get_edge_violations_for_plaquettes(
        &mut self,
        plaquettes: Option<Array6<i32>>,
    ) -> Result<Array6<i32>, CudaError> {
        let (t, x, y, z) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);
        let num_plaquettes = self.nreplicas * t * x * y * z * 6;
        let num_edges = self.nreplicas * t * x * y * z * 4;

        let mut plaquette_buffer = self
            .device
            .alloc_zeros::<i32>(num_plaquettes)
            .map_err(CudaError::from)?;

        let mut edge_buffer = self
            .device
            .alloc_zeros::<i32>(num_edges)
            .map_err(CudaError::from)?;

        if let Some(plaquettes) = plaquettes {
            self.device.htod_sync_copy_into(
                &plaquettes.into_iter().collect::<Vec<_>>(),
                &mut plaquette_buffer,
            )?;
        } else {
            self.calculate_plaquettes(&mut plaquette_buffer)?;
        }

        self.calculate_edge_violations(&mut edge_buffer, &plaquette_buffer)?;

        let output = self
            .device
            .dtoh_sync_copy(&edge_buffer)
            .map_err(CudaError::from)?;

        let edges = Array6::from_shape_vec((self.nreplicas, t, x, y, z, 4), output).unwrap();
        Ok(edges)
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
        let single_local_update = self
            .device
            .get_func("local_updates", "single_local_update")
            .unwrap();

        let n = self.simultaneous_local_updates();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // TODO: fix potential vector size.
        let potential_vector_size = 1u32;

        unsafe {
            single_local_update
                .launch(
                    cfg,
                    (
                        &mut self.state_buffer,
                        &mut self.potential_buffer,
                        potential_vector_size,
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
    fn state_volumes(&self) -> usize {
        Self::calculate_simultaneous_local_updates(self.nreplicas as usize, &self.bounds)
    }

    fn calculate_state_volumes(nreplicas: usize, bounds: &SiteIndex) -> usize {
        nreplicas * Self::calculate_state_volumes_per_replica(bounds)
    }

    fn calculate_state_volumes_per_replica(bounds: &SiteIndex) -> usize {
        bounds.t * bounds.x * bounds.y * bounds.z * 4
    }

    fn simultaneous_local_updates(&self) -> usize {
        Self::calculate_simultaneous_local_updates(self.nreplicas as usize, &self.bounds)
    }

    fn calculate_simultaneous_local_updates(nreplicas: usize, bounds: &SiteIndex) -> usize {
        nreplicas * Self::calculate_simultaneous_local_updates_per_replica(bounds)
    }

    fn calculate_simultaneous_local_updates_per_replica(bounds: &SiteIndex) -> usize {
        bounds.t * bounds.x * bounds.y * bounds.z / 2
    }

    fn global_updates_planes(&self) -> usize {
        Self::calculate_global_updates_planes(self.nreplicas as usize, &self.bounds)
    }

    fn calculate_global_updates_planes(nreplicas: usize, bounds: &SiteIndex) -> usize {
        nreplicas * Self::calculate_global_updates_planes_per_replica(bounds)
    }

    fn calculate_global_updates_planes_per_replica(bounds: &SiteIndex) -> usize {
        // TODO
        1
    }

    fn calculate_edge_violations(
        &mut self,
        edge_buffer: &mut CudaSlice<i32>,
        plaquette_buffer: &CudaSlice<i32>,
    ) -> Result<(), CudaError> {
        let (t, x, y, z) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);
        let num_edges = self.nreplicas * t * x * y * z * 4;

        let calculate_edge_sums = self
            .device
            .get_func("local_updates", "calculate_edge_sums")
            .unwrap();

        let cfg = LaunchConfig::for_num_elems(num_edges as u32);
        unsafe {
            calculate_edge_sums
                .launch(
                    cfg,
                    (
                        plaquette_buffer,
                        edge_buffer,
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

    fn calculate_plaquettes(
        &mut self,
        plaquette_buffer: &mut CudaSlice<i32>,
    ) -> Result<(), CudaError> {
        let (t, x, y, z) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);
        let num_plaquettes = self.nreplicas * t * x * y * z * 6;

        let single_local_update = self
            .device
            .get_func("local_updates", "calculate_plaquettes")
            .unwrap();

        let cfg = LaunchConfig::for_num_elems(num_plaquettes as u32);
        unsafe {
            single_local_update
                .launch(
                    cfg,
                    (
                        &self.state_buffer,
                        plaquette_buffer,
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
                    Some((
                        (r, s, d),
                        poss.into_iter().chain(negs.into_iter()).collect::<Vec<_>>(),
                    ))
                } else {
                    None
                }
            })
            .collect();
        Ok(res)
    }
}

#[derive(Debug)]
enum CudaError {
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
    use ndarray::Array;
    use num_traits::Zero;

    fn make_potentials(nreplicas: usize, npots: usize) -> Array2<f32> {
        let mut pots = Array2::zeros((nreplicas, npots));
        ndarray::Zip::indexed(&mut pots).for_each(|(_, p), x| *x = p.pow(2) as f32);

        pots
    }

    #[test]
    fn test_construction() -> Result<(), CudaError> {
        let state = CudaBackend::new(
            SiteIndex::new(6, 8, 10, 12),
            make_potentials(4, 32),
            None,
            None,
            None,
        )?;
        Ok(())
    }
    #[test]
    fn test_simple_launch() -> Result<(), CudaError> {
        let mut state = CudaBackend::new(
            SiteIndex::new(4, 4, 4, 4),
            make_potentials(1, 2),
            None,
            None,
            None,
        )?;
        state.run_single_local_update_sweep(0, false)
    }

    #[test]
    fn test_get_plaquettes() -> Result<(), CudaError> {
        let mut state = CudaBackend::new(
            SiteIndex::new(4, 4, 4, 4),
            make_potentials(1, 2),
            None,
            None,
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
        let state = DualVolumesState {
            mv: state,
            spanning_planes: Array::zeros((r, 4, t, t)),
        };
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_potentials(r, 2),
            Some(state),
            None,
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
        let state = DualVolumesState {
            mv: state,
            spanning_planes: Array::zeros((r, 4, t, t)),
        };
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_potentials(r, 2),
            Some(state),
            None,
            None,
        )?;
        let plaquettes = state.get_plaquettes()?;
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
            let state = DualVolumesState {
                mv: state,
                spanning_planes: Array::zeros((r, 4, t, t)),
            };
            let mut state = CudaBackend::new(
                SiteIndex::new(t, x, y, z),
                make_potentials(r, 2),
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
    fn test_get_edges_constructed() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 4, 4, 4, 4);
        let mut plaquette_state = Array::zeros((r, t, x, y, z, 6));
        plaquette_state[[0, 0, 0, 0, 0, 0]] = 1;
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_potentials(r, 32),
            None,
            None,
            None,
        )?;
        let edges = state.get_edge_violations_for_plaquettes(Some(plaquette_state))?;
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
            make_potentials(r, 32),
            None,
            None,
            None,
        )?;
        let edges = state.get_edge_violations_for_plaquettes(Some(plaquette_state))?;
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
            make_potentials(r, 32),
            None,
            None,
            None,
        )?;
        let edges = state.get_edge_violations_for_plaquettes(Some(plaquette_state))?;
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
            make_potentials(r, 32),
            None,
            None,
            None,
        )?;
        let edges = state.get_edge_violations_for_plaquettes(Some(plaquette_state))?;
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
            make_potentials(r, 32),
            None,
            None,
            None,
        )?;
        let edges = state.get_edge_violations_for_plaquettes(Some(plaquette_state))?;
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
            make_potentials(r, 32),
            None,
            None,
            None,
        )?;
        let edges = state.get_edge_violations_for_plaquettes(Some(plaquette_state))?;
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
            make_potentials(r, 32),
            None,
            None,
            None,
        )?;
        let edges = state.get_edge_violations_for_plaquettes(Some(plaquette_state))?;
        let nonzero = edges.iter().filter(|x| !x.is_zero()).count();

        assert_eq!(nonzero, 0);
        Ok(())
    }

    #[test]
    fn test_get_edges_single_inc() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (1, 6, 6, 6, 6);
        let mut state = Array::zeros((r, t, x, y, z, 4));
        state[[0, 0, 0, 0, 0, 0]] = 1;
        let state = DualVolumesState {
            mv: state,
            spanning_planes: Array::zeros((r, 4, t, t)),
        };
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_potentials(r, 32),
            Some(state),
            None,
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
        let state = DualVolumesState {
            mv: state,
            spanning_planes: Array::zeros((r, 4, t, t)),
        };
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_potentials(r, 32),
            Some(state),
            None,
            None,
        )?;
        let plaquettes = state.get_plaquettes()?;
        let nonzero = plaquettes.iter().filter(|x| !x.is_zero()).count();
        let violations = state.get_edges_with_violations()?;
        assert_eq!(nonzero, 10);
        assert!(violations.is_empty());
        Ok(())
    }
    #[test]
    fn test_simple_launch_replicas() -> Result<(), CudaError> {
        let mut state = CudaBackend::new(
            SiteIndex::new(4, 4, 4, 4),
            make_potentials(3, 32),
            None,
            None,
            None,
        )?;
        state.run_single_local_update_sweep(0, false)
    }
}
