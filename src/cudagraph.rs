use crate::util::MeshIterator;
use crate::{Dimension, NDDualGraph, SiteIndex};
use cudarc::curand::result::CurandError;
use cudarc::curand::CudaRng;
use cudarc::driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx, CompileError};
use ndarray::{Array1, Array2, Array6, ArrayView1, ArrayView6, Axis};
use ndarray_rand::rand::prelude::SliceRandom;
use ndarray_rand::rand::{random, thread_rng, Rng};
use rayon::prelude::*;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

pub struct CudaBackend {
    nreplicas: usize,
    bounds: SiteIndex,
    potential_size: usize,
    device: Arc<CudaDevice>,
    state: CudaSlice<i32>,
    potential_redirect_buffer: CudaSlice<u32>,
    potential_redirect_array: RedirectArrays,
    potential_buffer: CudaSlice<f32>,
    winding_chemical_potential_buffer: CudaSlice<f32>, // potential per plane
    using_chemical_potential: Option<Array1<f32>>,
    rng_buffer: CudaSlice<f32>,
    cuda_rng: CudaRng,
    local_update_types: Option<Vec<(u16, bool)>>,
}

enum RedirectArrays {
    None,
    Redirect {
        redirect: Array1<u32>,
        inverse: Array1<u32>,
    },
}

impl RedirectArrays {
    fn unwrap(&mut self, n: usize) -> (ArrayView1<u32>, ArrayView1<u32>) {
        match self {
            RedirectArrays::None => {
                *self = Self::new((0..n as u32).collect::<Vec<_>>());
                self.unwrap(n)
            }
            RedirectArrays::Redirect { redirect, inverse } => (redirect.view(), inverse.view()),
        }
    }

    fn get_redirect(&self) -> Option<&[u32]> {
        match self {
            RedirectArrays::None => None,
            RedirectArrays::Redirect { redirect, .. } => redirect.as_slice(),
        }
    }

    fn new<Arr>(redirect: Arr) -> Self
    where
        Arr: Into<Array1<u32>>,
    {
        let redirect = redirect.into();
        let mut inverse = redirect.clone();

        for (i, r) in redirect.iter().enumerate() {
            inverse[*r as usize] = i as u32;
        }
        Self::new_both(redirect, inverse)
    }

    fn new_both<Arr1, Arr2>(redirect: Arr1, inverse: Arr2) -> Self
    where
        Arr1: Into<Array1<u32>>,
        Arr2: Into<Array1<u32>>,
    {
        let redirect = redirect.into();
        let inverse = inverse.into();

        debug_assert!(redirect
            .iter()
            .enumerate()
            .all(|(i, x)| inverse[*x as usize] == i as u32));

        let redirect = Array1::from_vec(redirect.into_iter().map(|x| x).collect::<Vec<_>>());
        let inverse = Array1::from_vec(inverse.into_iter().map(|x| x).collect::<Vec<_>>());

        Self::Redirect { redirect, inverse }
    }
}
impl Debug for RedirectArrays {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RedirectArrays::None => f.write_str("[Identity]"),
            RedirectArrays::Redirect { redirect, .. } => {
                f.write_str(&format!("R{:?}", redirect.as_slice().unwrap()))
            }
        }
    }
}

/// A state in the language of integer plaquettes
#[derive(Clone)]
pub struct DualState {
    // Shape (replicas, T, X, Y, Z, 5)
    plaquettes: Array6<i32>,
}

impl DualState {
    pub fn get_plaquettes(&self) -> ArrayView6<i32> {
        self.plaquettes.view()
    }

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
        chemical_potential: Option<Array1<f32>>,
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
                    "sum_winding",
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

        let num_plaquettes = nreplicas * t * x * y * z * 6;

        let mut plaquette_buffer = device
            .alloc_zeros::<i32>(num_plaquettes)
            .map_err(CudaError::from)?;

        if let Some(DualState { plaquettes }) = dual_initial_state {
            let plaquettes = plaquettes.into_iter().collect::<Vec<_>>();
            device
                .htod_copy_into(plaquettes, &mut plaquette_buffer)
                .map_err(CudaError::from)?;
        }

        // Set up the potentials
        let vn = vn.iter().copied().collect::<Vec<_>>();
        let mut potential_buffer = device
            .alloc_zeros::<f32>(vn.len())
            .map_err(CudaError::from)?;
        device
            .htod_copy_into(vn, &mut potential_buffer)
            .map_err(CudaError::from)?;

        let mut using_chemical_potential = None;
        let mut winding_chemical_potential_buffer = device
            .alloc_zeros::<f32>(nreplicas)
            .map_err(CudaError::from)?;
        if let Some(chemical_potential_arr) = chemical_potential {
            let chemical_potential = chemical_potential_arr.iter().copied().collect::<Vec<_>>();
            device
                .htod_copy_into(chemical_potential, &mut winding_chemical_potential_buffer)
                .map_err(CudaError::from)?;
            using_chemical_potential = Some(chemical_potential_arr);
        }

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
            state: plaquette_buffer,
            potential_redirect_buffer,
            potential_redirect_array: RedirectArrays::None,
            potential_buffer,
            winding_chemical_potential_buffer,
            using_chemical_potential,
            rng_buffer,
            cuda_rng,
            local_update_types: Some(local_update_types),
        })
    }

    pub fn permute_states(&mut self, permutation: &[usize]) -> Result<(), CudaError> {
        let in_order = Array1::from_vec((0..self.nreplicas as u32).collect::<Vec<_>>());
        let (plaquette_to_potential, potential_to_plaquette) =
            self.potential_redirect_array.unwrap(self.nreplicas);

        let mut new_redirect = vec![0; self.nreplicas];
        for (pot_i, pot_j) in permutation.iter().copied().enumerate() {
            let state_i = potential_to_plaquette[pot_i];
            let state_j = potential_to_plaquette[pot_j];
            new_redirect[state_i as usize] = plaquette_to_potential[state_j as usize];
        }

        self.device
            .htod_copy_into(new_redirect.clone(), &mut self.potential_redirect_buffer)?;
        let new_redirect = Array1::from_vec(new_redirect);
        if new_redirect == in_order {
            self.potential_redirect_array = RedirectArrays::None;
        } else {
            self.potential_redirect_array = RedirectArrays::new(new_redirect);
        }

        Ok(())
    }

    pub fn parallel_tempering_step(&mut self, swaps: &[(usize, usize)]) -> Result<(), CudaError> {
        let mut rng = thread_rng();
        let rand_values = (0..self.nreplicas).map(|_| rng.gen()).collect::<Vec<f32>>();
        self.parallel_tempering_step_with_rand(swaps, &rand_values)
    }

    fn parallel_tempering_step_with_rand(
        &mut self,
        swaps: &[(usize, usize)],
        random_values: &[f32],
    ) -> Result<(), CudaError> {
        let base_energies = self.get_action_per_replica()?;
        let mut perm = (0..self.nreplicas).collect::<Vec<_>>();
        for (a, b) in swaps.iter().copied() {
            perm[a] = b;
            perm[b] = a;
        }
        self.permute_states(&perm)?;
        let swap_energies = self.get_action_per_replica()?;

        for ((a, b), rng_value) in swaps.iter().copied().zip(random_values.iter().copied()) {
            let base_energy = base_energies[a] + base_energies[b];
            let swap_energy = swap_energies[a] + swap_energies[b];
            let weight = if swap_energy < base_energy {
                1.0 + f32::EPSILON
            } else {
                (-(swap_energy - base_energy)).exp()
            };
            // If swap energy is larger than base_energy: rng_value must be very small.

            let should_swap = weight >= rng_value;
            if should_swap {
                // If swap desired, leave as is.
                perm[a] = a;
                perm[b] = b;
            } else {
                // If no swap desired, undo the previously added swap.
                perm[a] = b;
                perm[b] = a;
            }
        }
        self.permute_states(&perm)?;

        Ok(())
    }

    pub fn wait_for_gpu(&mut self) -> Result<(), CudaError> {
        self.device.synchronize().map_err(CudaError::from)
    }

    pub fn get_plaquettes(&mut self) -> Result<Array6<i32>, CudaError> {
        let (t, x, y, z) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);
        let output = self
            .device
            .dtoh_sync_copy(&self.state)
            .map_err(CudaError::from)?;
        let mut plaquettes =
            Array6::from_shape_vec((self.nreplicas, t, x, y, z, 6), output).unwrap();

        if let Some(redirect) = self.potential_redirect_array.get_redirect() {
            let plaquettes_clone = plaquettes.clone();
            plaquettes_clone
                .axis_iter(Axis(0))
                .enumerate()
                .for_each(|(ir, x)| {
                    plaquettes
                        .index_axis_mut(Axis(0), redirect[ir] as usize)
                        .iter_mut()
                        .zip(x)
                        .for_each(|(x, y)| {
                            *x = *y;
                        });
                });
        }
        Ok(plaquettes)
    }

    pub fn get_edge_violations(&mut self) -> Result<Array6<i32>, CudaError> {
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

        let mut edges = Array6::from_shape_vec((self.nreplicas, t, x, y, z, 4), output).unwrap();

        if let Some(redirect) = self.potential_redirect_array.get_redirect() {
            let edges_clone = edges.clone();
            edges_clone
                .axis_iter(Axis(0))
                .enumerate()
                .for_each(|(ir, x)| {
                    edges
                        .index_axis_mut(Axis(0), redirect[ir] as usize)
                        .iter_mut()
                        .zip(x)
                        .for_each(|(x, y)| {
                            *x = *y;
                        });
                });
        }

        Ok(edges)
    }

    pub fn run_global_update_sweep(&mut self) -> Result<(), CudaError> {
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
                        &mut self.state,
                        &self.potential_buffer,
                        &self.winding_chemical_potential_buffer,
                        &self.potential_redirect_buffer,
                        self.potential_size,
                        &self.rng_buffer,
                        self.nreplicas,
                        self.bounds.t,
                        self.bounds.x,
                        self.bounds.y,
                        self.bounds.z,
                    ),
                )
                .map_err(CudaError::from)?
        };

        debug_assert_eq!(
            self.get_edge_violations()
                .map(|x| x.iter().map(|x| x.abs()).sum()),
            Ok(0)
        );

        Ok(())
    }

    pub fn get_winding_per_replica(&mut self) -> Result<Array2<i32>, CudaError> {
        let threads_to_sum = self.nreplicas * 6;
        let mut sum_buffer = self
            .device
            .alloc_zeros::<i32>(threads_to_sum)
            .map_err(CudaError::from)?;

        let cfg = LaunchConfig::for_num_elems(threads_to_sum as u32);
        let partial_sum_energies = self.device.get_func("gauge_kernel", "sum_winding").unwrap();

        unsafe {
            partial_sum_energies
                .launch(
                    cfg,
                    (
                        &self.state,
                        &mut sum_buffer,
                        self.nreplicas,
                        self.bounds.t,
                        self.bounds.x,
                        self.bounds.y,
                        self.bounds.z,
                    ),
                )
                .map_err(CudaError::from)?
        };
        let windings = self
            .device
            .dtoh_sync_copy(&sum_buffer)
            .map(|x| Array2::from_shape_vec((self.nreplicas, 6), x).unwrap())
            .map_err(CudaError::from)?;

        if let Some(redirect) = self.potential_redirect_array.get_redirect() {
            let mut result = windings.clone();
            windings
                .axis_iter(Axis(0))
                .enumerate()
                .for_each(|(ir, windings)| {
                    let rr = redirect[ir];
                    let mut write_axis = result.index_axis_mut(Axis(0), rr as usize);
                    write_axis
                        .iter_mut()
                        .zip(windings)
                        .for_each(|(x, y)| *x = *y);
                });
            Ok(result)
        } else {
            Ok(windings)
        }
    }

    pub fn get_action_per_replica(&mut self) -> Result<Array1<f32>, CudaError> {
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
        let energies = self
            .device
            .dtoh_sync_copy(&subslice)
            .map(Array1::from_vec)
            .map_err(CudaError::from)?;

        let mut energies = if let Some(redirect) = self.potential_redirect_array.get_redirect() {
            let mut result = energies.clone();
            energies.iter().enumerate().for_each(|(ir, windings)| {
                let rr = redirect[ir];
                result[rr as usize] = *windings;
            });
            result
        } else {
            energies
        };

        if let Some(chemical_potentials) = self.using_chemical_potential.take() {
            let windings = self.get_winding_per_replica()?;
            energies
                .iter_mut()
                .zip(windings.axis_iter(Axis(0)))
                .zip(chemical_potentials.iter())
                .map(|((a, b), c)| (a, b, c))
                .enumerate()
                .for_each(|(_r, (e, w, mu))| *e += w.sum() as f32 * (*mu));
            self.using_chemical_potential = Some(chemical_potentials);
        }

        Ok(energies)
    }

    pub fn run_local_update_sweep(&mut self) -> Result<(), CudaError> {
        let mut rng = thread_rng();
        let mut local_update_types = self.local_update_types.take().unwrap();
        local_update_types.shuffle(&mut rng);
        let res = local_update_types
            .iter()
            .copied()
            .try_for_each(|(volume, offset)| self.run_single_local_update_single(volume, offset));
        self.local_update_types = Some(local_update_types);

        debug_assert_eq!(
            self.get_edge_violations()
                .map(|x| x.iter().map(|x| x.abs()).sum()),
            Ok(0)
        );

        res
    }

    pub fn run_single_local_update_single(
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
                .map_err(CudaError::from)?
        };

        debug_assert_eq!(
            self.get_edge_violations()
                .map(|x| x.iter().map(|x| x.abs()).sum()),
            Ok(0)
        );

        Ok(())
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

    pub fn calculate_edge_violations(
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

#[derive(Debug, PartialEq)]
pub enum CudaError {
    Value(String),
    Compile(CompileError),
    Driver(DriverError),
    Rand(CurandError),
}

impl Display for CudaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Not great but it'll do. TODO fix.
        f.write_str(&format!("{:?}", self))
    }
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
    use ndarray::{arr1, arr2, s, Array, Axis};
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
            None,
        )?;
        state.run_single_local_update_single(0, false)
    }

    #[test]
    fn test_repeated_launch() -> Result<(), CudaError> {
        let mut state = CudaBackend::new(
            SiteIndex::new(8, 8, 8, 8),
            make_custom_simple_potentials(6, 32, |r, n| {
                0.25 * (r as f32 + 1.0) * (n.pow(2) as f32)
            }),
            None,
            Some(31415),
            None,
            None,
        )?;

        for _ in 0..100 {
            state.run_local_update_sweep()?
        }

        Ok(())
    }

    #[test]
    fn test_get_plaquettes() -> Result<(), CudaError> {
        let mut state = CudaBackend::new(
            SiteIndex::new(4, 4, 4, 4),
            make_simple_potentials(1, 2),
            None,
            Some(31415),
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
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 2),
            Some(DualState::new_volumes(state)),
            Some(31415),
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
        let state = DualState::new_volumes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 2),
            Some(state),
            Some(31415),
            None,
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
            None,
        )?;
        let edges = state.get_edge_violations()?;

        assert!(edges.iter().all(|x| x.is_zero()));

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
            None,
        )?;
        state.run_single_local_update_single(0, false)?;

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
            None,
        )?;
        state.run_single_local_update_single(0, false)?;

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
            None,
        )?;
        state.run_global_update_sweep()?;

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
        for nonzero_val in &nonzero[1..r] {
            assert_eq!(*nonzero_val, 0);
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
            None,
        )?;
        state.run_global_update_sweep()?;

        let plaqs = state.get_plaquettes()?;
        // We should get global planes in replica 0 and nowhere else.
        let nonzero = plaqs
            .axis_iter(Axis(0))
            .map(|x| x.iter().copied().filter(|x| !x.is_zero()).count())
            .collect::<Vec<_>>();
        assert_eq!(nonzero.len(), r);
        for nonzero_val in nonzero {
            assert_eq!(nonzero_val, 0);
        }
        Ok(())
    }

    #[test]
    fn test_winding_counts() -> Result<(), CudaError> {
        let (r, d) = (3, 8);

        let mut state = Array6::zeros((r, d, d, d, d, 6));
        state
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(rr, mut x)| {
                for i in 0..rr {
                    x.slice_mut(s![.., .., i, i, 0])
                        .iter_mut()
                        .for_each(|x| *x = 1);
                }
            });
        let state = DualState::new_plaquettes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_simple_potentials(r, 32),
            Some(state),
            Some(31415),
            None,
            None,
        )?;
        let windings = state.get_winding_per_replica()?;
        assert_eq!(
            windings,
            arr2(&[[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]])
        );
        Ok(())
    }

    #[test]
    fn test_get_edges_single_inc_rotate() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (5, 6, 6, 6, 6);
        let mut state = Array::zeros((r, t, x, y, z, 4));
        state[[0, 0, 0, 0, 0, 0]] = 1;
        let state = DualState::new_volumes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(state),
            Some(31415),
            None,
            None,
        )?;

        let perm = (1..r + 1).map(|x| x % r).collect::<Vec<_>>();
        for i in 0..2 * r {
            let mut correct_state = Array::zeros((r, t, x, y, z, 4));
            correct_state[[i % r, 0, 0, 0, 0, 0]] = 1;
            let correct_state = DualState::new_volumes(correct_state);
            let correct_plaquettes = correct_state.get_plaquettes().to_owned();
            let check_plaquettes = state.get_plaquettes()?;
            assert_eq!(check_plaquettes, correct_plaquettes);

            let mut energies_correct = Array1::from_vec(vec![0.0; r]);
            energies_correct[i % r] = 3.0;

            let energies = state.get_action_per_replica()?;
            assert_eq!(energies, energies_correct);

            state.permute_states(&perm)?;
        }

        Ok(())
    }

    #[test]
    fn test_get_edges_single_inc_rotate_swaps() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (5, 6, 6, 6, 6);
        let mut state = Array::zeros((r, t, x, y, z, 4));
        state[[0, 0, 0, 0, 0, 0]] = 1;
        let state = DualState::new_volumes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_simple_potentials(r, 32),
            Some(state),
            Some(31415),
            None,
            None,
        )?;

        assert_eq!(state.get_action_per_replica()?, arr1(&[3., 0., 0., 0., 0.]));
        state.permute_states(&[1, 0, 2, 3, 4])?;
        assert_eq!(state.get_action_per_replica()?, arr1(&[0., 3., 0., 0., 0.]));
        state.permute_states(&[0, 2, 1, 3, 4])?;
        assert_eq!(state.get_action_per_replica()?, arr1(&[0., 0., 3., 0., 0.]));
        state.permute_states(&[0, 1, 3, 2, 4])?;
        assert_eq!(state.get_action_per_replica()?, arr1(&[0., 0., 0., 3., 0.]));
        state.permute_states(&[0, 1, 2, 4, 3])?;
        assert_eq!(state.get_action_per_replica()?, arr1(&[0., 0., 0., 0., 3.]));

        Ok(())
    }

    #[test]
    fn test_reverse_order() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (5, 6, 6, 6, 6);
        let mut state = Array::zeros((r, t, x, y, z, 4));
        for rr in 0..r {
            state[[rr, 0, 0, 0, 0, 0]] = (rr + 1) as i32;
        }
        let state = DualState::new_volumes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_custom_simple_potentials(r, 32, |r, n| ((r + 1) * n) as f32),
            Some(state),
            Some(31415),
            None,
            None,
        )?;

        let expected_energies = (0..r)
            .map(|rr| 6 * (rr + 1) * (rr + 1))
            .map(|x| x as f32)
            .collect::<Vec<_>>();
        let energies = state.get_action_per_replica()?;

        assert_eq!(energies, Array1::from_vec(expected_energies));

        let mut perm = (0..r).collect::<Vec<_>>();
        perm.reverse();
        state.permute_states(&perm)?;
        let energies = state.get_action_per_replica()?;

        let expected_energies = (0..r)
            .map(|rr| 6 * (rr + 1) * ((r + 1) - (rr + 1)))
            .map(|x| x as f32)
            .collect::<Vec<_>>();
        assert_eq!(energies, Array1::from_vec(expected_energies));

        Ok(())
    }

    #[test]
    fn test_tempering() -> Result<(), CudaError> {
        let (r, t, x, y, z) = (6, 8, 8, 8, 8);

        // Make a state in reverse order: biggest ns in the last replica.
        let mut state = Array::zeros((r, t, x, y, z, 4));
        for rr in 0..r {
            state[[rr, 0, 0, 0, 0, 0]] = (1 + rr) as i32;
        }
        let state = DualState::new_volumes(state);

        // Make a potential which prefers big ns in the first replica.
        let mut state = CudaBackend::new(
            SiteIndex::new(t, x, y, z),
            make_custom_simple_potentials(r, 32, |r, n| ((r + 1) * n) as f32),
            Some(state),
            Some(31415),
            None,
            None,
        )?;

        let perms_a = (0..r / 2).map(|x| (2 * x, 2 * x + 1)).collect::<Vec<_>>();
        let perms_b = (0..(r - 1) / 2)
            .map(|x| (2 * x + 1, 2 * (x + 1)))
            .collect::<Vec<_>>();

        let rands = (0..r / 2).map(|_| 0.5).collect::<Vec<_>>();
        for i in 0..r {
            state.parallel_tempering_step_with_rand(
                if i % 2 == 0 { &perms_a } else { &perms_b },
                &rands,
            )?;
        }

        let plaquettes = state.get_plaquettes()?;
        let sums = plaquettes
            .axis_iter(Axis(0))
            .map(|x| x.iter().map(|y| y.abs()).sum::<i32>() / 6)
            .collect::<Vec<_>>();

        let mut against = (0..r as i32).map(|x| x + 1).collect::<Vec<_>>();
        against.reverse();
        assert_eq!(sums, against);

        // Run it again and make sure it doesn't change.

        for i in 0..r {
            state.parallel_tempering_step_with_rand(
                if i % 2 == 0 { &perms_a } else { &perms_b },
                &rands,
            )?;
        }

        let plaquettes = state.get_plaquettes()?;
        let sums = plaquettes
            .axis_iter(Axis(0))
            .map(|x| x.iter().map(|y| y.abs()).sum::<i32>() / 6)
            .collect::<Vec<_>>();

        assert_eq!(sums, against);

        Ok(())
    }

    #[test]
    fn test_chemical_potential_windings() -> Result<(), CudaError> {
        let (r, d) = (3, 4);

        let mut initial_state = Array6::zeros((r, d, d, d, d, 6));
        initial_state
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(rr, mut x)| {
                for i in 0..rr {
                    x.slice_mut(s![.., .., i, i, 0])
                        .iter_mut()
                        .for_each(|x| *x = 1);
                }
            });
        let state = DualState::new_plaquettes(initial_state.clone());
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_custom_simple_potentials(r, 4, |_, _| 0.0),
            Some(state),
            Some(31415),
            None,
            Some(Array1::from_vec((0..r).map(|_| 5.0).collect())),
        )?;

        state.run_global_update_sweep()?;

        let initial_rep_winds = initial_state
            .sum_axis(Axis(1))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1));

        let plaqs = state.get_plaquettes()?;
        let rep_winds = plaqs
            .sum_axis(Axis(1))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1));

        assert_eq!(initial_rep_winds + d.pow(4) as i32, rep_winds);

        let windings = state.get_winding_per_replica()?;

        assert_eq!(windings * d.pow(2) as i32, rep_winds);

        Ok(())
    }

    #[test]
    fn test_winding_chemical_potential() -> Result<(), CudaError> {
        let (r, d) = (3, 8);

        let mut state = Array6::zeros((r, d, d, d, d, 6));
        state
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(rr, mut x)| {
                for i in 0..rr {
                    x.slice_mut(s![.., .., i, i, 0])
                        .iter_mut()
                        .for_each(|x| *x = 1);
                }
            });
        let state = DualState::new_plaquettes(state);
        let mut state = CudaBackend::new(
            SiteIndex::new(d, d, d, d),
            make_custom_simple_potentials(r, 4, |_, _| 0.0),
            Some(state),
            Some(31415),
            None,
            Some(Array1::from_vec((0..r).map(|rr| (rr + 1) as f32).collect())),
        )?;
        let energies = state.get_action_per_replica()?;
        assert_eq!(energies, arr1(&[0.0 * 1.0, 1.0 * 2.0, 2.0 * 3.0]));

        state.permute_states(&[2, 1, 0])?;
        let energies = state.get_action_per_replica()?;
        assert_eq!(energies, arr1(&[2.0 * 1.0, 1.0 * 2.0, 0.0 * 3.0]));

        Ok(())
    }
}
