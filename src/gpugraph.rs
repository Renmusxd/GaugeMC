use crate::util::MeshIterator;
use crate::{Dimension, NDDualGraph, SiteIndex};
use bytemuck;
use log::{info, warn};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, Array6, ArrayView5, Axis};
use ndarray_rand::rand::rngs::SmallRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use std::borrow::Cow;
use std::cmp::max;
use std::iter::repeat;
use wgpu;
use wgpu::util::DeviceExt;
use wgpu::DeviceType;

pub enum WindingNumsOption {
    Gpu,
    Cpu,
    OldCpu,
}

impl Default for WindingNumsOption {
    fn default() -> Self {
        Self::Gpu
    }
}

pub enum EnergyOption {
    Gpu,
    Cpu,
    CpuIfPresent,
}

impl Default for EnergyOption {
    fn default() -> Self {
        Self::Gpu
    }
}

pub enum ParallelTemperingOption {
    Gpu,
    Cpu,
    CpuIfPresent,
}

impl Default for ParallelTemperingOption {
    fn default() -> Self {
        Self::Gpu
    }
}

pub struct GPUBackend {
    rng: Option<SmallRng>,
    // In order of vn, not matching GPU internal state.
    state: Option<(usize, Array6<i32>)>,
    shape: SiteIndex,
    num_replicas: usize,
    num_pcgs: usize,
    vn: Array2<f32>,
    replica_index_to_vn_index: Vec<usize>,
    vn_index_to_replica_index: Vec<usize>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    state_buffer: wgpu::Buffer,
    vn_buffer: wgpu::Buffer,
    pcgstate_buffer: wgpu::Buffer,
    sum_buffer: wgpu::Buffer,
    localupdate: LocalUpdatePipeline,
    globalupdate: GlobalUpdatePipeline,
    pcgupdate: PCGRotatePipeline,
    sum_energy_planes: SumEnergyPipeline,
    sum_winding_planes: WindingPipeline,
    copy_state_pipeline: CopyPipeline,
    bindgroup: wgpu::BindGroup,
    // Debugging information
    tempering_debug: Option<ParallelTemperingDebug>,
    winding_nums_option: WindingNumsOption,
    energy_option: EnergyOption,
    parallel_tempering_option: ParallelTemperingOption,

    optimize_writing_args: Option<bool>,
    written_arguments: Option<Vec<u32>>,
    heatbath: Option<bool>,
}

struct ParallelTemperingDebug {
    swap_attempts: Vec<u64>,
    swap_successes: Vec<u64>,
}

impl ParallelTemperingDebug {
    fn new(num_replicas: usize) -> Self {
        Self {
            swap_attempts: vec![0; num_replicas - 1],
            swap_successes: vec![0; num_replicas - 1],
        }
    }

    fn get_swap_probs(&self) -> Vec<f64> {
        self.swap_successes
            .iter()
            .zip(self.swap_attempts.iter())
            .map(|(s, a)| {
                if *a > 0 {
                    (*s as f64) / (*a as f64)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// A swap attempt between lower_replica and lower_replica + 1
    fn add_attempt(&mut self, lower_replica: usize, success: bool) {
        self.swap_attempts[lower_replica] += 1;
        if success {
            self.swap_successes[lower_replica] += 1;
        }
    }

    fn clear(&mut self) {
        self.swap_attempts.iter_mut().for_each(|x| *x = 0);
        self.swap_successes.iter_mut().for_each(|x| *x = 0);
    }
}

struct LocalUpdatePipeline {
    index_buffer: wgpu::Buffer,
    update_pipeline: wgpu::ComputePipeline,
}

struct GlobalUpdatePipeline {
    update_pipeline: wgpu::ComputePipeline,
}

struct PCGRotatePipeline {
    update_pipeline: wgpu::ComputePipeline,
}

struct SumEnergyPipeline {
    init_sum_pipeline: wgpu::ComputePipeline,
    inc_sum_pipeline: wgpu::ComputePipeline,
}

struct WindingPipeline {
    winding_sum_pipeline: wgpu::ComputePipeline,
}

struct CopyPipeline {
    copy_state_pipeline: wgpu::ComputePipeline,
}

const WORKGROUP: usize = 256;
const INIT_REDUX: usize = 16;

impl GPUBackend {
    pub async fn new_async(
        bounds: SiteIndex,
        vn: Array2<f32>,
        initial_state: Option<Array6<i32>>,
        seed: Option<u64>,
        device_id: Option<u32>,
    ) -> Result<Self, String> {
        let (t, x, y, z) = (bounds.t, bounds.x, bounds.y, bounds.z);
        for d in [t, x, y, z] {
            if d % 2 == 1 {
                return Err(format!(
                    "Expected all dims to be even, found: {:?}",
                    [t, x, y, z]
                ));
            }
        }

        let num_replicas = vn.shape()[0];
        if num_replicas == 0 {
            return Err("Replica number must be larger than 0".to_string());
        }

        let n_faces = num_replicas * t * x * y * z * 6;
        if let Some(initial_state) = &initial_state {
            if [num_replicas, t, x, y, z, 6] != initial_state.shape() {
                return Err(format!(
                    "Expected initial state with shape: {:?} found {:?}",
                    [num_replicas, t, x, y, z, 6],
                    initial_state.shape()
                ));
            }
        }

        let n_planes = num_replicas * (t * (x + y + z) + x * (y + z) + y * z);
        let num_pcgs = max(n_faces / 2, n_planes);

        let int_size = std::mem::size_of::<i32>();
        let float_size = std::mem::size_of::<f32>();
        let index_size = std::mem::size_of::<u32>();

        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = if let Some(device_id) = device_id {
            let mut adapters = instance
                .enumerate_adapters(wgpu::Backends::all()).into_iter()
                .filter(|x| x.get_info().device_type != DeviceType::Cpu)
                .filter(|x| x.get_info().device == device_id)
                .collect::<Vec<_>>();
            info!(
                "Found {} adapters with device_id={}",
                adapters.len(),
                device_id
            );
            if let Some(a) = adapters.pop() {
                Ok(a)
            } else {
                let adapters = instance
                    .enumerate_adapters(wgpu::Backends::all()).into_iter()
                    .map(|a| a.get_info())
                    .collect::<Vec<_>>();
                warn!(
                    "No adapters found with device_id={} (compared to {} unfiltered)",
                    device_id,
                    adapters.len()
                );
                info!("List of adapters: {:?}", adapters);
                Err(format!("No adapter found for device_id={}", device_id))
            }
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .ok_or_else(|| "GPU: Instance was not able to request an adapter".to_string())
        }?;
        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_bind_groups = 5;
        limits.max_storage_buffers_per_shader_stage = 5;
        limits.max_storage_buffer_binding_size = max(
            limits.max_storage_buffer_binding_size,
            (n_planes * int_size) as u32,
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .map_err(|f| format!("GPU Error: {:?}", f))?;

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("kernels/shader.wgsl"))),
        });

        let index_buffer_entries = 10 + num_replicas;

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((n_faces * int_size) as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((vn.len() * float_size) as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (index_buffer_entries * index_size) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((n_faces / 2) * int_size) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((n_faces / INIT_REDUX) * float_size) as _,
                            ),
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DualGaugeLayout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });
        let localupdate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("localupdate pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "main_local",
                compilation_options: Default::default(),
            });
        let globalupdate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("globalupdate pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "main_global",
                compilation_options: Default::default(),
            });
        let rotate_pcg_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rotate pcg pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "rotate_pcg",
                compilation_options: Default::default(),
            });
        let init_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("initial energy summation pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "initial_sum_energy",
            compilation_options: Default::default(),
        });
        let inc_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("incremental energy summation pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "incremental_sum_energy",
            compilation_options: Default::default(),
        });
        let winding_sum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("winding summation pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "calculate_winding_numbers",
                compilation_options: Default::default(),
            });
        let copy_state_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("copy state pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "copy_replica",
                compilation_options: Default::default(),
            });

        let initial_state = if let Some(initial_state) = initial_state {
            initial_state
        } else {
            Array6::zeros((num_replicas, t, x, y, z, 6))
        };

        let state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plaquette Buffer"),
            contents: bytemuck::cast_slice(
                initial_state
                    .as_slice()
                    .expect("Initial state not contiguous in memory!"),
            ),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vn_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vn Buffer"),
            contents: bytemuck::cast_slice(
                vn.as_slice().expect("Potentials not contiguous in memory!"),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&vec![0_u32; index_buffer_entries]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let mut rng = if let Some(seed) = seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };

        let contents = (0..num_pcgs).map(|_| rng.gen()).collect::<Vec<u32>>();
        let pcgstate_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PCG Buffer"),
            contents: bytemuck::cast_slice(&contents),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sum_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sum Buffer"),
            contents: bytemuck::cast_slice(&vec![0_u32; n_faces / INIT_REDUX]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vn_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pcgstate_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sum_buffer.as_entire_binding(),
                },
            ],
            label: None,
        });

        Ok(Self {
            rng: Some(rng),
            state: Some((num_replicas, initial_state)),
            shape: bounds,
            num_replicas,
            num_pcgs,
            vn,
            replica_index_to_vn_index: (0..num_replicas).collect(),
            vn_index_to_replica_index: (0..num_replicas).collect(),
            device,
            queue,
            state_buffer,
            vn_buffer,
            pcgstate_buffer,
            sum_buffer,
            bindgroup,
            localupdate: LocalUpdatePipeline {
                index_buffer,
                update_pipeline: localupdate_pipeline,
            },
            globalupdate: GlobalUpdatePipeline {
                update_pipeline: globalupdate_pipeline,
            },
            pcgupdate: PCGRotatePipeline {
                update_pipeline: rotate_pcg_pipeline,
            },
            sum_energy_planes: SumEnergyPipeline {
                init_sum_pipeline,
                inc_sum_pipeline,
            },
            sum_winding_planes: WindingPipeline {
                winding_sum_pipeline,
            },
            copy_state_pipeline: CopyPipeline {
                copy_state_pipeline,
            },
            tempering_debug: Some(ParallelTemperingDebug::new(num_replicas)),
            winding_nums_option: WindingNumsOption::default(),
            energy_option: EnergyOption::default(),
            parallel_tempering_option: ParallelTemperingOption::default(),
            optimize_writing_args: None,
            written_arguments: None,
            heatbath: None,
        })
    }

    pub fn set_heatbath(&mut self, use_heatbath: Option<bool>) {
        self.heatbath = use_heatbath;
    }

    pub fn set_optimize_args(&mut self, use_optimized_args: Option<bool>) {
        self.optimize_writing_args = use_optimized_args;
    }

    pub fn get_bounds(&self) -> SiteIndex {
        self.shape.clone()
    }

    pub fn get_num_replicas(&self) -> usize {
        self.num_replicas
    }

    pub fn get_potentials(&self) -> &Array2<f32> {
        &self.vn
    }

    pub fn num_planes(&self) -> usize {
        let (t, x, y, z) = (self.shape.t, self.shape.x, self.shape.y, self.shape.z);
        let tnums = t * (x + y + z);
        let xnums = x * (y + z);
        tnums + xnums + y * z
    }

    pub fn run_pcg_rotate(&mut self) {
        self.run_pcg_rotate_offset(false);
        self.run_pcg_rotate_offset(true);
    }

    pub fn run_pcg_rotate_offset(&mut self, offset: bool) {
        self.queue.write_buffer(
            &self.localupdate.index_buffer,
            0_u64,
            bytemuck::cast_slice(&[self.num_pcgs as u32, if offset { 1 } else { 0 }]),
        );
        self.written_arguments = None;

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute pass
        command_encoder.push_debug_group(&format!("PCG Rotate Sweep: {}", offset));

        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Run PCG update"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pcgupdate.update_pipeline);
            cpass.set_bind_group(0, &self.bindgroup, &[]);

            let nneeded = self.num_pcgs / 2;
            let ndispatch = ((nneeded + (WORKGROUP - 1)) / WORKGROUP) as u32;
            cpass.dispatch_workgroups(ndispatch, 1, 1);
        }
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));
    }

    pub fn run_local_sweep(&mut self, dims: &[Dimension; 3], leftover: Dimension, offset: bool) {
        self.state = None;

        self.write_arguments(
            dims.iter()
                .map(|d| usize::from(*d).try_into().unwrap())
                .chain(Some(usize::from(leftover).try_into().unwrap()))
                .chain(Some(if offset { 1 } else { 0 })),
            None,
        );

        let mu = self.shape.get(dims[0]) as u32;
        let nu = self.shape.get(dims[1]) as u32;
        let sigma = self.shape.get(dims[2]) as u32;
        let rho = self.shape.get(leftover) as u32;

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute pass
        command_encoder.push_debug_group(&format!("Local Sweep: {:?}", dims));

        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Run local sweep"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.localupdate.update_pipeline);
            cpass.set_bind_group(0, &self.bindgroup, &[]);

            let cubes_per_replica = rho * mu * nu * (sigma / 2);
            let nneeded = self.num_replicas * cubes_per_replica as usize;
            let ndispatch = ((nneeded + (WORKGROUP - 1)) / WORKGROUP) as u32;
            cpass.dispatch_workgroups(ndispatch, 1, 1);
        }
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));
        debug_assert_eq!(self.get_edges_with_violations(), Ok(vec![]));
    }

    pub fn run_global_sweep(&mut self, skip_last: Option<usize>) {
        self.state = None;
        let num_replicas = self.num_replicas - skip_last.unwrap_or_default();
        self.write_arguments(None, Some(num_replicas));

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute pass
        command_encoder.push_debug_group("Global Sweep");

        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Run global sweep"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.globalupdate.update_pipeline);
            cpass.set_bind_group(0, &self.bindgroup, &[]);

            let (t, x, y, z) = (self.shape.t, self.shape.x, self.shape.y, self.shape.z);
            let planes_per_replica = t * (x + y + z) + x * (y + z) + y * z;
            let nneeded = num_replicas * planes_per_replica;
            let ndispatch = ((nneeded + (WORKGROUP - 1)) / WORKGROUP) as u32;

            cpass.dispatch_workgroups(ndispatch, 1, 1);
        }
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));
        debug_assert_eq!(self.get_edges_with_violations(), Ok(vec![]));
    }

    pub fn swap_replica_potentials<It>(&mut self, start: usize, stop: usize, it: It)
        where
            It: IntoIterator<Item=bool>,
    {
        let entries = &mut self.vn_index_to_replica_index[start..stop];
        let state = self
            .state
            .as_mut()
            .and_then(|(i, s)| if *i == stop { Some(s) } else { None });
        if let Some(state) = state {
            state
                .slice_axis_mut(Axis(0), ndarray::Slice::from(start..stop))
                .axis_chunks_iter_mut(Axis(0), 2)
                .zip(entries.chunks_exact_mut(2))
                .zip(it)
                .filter_map(|((a, b), c)| {
                    if c && a.shape()[0] == 2 {
                        Some((a, b))
                    } else {
                        None
                    }
                })
                .for_each(|(state_chunk, vn_chunk)| {
                    let a = vn_chunk[0];
                    let b = vn_chunk[1];
                    vn_chunk[0] = b;
                    vn_chunk[1] = a;

                    let (a_s, b_s) = state_chunk.split_at(Axis(0), 1);
                    ndarray::Zip::from(a_s)
                        .and(b_s)
                        .into_par_iter()
                        .for_each(|(a, b)| std::mem::swap(&mut (*a), &mut (*b)));
                });
        } else {
            entries
                .chunks_exact_mut(2)
                .zip(it)
                .filter(|(_, b)| *b)
                .for_each(|(chunk, _)| {
                    let a = chunk[0];
                    let b = chunk[1];
                    chunk[0] = b;
                    chunk[1] = a;
                });
        }
        for (i, v) in self.vn_index_to_replica_index.iter().copied().enumerate() {
            self.replica_index_to_vn_index[v] = i;
        }
    }

    pub fn get_base_and_modified_energies(
        &mut self,
        start: usize,
        stop: usize,
        skip_last: Option<usize>,
    ) -> Result<(Array1<f32>, Array1<f32>), String> {
        let reset_to = match self.parallel_tempering_option {
            // Remove precalculated state so it cannot be used.
            ParallelTemperingOption::Gpu => Some(self.state.take()),
            // Calculate the state to force it to be present.
            ParallelTemperingOption::Cpu => {
                self.calculate_state(skip_last)?;
                None
            }
            // Do nothing, use CPU if available otherwise use gpu.
            ParallelTemperingOption::CpuIfPresent => None,
        };

        let base_energies = self.get_energy(skip_last)?;
        self.swap_replica_potentials(start, stop, repeat(true));
        let modified_energies = self.get_energy(skip_last)?;
        self.swap_replica_potentials(start, stop, repeat(true));
        // Should now be back to where we were before.
        debug_assert_eq!(base_energies, self.get_energy(skip_last)?);

        // Put precalculated state back if it was stored.
        if let Some(reset_to) = reset_to {
            self.state = reset_to;
        }

        Ok((base_energies, modified_energies))
    }

    pub fn run_parallel_tempering_sweep(
        &mut self,
        offset: bool,
        skip_last: Option<usize>,
    ) -> Result<(), String> {
        let start = if offset { 1 } else { 0 };
        let stop = self.num_replicas - skip_last.unwrap_or_default();
        let heatbath = self.heatbath.unwrap_or_default();

        let (base_energies, modified_energies) =
            self.get_base_and_modified_energies(start, stop, skip_last)?;

        let mut rng = self.rng.take().unwrap();
        let mut tempering_debug = self.tempering_debug.take().unwrap();
        let bases = &base_energies.as_slice().unwrap()[start..stop];
        let modifieds = &modified_energies.as_slice().unwrap()[start..stop];

        let swap_choices = modifieds
            .chunks_exact(2)
            .zip(bases.chunks_exact(2))
            .map(|(chunk_mod, chunk_base)| {
                // mod - base
                let action = (chunk_mod[0] - chunk_base[0]) + (chunk_mod[1] - chunk_base[1]);
                if heatbath {
                    // e(-mod) / e(-base) + e(-mod) == e(-mod + base) / 1 + e(-mod + base);
                    let weight = (-action).exp();
                    weight / (1.0 + weight)
                } else {
                    // min(1.0, e(-mod)/e(-base)) = min(1.0, e(-mod + base))
                    // = e(max(0.0, -mod+base))
                    let metro_minus_action = if action < 0.0 { 0.0 } else { -action };
                    metro_minus_action.exp()
                }
            })
            .map(|prob| rng.gen_bool(prob as f64))
            .enumerate()
            .map(|(i, b)| {
                // Log the bools
                let r = 2 * i + start;
                tempering_debug.add_attempt(r, b);
                b
            });

        self.swap_replica_potentials(start, stop, swap_choices);
        self.rng = Some(rng);
        self.tempering_debug = Some(tempering_debug);
        debug_assert_eq!(self.get_edges_with_violations(), Ok(vec![]));
        Ok(())
    }

    pub fn get_parallel_tempering_success_rate(&self) -> Vec<f64> {
        self.tempering_debug.as_ref().unwrap().get_swap_probs()
    }

    /// Force clear the stored state.
    pub fn clear_stored_state(&mut self) {
        self.state = None;
    }

    pub fn get_energy_from_cpu(&mut self, skip_last: Option<usize>) -> Result<Array1<f32>, String> {
        // If forced and not calculated.
        self.calculate_state(skip_last)?;
        // sum t, x, y, z
        let potentials = self.get_potentials();
        let state = self.get_precalculated_state().unwrap();

        let state_subslice = state.slice_axis(
            Axis(0),
            ndarray::Slice::from(..self.num_replicas - skip_last.unwrap_or_default()),
        );
        let potential_subslice = potentials.slice_axis(
            Axis(0),
            ndarray::Slice::from(..self.num_replicas - skip_last.unwrap_or_default()),
        );

        let res = state_subslice
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(potential_subslice.axis_iter(Axis(0)).into_par_iter())
            .map(|(s, potential)| {
                s.into_par_iter()
                    .map(|s| potential[s.unsigned_abs() as usize])
                    .sum()
            })
            .collect::<Vec<_>>();
        let res = Array1::from_vec(res);
        Ok(res)
    }

    pub fn get_energy_from_gpu(&mut self, skip_last: Option<usize>) -> Result<Array1<f32>, String> {
        let (t, x, y, z, p) = (self.shape.t, self.shape.x, self.shape.y, self.shape.z, 6);
        let buff_size = t * x * y * z * p / INIT_REDUX;
        let mut threads_per_replica = buff_size;

        self.write_arguments(None, None);

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute pass
        command_encoder.push_debug_group("Initial Energy Summation");

        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Run initial sum pipeline"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sum_energy_planes.init_sum_pipeline);
            cpass.set_bind_group(0, &self.bindgroup, &[]);

            let nneeded = self.num_replicas * threads_per_replica;
            let ndispatch = ((nneeded + (WORKGROUP - 1)) / WORKGROUP) as u32;

            cpass.dispatch_workgroups(ndispatch, 1, 1);
        }

        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));

        while threads_per_replica > 1 {
            self.write_arguments(Some(threads_per_replica as u32), None);

            // get command encoder
            let mut command_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // compute pass
            command_encoder.push_debug_group("Incremental Energy Summation");
            {
                let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Run incremental sum pipeline"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.sum_energy_planes.inc_sum_pipeline);
                cpass.set_bind_group(0, &self.bindgroup, &[]);

                threads_per_replica = threads_per_replica / 2 + threads_per_replica % 2;
                let nneeded = self.num_replicas * threads_per_replica;
                let ndispatch = ((nneeded + (WORKGROUP - 1)) / WORKGROUP) as u32;
                cpass.dispatch_workgroups(ndispatch, 1, 1);
            }
            command_encoder.pop_debug_group();

            self.queue.submit(Some(command_encoder.finish()));
        }

        let num_replicas = self.num_replicas - skip_last.unwrap_or_default();
        self.read_energy_from_gpu(Some(num_replicas))
    }

    /// Get the energy of each replica, allow forcing from stored state, or not.
    pub fn get_energy(&mut self, skip_last: Option<usize>) -> Result<Array1<f32>, String> {
        match self.energy_option {
            EnergyOption::Gpu => self.get_energy_from_gpu(skip_last),
            EnergyOption::CpuIfPresent if self.state.is_none() => {
                self.get_energy_from_gpu(skip_last)
            }
            _ => self.get_energy_from_cpu(skip_last),
        }
    }

    pub fn set_winding_num_method(&mut self, method: WindingNumsOption) {
        self.winding_nums_option = method;
    }
    pub fn set_energy_method(&mut self, method: EnergyOption) {
        self.energy_option = method;
    }

    pub fn set_parallel_tempering_method(&mut self, method: ParallelTemperingOption) {
        self.parallel_tempering_option = method;
    }

    pub fn get_winding_nums(&mut self, skip_last: Option<usize>) -> Result<Array2<i32>, String> {
        match self.winding_nums_option {
            WindingNumsOption::Gpu => self.get_winding_nums_gpu(skip_last),
            WindingNumsOption::Cpu => self.get_winding_nums_cpu(skip_last),
            WindingNumsOption::OldCpu => self.get_winding_nums_cpu_old(skip_last),
        }
    }

    pub fn get_winding_nums_cpu_old(
        &mut self,
        skip_last: Option<usize>,
    ) -> Result<Array2<i32>, String> {
        let mut sum = self
            .get_state(skip_last)?
            .sum_axis(Axis(1))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1));
        let bounds = &self.shape;
        let plane_sizes = [
            bounds.t * bounds.x,
            bounds.t * bounds.y,
            bounds.t * bounds.z,
            bounds.x * bounds.y,
            bounds.x * bounds.z,
            bounds.y * bounds.z,
        ];
        sum.axis_iter_mut(Axis(0)).for_each(|mut arr| {
            arr.iter_mut().zip(plane_sizes.iter()).for_each(|(s, n)| {
                *s /= *n as i32;
            })
        });
        Ok(sum)
    }

    pub fn get_winding_nums_cpu(
        &mut self,
        skip_last: Option<usize>,
    ) -> Result<Array2<i32>, String> {
        let num_replicas = self.num_replicas - skip_last.unwrap_or_default();
        let mut res = Array2::zeros((num_replicas, 6));
        let state = self.get_state(skip_last)?;
        ndarray::Zip::indexed(&mut res)
            .into_par_iter()
            .for_each(|((rep, p), res)| {
                let subslice = match p {
                    0 => state.slice(s![rep, 0, 0, .., .., p]),
                    1 => state.slice(s![rep, 0, .., 0, .., p]),
                    2 => state.slice(s![rep, 0, .., .., 0, p]),
                    3 => state.slice(s![rep, .., 0, 0, .., p]),
                    4 => state.slice(s![rep, .., 0, .., 0, p]),
                    5 => state.slice(s![rep, .., .., 0, 0, p]),
                    _ => unreachable!(),
                };
                *res = subslice.iter().copied().sum();
            });

        Ok(res)
    }

    pub fn get_winding_nums_gpu(
        &mut self,
        skip_last: Option<usize>,
    ) -> Result<Array2<i32>, String> {
        let p = 6;
        let threads_per_replica = p;
        let num_replicas = self.num_replicas - skip_last.unwrap_or_default();

        self.write_arguments(None, Some(num_replicas));

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute pass
        command_encoder.push_debug_group("Winding Num Summation");

        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Run winding sum pipeline"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sum_winding_planes.winding_sum_pipeline);
            cpass.set_bind_group(0, &self.bindgroup, &[]);

            let nneeded = num_replicas * threads_per_replica;
            let ndispatch = ((nneeded + (WORKGROUP - 1)) / WORKGROUP) as u32;

            cpass.dispatch_workgroups(ndispatch, 1, 1);
        }

        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));

        let raw_windings = self
            .read_sumbuffer_from_gpu(p * num_replicas)
            .into_iter()
            .map(|f| f as i32)
            .collect::<Array1<i32>>()
            .into_shape((num_replicas, p))
            .map_err(|se| se.to_string())?;
        // Now reorder by parallel temperings.
        let mut result = Array2::zeros((num_replicas, p));
        ndarray::Zip::indexed(&mut result)
            .into_par_iter()
            .for_each(|((ir, ip), v)| {
                let ir = self.vn_index_to_replica_index[ir];
                *v = raw_windings[[ir, ip]];
            });
        Ok(result)
    }

    pub fn write_potentials(&mut self, vn: Array2<f32>) {
        self.queue.write_buffer(
            &self.vn_buffer,
            0_u64,
            bytemuck::cast_slice(vn.as_slice().expect("Memory not contiguous")),
        );
        self.vn = vn;
    }

    pub fn write_state(&mut self, replica: usize, state: ArrayView5<i32>) -> Result<(), String> {
        let own_shape = self.shape.shape();
        let own_shape = own_shape.into_iter().chain([6]).collect::<Vec<_>>();
        if own_shape != state.shape() {
            return Err(format!(
                "Given state had shape {:?} but expected {:?}",
                state.shape(),
                own_shape.as_slice()
            ));
        }

        let volume = self.shape.volume() * 6;
        let offset = (std::mem::size_of::<u32>() * replica * volume) as u64;
        let state = state.as_slice().ok_or("state not contiguous")?;
        self.queue
            .write_buffer(&self.state_buffer, offset, bytemuck::cast_slice(state));
        if let Some((_, old_state)) = self.state.as_mut() {
            if replica < old_state.shape()[0] {
                old_state
                    .index_axis_mut(Axis(0), replica)
                    .iter_mut()
                    .zip(state.iter().copied())
                    .for_each(|(v, x)| *v = x)
            }
        }

        Ok(())
    }

    pub fn write_arguments<It>(&mut self, it: It, num_replicas_overwrite: Option<usize>)
        where
            It: IntoIterator<Item=u32>,
    {
        let vals = [
            self.shape.t as u32,
            self.shape.x as u32,
            self.shape.y as u32,
            self.shape.z as u32,
            num_replicas_overwrite.unwrap_or(self.num_replicas) as u32,
        ]
            .into_iter()
            // Put in all the vn offsets.
            .chain((0..self.num_replicas).map(|r| {
                let r_rot = self.replica_index_to_vn_index[r];
                (self.vn.shape()[1] * r_rot) as u32
            }))
            // Add the additional arguments
            .chain(it)
            .collect::<Vec<u32>>();
        let nvals = vals.len();

        let optimize_args = self.optimize_writing_args.unwrap_or(true);
        if optimize_args {
            if Some(&vals) != self.written_arguments.as_ref() {
                if let Some(old_vals) = self.written_arguments.as_ref() {
                    let first_diff = vals
                        .iter()
                        .zip(old_vals.iter())
                        .enumerate()
                        .find(|(_, (a, b))| a.ne(b))
                        .map(|(i, _)| i)
                        .unwrap_or(old_vals.len());
                    if first_diff < vals.len() {
                        let offset = (std::mem::size_of::<u32>() * first_diff) as u64;
                        self.queue.write_buffer(
                            &self.localupdate.index_buffer,
                            offset,
                            bytemuck::cast_slice(&vals[first_diff..]),
                        );
                        self.written_arguments = Some(vals);
                    }
                    // If the difference comes after the end of vals (vals is a subsequence of old_vals)
                    // then no need to write.
                } else {
                    self.queue.write_buffer(
                        &self.localupdate.index_buffer,
                        0_u64,
                        bytemuck::cast_slice(vals.as_slice()),
                    );
                    self.written_arguments = Some(vals);
                }
            }
        } else {
            self.queue.write_buffer(
                &self.localupdate.index_buffer,
                0_u64,
                bytemuck::cast_slice(vals.as_slice()),
            );
            self.written_arguments = Some(vals);
        }

        debug_assert_eq!(
            Some(
                self.read_argbuffer_from_gpu(max(
                    nvals,
                    self.written_arguments
                        .as_ref()
                        .map(|x| x.len())
                        .unwrap_or_default(),
                ))
            ),
            self.written_arguments
        );
    }

    pub fn read_argbuffer_from_gpu(&mut self, read_num_values: usize) -> Vec<u32> {
        let to_read = read_num_values * std::mem::size_of::<u32>();

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: to_read as _,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            command_encoder.push_debug_group("Copy args");
            command_encoder.copy_buffer_to_buffer(
                &self.localupdate.index_buffer,
                0,
                &staging_buffer,
                0,
                to_read as _,
            );
            command_encoder.pop_debug_group();
        }
        self.queue.submit(Some(command_encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Gets the future representing when `staging_buffer` can be read from
        buffer_slice.map_async(wgpu::MapMode::Read, |_| ());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(wgpu::Maintain::Wait);

        // Gets contents of buffer
        // location R uses potential Vi = rotation[R]
        let data = buffer_slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }

    pub fn wait_for_gpu(&mut self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn read_sumbuffer_from_gpu(&mut self, read_num_values: usize) -> Vec<f32> {
        let to_read = read_num_values * std::mem::size_of::<f32>();

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: to_read as _,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            command_encoder.push_debug_group("Copy energies");
            command_encoder.copy_buffer_to_buffer(
                &self.sum_buffer,
                0,
                &staging_buffer,
                0,
                to_read as _,
            );
            command_encoder.pop_debug_group();
        }
        self.queue.submit(Some(command_encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Gets the future representing when `staging_buffer` can be read from
        buffer_slice.map_async(wgpu::MapMode::Read, |_| ());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(wgpu::Maintain::Wait);

        // Gets contents of buffer
        // location R uses potential Vi = rotation[R]
        let data = buffer_slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }

    pub fn read_energy_from_gpu(
        &mut self,
        read_num_values: Option<usize>,
    ) -> Result<Array1<f32>, String> {
        let read_num_values = read_num_values.unwrap_or(self.num_replicas);
        let data_vec = self.read_sumbuffer_from_gpu(read_num_values);
        let mut result = Array1::zeros((read_num_values, ));
        result.iter_mut().enumerate().for_each(|(r, v)| {
            let r_rot = self.vn_index_to_replica_index[r];
            *v = data_vec[r_rot];
        });

        // // // With the current interface, we have to make sure all mapped views are
        // // // dropped before we unmap the buffer.
        // drop(data);
        // buf.unmap(); // Unmaps buffer from memory
        Ok(result)
    }

    pub fn copy_state_on_gpu(&mut self, from: usize, to: usize, swap: bool) {
        self.state = None;
        self.write_arguments([from as u32, to as u32, if swap { 1 } else { 0 }], None);

        let buff_size = self.shape.t * self.shape.x * self.shape.y * self.shape.z;
        let nneeded = buff_size;

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // compute pass
        command_encoder.push_debug_group("Copy replica.");

        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Copy replica onto device"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.copy_state_pipeline.copy_state_pipeline);
            cpass.set_bind_group(0, &self.bindgroup, &[]);

            let ndispatch = ((nneeded + (WORKGROUP - 1)) / WORKGROUP) as u32;

            cpass.dispatch_workgroups(ndispatch, 1, 1);
        }

        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));
    }

    pub fn calculate_state(&mut self, skip_last: Option<usize>) -> Result<(), String> {
        let num_replicas = self.num_replicas - skip_last.unwrap_or_default();
        let should_run = self
            .state
            .as_ref()
            .map(|(i, _)| num_replicas.ne(i))
            .unwrap_or(true);

        if should_run {
            let (t, x, y, z, p) = (self.shape.t, self.shape.x, self.shape.y, self.shape.z, 6);
            let n_faces = num_replicas * t * x * y * z * p;
            let int_size = std::mem::size_of::<i32>();
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n_faces * int_size) as _,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // get command encoder
            let mut command_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                command_encoder.push_debug_group("Copy state");
                command_encoder.copy_buffer_to_buffer(
                    &self.state_buffer,
                    0,
                    &staging_buffer,
                    0,
                    (n_faces * int_size) as _,
                );
                command_encoder.pop_debug_group();
            }
            self.queue.submit(Some(command_encoder.finish()));

            // Note that we're not calling `.await` here.
            let buffer_slice = staging_buffer.slice(..);
            // Gets the future representing when `staging_buffer` can be read from
            buffer_slice.map_async(wgpu::MapMode::Read, |_| ());

            // Poll the device in a blocking manner so that our future resolves.
            // In an actual application, `device.poll(...)` should
            // be called in an event loop or on another thread.
            self.device.poll(wgpu::Maintain::Wait);

            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            let data_vec: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
            let mut result = Array6::zeros((num_replicas, t, x, y, z, 6));
            ndarray::Zip::indexed(&mut result).into_par_iter().for_each(
                |((ir, it, ix, iy, iz, ip), v)| {
                    let ir = self.vn_index_to_replica_index[ir];
                    let replica_offset = ir * (t * x * y * z * p);
                    let t_offset = it * x * y * z * p;
                    let x_offset = ix * y * z * p;
                    let y_offset = iy * z * p;
                    let z_offset = iz * p;
                    let p_offset = ip;
                    let indx =
                        replica_offset + t_offset + x_offset + y_offset + z_offset + p_offset;
                    *v = data_vec[indx];
                },
            );
            self.state = Some((num_replicas, result));

            // // // With the current interface, we have to make sure all mapped views are
            // // // dropped before we unmap the buffer.
            // drop(data);
            // buf.unmap(); // Unmaps buffer from memory

            Ok(())
        } else {
            Ok(())
        }
    }

    pub fn get_state(&mut self, skip_last: Option<usize>) -> Result<&Array6<i32>, String> {
        self.calculate_state(skip_last)?;
        // Returns data from buffer
        self.state
            .as_ref()
            .map(|(_, s)| s)
            .ok_or_else(|| "State not stored".to_string())
    }

    pub fn get_precalculated_state(&self) -> Option<&Array6<i32>> {
        self.state.as_ref().map(|(_, s)| s)
    }

    pub fn clear_parallel_tempering_data(&mut self) {
        if let Some(x) = self.tempering_debug.as_mut() {
            x.clear()
        }
    }

    pub fn get_edges_with_violations(
        &mut self,
    ) -> Result<Vec<((usize, SiteIndex, Dimension), Vec<(SiteIndex, usize)>)>, String> {
        let shape = self.shape.clone();
        let edge_iterator = MeshIterator::new([
            self.get_num_replicas(),
            shape.t,
            shape.x,
            shape.y,
            shape.z,
            4,
        ])
            .build_parallel_iterator()
            .map(|[r, t, x, y, z, d]| (r, SiteIndex { t, x, y, z }, Dimension::from(d)));

        self.calculate_state(None)?;
        let state = self.get_precalculated_state().unwrap();
        let res = edge_iterator
            .filter_map(|(r, s, d)| {
                let (poss, negs) = NDDualGraph::plaquettes_next_to_edge(&s, d, &self.shape);
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

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use ndarray::{Array, Array5};

    #[test]
    fn test_run_global_except_last() -> Result<(), String> {
        let (r, t, x, y, z) = (6, 4, 4, 4, 4);
        let mut vn = Array2::zeros((r, 3));
        vn.axis_iter_mut(Axis(0))
            .for_each(|mut ax| {
                ax.iter_mut().enumerate().for_each(|(i, v)| {
                    if i == 0 {
                        *v = 0.0;
                    } else {
                        *v = 10.0;
                    }
                })
            });
        let mut graph = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(t, x, y, z),
            vn,
            Some(Array6::ones((r, t, x, y, z, 6))),
            None,
            None,
        ))?;

        graph.run_global_sweep(Some(1));
        let state = graph.get_state(Some(1))?;
        state.indexed_iter().for_each(|((_ir, _, _, _, _, _), v)| {
            assert_eq!(*v, 0);
        });
        let state = graph.get_state(None)?;
        state.indexed_iter().for_each(|((ir, _, _, _, _, _), v)| {
            if ir == r - 1 {
                assert_eq!(*v, 1);
            } else {
                assert_eq!(*v, 0);
            }
        });
        Ok(())
    }

    #[test]
    fn test_global_respect_replica_potential() -> Result<(), String> {
        let (r, t, x, y, z) = (4, 4, 4, 4, 4);
        let mut vn = Array2::zeros((r, 3));
        vn.axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(r, mut ax)| {
                ax.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = 6.0 * ((r + 1) as f32) * (i.pow(2)) as f32;
                    if r % 2 == 1 {
                        *v = -*v;
                    }
                })
            });
        let mut graph = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(t, x, y, z),
            vn,
            None,
            None,
            None,
        ))?;

        graph.run_global_sweep(None);

        let state = graph.get_state(None)?;
        println!("{:?}", state);

        state.indexed_iter().for_each(|((r, _, _, _, _, _), v)| {
            if r % 2 == 0 {
                assert_eq!(*v, 0);
            } else {
                assert_eq!(v.abs(), 1);
            }
        });

        graph.run_global_sweep(None);

        let state = graph.get_state(None)?;

        state.indexed_iter().for_each(|((r, _, _, _, _, _), v)| {
            if r % 2 == 0 {
                assert_eq!(*v, 0);
            } else {
                assert_eq!(v.abs(), 2);
            }
        });
        Ok(())
    }

    fn make_replica_value_state(
        r: usize,
        t: usize,
        x: usize,
        y: usize,
        z: usize,
    ) -> Result<GPUBackend, String> {
        let mut vn = Array2::zeros((r, 10));
        vn.axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(r, mut ax)| {
                ax.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = ((r + 1) as f32) * (i.pow(2)) as f32;
                })
            });
        let mut state = Array6::zeros((r, t, x, y, z, 6));
        state
            .indexed_iter_mut()
            .for_each(|((r, _, _, _, _, _), v)| {
                *v = r as i32;
            });
        pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(t, x, y, z),
            vn,
            Some(state),
            None,
            None,
        ))
    }

    #[test]
    fn test_winding_num_calc() -> Result<(), String> {
        let mut s = make_replica_value_state(3, 4, 4, 4, 4)?;
        for i in 0..100 {
            s.run_global_sweep(None);
            s.swap_replica_potentials(i % 2, 3, repeat(true));
            s.swap_replica_potentials(i % 2, 3, repeat(true));
            let cpu_w = s.get_winding_nums_cpu(None)?;
            let gpu_w = s.get_winding_nums_gpu(None)?;
            let old_w = s.get_winding_nums_cpu_old(None)?;
            assert_eq!(cpu_w, gpu_w);
            assert_eq!(cpu_w, old_w);
        }
        println!("{:?}", s.get_parallel_tempering_success_rate());
        Ok(())
    }

    #[test]
    fn test_winding_num_calc_skip() -> Result<(), String> {
        let mut s = make_replica_value_state(8, 4, 4, 4, 4)?;
        for i in 0..100 {
            s.run_global_sweep(None);
            s.swap_replica_potentials(i % 2, 3, repeat(true));
            s.swap_replica_potentials(i % 2, 3, repeat(true));
            let cpu_w = s.get_winding_nums_cpu(Some(5))?;
            let gpu_w = s.get_winding_nums_gpu(Some(5))?;
            let old_w = s.get_winding_nums_cpu_old(Some(5))?;
            assert_eq!(cpu_w, gpu_w);
            assert_eq!(cpu_w, old_w);
        }
        println!("{:?}", s.get_parallel_tempering_success_rate());
        Ok(())
    }

    #[test]
    fn test_energy_calc() -> Result<(), String> {
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        let energies = s.get_energy_from_gpu(None)?;
        s.swap_replica_potentials(0, 4, repeat(true));
        let swapped_state = s.get_energy_from_gpu(None)?;

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.swap_replica_potentials(0, 4, repeat(true));
        let calced_swapped_state = s.get_energy_from_gpu(None)?;

        let n_faces = 4 * 4 * 4 * 4 * 6;
        let energies_slice = energies
            .as_slice()
            .ok_or_else(|| "Not a slice".to_string())?;
        let swapped_energies_slice = swapped_state
            .as_slice()
            .ok_or_else(|| "Not a slice".to_string())?;
        let calced_swapped_energies_slice = calced_swapped_state
            .as_slice()
            .ok_or_else(|| "Not a slice".to_string())?;
        assert_eq!(
            energies_slice,
            &[
                1.0 * 0.0 * (n_faces as f32),
                2.0 * 1.0 * (n_faces as f32),
                3.0 * 4.0 * (n_faces as f32),
                4.0 * 9.0 * (n_faces as f32),
            ]
        );
        assert_eq!(
            swapped_energies_slice,
            &[
                1.0 * 1.0 * (n_faces as f32),
                2.0 * 0.0 * (n_faces as f32),
                3.0 * 9.0 * (n_faces as f32),
                4.0 * 4.0 * (n_faces as f32),
            ]
        );
        assert_eq!(calced_swapped_energies_slice, swapped_energies_slice);

        Ok(())
    }

    #[test]
    fn test_energy_calc_skip_last() -> Result<(), String> {
        for skip in [0, 1, 2] {
            let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
            let energies = s.get_energy_from_gpu(Some(skip))?;

            let n_faces = 4 * 4 * 4 * 4 * 6;
            let energies_slice = energies
                .as_slice()
                .ok_or_else(|| "Not a slice".to_string())?;
            assert_eq!(
                energies_slice,
                &[
                    1.0 * 0.0 * (n_faces as f32),
                    2.0 * 1.0 * (n_faces as f32),
                    3.0 * 4.0 * (n_faces as f32),
                    4.0 * 9.0 * (n_faces as f32),
                ][..4 - skip]
            );
        }
        Ok(())
    }

    #[test]
    fn test_energy_calc_cpu() -> Result<(), String> {
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        let energies = s.get_energy_from_cpu(None)?;
        s.swap_replica_potentials(0, 4, repeat(true));
        let swapped_state = s.get_energy_from_cpu(None)?;

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.swap_replica_potentials(0, 4, repeat(true));
        let calced_swapped_state = s.get_energy_from_cpu(None)?;

        let n_faces = 4 * 4 * 4 * 4 * 6;
        let energies_slice = energies
            .as_slice()
            .ok_or_else(|| "Not a slice".to_string())?;
        let swapped_energies_slice = swapped_state
            .as_slice()
            .ok_or_else(|| "Not a slice".to_string())?;
        let calced_swapped_energies_slice = calced_swapped_state
            .as_slice()
            .ok_or_else(|| "Not a slice".to_string())?;
        assert_eq!(
            energies_slice,
            &[
                1.0 * 0.0 * (n_faces as f32),
                2.0 * 1.0 * (n_faces as f32),
                3.0 * 4.0 * (n_faces as f32),
                4.0 * 9.0 * (n_faces as f32),
            ]
        );
        assert_eq!(
            swapped_energies_slice,
            &[
                1.0 * 1.0 * (n_faces as f32),
                2.0 * 0.0 * (n_faces as f32),
                3.0 * 9.0 * (n_faces as f32),
                4.0 * 4.0 * (n_faces as f32),
            ]
        );
        assert_eq!(calced_swapped_energies_slice, swapped_energies_slice);

        Ok(())
    }

    #[test]
    fn test_energy_calc_cpu_skip_last() -> Result<(), String> {
        for skip in [0, 1, 2] {
            let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
            let energies = s.get_energy_from_cpu(Some(skip))?;

            let n_faces = 4 * 4 * 4 * 4 * 6;
            let energies_slice = energies
                .as_slice()
                .ok_or_else(|| "Not a slice".to_string())?;
            assert_eq!(
                energies_slice,
                &[
                    1.0 * 0.0 * (n_faces as f32),
                    2.0 * 1.0 * (n_faces as f32),
                    3.0 * 4.0 * (n_faces as f32),
                    4.0 * 9.0 * (n_faces as f32),
                ][..4 - skip]
            );
        }
        Ok(())
    }

    #[test]
    fn test_swap_states() -> Result<(), String> {
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        let basic_state = s.get_state(None)?.clone();
        s.clear_stored_state();
        s.swap_replica_potentials(0, 4, repeat(true));
        let cleared_swapped_state = s.get_state(None)?.clone();

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.calculate_state(None)?;
        s.swap_replica_potentials(0, 4, repeat(true));
        let calced_swapped_state = s.get_state(None)?.clone();

        let rubrick = [0, 1, 2, 3];
        basic_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        let rubrick = [1, 0, 3, 2];
        cleared_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        calced_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        Ok(())
    }

    #[test]
    fn test_offset_swap_states() -> Result<(), String> {
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        let basic_state = s.get_state(None)?.clone();
        s.clear_stored_state();
        s.swap_replica_potentials(1, 4, repeat(true));
        let cleared_swapped_state = s.get_state(None)?.clone();

        let mut s = make_replica_value_state(3, 4, 4, 4, 4)?;
        s.calculate_state(None)?;
        s.swap_replica_potentials(1, 3, repeat(true));
        let calced_swapped_state = s.get_state(None)?.clone();

        let rubrick = [0, 1, 2, 3];
        basic_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        let rubrick = [0, 2, 1, 3];
        cleared_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        calced_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        Ok(())
    }

    #[test]
    fn test_swap_states_some() -> Result<(), String> {
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        let basic_state = s.get_state(None)?.clone();
        s.clear_stored_state();
        s.swap_replica_potentials(0, 4, [true, false]);
        let cleared_swapped_state = s.get_state(None)?.clone();

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.calculate_state(None)?;
        s.swap_replica_potentials(0, 4, [true, false]);
        let calced_swapped_state = s.get_state(None)?.clone();

        let rubrick = [0, 1, 2, 3];
        basic_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        let rubrick = [1, 0, 2, 3];
        cleared_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        calced_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        let basic_state = s.get_state(None)?.clone();
        s.clear_stored_state();
        s.swap_replica_potentials(0, 4, [false, true]);
        let cleared_swapped_state = s.get_state(None)?.clone();

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.calculate_state(None)?;
        s.swap_replica_potentials(0, 4, [false, true]);
        let calced_swapped_state = s.get_state(None)?.clone();

        let rubrick = [0, 1, 2, 3];
        basic_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        let rubrick = [0, 1, 3, 2];
        cleared_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        calced_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        Ok(())
    }

    #[test]
    fn test_offset_swap_states_some() -> Result<(), String> {
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        let basic_state = s.get_state(None)?.clone();
        s.clear_stored_state();
        s.swap_replica_potentials(1, 4, [true]);
        let cleared_swapped_state = s.get_state(None)?.clone();

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.calculate_state(None)?;
        s.swap_replica_potentials(1, 4, [true]);
        let calced_swapped_state = s.get_state(None)?.clone();

        let rubrick = [0, 1, 2, 3];
        basic_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        let rubrick = [0, 2, 1, 3];
        cleared_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        calced_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        let basic_state = s.get_state(None)?.clone();
        s.clear_stored_state();
        s.swap_replica_potentials(0, 4, [false]);
        let cleared_swapped_state = s.get_state(None)?.clone();

        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.calculate_state(None)?;
        s.swap_replica_potentials(0, 4, [false]);
        let calced_swapped_state = s.get_state(None)?.clone();

        let rubrick = [0, 1, 2, 3];
        basic_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        cleared_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        calced_swapped_state
            .indexed_iter()
            .for_each(|((r, _, _, _, _, _), v)| {
                assert_eq!(*v, rubrick[r]);
            });
        Ok(())
    }

    #[test]
    fn test_parallel_swap_back_gpu() -> Result<(), String> {
        // Make a state where np is higher for larger potentials,
        // energetically preferable to swap back.
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.set_parallel_tempering_method(ParallelTemperingOption::Gpu);
        let _energies = s.get_energy(None)?;
        s.swap_replica_potentials(0, 4, repeat(true));
        let swapped_energy = s.get_energy(None)?;
        let swap_state = s.get_state(None)?;

        // Reinit, and check if it swaps.
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.run_parallel_tempering_sweep(false, None)?;

        assert_eq!(swapped_energy, s.get_energy(None)?);
        assert_eq!(swap_state, s.get_state(None)?);

        Ok(())
    }

    #[test]
    fn test_parallel_swap_back_cpu() -> Result<(), String> {
        // Make a state where np is higher for larger potentials,
        // energetically preferable to swap back.
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.set_parallel_tempering_method(ParallelTemperingOption::Cpu);

        let _energies = s.get_energy(None)?;
        s.swap_replica_potentials(0, 4, repeat(true));
        let swapped_energy = s.get_energy(None)?;
        let swap_state = s.get_state(None)?;

        // Reinit, and check if it swaps.
        let mut s = make_replica_value_state(4, 4, 4, 4, 4)?;
        s.run_parallel_tempering_sweep(false, None)?;

        assert_eq!(swapped_energy, s.get_energy(None)?);
        assert_eq!(swap_state, s.get_state(None)?);

        Ok(())
    }

    #[test]
    fn test_copy_replica() -> Result<(), String> {
        let (r, t, x, y, z) = (3, 2, 2, 2, 2);
        let vn = Array2::zeros((r, 3));
        let mut init_state = Array::zeros((r, t, x, y, z, 6));
        init_state.iter_mut().enumerate().for_each(|(i, x)| {
            *x = i as i32;
        });

        let mut graph = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(t, x, y, z),
            vn,
            Some(init_state),
            None,
            None,
        ))?;

        let state = graph.get_state(None)?;
        assert_ne!(state.index_axis(Axis(0), 0), state.index_axis(Axis(0), 1));
        assert_ne!(state.index_axis(Axis(0), 0), state.index_axis(Axis(0), 2));
        assert_ne!(state.index_axis(Axis(0), 1), state.index_axis(Axis(0), 2));
        graph.copy_state_on_gpu(0, 2, false);
        let state = graph.get_state(None)?;
        assert_ne!(state.index_axis(Axis(0), 0), state.index_axis(Axis(0), 1));
        assert_eq!(state.index_axis(Axis(0), 0), state.index_axis(Axis(0), 2));
        assert_ne!(state.index_axis(Axis(0), 1), state.index_axis(Axis(0), 2));
        graph.copy_state_on_gpu(0, 1, false);
        let state = graph.get_state(None)?;
        assert_eq!(state.index_axis(Axis(0), 0), state.index_axis(Axis(0), 1));
        assert_eq!(state.index_axis(Axis(0), 0), state.index_axis(Axis(0), 2));
        assert_eq!(state.index_axis(Axis(0), 1), state.index_axis(Axis(0), 2));
        Ok(())
    }

    #[test]
    fn test_skip_tempering() -> Result<(), String> {
        let (r, t, x, y, z) = (16, 4, 4, 4, 4);
        let vn = Array2::zeros((r, 32));
        let mut init_state = Array::zeros((r, t, x, y, z, 6));
        ndarray::Zip::indexed(&mut init_state).for_each(|i, x| {
            *x = i.0 as i32;
        });

        let mut graph = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(t, x, y, z),
            vn,
            Some(init_state),
            None,
            None,
        ))?;

        let skip_last = 3;
        for i in 0..10 {
            graph.clear_stored_state();
            graph.run_parallel_tempering_sweep(i % 2 == 0, Some(skip_last))?;
        }
        let state = graph.get_state(None)?;
        ndarray::Zip::indexed(state).for_each(|(ir, _, _, _, _, _), val| {
            if ir >= r - skip_last {
                assert_eq!(*val, ir as i32);
            }
        });
        Ok(())
    }

    #[test]
    fn test_set_state() -> Result<(), String> {
        let (r, t, x, y, z) = (6, 2, 2, 2, 2);
        let vn = Array2::zeros((r, 3));
        let mut init_state = Array::zeros((r, t, x, y, z, 6));
        init_state.iter_mut().enumerate().for_each(|(i, x)| {
            *x = i as i32;
        });

        let mut graph = pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(t, x, y, z),
            vn,
            Some(init_state),
            None,
            None,
        ))?;

        let mut set_to = Array5::zeros((t, x, y, z, 6));
        for ir in 0..r {
            set_to.iter_mut().for_each(|x| *x = ir as i32);
            graph.write_state(ir, set_to.view())?;
        }
        graph.clear_stored_state();

        let gpu_state = graph.get_state(None)?;
        ndarray::Zip::indexed(gpu_state).for_each(|(r, _, _, _, _, _), v| {
            assert_eq!(*v, r as i32);
        });
        graph.clear_stored_state();

        for ir in (0..r).rev() {
            set_to.iter_mut().for_each(|x| *x = ir as i32);
            graph.write_state(ir, set_to.view())?;
        }
        graph.clear_stored_state();

        let gpu_state = graph.get_state(None)?;
        ndarray::Zip::indexed(gpu_state).for_each(|(r, _, _, _, _, _), v| {
            assert_eq!(*v, r as i32);
        });

        Ok(())
    }
}
