use crate::{Dimension, NDDualGraph, SiteIndex};
use bytemuck;
use ndarray::{Array1, Array2, Array6, Axis};
use ndarray_rand::rand::rngs::SmallRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::max;
use wgpu;
use wgpu::util::DeviceExt;

pub struct GPUBackend {
    state: Option<Array6<i32>>,
    shape: SiteIndex,
    num_replicas: usize,
    num_pcgs: usize,
    vn: Vec<f32>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    state_buffer: wgpu::Buffer,
    vn_buffer: wgpu::Buffer,
    pcgstate_buffer: wgpu::Buffer,
    localupdate: LocalUpdatePipeline,
    globalupdate: GlobalUpdatePipeline,
    pcgupdate: PCGRotatePipeline,
}

struct LocalUpdatePipeline {
    index_buffer: wgpu::Buffer,
    update_pipeline: wgpu::ComputePipeline,
    bindgroup: wgpu::BindGroup,
}
struct GlobalUpdatePipeline {
    update_pipeline: wgpu::ComputePipeline,
    bindgroup: wgpu::BindGroup,
}
struct PCGRotatePipeline {
    update_pipeline: wgpu::ComputePipeline,
    bindgroup: wgpu::BindGroup,
}

impl GPUBackend {
    pub async fn new_async(
        t: usize,
        x: usize,
        y: usize,
        z: usize,
        vn: Vec<f32>,
        num_replicas: Option<usize>,
        initial_state: Option<Array6<i32>>,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        for d in [t, x, y, z] {
            if d % 2 == 1 {
                return Err(format!(
                    "Expected all dims to be even, found: {:?}",
                    [t, x, y, z]
                ));
            }
        }
        let num_replicas = num_replicas.unwrap_or(1);
        if num_replicas == 0 {
            return Err("num_replicas must be larger than 0".to_string());
        }

        let bounds = SiteIndex { t, x, y, z };
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

        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let compute_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let int_size = std::mem::size_of::<i32>();
        let float_size = std::mem::size_of::<f32>();
        let index_size = std::mem::size_of::<u32>();

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
                            min_binding_size: wgpu::BufferSize::new((10 * index_size) as _),
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
                ],
                label: None,
            });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("localupdate"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let localupdate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("localupdate pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "main",
            });
        let globalupdate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("globalupdate pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "main_global",
            });
        let rotate_pcg_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rotate pcg pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "rotate_pcg",
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let vn_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vn Buffer"),
            contents: bytemuck::cast_slice(&vn),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&[0_u32; 10]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        let mut bindgroups = (0..3).map(|_| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                ],
                label: None,
            })
        });
        let local_b = bindgroups.next().unwrap();
        let global_b = bindgroups.next().unwrap();
        let pcg_b = bindgroups.next().unwrap();

        Ok(Self {
            state: None,
            shape: bounds,
            num_replicas,
            num_pcgs,
            vn,
            device,
            queue,
            state_buffer,
            vn_buffer,
            pcgstate_buffer,
            localupdate: LocalUpdatePipeline {
                index_buffer,
                update_pipeline: localupdate_pipeline,
                bindgroup: local_b,
            },
            globalupdate: GlobalUpdatePipeline {
                update_pipeline: globalupdate_pipeline,
                bindgroup: global_b,
            },
            pcgupdate: PCGRotatePipeline {
                update_pipeline: rotate_pcg_pipeline,
                bindgroup: pcg_b,
            },
        })
    }

    pub fn get_bounds(&self) -> SiteIndex {
        self.shape.clone()
    }

    pub fn get_num_replicas(&self) -> usize {
        self.num_replicas
    }
    pub fn get_potential(&self) -> &[f32] {
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
        let ndims = [
            self.shape.t,
            self.shape.x,
            self.shape.y,
            self.shape.z,
            self.num_replicas,
            self.num_pcgs,
            if offset { 1 } else { 0 },
        ]
        .into_iter()
        .map(|x| x as u32)
        .collect::<Vec<_>>();

        self.queue.write_buffer(
            &self.localupdate.index_buffer,
            0_u64,
            bytemuck::cast_slice(&ndims),
        );

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute pass
        command_encoder.push_debug_group(&format!("PCG Rotate Sweep: {}", offset));

        {
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pcgupdate.update_pipeline);
            cpass.set_bind_group(0, &self.pcgupdate.bindgroup, &[]);

            let nneeded = self.num_pcgs / 2;
            let ndispatch = ((nneeded + 255) / 256) as u32;
            cpass.dispatch(ndispatch, 1, 1);
        }
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));
    }

    pub fn run_local_sweep(&mut self, dims: &[Dimension; 3], leftover: Dimension, offset: bool) {
        self.state = None;
        let ndims = [
            self.shape.t,
            self.shape.x,
            self.shape.y,
            self.shape.z,
            self.num_replicas,
        ]
        .into_iter()
        .map(|d| d as u32)
        .chain(dims.iter().map(|d| usize::from(*d).try_into().unwrap()))
        .chain(Some(usize::from(leftover).try_into().unwrap()))
        .chain(Some(if offset { 1 } else { 0 }))
        .collect::<Vec<u32>>();

        self.queue.write_buffer(
            &self.localupdate.index_buffer,
            0_u64,
            bytemuck::cast_slice(&ndims),
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
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.localupdate.update_pipeline);
            cpass.set_bind_group(0, &self.localupdate.bindgroup, &[]);

            let cubes_per_replica = rho * mu * nu * (sigma / 2);
            let nneeded = self.num_replicas * cubes_per_replica as usize;
            let ndispatch = ((nneeded + 255) / 256) as u32;
            cpass.dispatch(ndispatch, 1, 1);
        }
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));
    }

    pub fn run_global_sweep(&mut self) {
        self.state = None;
        // Write bounds in case they aren't there.
        self.queue.write_buffer(
            &self.localupdate.index_buffer,
            0_u64,
            bytemuck::cast_slice(&[
                self.shape.t as u32,
                self.shape.x as u32,
                self.shape.y as u32,
                self.shape.z as u32,
                self.num_replicas as u32,
            ]),
        );

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute pass
        command_encoder.push_debug_group("Global Sweep");

        {
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.globalupdate.update_pipeline);
            cpass.set_bind_group(0, &self.globalupdate.bindgroup, &[]);

            let (t, x, y, z) = (self.shape.t, self.shape.x, self.shape.y, self.shape.z);
            let planes_per_replica = t * (x + y + z) + x * (y + z) + y * z;
            let nneeded = self.num_replicas * planes_per_replica;
            let ndispatch = ((nneeded + 255) / 256) as u32;

            cpass.dispatch(ndispatch, 1, 1);
        }
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));
    }

    pub fn get_energy(&mut self) -> Result<Array1<f32>, String> {
        // sum t, x, y, z
        self.calculate_state()?;
        let potential = self.get_potential();
        let res = self
            .get_precalculated_state()
            .unwrap()
            .axis_iter(Axis(0))
            .map(|s| s.iter().map(|s| potential[s.abs() as usize]).sum())
            .collect();
        Ok(res)
    }

    pub fn get_winding_nums(&mut self) -> Result<Array2<i32>, String> {
        let mut sum = self
            .get_state()?
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

    pub fn calculate_state(&mut self) -> Result<(), String> {
        if self.state.is_none() {
            let (t, x, y, z) = (self.shape.t, self.shape.x, self.shape.y, self.shape.z);
            let n_faces = self.num_replicas * t * x * y * z * 6;
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
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

            // Poll the device in a blocking manner so that our future resolves.
            // In an actual application, `device.poll(...)` should
            // be called in an event loop or on another thread.
            self.device.poll(wgpu::Maintain::Wait);

            // Awaits until `buffer_future` can be read from
            if let Ok(()) = pollster::block_on(buffer_future) {
                // Gets contents of buffer
                let data = buffer_slice.get_mapped_range();
                let result: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
                let result = Array6::from_shape_vec((self.num_replicas, t, x, y, z, 6), result)
                    .expect("Incorrect shape");
                self.state = Some(result);

                // // // With the current interface, we have to make sure all mapped views are
                // // // dropped before we unmap the buffer.
                // drop(data);
                // buf.unmap(); // Unmaps buffer from memory

                Ok(())
            } else {
                Err("failed to run compute on gpu!".to_string())
            }
        } else {
            Ok(())
        }
    }

    pub fn get_state(&mut self) -> Result<&Array6<i32>, String> {
        self.calculate_state()?;
        // Returns data from buffer
        self.state
            .as_ref()
            .ok_or_else(|| "State not stored".to_string())
    }

    pub fn get_precalculated_state(&self) -> Option<&Array6<i32>> {
        self.state.as_ref()
    }

    pub fn get_edges_with_violations(
        &mut self,
    ) -> Result<Vec<((usize, SiteIndex, Dimension), Vec<(SiteIndex, usize)>)>, String> {
        let shape = self.shape.clone();
        let edge_iterator = (0..self.get_num_replicas()).flat_map(|r| {
            (0..shape.t).flat_map(move |t| {
                (0..shape.x).flat_map(move |x| {
                    (0..shape.y).flat_map(move |y| {
                        (0..shape.z).flat_map(move |z| {
                            (0..4usize)
                                .map(Dimension::from)
                                .map(move |d| (r, SiteIndex { t, x, y, z }, d))
                        })
                    })
                })
            })
        });
        self.calculate_state()?;
        let state = self.get_precalculated_state().unwrap();
        let res = edge_iterator
            .par_bridge()
            .filter_map(|(r, s, d)| {
                let (poss, negs) = NDDualGraph::plaquettes_next_to_edge(&s, d, &self.shape);
                let sum = poss
                    .iter()
                    .cloned()
                    .map(|p| (p, 1))
                    .chain(negs.iter().cloned().map(|n| (n, -1)))
                    .map(|((site, p), mult)| {
                        state
                            .get(ndarray::Ix6(r, site.t, site.x, site.y, site.z, p))
                            .unwrap()
                            * mult
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
