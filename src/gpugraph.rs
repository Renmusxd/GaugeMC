use crate::{Dimension, SiteIndex};
use bytemuck;
use std::borrow::Cow;
use wgpu;
use wgpu::util::DeviceExt;

pub struct GPUBackend {
    shape: SiteIndex,
    device: wgpu::Device,
    queue: wgpu::Queue,
    state_buffer: wgpu::Buffer,
    vn_buffer: wgpu::Buffer,
    localupdate: LocalUpdatePipeline,
}

struct LocalUpdatePipeline {
    index_buffer: wgpu::Buffer,
    localupdate_pipeline: wgpu::ComputePipeline,
    bindgroup: wgpu::BindGroup,
}

impl GPUBackend {
    pub async fn new_async(t: usize, x: usize, y: usize, z: usize, vn: Vec<f32>) -> Self {
        let bounds = SiteIndex { t, x, y, z };
        let n_faces = t * x * y * z * 6;

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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("localupdates.wgsl"))),
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
                            min_binding_size: wgpu::BufferSize::new((4 * index_size) as _),
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

        let state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plaquette Buffer"),
            contents: bytemuck::cast_slice(&vec![0; n_faces]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let vn_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vn Buffer"),
            contents: bytemuck::cast_slice(&vn),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&vec![0_u32; 9]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // create two bind groups, one for each buffer as the src
        // where the alternate buffer is used as the dst

        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
            label: None,
        });

        Self {
            shape: bounds,
            device,
            queue,
            state_buffer,
            vn_buffer,
            localupdate: LocalUpdatePipeline {
                index_buffer,
                localupdate_pipeline,
                bindgroup,
            },
        }
    }

    pub fn run_local_sweep(&mut self, dims: &[Dimension; 3], leftover: Dimension, offset: bool) {
        let ndims = [self.shape.t, self.shape.x, self.shape.y, self.shape.z]
            .into_iter()
            .map(|d| d as u32)
            .chain(dims.iter().map(|d| usize::from(*d).try_into().unwrap()))
            .chain(Some(usize::from(leftover).try_into().unwrap()))
            .chain(Some(if offset { 1 } else { 0 }))
            .collect::<Vec<u32>>();

        self.queue.write_buffer(
            &self.localupdate.index_buffer,
            0 as _,
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
            cpass.set_pipeline(&self.localupdate.localupdate_pipeline);
            cpass.set_bind_group(0, &self.localupdate.bindgroup, &[]);
            // cpass.dispatch(rho * mu * nu * (sigma / 2), 1, 1);
            cpass.dispatch(1, 1, 1);
        }
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));
    }

    pub fn get_state(&mut self) -> Vec<i32> {
        let n_faces = self.shape.t * self.shape.x * self.shape.y * self.shape.z * 6;
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

            // // // With the current interface, we have to make sure all mapped views are
            // // // dropped before we unmap the buffer.
            // drop(data);
            // buf.unmap(); // Unmaps buffer from memory

            // Returns data from buffer
            result
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}
