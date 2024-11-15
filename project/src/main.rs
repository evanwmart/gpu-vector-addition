use wgpu::util::DeviceExt;
use std::time::Instant;
use std::convert::TryInto;
use futures_intrusive::channel::shared::oneshot_channel;

async fn gpu_vector_addition(a: &[f32], b: &[f32]) -> Vec<f32> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        #[cfg(not(target_arch = "wasm32"))]
        backends: wgpu::Backends::PRIMARY,
        #[cfg(target_arch = "wasm32")]
        backends: wgpu::Backends::GL,
        ..Default::default()
    });

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None).await
        .unwrap();

    let max_chunk_size = 134_217_728 / std::mem::size_of::<f32>(); // Maximum elements per chunk based on buffer limit
    let mut result = Vec::with_capacity(a.len());

    for chunk_start in (0..a.len()).step_by(max_chunk_size) {
        let chunk_end = (chunk_start + max_chunk_size).min(a.len());
        let a_chunk = &a[chunk_start..chunk_end];
        let b_chunk = &b[chunk_start..chunk_end];
        let chunk_size = ((chunk_end - chunk_start) * std::mem::size_of::<f32>())
            .try_into()
            .unwrap();

        // Create buffers for the current chunk
        let a_buffer = device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("A Buffer"),
                contents: bytemuck::cast_slice(a_chunk),
                usage: wgpu::BufferUsages::STORAGE,
            })
        );

        let b_buffer = device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("B Buffer"),
                contents: bytemuck::cast_slice(b_chunk),
                usage: wgpu::BufferUsages::STORAGE,
            })
        );

        let result_buffer = device.create_buffer(
            &(wgpu::BufferDescriptor {
                label: Some("Result Buffer"),
                size: chunk_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        );

        let staging_buffer = device.create_buffer(
            &(wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: chunk_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        );

        // Shader code with offset handling
        let shader_code =
            r#"
            struct Vector {
                data: array<f32>,
            };

            struct Params {
                offset: u32,
            };

            @group(0) @binding(0) var<storage, read> a: Vector;
            @group(0) @binding(1) var<storage, read> b: Vector;
            @group(0) @binding(2) var<storage, read_write> result: Vector;
            @group(0) @binding(3) var<uniform> params: Params;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x + params.offset;
                result.data[index] = a.data[index] + b.data[index];
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vector Addition Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(
            &(wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        );

        let params_buffer = device.create_buffer(
            &(wgpu::BufferDescriptor {
                label: Some("Params Buffer"),
                size: std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );

        let bind_group = device.create_bind_group(
            &(wgpu::BindGroupDescriptor {
                label: Some("Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: result_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            })
        );

        let pipeline_layout = device.create_pipeline_layout(
            &(wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })
        );

        let compute_pipeline = device.create_compute_pipeline(
            &(wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        );

        let mut encoder = device.create_command_encoder(
            &(wgpu::CommandEncoderDescriptor { label: Some("Command Encoder") })
        );

        let max_dispatch_size = 65535;
        let mut remaining_elements = a_chunk.len() as u32;
        let mut offset = chunk_start as u32;

        while remaining_elements > 0 {
            let dispatch_size = remaining_elements.min(max_dispatch_size);

            // Update the offset in the params buffer
            queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[offset]));

            {
                let mut compute_pass = encoder.begin_compute_pass(
                    &(wgpu::ComputePassDescriptor {
                        label: Some("Compute Pass"),
                        timestamp_writes: None,
                    })
                );
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            }

            offset += dispatch_size;
            remaining_elements -= dispatch_size;
        }

        encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, chunk_size);
        queue.submit(Some(encoder.finish()));

        // Read buffer data with callback
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);

        receiver.receive().await.unwrap().expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let chunk_result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result.extend(chunk_result);
    }

    result
}

fn main() {
    let a: Vec<f32> = vec![1.0; 100_000_000];
    let b: Vec<f32> = vec![2.0; 100_000_000];

    let start_gpu = Instant::now(); // Record the start time
    let _result_gpu = pollster::block_on(gpu_vector_addition(&a, &b));
    let duration_gpu = start_gpu.elapsed(); // Calculate the elapsed time

    println!("COmpleted Vector Addition in {:?}", duration_gpu);
}
