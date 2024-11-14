use std::borrow::Cow;
use wgpu::util::DeviceExt;

const MAX_CHUNK_SIZE: usize = 134_217_728 / 4; // Maximum chunk size in f32 elements (based on 134 MB buffer limit)

pub async fn add_vectors_gpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    // Validate input vectors have the same length
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");

    // Create GPU instance, adapter, device, and queue
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .unwrap();

    // Load and create compute shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Vector Addition Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/vector_addition.wgsl"))),
    });

    // Create pipeline and bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Vector Addition Bind Group Layout"),
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Vector Addition Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Vector Addition Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        cache: None,  
        compilation_options: wgpu::PipelineCompilationOptions::default(),  
    });

    // Prepare result vector
    let mut result = Vec::with_capacity(a.len());

    // Process each chunk
    for chunk_start in (0..a.len()).step_by(MAX_CHUNK_SIZE) {
        let chunk_end = (chunk_start + MAX_CHUNK_SIZE).min(a.len());
        let a_chunk = &a[chunk_start..chunk_end];
        let b_chunk = &b[chunk_start..chunk_end];
        let chunk_size = ((chunk_end - chunk_start) * std::mem::size_of::<f32>()) as u64;

        // Create input buffers for each chunk
        let input_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer A"),
            contents: bytemuck::cast_slice(a_chunk),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let input_buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer B"),
            contents: bytemuck::cast_slice(b_chunk),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create output and staging buffers for each chunk
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: chunk_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: chunk_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group for each chunk
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vector Addition Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder for each chunk
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Vector Addition Command Encoder"),
        });

        // Dispatch compute pipeline with chunked workgroups
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Vector Addition Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate the total number of elements in this chunk
            let num_elements = chunk_end - chunk_start;
            let workgroup_size = 256;  // Assume each workgroup processes 256 elements
            let max_workgroups = 65535;

            // Calculate how many elements each dispatch can handle
            let elements_per_dispatch = max_workgroups * workgroup_size;

            // Divide the workload into manageable dispatches
            let mut offset = 0;
            while offset < num_elements {
                let elements_left = num_elements - offset;
                let dispatch_size = (elements_left + workgroup_size - 1) / workgroup_size;

                // Ensure we don't exceed the maximum workgroups in a single dispatch
                let dispatch_groups = dispatch_size.min(max_workgroups);

                compute_pass.dispatch_workgroups(dispatch_groups as u32, 1, 1);

                offset += dispatch_groups * workgroup_size;
            }
        }

        // Copy output to staging buffer and submit
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, chunk_size);
        queue.submit(Some(encoder.finish()));

        // Map staging buffer to read back results
        let slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);

        if receiver.receive().await.unwrap().is_ok() {
            let data = slice.get_mapped_range();
            let chunk_result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();

            // Append the chunk result to the final result
            result.extend(chunk_result);
        } else {
            panic!("Failed to run compute on GPU!");
        }
    }

    result
}
