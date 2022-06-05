use std::{io::Cursor, sync::Arc, time::Instant};

use nalgebra::{Matrix4, Point3, Vector3};
use vulkano::{
    buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImmutableImage,
        MipmapsCount, SwapchainImage,
    },
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint, StateMode,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::window::Window;

use crate::utils::{repeat_element, InstanceData, Normal, TexCoord, Vertex, SIZE};

pub struct MainPipeline {
    vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    normals_buffer: Arc<ImmutableBuffer<[Normal]>>,
    texture_coordinate_buffer: Arc<ImmutableBuffer<[TexCoord]>>,
    instance_buffer: Arc<ImmutableBuffer<[InstanceData]>>,
    pipeline: Arc<GraphicsPipeline>,
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    sampler: Arc<Sampler>,
    texture: Arc<ImageView<ImmutableImage>>,
    framebuffers: Vec<Arc<Framebuffer>>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    pub recreate_swapchain: bool,
    swapchain: Arc<Swapchain<Window>>,
    surface: Arc<Surface<Window>>,
    rotation_start: Instant,
    queue: Arc<Queue>,

    pub world_position: Matrix4<f32>,

    pub pan_up: bool,
    pub pan_left: bool,
    pub pan_down: bool,
    pub pan_right: bool,

    pub rotate_up: bool,
    pub rotate_left: bool,
    pub rotate_down: bool,
    pub rotate_right: bool,

    pub control: bool,
}

impl MainPipeline {
    pub fn new(
        device: Arc<Device>,
        swapchain: Arc<Swapchain<Window>>,
        surface: Arc<Surface<Window>>,
        queue: Arc<Queue>,
        images: Vec<Arc<SwapchainImage<Window>>>,
    ) -> Self {
        // these coords are stupid because they don't match screen

        // x right
        // y up?
        // z back

        // counter clockwise around the whole face

        const N_TOP: Normal = Normal {
            normal: [0.0, -SIZE, 0.0],
        };
        const N_BOTTOM: Normal = Normal {
            normal: [0.0, SIZE, 0.0],
        };
        const N_LEFT: Normal = Normal {
            normal: [-SIZE, 0.0, 0.0],
        };
        const N_RIGHT: Normal = Normal {
            normal: [SIZE, 0.0, 0.0],
        };
        const N_FRONT: Normal = Normal {
            normal: [0.0, 0.0, -SIZE],
        };
        const N_BACK: Normal = Normal {
            normal: [0.0, 0.0, SIZE],
        };

        let texs = vec![
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 1.0],
            },
            TexCoord {
                tex_coord: [1.0, 1.0],
            },
            TexCoord {
                tex_coord: [1.0, 1.0],
            },
            TexCoord {
                tex_coord: [1.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
        ];

        let vertices: Vec<(Vertex, Normal, TexCoord)> = vec![
            (
                Vertex {
                    position: [-SIZE, SIZE, -SIZE],
                },
                N_FRONT,
                texs[0],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, -SIZE],
                },
                N_FRONT,
                texs[1],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, -SIZE],
                },
                N_FRONT,
                texs[2],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, -SIZE],
                },
                N_FRONT,
                texs[3],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, -SIZE],
                },
                N_FRONT,
                texs[4],
            ),
            (
                Vertex {
                    position: [-SIZE, SIZE, -SIZE],
                },
                N_FRONT,
                texs[5],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, SIZE],
                },
                N_BACK,
                texs[0],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, SIZE],
                },
                N_BACK,
                texs[1],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, SIZE],
                },
                N_BACK,
                texs[2],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, SIZE],
                },
                N_BACK,
                texs[3],
            ),
            (
                Vertex {
                    position: [-SIZE, SIZE, SIZE],
                },
                N_BACK,
                texs[4],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, SIZE],
                },
                N_BACK,
                texs[5],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, -SIZE],
                },
                N_RIGHT,
                texs[0],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, -SIZE],
                },
                N_RIGHT,
                texs[1],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, SIZE],
                },
                N_RIGHT,
                texs[2],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, SIZE],
                },
                N_RIGHT,
                texs[3],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, SIZE],
                },
                N_RIGHT,
                texs[4],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, -SIZE],
                },
                N_RIGHT,
                texs[5],
            ),
            (
                Vertex {
                    position: [-SIZE, SIZE, SIZE],
                },
                N_LEFT,
                texs[0],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, SIZE],
                },
                N_LEFT,
                texs[1],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, -SIZE],
                },
                N_LEFT,
                texs[2],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, -SIZE],
                },
                N_LEFT,
                texs[3],
            ),
            (
                Vertex {
                    position: [-SIZE, SIZE, -SIZE],
                },
                N_LEFT,
                texs[4],
            ),
            (
                Vertex {
                    position: [-SIZE, SIZE, SIZE],
                },
                N_LEFT,
                texs[5],
            ),
            (
                Vertex {
                    position: [-SIZE, SIZE, SIZE],
                },
                N_TOP,
                texs[0],
            ),
            (
                Vertex {
                    position: [-SIZE, SIZE, -SIZE],
                },
                N_TOP,
                texs[1],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, -SIZE],
                },
                N_TOP,
                texs[2],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, -SIZE],
                },
                N_TOP,
                texs[3],
            ),
            (
                Vertex {
                    position: [SIZE, SIZE, SIZE],
                },
                N_TOP,
                texs[4],
            ),
            (
                Vertex {
                    position: [-SIZE, SIZE, SIZE],
                },
                N_TOP,
                texs[5],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, -SIZE],
                },
                N_BOTTOM,
                texs[0],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, SIZE],
                },
                N_BOTTOM,
                texs[1],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, SIZE],
                },
                N_BOTTOM,
                texs[2],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, SIZE],
                },
                N_BOTTOM,
                texs[3],
            ),
            (
                Vertex {
                    position: [SIZE, -SIZE, -SIZE],
                },
                N_BOTTOM,
                texs[4],
            ),
            (
                Vertex {
                    position: [-SIZE, -SIZE, -SIZE],
                },
                N_BOTTOM,
                texs[5],
            ),
        ];

        let (vertex_buffer, vertex_buffer_future) = ImmutableBuffer::from_iter(
            vertices.iter().map(|e| e.0),
            BufferUsage::all(),
            queue.clone(),
        )
        .unwrap();
        let (normals_buffer, normals_buffer_future) = ImmutableBuffer::from_iter(
            vertices.iter().map(|e| e.1),
            BufferUsage::all(),
            queue.clone(),
        )
        .unwrap();
        let (texture_coordinate_buffer, texture_coordinate_buffer_future) =
            ImmutableBuffer::from_iter(
                vertices.iter().map(|e| e.2),
                BufferUsage::all(),
                queue.clone(),
            )
            .unwrap();

        // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html

        // Now we create another buffer that will store the unique data per instance.
        // For this example, we'll have the instances form a 10x10 grid that slowly gets larger.
        let instances = {
            let mut data = Vec::new();
            for x in 0..1000 {
                for y in 0..1 {
                    for z in 0..1000 {
                        data.push(InstanceData {
                            position_offset: [x as f32 * 20.0, y as f32 * 20.0, z as f32 * 20.0],
                        });
                    }
                }
            }
            data
        };
        let (instance_buffer, instance_buffer_future) =
            ImmutableBuffer::from_iter(instances, BufferUsage::all(), queue.clone()).unwrap();

        let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();

        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )
        .unwrap();

        let (texture, tex_future) = {
            let png_bytes = include_bytes!("dirt.png").to_vec();
            let cursor = Cursor::new(png_bytes);
            let decoder = png::Decoder::new(cursor);
            let mut reader = decoder.read_info().unwrap();
            let info = reader.info();
            let dimensions = ImageDimensions::Dim2d {
                width: info.width,
                height: info.height,
                array_layers: 1,
            };
            let mut image_data = Vec::new();
            image_data.resize((info.width * info.height * 4) as usize, 0);
            let output = reader.next_frame(&mut image_data).unwrap();

            println!("{:?}", output);

            let (image, future) = ImmutableImage::from_iter(
                image_data,
                dimensions,
                MipmapsCount::One,
                Format::R8G8B8A8_SRGB,
                queue.clone(),
            )
            .unwrap();
            (ImageView::new_default(image).unwrap(), future)
        };

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let (pipeline, framebuffers) =
            window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());

        let rotation_start = Instant::now();

        Self {
            normals_buffer,
            texture_coordinate_buffer,
            vertex_buffer,
            instance_buffer,
            pipeline,
            uniform_buffer,
            sampler,
            texture,
            framebuffers,
            fs,
            vs,
            render_pass,
            device,
            previous_frame_end: Some(
                tex_future
                    .join(vertex_buffer_future)
                    .join(normals_buffer_future)
                    .join(texture_coordinate_buffer_future)
                    .join(instance_buffer_future)
                    .boxed(),
            ),
            recreate_swapchain: false,
            rotation_start,
            surface,
            swapchain,
            queue,
            pan_down: false,
            pan_left: false,
            pan_right: false,
            pan_up: false,
            rotate_down: false,
            rotate_left: false,
            rotate_right: false,
            rotate_up: false,
            control: false,
            world_position: Matrix4::new_scaling(1.0),
        }
    }

    pub fn render(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swapchain {
            let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
                image_extent: self.surface.window().inner_size().into(),
                ..self.swapchain.create_info()
            }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

            self.swapchain = new_swapchain;

            // this part here is pipeline specific - the part above not
            let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
                self.device.clone(),
                &self.vs,
                &self.fs,
                &new_images,
                self.render_pass.clone(),
            );
            self.pipeline = new_pipeline;
            self.framebuffers = new_framebuffers;
            self.recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        // this part here is pipeline-specific
        let uniform_buffer_subbuffer = {
            /*  let elapsed = self.rotation_start.elapsed();
                        let rotation =
                            elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                        let rotation_mat = Matrix4::from_angle_y(Rad(rotation as f32));
                        let rotation_mat2 = Matrix4::from_angle_x(Rad((rotation/10f64) as f32));

            */
            if self.rotate_left {
                self.world_position = Matrix4::new_rotation(Vector3::new(0.0, 0.02,0.0)) * self.world_position;
            }
            if self.rotate_right {
                self.world_position = Matrix4::new_rotation(Vector3::new(0.0, -0.02, 0.0)) * self.world_position;
            }
            if self.rotate_up {
                self.world_position = Matrix4::new_rotation(Vector3::new(0.02, 0.0, 0.0)) * self.world_position;
            }
            if self.rotate_down {
                self.world_position = Matrix4::new_rotation(Vector3::new(-0.02, 0.0, 0.0)) * self.world_position;
            }
            if self.pan_up {
                self.world_position = Matrix4::new_translation(&Vector3::new(0.0,0.0,2.0)) * self.world_position;
            }
            if self.pan_down {
                self.world_position = Matrix4::new_translation(&Vector3::new(0.0,0.0,-2.0))* self.world_position;
            }
            if self.pan_left {
                self.world_position = Matrix4::new_translation(&Vector3::new(-2.0,0.0,0.0)) * self.world_position;
            }
            if self.pan_right {
                self.world_position = Matrix4::new_translation(&Vector3::new(2.0,0.0,0.0)) * self.world_position;
            }

            // note: this teapot was meant for OpenGL where the origin is at the lower left
            //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
            let aspect_ratio =
                self.swapchain.image_extent()[0] as f32 / self.swapchain.image_extent()[1] as f32;
            let proj =
                Matrix4::new_perspective(aspect_ratio, std::f32::consts::FRAC_PI_2, 0.01, 100.0);
            let view = Matrix4::look_at_rh(
                &Point3::new(0.3, 0.3, 1.0),
                &Point3::new(0.0, 0.0, 0.0),
                &Vector3::new(0.0, -1.0, 0.0),
            );
            let scale = Matrix4::new_scaling(0.01);

            let uniform_data = vs::ty::Data {
                world: self.world_position.into(),
                view: (view * scale).into(),
                proj: proj.into(),
            };

            // TODO FIXMe check if this is ever dropped
            self.uniform_buffer.next(uniform_data).unwrap()
        };

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
        )
        .unwrap();

        let layout2 = self.pipeline.layout().set_layouts().get(1).unwrap();
        let set2 = PersistentDescriptorSet::new(
            layout2.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                self.texture.clone(),
                self.sampler.clone(),
            )],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .begin_render_pass(
                self.framebuffers[image_num].clone(),
                SubpassContents::Inline,
                vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                1,
                set2,
            )
            .bind_vertex_buffers(
                0,
                (
                    self.vertex_buffer.clone(),
                    self.normals_buffer.clone(),
                    self.texture_coordinate_buffer.clone(),
                    self.instance_buffer.clone(),
                ),
            )
            .draw(
                self.vertex_buffer.len() as u32,
                self.instance_buffer.len() as u32,
                0,
                0,
            )
            .unwrap()
            .end_render_pass()
            .unwrap();
        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &ShaderModule,
    fs: &ShaderModule,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>) {
    let dimensions = images[0].dimensions().width_height();

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    // In the triangle example we use a dynamic viewport, as its a simple example.
    // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
    // This allows the driver to optimize things, at the cost of slower window resizes.
    // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(
            BuffersDefinition::new()
                .vertex::<Vertex>()
                .vertex::<Normal>()
                .vertex::<TexCoord>()
                .instance::<InstanceData>(),
        )
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState {
            // primitive_restart_enable: StateMode::Fixed(true),
            // topology: PrimitiveTopology::TriangleStrip
            ..Default::default()
        })
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            },
        ]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .rasterization_state(RasterizationState {
            cull_mode: StateMode::Fixed(CullMode::Back),
            front_face: StateMode::Fixed(FrontFace::CounterClockwise),
            ..Default::default()
        })
        .build(device)
        .unwrap();

    (pipeline, framebuffers)
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]

        },
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/frag.glsl"
    }
}
