use std::{io::Cursor, sync::Arc};

use vulkano::{buffer::{CpuAccessibleBuffer, BufferUsage, CpuBufferPool}, format::Format, image::{ImageDimensions, ImmutableImage, MipmapsCount, view::ImageView, SwapchainImage, AttachmentImage}, sampler::{Sampler, SamplerCreateInfo, Filter, SamplerAddressMode}, device::{Device, Queue}, shader::ShaderModule, render_pass::{RenderPass, Framebuffer, FramebufferCreateInfo, Subpass}, pipeline::{GraphicsPipeline, graphics::{vertex_input::BuffersDefinition, input_assembly::InputAssemblyState, viewport::{ViewportState, Viewport}, depth_stencil::DepthStencilState}}, swapchain::Swapchain, command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer, pool::standard::StandardCommandPoolAlloc}, sync::NowFuture};
use winit::window::Window;
use vulkano::image::ImageAccess;
use crate::utils::{Vertex, repeat_element, SIZE, Normal, TexCoord, InstanceData};

pub struct PoritzCraftRenderer {

}

impl PoritzCraftRenderer {
    pub fn new(device: &Arc<Device>, swapchain: &Arc<Swapchain<Window>>, queue: &Arc<Queue>, images: &[Arc<SwapchainImage<Window>>]) -> (CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer<StandardCommandPoolAlloc>>, Self) {
        // TODO to render a cube we only need the three visible faces

        // every vertex is duplicated three times for the three normal directions
        let vertices: Vec<Vertex> = repeat_element(
            [
                Vertex {
                    position: [-SIZE, -SIZE, -SIZE],
                },
                Vertex {
                    position: [SIZE, -SIZE, -SIZE],
                },
                Vertex {
                    position: [SIZE, SIZE, -SIZE],
                },
                Vertex {
                    position: [-SIZE, SIZE, -SIZE],
                },
                Vertex {
                    position: [-SIZE, -SIZE, SIZE],
                },
                Vertex {
                    position: [SIZE, -SIZE, SIZE],
                },
                Vertex {
                    position: [SIZE, SIZE, SIZE],
                },
                Vertex {
                    position: [-SIZE, SIZE, SIZE],
                },
            ]
            .into_iter(),
            3,
        )
        .collect();

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

        let normals: Vec<Normal> = vec![
            N_LEFT, N_TOP, N_FRONT, N_RIGHT, N_TOP, N_FRONT, N_RIGHT, N_BOTTOM, N_FRONT, N_LEFT,
            N_BOTTOM, N_FRONT, // repeat with N_BACK
            N_LEFT, N_TOP, N_BACK, N_RIGHT, N_TOP, N_BACK, N_RIGHT, N_BOTTOM, N_BACK, N_LEFT,
            N_BOTTOM, N_BACK,
        ];

        // TODO FIXME this is wrong because every vertex occurs three times
        let texture_coordinates: Vec<TexCoord> = vec![
            // top left of front face
            TexCoord {
                tex_coord: [1.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            // top right of front face
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [1.0, 0.0],
            },
            // bottom right of front face
            TexCoord {
                tex_coord: [0.0, 1.0],
            },
            TexCoord {
                tex_coord: [1.0, 0.0],
            },
            TexCoord {
                tex_coord: [1.0, 1.0],
            },
            // bottom left of front face
            TexCoord {
                tex_coord: [1.0, 1.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 1.0],
            },
            // leftright, topbottom, frontback
            // top left (looking from front) so top right of back face
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [1.0, 0.0],
            },
            // top right (looking from front) so top left of back face
            TexCoord {
                tex_coord: [1.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            // bottom right (looking from front) so bottom left of back face
            TexCoord {
                tex_coord: [1.0, 1.0],
            },
            TexCoord {
                tex_coord: [0.0, 0.0],
            },
            TexCoord {
                tex_coord: [0.0, 1.0],
            },
            // bottom left (looking from front) so bottom right of back face
            TexCoord {
                tex_coord: [0.0, 1.0],
            },
            TexCoord {
                tex_coord: [1.0, 0.0],
            },
            TexCoord {
                tex_coord: [1.0, 1.0],
            },
        ];

        let indices: Vec<u16> = vec![
            2,
            3 + 2,
            2 * 3 + 2,
            2 * 3 + 2,
            3 * 3 + 2,
            2, // front
            /* 4 * 3 + 2,
            5 * 3 + 2,
            6 * 3 + 2,
            6 * 3 + 2,
            7 * 3 + 2,
            4 * 3 + 2, // back*/
            0,
            3 * 3,
            7 * 3,
            0,
            4 * 3,
            7 * 3, // left
            /* 3,
            2 * 3,
            5 * 3,
            2 * 3,
            5 * 3,
            6 * 3, // right*/
            1,
            3 + 1,
            4 * 3 + 1,
            3 + 1,
            4 * 3 + 1,
            5 * 3 + 1, // top
                       /*2 * 3 + 1,
                       6 * 3 + 1,
                       7 * 3 + 1,
                       2 * 3 + 1,
                       3 * 3 + 1,
                       7 * 3 + 1, // bottom*/
        ];

        // The start of this example is exactly the same as `triangle`. You should read the
        // `triangle` example if you haven't done so yet.

        let vertex_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices)
                .unwrap();
        let normals_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, normals)
                .unwrap();
        let texture_coordinate_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            texture_coordinates,
        )
        .unwrap();

        let index_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, indices)
                .unwrap();

        // Now we create another buffer that will store the unique data per instance.
        // For this example, we'll have the instances form a 10x10 grid that slowly gets larger.
        let instances = {
            let mut data = Vec::new();
            for x in 0..100 {
                for y in 0..1 {
                    for z in 0..100 {
                        data.push(InstanceData {
                            position_offset: [x as f32 * 20.0, y as f32 * 20.0, z as f32 * 20.0],
                        });
                    }
                }
            }
            data
        };
        let instance_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, instances)
                .unwrap();

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
            let png_bytes = include_bytes!("grass_block_side.png").to_vec();
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

        let (mut pipeline, mut framebuffers) =
            window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());

        (tex_future, Self {})
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
        .input_assembly_state(InputAssemblyState::new())
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
