use std::time::Instant;

use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    image::ImageUsage,
    instance::{Instance, InstanceCreateInfo},
    pipeline::PipelineBindPoint,
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use vulkano::sync::GpuFuture;
use crate::renderer::PoritzCraftRenderer;

// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub struct PoritzCraftWindow {}

impl PoritzCraftWindow {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(&self) {
        let required_extensions = vulkano_win::required_extensions();
        let instance = Instance::new(InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        })
        .unwrap();

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
            .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
            .filter_map(|p| {
                p.queue_families()
                    .find(|&q| {
                        q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false)
                    })
                    .map(|q| (p, q))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: physical_device
                    .required_extensions()
                    .union(&device_extensions),
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let (mut swapchain, images) = {
            let surface_capabilities = physical_device
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let image_format = Some(
                physical_device
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_extent: surface.window().inner_size().into(),
                    image_usage: ImageUsage::color_attachment(),
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let mut recreate_swapchain = false;

        let (tex_future, renderer) = PoritzCraftRenderer::new(&device, &swapchain, &queue, &images);

        let mut previous_frame_end = Some(tex_future.boxed());
        let rotation_start = Instant::now();

        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    recreate_swapchain = true;
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput { input, .. },
                    ..
                } => {
                    if let Some(key_code) = input.virtual_keycode {
                        /*
                        match key_code {
                            VirtualKeyCode::Escape => self.should_quit = state_is_pressed(input.state),
                            VirtualKeyCode::W => self.pan_up = state_is_pressed(input.state),
                            VirtualKeyCode::A => self.pan_left = state_is_pressed(input.state),
                            VirtualKeyCode::S => self.pan_down = state_is_pressed(input.state),
                            VirtualKeyCode::D => self.pan_right = state_is_pressed(input.state),
                            VirtualKeyCode::F => self.toggle_full_screen = state_is_pressed(input.state),
                            VirtualKeyCode::Return => self.randomize_palette = state_is_pressed(input.state),
                            VirtualKeyCode::Equals => self.increase_iterations = state_is_pressed(input.state),
                            VirtualKeyCode::Minus => self.decrease_iterations = state_is_pressed(input.state),
                            VirtualKeyCode::Space => self.toggle_julia = state_is_pressed(input.state),
                            _ => (),
                        }
                        */
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::MouseInput { state, button, .. },
                    ..
                } => {}
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { position, .. },
                    ..
                } => {}
                Event::WindowEvent {
                    event: WindowEvent::MouseWheel { delta, .. },
                    ..
                } => {}
                Event::RedrawEventsCleared => {
                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if recreate_swapchain {
                        let (new_swapchain, new_images) =
                            match swapchain.recreate(SwapchainCreateInfo {
                                image_extent: surface.window().inner_size().into(),
                                ..swapchain.create_info()
                            }) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                                    return
                                }
                                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                            };

                        swapchain = new_swapchain;
                        let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
                            device.clone(),
                            &vs,
                            &fs,
                            &new_images,
                            render_pass.clone(),
                        );
                        pipeline = new_pipeline;
                        framebuffers = new_framebuffers;
                        recreate_swapchain = false;
                    }

                    let uniform_buffer_subbuffer = {
                        let elapsed = rotation_start.elapsed();
                        let rotation = elapsed.as_secs() as f64
                            + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                        let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                        // note: this teapot was meant for OpenGL where the origin is at the lower left
                        //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                        let aspect_ratio =
                            swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
                        let proj = cgmath::perspective(
                            Rad(std::f32::consts::FRAC_PI_2),
                            aspect_ratio,
                            0.01,
                            100.0,
                        );
                        let view = Matrix4::look_at_rh(
                            Point3::new(0.3, 0.3, 1.0),
                            Point3::new(0.0, 0.0, 0.0),
                            Vector3::new(0.0, -1.0, 0.0),
                        );
                        let scale = Matrix4::from_scale(0.01);

                        let uniform_data = vs::ty::Data {
                            world: Matrix4::from(rotation).into(),
                            view: (view * scale).into(),
                            proj: proj.into(),
                        };

                        uniform_buffer.next(uniform_data).unwrap()
                    };

                    let (image_num, suboptimal, acquire_future) =
                        match acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {:?}", e),
                        };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    let future = render();

                    match future {
                        Ok(future) => {
                            previous_frame_end = Some(future.boxed());
                        }
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(sync::now(device.clone()).boxed());
                        }
                        Err(e) => {
                            println!("Failed to flush future: {:?}", e);
                            previous_frame_end = Some(sync::now(device.clone()).boxed());
                        }
                    }
                }
                _ => (),
            }
        });
    }
}

pub fn render() {
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
    )
    .unwrap();

    let layout2 = pipeline.layout().set_layouts().get(1).unwrap();
    let set2 = PersistentDescriptorSet::new(
        layout2.clone(),
        [WriteDescriptorSet::image_view_sampler(
            0,
            texture.clone(),
            sampler.clone(),
        )],
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .begin_render_pass(
            framebuffers[image_num].clone(),
            SubpassContents::Inline,
            vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            set,
        )
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            1,
            set2,
        )
        .bind_vertex_buffers(
            0,
            (
                vertex_buffer.clone(),
                normals_buffer.clone(),
                texture_coordinate_buffer.clone(),
                instance_buffer.clone(),
            ),
        )
        .bind_index_buffer(index_buffer.clone())
        .draw_indexed(
            index_buffer.len() as u32,
            instance_buffer.len() as u32,
            0,
            0,
            0,
        )
        .unwrap()
        .end_render_pass()
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let future = previous_frame_end
        .take()
        .unwrap()
        .join(acquire_future)
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
        .then_signal_fence_and_flush();

    future
}
