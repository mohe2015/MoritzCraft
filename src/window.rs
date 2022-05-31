// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


use crate::renderer::PoritzCraftRenderer;


use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    image::ImageUsage,
    instance::{Instance, InstanceCreateInfo},
    swapchain::{
        Swapchain, SwapchainCreateInfo,
    },
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

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

        let (swapchain, images) = {
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

        let mut renderer = PoritzCraftRenderer::new(device, swapchain, queue, &images, surface);

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
                    renderer.recreate_swapchain = true;
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput { input, .. },
                    ..
                } => {
                    if let Some(_key_code) = input.virtual_keycode {
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
                    event: WindowEvent::MouseInput { state: _, button: _, .. },
                    ..
                } => {}
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { position: _, .. },
                    ..
                } => {}
                Event::WindowEvent {
                    event: WindowEvent::MouseWheel { delta: _, .. },
                    ..
                } => {}
                Event::RedrawEventsCleared => {
                    renderer.render();
                }
                _ => (),
            }
        });
    }
}
