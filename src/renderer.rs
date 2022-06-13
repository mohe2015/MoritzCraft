// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use crate::main_pipeline::MainPipeline;

use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo};
use vulkano::image::ImageUsage;
use vulkano::instance::debug::{DebugUtilsMessenger, DebugUtilsMessengerCreateInfo, Message};
use vulkano::instance::{layers_list, Instance, InstanceCreateInfo};

use vulkano::{
    device::Device,
    swapchain::{Swapchain, SwapchainCreateInfo},
};

use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

pub struct PoritzCraftRenderer {
    pub main_pipeline: MainPipeline,
}

impl PoritzCraftRenderer {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let required_extensions = vulkano_win::required_extensions();

        println!("List of Vulkan debugging layers available to use:");
        let mut layers = layers_list().unwrap();
        while let Some(l) = layers.next() {
            //println!("\t{}", l.name());
        }

        let instance = Instance::new(InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_owned()],
            ..Default::default()
        })
        .unwrap();

        let debug_create_info =
            DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg: &Message| {
                println!("Debug callback: {:?}", msg.description);
            }));
        let _callback = unsafe { DebugUtilsMessenger::new(instance.clone(), debug_create_info) };

        let window = WindowBuilder::new()
            .with_title("PoritzCraft")
            .build(event_loop)
            .unwrap();

        window.set_cursor_grab(true).unwrap();
        window.set_cursor_visible(false);

        let surface = vulkano_win::create_surface_from_handle(window, instance.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ext_descriptor_indexing: true,
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
            .max_by_key(|(p, _)| match p.properties().device_type {
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
                enabled_features: Features {
                    sampler_anisotropy: true,
                    descriptor_indexing: true,
                    descriptor_binding_variable_descriptor_count: true,
                    // https://chunkstories.xyz/blog/a-note-on-descriptor-indexing/
                    // vulkaninfo
                    shader_sampled_image_array_dynamic_indexing: true,
                    shader_sampled_image_array_non_uniform_indexing: true,
                    runtime_descriptor_array: true,
                    ..Features::none()
                },
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

        Self {
            main_pipeline: MainPipeline::new(device, swapchain, surface, queue, images),
        }
    }
}
