// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
pub mod renderer;
pub mod window;

use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use std::io::Cursor;
use std::{sync::Arc, time::Instant};
use vulkano::image::{ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::impl_vertex;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::event::{ElementState, VirtualKeyCode};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::window::PoritzCraftWindow;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    position: [f32; 3],
}

impl_vertex!(Vertex, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct InstanceData {
    position_offset: [f32; 3],
}
impl_vertex!(InstanceData, position_offset);

const SIZE: f32 = 10.0;

// x to the right
// y down
// z inwards

fn repeat_element<T: Clone>(it: impl Iterator<Item = T>, cnt: usize) -> impl Iterator<Item = T> {
    it.flat_map(move |n| std::iter::repeat(n).take(cnt))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Normal {
    normal: [f32; 3],
}

impl_vertex!(Normal, normal);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct TexCoord {
    tex_coord: [f32; 2],
}

impl_vertex!(TexCoord, tex_coord);

fn state_is_pressed(state: ElementState) -> bool {
    match state {
        ElementState::Pressed => true,
        ElementState::Released => false,
    }
}
