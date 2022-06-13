// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
use bytemuck::{Pod, Zeroable};
use vulkano::impl_vertex;
use winit::event::ElementState;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
}

impl_vertex!(Vertex, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct InstanceData {
    pub position_offset: [f32; 3],
    pub block_type: u32,
}
impl_vertex!(InstanceData, position_offset, block_type);

pub const SIZE: f32 = 10.0;

// x to the right
// y down
// z inwards

pub fn repeat_element<T: Clone>(
    it: impl Iterator<Item = T>,
    cnt: usize,
) -> impl Iterator<Item = T> {
    it.flat_map(move |n| std::iter::repeat(n).take(cnt))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Normal {
    pub normal: [f32; 3],
}

impl_vertex!(Normal, normal);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct TexCoord {
    pub tex_coord: [f32; 2],
}

impl_vertex!(TexCoord, tex_coord);

pub fn state_is_pressed(state: ElementState) -> bool {
    match state {
        ElementState::Pressed => true,
        ElementState::Released => false,
    }
}
