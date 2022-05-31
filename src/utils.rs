use bytemuck::{Zeroable, Pod};
use vulkano::impl_vertex;
use winit::event::ElementState;


#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    position: [f32; 3],
}

impl_vertex!(Vertex, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct InstanceData {
    position_offset: [f32; 3],
}
impl_vertex!(InstanceData, position_offset);

pub const SIZE: f32 = 10.0;

// x to the right
// y down
// z inwards

pub fn repeat_element<T: Clone>(it: impl Iterator<Item = T>, cnt: usize) -> impl Iterator<Item = T> {
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

pub fn state_is_pressed(state: ElementState) -> bool {
    match state {
        ElementState::Pressed => true,
        ElementState::Released => false,
    }
}
