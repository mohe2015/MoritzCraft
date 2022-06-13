#version 450

// vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

// per-instance data
layout(location = 3) in vec3 position_offset;
layout(location = 4) in uint block_type;

layout(location = 0) flat out vec3 v_normal;
layout(location = 1) out vec2 v_tex_coord;
layout(location = 2) out flat uint v_block_type;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = mat3(uniforms.world) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position + position_offset, 1.0);
    v_tex_coord = tex_coord;
    v_block_type = block_type;
}