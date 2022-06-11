#version 450

#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D tex[];

const vec3 LIGHT = vec3(1.0, 5.0, 1.0);

void main() {
    float brightness = clamp(dot(normalize(v_normal), normalize(LIGHT)), 0.6, 1);
  
    vec4 texture_color = texture(tex[0], tex_coords);
    f_color = texture_color * 2.0 * brightness;
}