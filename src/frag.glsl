#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D tex;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    float brightness = clamp(dot(normalize(v_normal), normalize(LIGHT)), 0.5, 1);
  
    vec4 texture_color = texture(tex, tex_coords);
    f_color = texture_color * brightness;
}