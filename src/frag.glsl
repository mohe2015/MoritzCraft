#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D tex;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));
    vec3 dark_color = vec3(0.6, 0.0, 0.0);
    vec3 regular_color = vec3(1.0, 0.0, 0.0);

    f_color = texture(tex, tex_coords);
    //f_color = vec4(mix(dark_color, regular_color, brightness), 1.0);
}