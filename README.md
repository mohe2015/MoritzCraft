

 WINIT_UNIX_BACKEND=x11 QT_QPA_PLATFORM=xcb qrenderdoc


vulkan-validation-layers
amd radv vs amdvlk
lavapipe (vulkan-swrast)

// https://arewegameyet.rs/ecosystem/math/

// https://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/


Refactor as close as possible to https://github.com/vulkano-rs/vulkano/blob/v0.29.0/examples/src/bin/interactive_fractal/renderer.rs


Important ones:
https://github.com/vulkano-rs/vulkano/blob/v0.29.0/examples/src/bin/image/main.rs
https://github.com/vulkano-rs/vulkano/blob/v0.29.0/examples/src/bin/instancing.rs
https://github.com/vulkano-rs/vulkano/tree/v0.29.0/examples/src/bin/teapot

https://github.com/vulkano-rs/vulkano/blob/v0.29.0/examples/src/bin/immutable-sampler/main.rs

// if you measure performance, compile release mode

for interactiveness: https://github.com/vulkano-rs/vulkano/blob/v0.29.0/examples/src/bin/interactive_fractal/app.rs

important: https://github.com/vulkano-rs/vulkano/blob/v0.29.0/examples/src/bin/debug.rs

important: moving around in the world

for worldgen: https://github.com/vulkano-rs/vulkano/blob/v0.29.0/examples/src/bin/basic-compute-shader.rs

for the transformation matrix: https://github.com/vulkano-rs/vulkano/blob/v0.29.0/examples/src/bin/push-constants.rs

rendering multiple tings: https://vkguide.dev/docs/chapter-3/scene_management/

https://developer.nvidia.com/blog/vulkan-dos-donts/


https://vkguide.dev/docs/gpudriven/mesh_rendering/
https://vkguide.dev/docs/gpudriven/compute_culling/
https://vkguide.dev/docs/extra-chapter/asset_system/
https://vkguide.dev/docs/extra-chapter/multithreading/



// the cube is a bad start
// https://www.reddit.com/r/opengl/comments/4oozww/indexed_buffers_and_face_normals/

// texture coordinate interpolation
we currently have another problem but will soon have that problem

https://www.khronos.org/opengl/wiki/Vertex_Shader
User-defined output variables can have interpolation qualifiers (though these only matter if the output is being passed directly to the Vertex Post-Processing stage). Vertex shader outputs can also be aggregated into Interface Blocks.

https://vulkan-tutorial.com/Texture_mapping/Images
https://en.wikipedia.org/wiki/Texture_mapping

https://stackoverflow.com/questions/15242507/perspective-correct-texturing-of-trapezoid-in-opengl-es-2-0
https://stackoverflow.com/questions/10670092/perspective-correction-texture-interpolation-opengl

https://www.khronos.org/opengl/wiki/Fragment_Shader

// https://github.com/bwasty/vulkan-tutorial-rs
// https://vulkan-tutorial.com/Introduction
// https://vulkan-tutorial.com/Uniform_buffers/Descriptor_layout_and_buffer
// https://github.com/khronosGroup/Vulkan-samples
// https://github.com/SaschaWillems/Vulkan
// https://vkguide.dev/docs/gpudriven/gpu_driven_engines/

// TODO https://vulkan-tutorial.com/Uniform_buffers/Descriptor_layout_and_buffer
// https://www.vulkan.org/learn#vulkan-tutorials
// https://vkguide.dev/docs/chapter-3/triangle_mesh/

.minecraft/versions/x/x.jar
/assets/minecraft/textures/block/grass_block_side.png