// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::renderer::PoritzCraftRenderer;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

pub struct PoritzCraftWindow {}

impl PoritzCraftWindow {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(&self) {
        let event_loop = EventLoop::new();

        let mut renderer = PoritzCraftRenderer::new(&event_loop);

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
                    renderer.main_pipeline.recreate_swapchain = true;
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
                    event:
                        WindowEvent::MouseInput {
                            state: _,
                            button: _,
                            ..
                        },
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
                    renderer.main_pipeline.render();
                }
                _ => (),
            }
        });
    }
}
