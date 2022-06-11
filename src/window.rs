// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{renderer::PoritzCraftRenderer, utils::state_is_pressed};

use nalgebra::{Isometry3, Matrix4, Rotation3, Translation3, UnitQuaternion, Vector3};
use winit::{
    event::{DeviceEvent, Event, VirtualKeyCode, WindowEvent},
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

        event_loop.run(move |event, _, control_flow| match event {
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
                if let Some(key_code) = input.virtual_keycode {
                    match key_code {
                        VirtualKeyCode::W => {
                            let trans = Translation3::<f32>::new(0.0, 0.0, 10.0);
                            let d = renderer.main_pipeline.build_rotation().inverse() * trans;


                            // Translation3::from(renderer.main_pipeline.view_translation.vector + trans.vector)
                            renderer.main_pipeline.view_translation =
                                d.translation * renderer.main_pipeline.view_translation;
                        }
                        _ => (),
                    }
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
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
              
                // from_axis_angle
                //  Vector3::y_axis();
                // rotation_between

                renderer.main_pipeline.view_rotation_pitch += delta.0 as f32;
                renderer.main_pipeline.view_rotation_yaw += delta.1 as f32;
            }
            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta: _, .. },
                ..
            } => {}
            Event::RedrawEventsCleared => {
                renderer.main_pipeline.render();
            }
            _ => (),
        });
    }
}
