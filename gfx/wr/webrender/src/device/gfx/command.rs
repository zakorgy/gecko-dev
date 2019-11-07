/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal::command::{CommandBufferFlags, CommandBuffer};
use hal::device::Device as BackendDevice;
use hal::pool::CommandPool as HalCommandPool;

pub(super) enum CBStrategy {
    UseOne,
    AllocateNew,
}

pub struct CommandPool<B: hal::Backend> {
    command_pool: B::CommandPool,
    command_buffers: Vec<B::CommandBuffer>,
    strategy: CBStrategy,
    next_id: usize,
    begin: bool,
}

impl<B: hal::Backend> CommandPool<B> {
    pub(super) fn new(command_pool: B::CommandPool/*, strategy: CBStrategy*/) -> Self {
        CommandPool {
            command_pool,
            command_buffers: vec![],
            strategy: CBStrategy::AllocateNew,
            next_id: 0,
            begin: true,
        }
    }

    pub(super) fn buffer_mut(&mut self, inside_render_pass: bool) -> &mut B::CommandBuffer {
        match self.strategy {
            CBStrategy::UseOne => {
                if self.command_buffers.len() < 1 {
                    let mut command_buffer = unsafe {
                        self.command_pool.allocate_one(hal::command::Level::Primary)
                    };
                    self.command_buffers.push(command_buffer);
                }
                let command_buffer = self.command_buffers.get_mut(0).unwrap();
                if self.begin {
                    unsafe { command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT) };
                    self.begin = false;
                }
                command_buffer

            }
            CBStrategy::AllocateNew => {
                if let Some(command_buffer) = self.command_buffers.get_mut(self.next_id.max(1) - 1) {
                    if self.next_id != 0 && !inside_render_pass {
                        unsafe { command_buffer.finish() };
                    }
                }
                let next_id = if inside_render_pass {
                    self.next_id - 1
                } else {
                    self.next_id
                };
                if self.command_buffers.len() <= next_id {
                    let command_buffer = unsafe {
                        self.command_pool.allocate_one(hal::command::Level::Primary)
                    };
                    self.command_buffers.push(command_buffer);
                }
                let command_buffer = self.command_buffers.get_mut(next_id).unwrap();
                if !inside_render_pass {
                    unsafe { command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT) };
                    self.next_id += 1;
                }
                command_buffer
            }
        }
    }

    pub(super) fn buffers_to_submit(&mut self) -> &[B::CommandBuffer] {
        match self.strategy {
            CBStrategy::UseOne => {
                unsafe { self.command_buffers.get_mut(0).expect("No command buffer allocated?").finish() };
                &self.command_buffers
            }
            CBStrategy::AllocateNew => {
                unsafe { self.command_buffers.get_mut(self.next_id - 1).expect("No command buffer allocated?").finish() };
                &self.command_buffers[0..self.next_id]
            }
        }
    }

    pub(super) fn create_command_buffer(&mut self) {
        if self.command_buffers.is_empty() {
            let command_buffer = unsafe {
                self.command_pool.allocate_one(hal::command::Level::Primary)
            };
            self.command_buffers.push(command_buffer);
        }
    }

    pub fn remove_cmd_buffer(&mut self) -> B::CommandBuffer {
        self.command_buffers.remove(0)
    }

    pub fn return_cmd_buffer(&mut self, cmd_buffer: B::CommandBuffer) {
        self.command_buffers.insert(0, cmd_buffer);
    }

    pub(super) unsafe fn reset(&mut self) {
        self.command_pool.reset(false);
        self.next_id = 0;
        self.begin = true;
    }

    pub(super) unsafe fn destroy(self, device: &B::Device) {
        device.destroy_command_pool(self.command_pool);
    }
}
