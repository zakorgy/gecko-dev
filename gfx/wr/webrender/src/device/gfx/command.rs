/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal::Device as BackendDevice;
use hal::pool::RawCommandPool;

pub struct CommandPool<B: hal::Backend> {
    command_pool: B::CommandPool,
    command_buffers: Vec<B::CommandBuffer>,
}

impl<B: hal::Backend> CommandPool<B> {
    pub(super) fn new(command_pool: B::CommandPool) -> Self {
        CommandPool {
            command_pool,
            command_buffers: vec![],
        }
    }

    pub(super) fn create_command_buffer(&mut self) {
        if self.command_buffers.is_empty() {
            let command_buffer = self
                .command_pool
                .allocate_one(hal::command::RawLevel::Primary);
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
    }

    pub(super) unsafe fn destroy(self, device: &B::Device) {
        device.destroy_command_pool(self.command_pool);
    }
}
