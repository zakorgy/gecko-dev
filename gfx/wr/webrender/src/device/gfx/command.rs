/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal::Device as BackendDevice;
use hal::pool::RawCommandPool;

pub struct CommandPool<B: hal::Backend> {
    command_pool: B::CommandPool,
    command_buffers: Vec<B::CommandBuffer>,
    size: usize,
}

impl<B: hal::Backend> CommandPool<B> {
    pub(super) fn new(mut command_pool: B::CommandPool) -> Self {
        let command_buffer = command_pool.allocate_one(hal::command::RawLevel::Primary);
        CommandPool {
            command_pool,
            command_buffers: vec![command_buffer],
            size: 0,
        }
    }

    pub(super) fn acquire_command_buffer(
        &mut self,
    ) -> &mut B::CommandBuffer {
        if self.size >= self.command_buffers.len() {
            let command_buffer = self
                .command_pool
                .allocate_one(hal::command::RawLevel::Primary);
            self.command_buffers.push(command_buffer);
        }
        self.size += 1;
        &mut self.command_buffers[self.size - 1]
    }

    pub(super) fn command_buffers(&self) -> &[B::CommandBuffer] {
        &self.command_buffers[0 .. self.size]
    }

    pub(super) unsafe fn reset(&mut self) {
        self.command_pool.reset();
        self.size = 0;
    }

    pub(super) unsafe fn destroy(self, device: &B::Device) {
        device.destroy_command_pool(self.command_pool);
    }
}
