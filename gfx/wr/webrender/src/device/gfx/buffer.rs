/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal;
use hal::Device as BackendDevice;
use rendy_memory::{Block, Heaps, MemoryBlock, MemoryUsageValue, Write};

use std::cell::Cell;
use std::mem;

pub const MAX_INSTANCE_COUNT: usize = 8192;
pub const TEXTURE_CACHE_SIZE: usize = 128 << 20; // 128MB

pub(super) struct Buffer<B: hal::Backend> {
    pub(super) memory_block: MemoryBlock<B>,
    pub(super) buffer: B::Buffer,
    pub(super) buffer_size: usize,
    pub(super) buffer_len: usize,
    stride: usize,
    state: Cell<hal::buffer::State>,
}

impl<B: hal::Backend> Buffer<B> {
    pub(super) fn new(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        memory_usage: MemoryUsageValue,
        buffer_usage: hal::buffer::Usage,
        alignment_mask: usize,
        data_len: usize,
        stride: usize,
    ) -> Self {
        let buffer_size = (data_len * stride + alignment_mask) & !alignment_mask;
        let mut buffer = unsafe {
            device
                .create_buffer(buffer_size as u64, buffer_usage)
                .expect("create_buffer failed")
        };
        let requirements = unsafe { device.get_buffer_requirements(&buffer) };
        let alignment = ((requirements.alignment - 1) | (alignment_mask as u64)) + 1;
        let memory_block = heaps
            .allocate(
                device,
                requirements.type_mask as u32,
                memory_usage,
                requirements.size,
                alignment,
            )
            .expect("Allocate memory failed");

        unsafe {
            device.bind_buffer_memory(
                &memory_block.memory(),
                memory_block.range().start,
                &mut buffer,
            )
        }
        .expect("Bind buffer memory failed");

        Buffer {
            memory_block,
            buffer,
            buffer_size: requirements.size as _,
            buffer_len: data_len,
            stride,
            state: Cell::new(hal::buffer::Access::empty()),
        }
    }

    fn update_all<T: Copy>(
        &mut self,
        device: &B::Device,
        data: &[T],
        non_coherent_atom_size_mask: u64,
    ) {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let range = 0 .. ((size + non_coherent_atom_size_mask) & !non_coherent_atom_size_mask);
        unsafe {
            let mut mapped = self
                .memory_block
                .map(device, range.clone())
                .expect("Mapping memory block failed");
            mapped
                .write(device, 0 .. size)
                .expect("Writer creation failed")
                .write(&data);
        }
        self.memory_block.unmap(device);
    }

    fn update<T: Copy>(
        &mut self,
        device: &B::Device,
        data: &[T],
        offset: usize,
        non_coherent_atom_size_mask: u64,
    ) -> usize {
        let offset = (offset * self.stride) as u64;
        let size = (data.len() * self.stride) as u64;
        let range = offset
            .. ((offset + size + non_coherent_atom_size_mask) & !non_coherent_atom_size_mask);
        unsafe {
            let mut mapped = self
                .memory_block
                .map(device, range)
                .expect("Mapping memory block failed");
            mapped
                .write(device, 0 .. size)
                .expect("Writer creation failed")
                .write(&data);
        }
        self.memory_block.unmap(device);
        size as usize
    }

    pub(super) fn transit(&self, access: hal::buffer::Access) -> Option<hal::memory::Barrier<B>> {
        let src_state = self.state.get();
        if src_state == access {
            None
        } else {
            self.state.set(access);
            Some(hal::memory::Barrier::Buffer {
                states: src_state .. access,
                target: &self.buffer,
                families: None,
                range: None .. None,
            })
        }
    }

    pub(super) fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        unsafe {
            device.destroy_buffer(self.buffer);
            heaps.free(device, self.memory_block);
        }
    }
}

pub(super) struct BufferPool<B: hal::Backend> {
    buffer: Buffer<B>,
    data_stride: usize,
    non_coherent_atom_size_mask: usize,
    copy_alignment_mask: usize,
    offset: usize,
    size: usize,
    pub(super) buffer_offset: usize,
}

impl<B: hal::Backend> BufferPool<B> {
    pub(super) fn new(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        buffer_usage: hal::buffer::Usage,
        data_stride: usize,
        non_coherent_atom_size_mask: usize,
        pitch_alignment_mask: usize,
        copy_alignment_mask: usize,
    ) -> Self {
        let buffer = Buffer::new(
            device,
            heaps,
            MemoryUsageValue::Upload,
            buffer_usage,
            pitch_alignment_mask | non_coherent_atom_size_mask,
            TEXTURE_CACHE_SIZE,
            data_stride,
        );
        BufferPool {
            buffer,
            data_stride,
            non_coherent_atom_size_mask,
            copy_alignment_mask,
            offset: 0,
            size: 0,
            buffer_offset: 0,
        }
    }

    pub(super) fn add<T: Copy>(&mut self, device: &B::Device, data: &[T]) {
        assert!(
            mem::size_of::<T>() <= self.data_stride,
            "mem::size_of::<T>()={:?} <= self.data_stride={:?}",
            mem::size_of::<T>(),
            self.data_stride
        );
        let buffer_len = data.len() * self.data_stride;
        assert!(
            self.offset * self.data_stride + buffer_len < self.buffer.buffer_size,
            "offset({:?}) * data_stride({:?}) + buffer_len({:?}) < buffer_size({:?})",
            self.offset,
            self.data_stride,
            buffer_len,
            self.buffer.buffer_size
        );
        self.size = self.buffer.update(
            device,
            data,
            self.offset,
            self.non_coherent_atom_size_mask as u64,
        );
        self.buffer_offset = self.offset;
        self.offset += (self.size + self.copy_alignment_mask) & !self.copy_alignment_mask;
    }

    pub(super) fn buffer(&self) -> &Buffer<B> {
        &self.buffer
    }

    pub(super) fn reset(&mut self) {
        self.offset = 0;
        self.size = 0;
    }

    pub(super) fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        self.buffer.deinit(device, heaps);
    }
}

pub(super) struct InstancePoolBuffer<B: hal::Backend> {
    pub(super) buffer: Buffer<B>,
    pub(super) offset: usize,
    pub(super) last_update_size: usize,
    non_coherent_atom_size_mask: usize,
}

impl<B: hal::Backend> InstancePoolBuffer<B> {
    fn new(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        buffer_usage: hal::buffer::Usage,
        data_stride: usize,
        alignment_mask: usize,
        non_coherent_atom_size_mask: usize,
    ) -> Self {
        let buffer = Buffer::new(
            device,
            heaps,
            MemoryUsageValue::Dynamic,
            buffer_usage,
            alignment_mask,
            MAX_INSTANCE_COUNT,
            data_stride,
        );
        InstancePoolBuffer {
            buffer,
            offset: 0,
            last_update_size: 0,
            non_coherent_atom_size_mask,
        }
    }

    fn update<T: Copy>(&mut self, device: &B::Device, data: &[T]) {
        self.buffer.update(
            device,
            data,
            self.offset,
            self.non_coherent_atom_size_mask as u64,
        );
        self.last_update_size = data.len();
        self.offset += self.last_update_size;
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.last_update_size = 0;
    }

    fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        self.buffer.deinit(device, heaps);
    }
}

pub(super) struct InstanceBufferHandler<B: hal::Backend> {
    pub(super) buffers: Vec<InstancePoolBuffer<B>>,
    data_stride: usize,
    alignment_mask: usize,
    non_coherent_atom_size_mask: usize,
    pub(super) current_buffer_index: usize,
}

impl<B: hal::Backend> InstanceBufferHandler<B> {
    pub(super) fn new(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        data_stride: usize,
        non_coherent_atom_size_mask: usize,
        alignment_mask: usize,
    ) -> Self {
        let buffers = vec![InstancePoolBuffer::new(
            device,
            heaps,
            hal::buffer::Usage::VERTEX,
            data_stride,
            alignment_mask,
            non_coherent_atom_size_mask,
        )];

        InstanceBufferHandler {
            buffers,
            data_stride,
            alignment_mask,
            non_coherent_atom_size_mask,
            current_buffer_index: 0,
        }
    }

    pub(super) fn add<T: Copy>(
        &mut self,
        device: &B::Device,
        mut data: &[T],
        heaps: &mut Heaps<B>,
    ) {
        assert_eq!(self.data_stride, mem::size_of::<T>());
        while !data.is_empty() {
            if self.current_buffer().buffer.buffer_size
                == self.current_buffer().offset * self.data_stride
            {
                self.current_buffer_index += 1;
                if self.buffers.len() <= self.current_buffer_index {
                    self.buffers.push(InstancePoolBuffer::new(
                        device,
                        heaps,
                        hal::buffer::Usage::VERTEX,
                        self.data_stride,
                        self.alignment_mask,
                        self.non_coherent_atom_size_mask,
                    ))
                }
            }

            let update_size = if (self.current_buffer().offset + data.len()) * self.data_stride
                > self.current_buffer().buffer.buffer_size
            {
                self.current_buffer().buffer.buffer_size / self.data_stride
                    - self.current_buffer().offset
            } else {
                data.len()
            };

            self.buffers[self.current_buffer_index].update(device, &data[0 .. update_size]);

            data = &data[update_size ..]
        }
    }

    fn current_buffer(&self) -> &InstancePoolBuffer<B> {
        &self.buffers[self.current_buffer_index]
    }

    pub(super) fn reset(&mut self) {
        for buffer in &mut self.buffers {
            buffer.reset();
        }
        self.current_buffer_index = 0;
    }

    pub(super) fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        for buffer in self.buffers {
            buffer.deinit(device, heaps);
        }
    }
}

pub(super) struct UniformBufferHandler<B: hal::Backend> {
    buffers: Vec<Buffer<B>>,
    offset: usize,
    buffer_usage: hal::buffer::Usage,
    data_stride: usize,
    pitch_alignment_mask: usize,
    non_coherent_atom_size_mask: usize,
}

impl<B: hal::Backend> UniformBufferHandler<B> {
    pub(super) fn new(
        buffer_usage: hal::buffer::Usage,
        data_stride: usize,
        pitch_alignment_mask: usize,
        non_coherent_atom_size_mask: usize,
    ) -> Self {
        UniformBufferHandler {
            buffers: vec![],
            offset: 0,
            buffer_usage,
            data_stride,
            pitch_alignment_mask,
            non_coherent_atom_size_mask,
        }
    }

    pub(super) fn add<T: Copy>(&mut self, device: &B::Device, data: &[T], heaps: &mut Heaps<B>) {
        if self.buffers.len() == self.offset {
            self.buffers.push(Buffer::new(
                device,
                heaps,
                MemoryUsageValue::Dynamic,
                self.buffer_usage,
                self.pitch_alignment_mask,
                data.len(),
                self.data_stride,
            ));
        }
        self.buffers[self.offset].update_all(device, data, self.non_coherent_atom_size_mask as u64);
        self.offset += 1;
    }

    pub(super) fn buffer(&self) -> &Buffer<B> {
        &self.buffers[self.offset - 1]
    }

    pub(super) fn reset(&mut self) {
        self.offset = 0;
    }

    pub(super) fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        for buffer in self.buffers {
            buffer.deinit(device, heaps);
        }
    }
}

pub(super) struct VertexBufferHandler<B: hal::Backend> {
    buffer: Buffer<B>,
    buffer_usage: hal::buffer::Usage,
    data_stride: usize,
    pitch_alignment_mask: usize,
    non_coherent_atom_size_mask: usize,
    pub(super) buffer_len: usize,
}

impl<B: hal::Backend> VertexBufferHandler<B> {
    pub(super) fn new<T: Copy>(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        buffer_usage: hal::buffer::Usage,
        data: &[T],
        data_stride: usize,
        pitch_alignment_mask: usize,
        non_coherent_atom_size_mask: usize,
    ) -> Self {
        let mut buffer = Buffer::new(
            device,
            heaps,
            MemoryUsageValue::Dynamic,
            buffer_usage,
            pitch_alignment_mask,
            data.len(),
            data_stride,
        );
        buffer.update_all(device, data, non_coherent_atom_size_mask as u64);
        VertexBufferHandler {
            buffer_len: buffer.buffer_len,
            buffer,
            buffer_usage,
            data_stride,
            pitch_alignment_mask,
            non_coherent_atom_size_mask,
        }
    }

    pub(super) fn update<T: Copy>(&mut self, device: &B::Device, data: &[T], heaps: &mut Heaps<B>) {
        let buffer_len = data.len() * self.data_stride;
        if self.buffer.buffer_len != buffer_len {
            let old_buffer = mem::replace(
                &mut self.buffer,
                Buffer::new(
                    device,
                    heaps,
                    MemoryUsageValue::Dynamic,
                    self.buffer_usage,
                    self.pitch_alignment_mask,
                    data.len(),
                    self.data_stride,
                ),
            );
            old_buffer.deinit(device, heaps);
        }
        self.buffer
            .update_all(device, data, self.non_coherent_atom_size_mask as u64);
        self.buffer_len = buffer_len;
    }

    pub(super) fn buffer(&self) -> &Buffer<B> {
        &self.buffer
    }

    pub(super) fn reset(&mut self) {
        self.buffer_len = 0;
    }

    pub(super) fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        self.buffer.deinit(device, heaps);
    }
}
