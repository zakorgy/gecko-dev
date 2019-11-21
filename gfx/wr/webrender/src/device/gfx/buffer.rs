/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal;
use hal::device::Device as BackendDevice;
use rendy_memory::{Block, Heaps, Kind, MappedRange, MemoryBlock, MemoryUsage, MemoryUsageValue, Write};

use std::cell::Cell;
use std::sync::Arc;
use std::mem;

pub const DOWNLOAD_BUFFER_SIZE: usize = 10 << 20; // 10MB

#[derive(MallocSizeOf)]
pub struct BufferMemorySlice {
    #[ignore_malloc_size_of = "ptr"]
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for BufferMemorySlice {}
unsafe impl Sync for BufferMemorySlice {}

impl BufferMemorySlice {
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        BufferMemorySlice { ptr, size }
    }

    pub(crate) fn slice_mut<T>(&mut self) -> &mut [T] {
        use std::slice;
        let size_of_t = mem::size_of::<T>();
        assert_eq!(self.ptr as usize % size_of_t, 0);
        assert_eq!(self.size % size_of_t, 0);
        unsafe { slice::from_raw_parts_mut(self.ptr as _, self.size / size_of_t) }
    }
}

#[derive(Clone, Copy, Debug)]
struct PMBufferMemoryUsage;
impl MemoryUsage for PMBufferMemoryUsage {
    fn properties_required(&self) -> hal::memory::Properties {
        hal::memory::Properties::CPU_VISIBLE
    }

    #[inline]
    fn memory_fitness(&self, properties: hal::memory::Properties) -> u32 {
        assert!(properties.contains(hal::memory::Properties::CPU_VISIBLE));
        assert!(!properties.contains(hal::memory::Properties::LAZILY_ALLOCATED));

        0 | (properties.contains(hal::memory::Properties::CPU_CACHED) as u32) << 2
            | (properties.contains(hal::memory::Properties::COHERENT) as u32) << 1
    }

    fn allocator_fitness(&self, kind: Kind) -> u32 {
        match kind {
            Kind::Dedicated => 2,
            Kind::Dynamic => 1,
            Kind::Linear => 0,
        }
    }
}

#[derive(MallocSizeOf)]
pub struct GpuCacheBuffer<B: hal::Backend> {
    #[ignore_malloc_size_of = "handle"]
    pub buffer: Arc<B::Buffer>,
    pub transit_range_end: u64,
    #[ignore_malloc_size_of = "State does not implement size_of"]
    pub state: Cell<hal::buffer::State>,
}

unsafe impl<B: hal::Backend> Send for GpuCacheBuffer<B> {}
unsafe impl<B: hal::Backend> Sync for GpuCacheBuffer<B> {}

impl<B: hal::Backend> GpuCacheBuffer<B> {
    pub fn update_transit_range(&mut self, range_end: u64) {
        if range_end > self.transit_range_end {
            self.transit_range_end = range_end;
        }
    }

    pub fn transit(
        &self,
        access: hal::buffer::Access,
        with_range: bool,
    ) -> Option<hal::memory::Barrier<B>> {
        let src_state = self.state.get();
        if src_state == access {
            None
        } else {
            self.state.set(access);
            Some(hal::memory::Barrier::Buffer {
                states: src_state..access,
                target: &self.buffer,
                families: None,
                range: None..if with_range {
                    Some(self.transit_range_end)
                } else {
                    None
                },
            })
        }
    }
}

#[derive(Debug, MallocSizeOf)]
pub struct PersistentlyMappedBuffer<B: hal::Backend> {
    #[ignore_malloc_size_of = "buffer handle"]
    pub buffer: Arc<B::Buffer>,
    #[ignore_malloc_size_of = "memory handle"]
    pub memory_block: MemoryBlock<B>,
    pub coherent: bool,
    pub height: u64,
    pub size: u64,
    pub non_coherent_atom_size_mask: u64,
}

unsafe impl<B: hal::Backend> Send for PersistentlyMappedBuffer<B> {}
unsafe impl<B: hal::Backend> Sync for PersistentlyMappedBuffer<B> {}

impl<B: hal::Backend> PersistentlyMappedBuffer<B> {
    pub(crate) fn new<T>(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        non_coherent_atom_size_mask: u64,
        width: u64,
        height: u64,
        copy_src: Option<&mut PersistentlyMappedBuffer<B>>,
    ) -> Self {
        let buffer_stride = std::mem::size_of::<T>() as u64;
        let buffer_len = height * width * buffer_stride;

        assert_ne!(buffer_len, 0);
        let mut storage_buffer =
            unsafe { device.create_buffer(buffer_len, hal::buffer::Usage::STORAGE) }.unwrap();

        let buffer_req = unsafe { device.get_buffer_requirements(&storage_buffer) };
        let alignment = ((buffer_req.alignment - 1) | non_coherent_atom_size_mask) + 1;
        let memory_block = heaps
            .allocate(
                device,
                buffer_req.type_mask as u32,
                PMBufferMemoryUsage,
                buffer_req.size,
                alignment,
            )
            .expect("Allocate memory failed");

        assert!(memory_block
            .properties()
            .contains(hal::memory::Properties::CPU_CACHED));
        let coherent = memory_block
            .properties()
            .contains(hal::memory::Properties::COHERENT);

        unsafe {
            device.bind_buffer_memory(
                &memory_block.memory(),
                memory_block.range().start,
                &mut storage_buffer,
            )
        }
        .expect("Bind buffer memory failed");
        let mut buffer = PersistentlyMappedBuffer {
            buffer: Arc::new(storage_buffer),
            memory_block,
            coherent,
            height,
            size: buffer_req.size,
            non_coherent_atom_size_mask,
        };

        if let Some(src) = copy_src {
            unsafe { buffer.copy_from(src, device) };
        }

        buffer
    }

    unsafe fn copy_from(&mut self, copy_src: &mut Self, device: &B::Device) {
        {
            let (mut mapped, read_size) = copy_src.map(device, None);
            let read_slice = mapped.read::<u8>(device, 0..read_size).unwrap();
            let (mut mapped, _) = self.map(device, Some(read_size));
            let mut writer = mapped.write::<u8>(device, 0..read_size).unwrap();
            let writer_slice = writer.slice();
            writer_slice.copy_from_slice(read_slice);
        }
        copy_src.unmap(device);
        self.unmap(device);
    }

    pub fn buffer_memory_slice(&mut self, device: &B::Device) -> BufferMemorySlice {
        let (mapped, size) = self.map(device, None);
        BufferMemorySlice::new(mapped.ptr().as_ptr(), size as usize)
    }

    pub fn map<'a>(
        &'a mut self,
        device: &B::Device,
        size: Option<u64>,
    ) -> (MappedRange<'a, B>, u64) {
        assert!(
            size.unwrap_or(0) <= self.size,
            "size {:?} <= self.size {:?}",
            size.unwrap_or(0),
            self.size
        );
        let size = size.unwrap_or(self.size);
        (
            self.memory_block
                .map(&device, 0..size)
                .expect("Mapping memory block failed"),
            size,
        )
    }

    pub fn unmap(&mut self, device: &B::Device) {
        self.memory_block.unmap(device);
    }

    pub fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        unsafe {
            device.destroy_buffer(Arc::try_unwrap(self.buffer).unwrap());
            heaps.free(device, self.memory_block);
        }
    }

    pub fn get_buffer_info(&self, transit_range_end: u64) -> GpuCacheBuffer<B> {
        GpuCacheBuffer {
            buffer: Arc::clone(&self.buffer),
            transit_range_end: transit_range_end,
            state: Cell::new(hal::buffer::Access::empty()),
        }
    }
}

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
        let range = 0..((size + non_coherent_atom_size_mask) & !non_coherent_atom_size_mask)
            .min(self.buffer_size as u64);
        unsafe {
            let mut mapped = self
                .memory_block
                .map(device, range.clone())
                .expect("Mapping memory block failed");
            mapped
                .write(device, 0..size)
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
        let range =
            offset..((offset + size + non_coherent_atom_size_mask) & !non_coherent_atom_size_mask);
        unsafe {
            let mut mapped = self
                .memory_block
                .map(device, range)
                .expect("Mapping memory block failed");
            mapped
                .write(device, 0..size)
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
                states: src_state..access,
                target: &self.buffer,
                families: None,
                range: None..None,
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
        texture_cache_size: usize,
    ) -> Self {
        let buffer = Buffer::new(
            device,
            heaps,
            MemoryUsageValue::Upload,
            buffer_usage,
            pitch_alignment_mask | non_coherent_atom_size_mask,
            texture_cache_size,
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

    pub(super) fn add<T: Copy>(&mut self, device: &B::Device, data: &[T], texel_size_mask: usize) {
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
        let alignment_mask = self.copy_alignment_mask | texel_size_mask;
        self.buffer_offset = (self.offset + alignment_mask) & !alignment_mask;
        self.size = self.buffer.update(
            device,
            data,
            self.buffer_offset,
            self.non_coherent_atom_size_mask as u64,
        );
        let diff = self.buffer_offset - self.offset;
        self.offset += (self.size + diff + alignment_mask) & !alignment_mask;
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
    pub(super) last_data_stride: usize,
    non_coherent_atom_size_mask: usize,
}

impl<B: hal::Backend> InstancePoolBuffer<B> {
    fn new(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        buffer_usage: hal::buffer::Usage,
        alignment_mask: usize,
        non_coherent_atom_size_mask: usize,
        size: usize,
    ) -> Self {
        let buffer = Buffer::new(
            device,
            heaps,
            MemoryUsageValue::Dynamic,
            buffer_usage,
            alignment_mask,
            size,
            mem::size_of::<u8>(),
        );
        InstancePoolBuffer {
            buffer,
            offset: 0,
            last_update_size: 0,
            last_data_stride: 0,
            non_coherent_atom_size_mask,
        }
    }

    fn update(&mut self, device: &B::Device, data: &[u8], last_data_stride: usize) {
        self.buffer.update(
            device,
            data,
            self.offset,
            self.non_coherent_atom_size_mask as u64,
        );
        self.last_data_stride = last_data_stride;
        self.last_update_size = data.len();
        self.offset += self.last_update_size;
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.last_update_size = 0;
    }

    pub(super) fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        self.buffer.deinit(device, heaps);
    }

    fn space_left(&self) -> usize {
        self.buffer.buffer_size - self.offset
    }

    fn can_store_data(&self, stride: usize) -> bool {
        let next_offset = self.next_aligned_offset(stride);
        next_offset < self.buffer.buffer_size && self.buffer.buffer_size - next_offset >= stride
    }

    fn align_offset_to(&mut self, stride: usize) {
        self.offset = self.next_aligned_offset(stride);
    }

    fn next_aligned_offset(&self, stride: usize) -> usize {
        let remainder = self.offset % stride;
        match remainder {
            0 => self.offset,
            _ => self.offset + stride - remainder,
        }
    }
}

pub(super) struct InstanceBufferHandler<B: hal::Backend> {
    pub(super) buffers: Vec<InstancePoolBuffer<B>>,
    alignment_mask: usize,
    non_coherent_atom_size_mask: usize,
    pub(super) next_buffer_index: usize,
    buffer_size: usize,
}

impl<B: hal::Backend> InstanceBufferHandler<B> {
    pub(super) fn new(
        non_coherent_atom_size_mask: usize,
        alignment_mask: usize,
        buffer_size: usize,
    ) -> Self {
        InstanceBufferHandler {
            buffers: Vec::new(),
            alignment_mask,
            non_coherent_atom_size_mask,
            next_buffer_index: 0,
            buffer_size,
        }
    }

    pub(super) fn add<T: Copy>(
        &mut self,
        device: &B::Device,
        mut instance_data: &[T],
        heaps: &mut Heaps<B>,
        free_buffers: &mut Vec<InstancePoolBuffer<B>>,
    ) -> std::ops::Range<usize> {
        fn instance_data_to_u8_slice<T: Copy>(data: &[T]) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * mem::size_of::<T>(),
                )
            }
        }

        let data_stride = mem::size_of::<T>();
        let mut range = 0..0;
        let mut first_iteration = true;
        while !instance_data.is_empty() {
            let need_new_buffer =
                self.buffers.is_empty() || !self.current_buffer().can_store_data(data_stride);
            if need_new_buffer {
                let buffer = match free_buffers.pop() {
                    Some(b) => b,
                    None => InstancePoolBuffer::new(
                        device,
                        heaps,
                        hal::buffer::Usage::VERTEX,
                        self.alignment_mask,
                        self.non_coherent_atom_size_mask,
                        self.buffer_size,
                    ),
                };
                self.buffers.push(buffer);
                self.next_buffer_index += 1;
            } else {
                self.current_buffer_mut().align_offset_to(data_stride);
            }
            if first_iteration {
                range.start = self.next_buffer_index - 1;
                first_iteration = false;
            }
            let update_size =
                (self.current_buffer().space_left() / data_stride).min(instance_data.len());
            self.current_buffer_mut().update(
                device,
                instance_data_to_u8_slice(&instance_data[0..update_size]),
                data_stride,
            );
            instance_data = &instance_data[update_size..];
        }
        range.end = self.next_buffer_index;
        range
    }

    fn current_buffer(&self) -> &InstancePoolBuffer<B> {
        &self.buffers[self.next_buffer_index - 1]
    }

    fn current_buffer_mut(&mut self) -> &mut InstancePoolBuffer<B> {
        &mut self.buffers[self.next_buffer_index - 1]
    }

    pub(super) fn reset(&mut self, free_buffers: &mut Vec<InstancePoolBuffer<B>>) {
        if !self.buffers.is_empty() {
            // Keep one buffer and move the others back to the free set pool.
            for mut buffer in self.buffers.drain(1..) {
                buffer.reset();
                free_buffers.push(buffer);
            }
            self.next_buffer_index = self.buffers.len();
        }
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
        pitch_alignment_mask: usize,
        non_coherent_atom_size_mask: usize,
    ) -> Self {
        let data_stride = mem::size_of::<T>();
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
        self.data_stride = mem::size_of::<T>();
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

pub(super) struct UniformBufferHandler<B: hal::Backend> {
    buffers: Vec<Buffer<B>>,
    offset: usize,
    buffer_usage: hal::buffer::Usage,
    data_stride: usize,
    pitch_alignment_mask: usize,
}

impl<B: hal::Backend> UniformBufferHandler<B> {
    pub(super) fn new(
        buffer_usage: hal::buffer::Usage,
        data_stride: usize,
        pitch_alignment_mask: usize,
    ) -> Self {
        UniformBufferHandler {
            buffers: vec![],
            offset: 0,
            buffer_usage,
            data_stride,
            pitch_alignment_mask,
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
        self.buffers[self.offset].update_all(device, data, self.pitch_alignment_mask as u64);
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
