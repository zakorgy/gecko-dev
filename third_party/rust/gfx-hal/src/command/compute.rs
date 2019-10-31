//! `CommandBuffer` methods for compute operations.

use std::borrow::Borrow;

use super::{CommandBuffer, DescriptorSetOffset, Level, RawCommandBuffer, Shot};
use crate::buffer::Offset;
use crate::queue::capability::{Compute, Supports};
use crate::{Backend, WorkGroupCount};

impl<B: Backend, C: Supports<Compute>, S: Shot, L: Level> CommandBuffer<B, C, S, L> {
    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn bind_compute_pipeline(&mut self, pipeline: &B::ComputePipeline) {
        self.raw.bind_compute_pipeline(pipeline)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn bind_compute_descriptor_sets<I, J>(
        &mut self,
        layout: &B::PipelineLayout,
        first_set: usize,
        sets: I,
        offsets: J,
    ) where
        I: IntoIterator,
        I::Item: Borrow<B::DescriptorSet>,
        J: IntoIterator,
        J::Item: Borrow<DescriptorSetOffset>,
    {
        self.raw
            .bind_compute_descriptor_sets(layout, first_set, sets, offsets)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn dispatch(&mut self, count: WorkGroupCount) {
        self.raw.dispatch(count)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn dispatch_indirect(&mut self, buffer: &B::Buffer, offset: Offset) {
        self.raw.dispatch_indirect(buffer, offset)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn push_compute_constants(
        &mut self,
        layout: &B::PipelineLayout,
        offset: u32,
        constants: &[u32],
    ) {
        self.raw.push_compute_constants(layout, offset, constants);
    }
}
