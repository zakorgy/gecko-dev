/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use arrayvec::ArrayVec;
use hal::Device;
use hal::pso::{DescriptorSetLayoutBinding, DescriptorType as DT, ShaderStageFlags as SSF};
use hal::command::RawCommandBuffer;
use internal_types::FastHashMap;
use rendy_descriptor::{DescriptorAllocator, DescriptorRanges, DescriptorSet};
use rendy_memory::Heaps;
use smallvec::SmallVec;
use std::clone::Clone;
use std::cmp::Eq;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};
use std::marker::Copy;
use super::buffer::{UniformBufferHandler, PMBuffer};
use super::image::Image;
use super::TextureId;
use super::super::{ShaderKind, TextureFilter, VertexArrayKind};

pub(super) const DESCRIPTOR_SET_PER_PASS: usize = 0;
pub(super) const DESCRIPTOR_SET_PER_GROUP: usize = 1;
pub(super) const DESCRIPTOR_SET_PER_DRAW: usize = 2;
pub(super) const DESCRIPTOR_SET_LOCALS: usize = 3;

pub(super) const DESCRIPTOR_COUNT: u32 = 96;
pub(super) const PER_DRAW_TEXTURE_COUNT: usize = 3; // Color0, Color1, Color2
pub(super) const PER_PASS_TEXTURE_COUNT: usize = 2; // PrevPassAlpha, PrevPassColor
pub(super) const PER_GROUP_TEXTURE_COUNT: usize = 6; // GpuCache, TransformPalette, RenderTasks, Dither, PrimitiveHeadersF, PrimitiveHeadersI
pub(super) const RENDERER_TEXTURE_COUNT: usize = 11;
pub(super) const PER_GROUP_RANGE_DEFAULT: std::ops::Range<usize> = 8..9; // Dither
pub(super) const PER_GROUP_RANGE_CLIP: std::ops::Range<usize> = 6..9; //TransformPalette, RenderTasks, Dither
pub(super) const PER_GROUP_RANGE_PRIMITIVE: std::ops::Range<usize> = 6..11; // TransformPalette, RenderTasks, Dither, PrimitiveHeadersF, PrimitiveHeadersI
const GPU_CACHE_BINDING: u32 = 5;

pub(super) const MAX_DESCRIPTOR_SET_COUNT: usize = 4;

const fn descriptor_set_layout_binding(
    binding: u32,
    ty: DT,
    stage_flags: SSF,
    immutable_samplers: bool,
) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        binding,
        ty,
        count: 1,
        stage_flags,
        immutable_samplers,
    }
}

pub(super) const DEFAULT_SET_1: &'static [DescriptorSetLayoutBinding] = &[
    // Dither
    descriptor_set_layout_binding(8, DT::CombinedImageSampler, SSF::ALL, true),
];

pub(super) const COMMON_SET_2: &'static [DescriptorSetLayoutBinding] = &[
    // Color0
    descriptor_set_layout_binding(0, DT::CombinedImageSampler, SSF::ALL, false),
    // Color1
    descriptor_set_layout_binding(1, DT::CombinedImageSampler, SSF::ALL, false),
    // Color2
    descriptor_set_layout_binding(2, DT::CombinedImageSampler, SSF::ALL, false),
];

pub(super) const COMMON_SET_3: &'static [DescriptorSetLayoutBinding] = &[
    // Locals
    descriptor_set_layout_binding(0, DT::UniformBuffer, SSF::VERTEX, false),
];

pub(super) const CLIP_SET_1: &'static [DescriptorSetLayoutBinding] = &[
    // GpuCache
    descriptor_set_layout_binding(5, DT::StorageBuffer, SSF::ALL, false),
    // TransformPalette
    descriptor_set_layout_binding(6, DT::CombinedImageSampler, SSF::VERTEX, true),
    // RenderTasks
    descriptor_set_layout_binding(7, DT::CombinedImageSampler, SSF::VERTEX, true),
    // Dither
    descriptor_set_layout_binding(8, DT::CombinedImageSampler, SSF::ALL, true),
];

pub(super) const PRIMITIVE_SET_1: &'static [DescriptorSetLayoutBinding] = &[
    // GpuCache
    descriptor_set_layout_binding(5, DT::StorageBuffer, SSF::ALL, false),
    // TransformPalette
    descriptor_set_layout_binding(6, DT::CombinedImageSampler, SSF::VERTEX, true),
    // RenderTasks
    descriptor_set_layout_binding(7, DT::CombinedImageSampler, SSF::VERTEX, true),
    // Dither
    descriptor_set_layout_binding(8, DT::CombinedImageSampler, SSF::ALL, true),
    // PrimitiveHeadersF
    descriptor_set_layout_binding(9, DT::CombinedImageSampler, SSF::VERTEX, true),
    // PrimitiveHeadersI
    descriptor_set_layout_binding(10, DT::CombinedImageSampler, SSF::VERTEX, true),
];

pub(super) const PRIMITIVE_SET_0: &'static [DescriptorSetLayoutBinding] = &[
    // PrevPassAlpha
    descriptor_set_layout_binding(3, DT::CombinedImageSampler, SSF::ALL, true),
    // PrevPassColor
    descriptor_set_layout_binding(4, DT::CombinedImageSampler, SSF::ALL, true),
];

pub(super) const EMPTY_SET_0: &'static [DescriptorSetLayoutBinding] = &[];

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub(super) enum DescriptorGroup {
    Default,
    Clip,
    Primitive,
}

impl From<ShaderKind> for DescriptorGroup {
    fn from(kind: ShaderKind) -> Self {
        match kind {
            ShaderKind::Cache(VertexArrayKind::Border) | ShaderKind::Cache(VertexArrayKind::LineDecoration)
                | ShaderKind::DebugColor | ShaderKind::DebugFont
                | ShaderKind::VectorStencil | ShaderKind::VectorCover => DescriptorGroup::Default,
            ShaderKind::ClipCache => DescriptorGroup::Clip,
            ShaderKind::Brush | ShaderKind::Cache(VertexArrayKind::Blur) | ShaderKind::Cache(VertexArrayKind::Scale)
                | ShaderKind::Primitive | ShaderKind::Text => DescriptorGroup::Primitive,
            _ => unimplemented!("No descriptor group for kind {:?}", kind),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[allow(non_snake_case)]
pub(super) struct Locals {
    pub(super) uTransform: [[f32; 4]; 4],
    pub(super) uMode: i32,
}

 impl Locals {
    fn transform_as_u32_slice(&self) -> &[u32; 16] {
        unsafe {
            std::mem::transmute::<&[[f32; 4]; 4], &[u32; 16]>(&self.uTransform)
        }
    }
}

 impl Hash for Locals {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.transform_as_u32_slice().hash(state);
        self.uMode.hash(state);
    }
}

 impl PartialEq for Locals {
    fn eq(&self, other: &Locals) -> bool {
        self.uTransform == other.uTransform &&
        self.uMode == other.uMode
    }
}

impl Eq for Locals {}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Default)]
pub(super) struct PerDrawBindings(pub [TextureId; PER_DRAW_TEXTURE_COUNT], pub [TextureFilter; PER_DRAW_TEXTURE_COUNT]);

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Default)]
pub(super) struct PerPassBindings(pub [TextureId; PER_PASS_TEXTURE_COUNT]);

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Default)]
pub(super) struct PerGroupBindings(pub [TextureId; PER_GROUP_TEXTURE_COUNT]);

pub(super) struct DescriptorGroupData<B: hal::Backend> {
    pub(super) set_layouts: ArrayVec<[B::DescriptorSetLayout; MAX_DESCRIPTOR_SET_COUNT]>,
    pub(super) ranges: ArrayVec<[DescriptorRanges; MAX_DESCRIPTOR_SET_COUNT]>,
    pub(super) pipeline_layout: B::PipelineLayout,
}

pub(super) struct DescriptorData<B: hal::Backend>(pub(super) FastHashMap<DescriptorGroup, DescriptorGroupData<B>>);

impl<B: hal::Backend> DescriptorData<B> {
    fn descriptor_layout(&self, group: &DescriptorGroup, group_idx: usize) -> &B::DescriptorSetLayout {
        &self.0[group].set_layouts[group_idx]
    }

    fn ranges(&self, group: &DescriptorGroup, group_idx: usize) -> DescriptorRanges {
        self.0[group].ranges[group_idx]
    }

    pub(super) fn pipeline_layout(&self, group: &DescriptorGroup) -> &B::PipelineLayout {
        &self.0[group].pipeline_layout
    }

    pub(super) unsafe fn deinit(self, device: &B::Device) {
        for (_, group_data) in self.0 {
            for layout in group_data.set_layouts {
                device.destroy_descriptor_set_layout(layout);
            }
            device.destroy_pipeline_layout(group_data.pipeline_layout);
        }
    }
}

// Trait for managing the container type of the free descriptor sets in the `DescriptorSetHandler` struct
// which could be a `Vec` or a `FastHashMap`
pub(super) trait FreeSets<B: hal::Backend> {
    // Mutable reference of the free descriptor sets, `group` is only used for `FastHashMap`
    fn get_mut(&mut self, group: &DescriptorGroup) -> &mut Vec<DescriptorSet<B>>;

    // Free the underlying descriptor sets of the container
    unsafe fn free(self, allocator: &mut DescriptorAllocator<B>);
}

impl<B: hal::Backend> FreeSets<B> for Vec<DescriptorSet<B>> {
    fn get_mut(&mut self, _group: &DescriptorGroup) -> &mut Vec<DescriptorSet<B>> {
        self
    }

    unsafe fn free(self, allocator: &mut DescriptorAllocator<B>) {
        allocator.free(self.into_iter())
    }
}

impl<B: hal::Backend> FreeSets<B> for FastHashMap<DescriptorGroup, Vec<DescriptorSet<B>>> {
    fn get_mut(&mut self, group: &DescriptorGroup) -> &mut Vec<DescriptorSet<B>> {
        self.get_mut(group).unwrap()
    }

    unsafe fn free(self, allocator: &mut DescriptorAllocator<B>) {
        allocator.free(self.into_iter().flat_map(|(_, sets)| sets.into_iter()))
    }
}


// Trait for managing the key of the `descriptor_bindings` `FastHashMap` in the `DescriptorSetHandler` struct
pub(super) trait DescGroupKey {
    // Get the corresponding `DescriptorGroup` of the key structure.
    // There are cases where we don't care the exact group that's the `DescriptorGroup::Default` for.
    fn desc_group(&self) -> &DescriptorGroup { &DescriptorGroup::Default }

    // Check if the key contains the requested `TextureId`.
    // This is used to decide if we can recycle a `DescriptorSet` after a texture is freed.
    // There are keys which are not texture related, in this case we return false.
    fn has_texture_id(&self, _id: &TextureId) -> bool { false }
}

impl DescGroupKey for PerDrawBindings {
    fn has_texture_id(&self, id: &TextureId) -> bool {
        self.0.contains(id)
    }
}

impl DescGroupKey for (DescriptorGroup, PerGroupBindings) {
    fn desc_group(&self) -> &DescriptorGroup {
        &self.0
    }

    fn has_texture_id(&self, id: &TextureId) -> bool {
        (self.1).0.contains(id)
    }
}

impl DescGroupKey for PerPassBindings {
    fn has_texture_id(&self, id: &TextureId) -> bool {
        self.0.contains(id)
    }
}

impl DescGroupKey for Locals {}

pub(super) struct DescriptorSetHandler<K, B: hal::Backend, F> {
    free_sets: F,
    descriptor_bindings: FastHashMap<K, DescriptorSet<B>>,
}

impl<K, B, F> DescriptorSetHandler<K, B, F>
    where
        K: Copy + Clone + Eq + Hash + DescGroupKey,
        B: hal::Backend,
        F: FreeSets<B> + Extend<DescriptorSet<B>> {
    pub(super) fn new(
        device: &B::Device,
        descriptor_allocator: &mut DescriptorAllocator<B>,
        group_data: &DescriptorData<B>,
        group: &DescriptorGroup,
        set_index: usize,
        descriptor_count: u32,
        mut free_sets: F,
    ) -> Self {
        unsafe {
            descriptor_allocator.allocate(
                device,
                group_data.descriptor_layout(group, set_index),
                group_data.ranges(group, set_index),
                descriptor_count,
                &mut free_sets
            )
        }.expect("Allocate descriptor sets failed");
        Self::from_existing(free_sets)
    }
}

impl<K, B, F> DescriptorSetHandler<K, B, F>
    where
        K: Copy + Clone + Eq + Hash + DescGroupKey,
        B: hal::Backend,
        F: FreeSets<B> {
    pub(super) fn from_existing(free_sets: F) -> Self {
        DescriptorSetHandler {
            free_sets,
            descriptor_bindings: FastHashMap::default(),
        }
    }

    pub(super) fn reset(&mut self) {
        for (key, desc_set) in self.descriptor_bindings.drain() {
            self.free_sets.get_mut(key.desc_group()).push(desc_set);
        }
    }

    pub(super) fn descriptor_set(&self, key: &K) -> &B::DescriptorSet {
        self.descriptor_bindings[key].raw()
    }

    pub(super) fn retain(&mut self, id: &TextureId) {
        let keys_to_remove: Vec<_> = self.descriptor_bindings
            .keys()
            .filter(|res| res.has_texture_id(&id)).cloned().collect();
        for key in keys_to_remove {
            let desc_set =self.descriptor_bindings.remove(&key).unwrap();
            self.free_sets.get_mut(key.desc_group()).push(desc_set);
        }
    }

    pub(super) unsafe fn free(self, allocator: &mut DescriptorAllocator<B>) {
        self.free_sets.free(allocator);
        allocator.free(self.descriptor_bindings.into_iter().map(|(_, set)| set));
    }

    pub(super) fn bind_textures(
        &mut self,
        bound_textures: &[u32; RENDERER_TEXTURE_COUNT],
        bound_samplers: &[TextureFilter; RENDERER_TEXTURE_COUNT],
        cmd_buffer: &mut B::CommandBuffer,
        bindings: K,
        images: &FastHashMap<TextureId, Image<B>>,
        storage_buffer: Option<&PMBuffer<B>>,
        desc_allocator: &mut DescriptorAllocator<B>,
        device: &B::Device,
        group_data: &DescriptorData<B>,
        group: &DescriptorGroup,
        set_index: usize,
        range: std::ops::Range<usize>,
        sampler_linear: &B::Sampler,
        sampler_nearest: &B::Sampler,
    ) {
        let new_set = match self.descriptor_bindings.entry(bindings) {
            Entry::Occupied(_) => None,
            Entry::Vacant(v) => {
                let free_sets = self.free_sets.get_mut(group);
                let desc_set = match free_sets.pop() {
                    Some(ds) => ds,
                    None => {
                        unsafe {
                            desc_allocator.allocate(
                                device,
                                group_data.descriptor_layout(group, set_index),
                                group_data.ranges(group, set_index),
                                DESCRIPTOR_COUNT,
                                free_sets,
                            )
                        }.expect("Allocate descriptor sets failed");
                        free_sets.pop().unwrap()
                    }
                };
                Some(v.insert(desc_set))
            }
        };
        let mut descriptor_writes: SmallVec<[hal::pso::DescriptorSetWrite<_, _>; RENDERER_TEXTURE_COUNT]> = SmallVec::new();
        if let Some(buffer) = storage_buffer {
            if let Some(ref set) = new_set {
                descriptor_writes.push(hal::pso::DescriptorSetWrite {
                    set: set.raw(),
                    binding: GPU_CACHE_BINDING,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Buffer(
                        &buffer.buffer,
                        None .. None,
                    )),
                });
            }
        }
        for index in range {
            let image = &images[&bound_textures[index]].core;
            // We need to transit the image, even though it's bound in the descriptor set
            let mut src_stage = Some(hal::pso::PipelineStage::empty());
            if let Some(barrier) = image.transit(
                hal::image::Access::SHADER_READ,
                hal::image::Layout::ShaderReadOnlyOptimal,
                image.subresource_range.clone(),
                src_stage.as_mut(),
            ) {
                unsafe {
                    // TODO(zakorgy): combine these barriers into a single one
                    cmd_buffer.pipeline_barrier(
                        src_stage.unwrap()
                            .. hal::pso::PipelineStage::FRAGMENT_SHADER | hal::pso::PipelineStage::VERTEX_SHADER,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }
            if let Some(ref set) = new_set {
                let sampler = if index < PER_DRAW_TEXTURE_COUNT {
                    match bound_samplers[index] {
                        TextureFilter::Linear | TextureFilter::Trilinear => sampler_linear,
                        TextureFilter::Nearest => sampler_nearest,
                    }
                } else if index < PER_DRAW_TEXTURE_COUNT + PER_PASS_TEXTURE_COUNT {
                    sampler_linear
                } else {
                    sampler_nearest
                };
                descriptor_writes.push(hal::pso::DescriptorSetWrite {
                    set: set.raw(),
                    binding: index as _,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::CombinedImageSampler(
                        &image.view,
                        hal::image::Layout::ShaderReadOnlyOptimal,
                        sampler,
                    )),
                });
            }
        }
        unsafe { device.write_descriptor_sets(descriptor_writes) };
    }

    pub(super) fn bind_locals(
        &mut self,
        bindings: K,
        device: &B::Device,
        desc_allocator: &mut DescriptorAllocator<B>,
        group_data: &DescriptorData<B>,
        locals_buffer: &mut UniformBufferHandler<B>,
        heaps: &mut Heaps<B>,
    ) {
        if let Entry::Vacant(v) = self.descriptor_bindings.entry(bindings) {
            locals_buffer.add(&device, &[bindings], heaps);
            let free_sets = self.free_sets.get_mut(&DescriptorGroup::Default);
            let desc_set = match free_sets.pop() {
                Some(ds) => ds,
                None => {
                    unsafe {
                        desc_allocator.allocate(
                            device,
                            group_data.descriptor_layout(&DescriptorGroup::Default, DESCRIPTOR_SET_LOCALS),
                            group_data.ranges(&DescriptorGroup::Default, DESCRIPTOR_SET_LOCALS),
                            DESCRIPTOR_COUNT,
                            free_sets,
                        )
                    }.expect("Allocate descriptor sets failed");
                    free_sets.pop().unwrap()
                }
            };
            let desc_set = v.insert(desc_set);
            unsafe {
                device.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                    set: desc_set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Buffer(
                        &locals_buffer.buffer().buffer,
                        Some(0) .. None,
                    )),
                }));
            }
        }
    }
}
