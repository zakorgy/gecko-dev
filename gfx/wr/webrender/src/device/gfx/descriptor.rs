/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal;
use hal::Device as BackendDevice;
use hal::DescriptorPool;
use hal::pso::{DescriptorRangeDesc, DescriptorSetLayoutBinding};
use internal_types::FastHashMap;

use super::PipelineRequirements;
use super::super::ShaderKind;

const DEBUG_DESCRIPTOR_COUNT: usize = 5;

// There are three kind of shader layout one for debug shaders, one for cache clip shaders
// and the rest has the same(default) layout.
// We use these shader names to get the layout for it's corresponding group from a HashMap.
const DEBUG_SHADER: &'static str = "debug_color";
const CACHE_CLIP_SHADER: &'static str = "cs_clip_rectangle";
const DEFAULT_SHADER: &'static str = "brush_solid";

struct DescPool<B: hal::Backend> {
    descriptor_pool: B::DescriptorPool,
    descriptor_set: Vec<B::DescriptorSet>,
    descriptor_set_layout: B::DescriptorSetLayout,
    current_descriptor_set_idx: usize,
    max_descriptor_set_size: usize,
}

impl<B: hal::Backend> DescPool<B> {
    fn new(
        device: &B::Device,
        max_size: usize,
        descriptor_range_descriptors: Vec<DescriptorRangeDesc>,
        descriptor_set_layout: Vec<DescriptorSetLayoutBinding>,
    ) -> Self {
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                max_size,
                descriptor_range_descriptors.as_slice(),
                hal::pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("create_descriptor_pool failed");
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_set_layout, &[]) }
                .expect("create_descriptor_set_layout failed");
        let mut dp = DescPool {
            descriptor_pool,
            descriptor_set: vec![],
            descriptor_set_layout,
            current_descriptor_set_idx: 0,
            max_descriptor_set_size: max_size,
        };
        dp.allocate();
        dp
    }

    fn descriptor_set(&self) -> &B::DescriptorSet {
        &self.descriptor_set[self.current_descriptor_set_idx]
    }

    fn descriptor_set_layout(&self) -> &B::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    fn next(&mut self) -> bool {
        self.current_descriptor_set_idx += 1;
        if self.current_descriptor_set_idx >= self.max_descriptor_set_size {
            return false;
        }
        if self.current_descriptor_set_idx == self.descriptor_set.len() {
            self.allocate();
        }
        true
    }

    fn allocate(&mut self) {
        let desc_set = unsafe {
            self.descriptor_pool
                .allocate_set(&self.descriptor_set_layout)
        }
        .expect(&format!(
            "Failed to allocate set with layout: {:?}!",
            self.descriptor_set_layout,
        ));
        self.descriptor_set.push(desc_set);
    }

    fn reset(&mut self) {
        self.current_descriptor_set_idx = 0;
    }

    fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout);
            device.destroy_descriptor_pool(self.descriptor_pool);
        }
    }
}

pub(super) struct DescriptorPools<B: hal::Backend> {
    debug_pool: DescPool<B>,
    cache_clip_pool: Vec<DescPool<B>>,
    cache_clip_pool_idx: usize,
    default_pool: Vec<DescPool<B>>,
    default_pool_idx: usize,
    descriptors_per_pool: usize,
    descriptor_group_id: usize,
}

impl<B: hal::Backend> DescriptorPools<B> {
        fn get_layout_and_range(
            pipeline_requirements: &FastHashMap<String, PipelineRequirements>,
            shader_name: &'static str,
            descriptor_group_id: usize,
            descriptors_per_pool: usize,
        ) -> (Vec<DescriptorSetLayoutBinding>, Vec<DescriptorRangeDesc>) {
            let requirement = pipeline_requirements
                .get(shader_name)
                .expect(&format!("{} missing", shader_name));

            let (layout, mut range) = (
                requirement.descriptor_set_layout_bindings[descriptor_group_id].clone(),
                requirement.descriptor_range_descriptors[descriptor_group_id].clone(),
            );

            for r in range.iter_mut() {
                r.count *= descriptors_per_pool;
            }

            (layout, range)
        }


    pub(super) fn new(
        device: &B::Device,
        descriptors_per_pool: usize,
        pipeline_requirements: &FastHashMap<String, PipelineRequirements>,
        descriptor_group_id: usize,
    ) -> Self {
        let (debug_layout, debug_layout_range) = Self::get_layout_and_range(
            pipeline_requirements,
            DEBUG_SHADER,
            descriptor_group_id,
            DEBUG_DESCRIPTOR_COUNT,
        );

        let (cache_clip_layout, cache_clip_layout_range) = Self::get_layout_and_range(
            pipeline_requirements,
            CACHE_CLIP_SHADER,
            descriptor_group_id,
            descriptors_per_pool,
        );

        let (default_layout, default_layout_range) = Self::get_layout_and_range(
            pipeline_requirements,
            DEFAULT_SHADER,
            descriptor_group_id,
            descriptors_per_pool,
        );

        DescriptorPools {
            debug_pool: DescPool::new(
                device,
                DEBUG_DESCRIPTOR_COUNT,
                debug_layout_range,
                debug_layout,
            ),
            cache_clip_pool: vec![DescPool::new(
                device,
                descriptors_per_pool,
                cache_clip_layout_range,
                cache_clip_layout,
            )],
            cache_clip_pool_idx: 0,
            default_pool: vec![DescPool::new(
                device,
                descriptors_per_pool,
                default_layout_range,
                default_layout,
            )],
            default_pool_idx: 0,
            descriptors_per_pool,
            descriptor_group_id,
        }
    }

    fn get_pool(&self, shader_kind: &ShaderKind) -> &DescPool<B> {
        match *shader_kind {
            ShaderKind::DebugColor | ShaderKind::DebugFont => &self.debug_pool,
            ShaderKind::ClipCache => &self.cache_clip_pool[self.cache_clip_pool_idx],
            _ => &self.default_pool[self.default_pool_idx],
        }
    }

    fn get_pool_mut(&mut self, shader_kind: &ShaderKind) -> &mut DescPool<B> {
        match *shader_kind {
            ShaderKind::DebugColor | ShaderKind::DebugFont => &mut self.debug_pool,
            ShaderKind::ClipCache => &mut self.cache_clip_pool[self.cache_clip_pool_idx],
            _ => &mut self.default_pool[self.default_pool_idx],
        }
    }

    pub(super) fn get(&self, shader_kind: &ShaderKind) -> &B::DescriptorSet {
        self.get_pool(shader_kind).descriptor_set()
    }

    pub(super) fn get_layout(&self, shader_kind: &ShaderKind) -> &B::DescriptorSetLayout {
        self.get_pool(shader_kind).descriptor_set_layout()
    }

    pub(super) fn next(
        &mut self,
        shader_kind: &ShaderKind,
        device: &B::Device,
        pipeline_requirements: &FastHashMap<String, PipelineRequirements>,
    ) {
        if self.get_pool_mut(shader_kind).next() {
            return;
        }
        match shader_kind {
            ShaderKind::DebugColor | ShaderKind::DebugFont => unimplemented!("We should have enough debug descriptors!"),
            ShaderKind::ClipCache => {
                self.cache_clip_pool_idx += 1;
                if self.cache_clip_pool_idx < self.cache_clip_pool.len() {
                    assert!(self.get_pool_mut(shader_kind).next());
                    return;
                }
                // In lot of cases when we need extra pools, we will need an enormous amount of descriptors (above 4000).
                // Because of this we double the size of each new pool compared to the previous one.
                let mul = 2_usize.pow(self.cache_clip_pool_idx as u32).min(4096);
                let descriptors_per_pool = self.descriptors_per_pool * mul;
                let (cache_clip_layout, cache_clip_layout_range) = Self::get_layout_and_range(
                    pipeline_requirements,
                    CACHE_CLIP_SHADER,
                    self.descriptor_group_id,
                    descriptors_per_pool,
                );

                self.cache_clip_pool.push(DescPool::new(
                    device,
                    descriptors_per_pool,
                    cache_clip_layout_range,
                    cache_clip_layout,
                ));
                assert!(self.get_pool_mut(shader_kind).next());
            },
                _ => {
                self.default_pool_idx += 1;
                if self.default_pool_idx < self.default_pool.len() {
                    assert!(self.get_pool_mut(shader_kind).next());
                    return;
                }
                let mul = 2_usize.pow(self.default_pool_idx as u32).min(4096);
                let descriptors_per_pool = self.descriptors_per_pool * mul;
                let (default_layout, mut default_layout_range) = Self::get_layout_and_range(
                    pipeline_requirements,
                    DEFAULT_SHADER,
                    self.descriptor_group_id,
                    descriptors_per_pool,
                );
                self.default_pool.push(DescPool::new(
                    device,
                    descriptors_per_pool,
                    default_layout_range,
                    default_layout,
                ));
                assert!(self.get_pool_mut(shader_kind).next());
            },
        }
    }

    pub(super) fn reset(&mut self, device: &B::Device) {
        self.debug_pool.reset();

        // Free descritor pools which were not used in the previous draw
        while self.cache_clip_pool_idx < self.cache_clip_pool.len() - 1 {
            let pool = self.cache_clip_pool.pop()
                .expect("No cache clip pool found");
            pool.deinit(device);
        }
        while self.default_pool_idx < self.default_pool.len() - 1 {
            let pool = self.default_pool.pop()
                .expect("No default pool found");;
            pool.deinit(device);
        }

        for pool in self.cache_clip_pool.iter_mut() {
            pool.reset()
        }
        self.cache_clip_pool_idx = 0;
        for pool in self.default_pool.iter_mut() {
            pool.reset()
        }
        self.default_pool_idx = 0;
    }

    pub(super) fn deinit(self, device: &B::Device) {
        self.debug_pool.deinit(device);
        for pool in self.cache_clip_pool {
            pool.deinit(device)
        }
        for pool in self.default_pool {
            pool.deinit(device)
        }
    }
}
