/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, DeviceIntRect, ImageFormat};
use euclid::Transform3D;
use hal::{self, Device as BackendDevice};
use internal_types::FastHashMap;
use smallvec::SmallVec;
use rendy_memory::Heaps;

use super::buffer::{InstanceBufferHandler, UniformBufferHandler, VertexBufferHandler};
use super::blend_state::SUBPIXEL_CONSTANT_TEXT_COLOR;
use super::descriptor::DescriptorPools;
use super::image::ImageCore;
use super::render_pass::RenderPass;
use super::vertex_types;
use super::PipelineRequirements;
use super::super::{ShaderKind, VertexArrayKind};
use super::super::super::shader_source;

use std::mem;

const ENTRY_NAME: &str = "main";
const MAX_INDEX_COUNT: usize = 4096;
// The number of specialization constants in each shader.
const SPECIALIZATION_CONSTANT_COUNT: usize = 5;
// Size of a specialization constant variable in bytes.
const SPECIALIZATION_CONSTANT_SIZE: usize = 4;
const SPECIALIZATION_FEATURES: &'static [&'static [&'static str]] = &[
    &["ALPHA_PASS"],
    &["COLOR_TARGET"],
    &["GLYPH_TRANSFORM"],
    &["DITHERING"],
    &["DEBUG_OVERDRAW"],
];
const QUAD: [vertex_types::Vertex; 6] = [
    vertex_types::Vertex {
        aPosition: [0.0, 0.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [1.0, 0.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [0.0, 1.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [0.0, 1.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [1.0, 0.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [1.0, 1.0, 0.0],
    },
];

impl ShaderKind {
    pub(super) fn is_debug(&self) -> bool {
        match *self {
            ShaderKind::DebugFont | ShaderKind::DebugColor => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct Locals {
    uTransform: [[f32; 4]; 4],
    uMode: i32,
}

pub(crate) struct Program<B: hal::Backend> {
    bindings_map: FastHashMap<String, u32>,
    pipelines: FastHashMap<(hal::pso::BlendState, hal::pso::DepthTest), B::GraphicsPipeline>,
    pub(super) vertex_buffer: SmallVec<[VertexBufferHandler<B>; 1]>,
    pub(super) index_buffer: Option<SmallVec<[VertexBufferHandler<B>; 1]>>,
    pub(super) instance_buffer: SmallVec<[InstanceBufferHandler<B>; 1]>,
    pub(super) locals_buffer: SmallVec<[UniformBufferHandler<B>; 1]>,
    pub(super) shader_name: String,
    pub(super) shader_kind: ShaderKind,
    pub(super) bound_textures: [u32; 16],
}

impl<B: hal::Backend> Program<B> {
    pub(super) fn create(
        pipeline_requirements: PipelineRequirements,
        device: &B::Device,
        pipeline_layout: &B::PipelineLayout,
        heaps: &mut Heaps<B>,
        limits: &hal::Limits,
        shader_name: &str,
        features: &[&str],
        shader_kind: ShaderKind,
        render_pass: &RenderPass<B>,
        frame_count: usize,
        shader_modules: &mut FastHashMap<String, (B::ShaderModule, B::ShaderModule)>,
        pipeline_cache: Option<&B::PipelineCache>,
        surface_format: ImageFormat,
    ) -> Program<B> {
        if !shader_modules.contains_key(shader_name) {
            let vs_file = format!("{}.vert.spv", shader_name);
            let vs_module = unsafe {
                device.create_shader_module(
                    shader_source::SPIRV_BINARIES
                        .get(vs_file.as_str())
                        .expect("create_shader_module failed"),
                )
            }
            .expect(&format!("Failed to create vs module for: {}!", vs_file));

            let fs_file = format!("{}.frag.spv", shader_name);
            let fs_module = unsafe {
                device.create_shader_module(
                    shader_source::SPIRV_BINARIES
                        .get(fs_file.as_str())
                        .expect("create_shader_module failed"),
                )
            }
            .expect(&format!("Failed to create vs module for: {}!", fs_file));
            shader_modules.insert(String::from(shader_name), (vs_module, fs_module));
        }

        let (vs_module, fs_module) = shader_modules.get(shader_name).unwrap();

        let mut constants = Vec::with_capacity(SPECIALIZATION_CONSTANT_COUNT);
        let mut specialization_data =
            vec![0; SPECIALIZATION_CONSTANT_COUNT * SPECIALIZATION_CONSTANT_SIZE];
        for i in 0 .. SPECIALIZATION_CONSTANT_COUNT {
            constants.push(hal::pso::SpecializationConstant {
                id: i as _,
                range: (SPECIALIZATION_CONSTANT_SIZE * i) as _
                    .. (SPECIALIZATION_CONSTANT_SIZE * (i + 1)) as _,
            });
            for (index, feature) in SPECIALIZATION_FEATURES[i].iter().enumerate() {
                if features.contains(feature) {
                    specialization_data[SPECIALIZATION_CONSTANT_SIZE * i] = (index + 1) as u8;
                }
            }
        }

        let pipelines = {
            let (vs_entry, fs_entry) = (
                hal::pso::EntryPoint::<B> {
                    entry: ENTRY_NAME,
                    module: &vs_module,
                    specialization: hal::pso::Specialization {
                        constants: &constants,
                        data: &specialization_data.as_slice(),
                    },
                },
                hal::pso::EntryPoint::<B> {
                    entry: ENTRY_NAME,
                    module: &fs_module,
                    specialization: hal::pso::Specialization {
                        constants: &constants,
                        data: &specialization_data.as_slice(),
                    },
                },
            );

            let shader_entries = hal::pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            use hal::pso::{BlendState, DepthTest};
            use super::blend_state::*;
            use super::{LESS_EQUAL_TEST, LESS_EQUAL_WRITE};

            let pipeline_states = match shader_kind {
                ShaderKind::Cache(VertexArrayKind::Scale) => [
                    (BlendState::Off, DepthTest::Off),
                    (BlendState::MULTIPLY, DepthTest::Off),
                ]
                .into_iter(),
                ShaderKind::Cache(VertexArrayKind::Blur) => [
                    (BlendState::Off, DepthTest::Off),
                    (BlendState::Off, LESS_EQUAL_TEST),
                ]
                .into_iter(),
                ShaderKind::Cache(VertexArrayKind::Border)
                | ShaderKind::Cache(VertexArrayKind::LineDecoration) => {
                    [(BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off)].into_iter()
                }
                ShaderKind::ClipCache => [(BlendState::MULTIPLY, DepthTest::Off)].into_iter(),
                ShaderKind::Text => {
                    if features.contains(&"DUAL_SOURCE_BLENDING") {
                        [
                            (BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off),
                            (BlendState::PREMULTIPLIED_ALPHA, LESS_EQUAL_TEST),
                            (SUBPIXEL_CONSTANT_TEXT_COLOR, DepthTest::Off),
                            (SUBPIXEL_CONSTANT_TEXT_COLOR, LESS_EQUAL_TEST),
                            (SUBPIXEL_PASS0, DepthTest::Off),
                            (SUBPIXEL_PASS0, LESS_EQUAL_TEST),
                            (SUBPIXEL_PASS1, DepthTest::Off),
                            (SUBPIXEL_PASS1, LESS_EQUAL_TEST),
                            (SUBPIXEL_WITH_BG_COLOR_PASS0, DepthTest::Off),
                            (SUBPIXEL_WITH_BG_COLOR_PASS0, LESS_EQUAL_TEST),
                            (SUBPIXEL_WITH_BG_COLOR_PASS1, DepthTest::Off),
                            (SUBPIXEL_WITH_BG_COLOR_PASS1, LESS_EQUAL_TEST),
                            (SUBPIXEL_WITH_BG_COLOR_PASS2, DepthTest::Off),
                            (SUBPIXEL_WITH_BG_COLOR_PASS2, LESS_EQUAL_TEST),
                            (SUBPIXEL_DUAL_SOURCE, DepthTest::Off),
                            (SUBPIXEL_DUAL_SOURCE, LESS_EQUAL_TEST),
                        ]
                        .into_iter()
                    } else {
                        [
                            (BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off),
                            (BlendState::PREMULTIPLIED_ALPHA, LESS_EQUAL_TEST),
                            (SUBPIXEL_CONSTANT_TEXT_COLOR, DepthTest::Off),
                            (SUBPIXEL_CONSTANT_TEXT_COLOR, LESS_EQUAL_TEST),
                            (SUBPIXEL_PASS0, DepthTest::Off),
                            (SUBPIXEL_PASS0, LESS_EQUAL_TEST),
                            (SUBPIXEL_PASS1, DepthTest::Off),
                            (SUBPIXEL_PASS1, LESS_EQUAL_TEST),
                            (SUBPIXEL_WITH_BG_COLOR_PASS0, DepthTest::Off),
                            (SUBPIXEL_WITH_BG_COLOR_PASS0, LESS_EQUAL_TEST),
                            (SUBPIXEL_WITH_BG_COLOR_PASS1, DepthTest::Off),
                            (SUBPIXEL_WITH_BG_COLOR_PASS1, LESS_EQUAL_TEST),
                            (SUBPIXEL_WITH_BG_COLOR_PASS2, DepthTest::Off),
                            (SUBPIXEL_WITH_BG_COLOR_PASS2, LESS_EQUAL_TEST),
                        ]
                        .into_iter()
                    }
                }
                ShaderKind::DebugColor | ShaderKind::DebugFont => {
                    [(BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off)].into_iter()
                }
                _ => [
                    (BlendState::Off, DepthTest::Off),
                    (BlendState::Off, LESS_EQUAL_TEST),
                    (BlendState::Off, LESS_EQUAL_WRITE),
                    (ALPHA, DepthTest::Off),
                    (ALPHA, LESS_EQUAL_TEST),
                    (ALPHA, LESS_EQUAL_WRITE),
                    (BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off),
                    (BlendState::PREMULTIPLIED_ALPHA, LESS_EQUAL_TEST),
                    (BlendState::PREMULTIPLIED_ALPHA, LESS_EQUAL_WRITE),
                    (PREMULTIPLIED_DEST_OUT, DepthTest::Off),
                    (PREMULTIPLIED_DEST_OUT, LESS_EQUAL_TEST),
                    (PREMULTIPLIED_DEST_OUT, LESS_EQUAL_WRITE),
                ]
                .into_iter(),
            };

            let format = match shader_kind {
                ShaderKind::ClipCache => ImageFormat::R8,
                ShaderKind::Cache(VertexArrayKind::Blur) if features.contains(&"ALPHA_TARGET") => {
                    ImageFormat::R8
                }
                ShaderKind::Cache(VertexArrayKind::Scale) if features.contains(&"ALPHA_TARGET") => {
                    ImageFormat::R8
                }
                _ => surface_format,
            };

            let create_desc = |(blend_state, depth_test)| {
                let subpass = hal::pass::Subpass {
                    index: 0,
                    main_pass: render_pass
                        .get_render_pass(format, depth_test != hal::pso::DepthTest::Off),
                };
                let mut pipeline_descriptor = hal::pso::GraphicsPipelineDesc::new(
                    shader_entries.clone(),
                    hal::Primitive::TriangleList,
                    hal::pso::Rasterizer::FILL,
                    &pipeline_layout,
                    subpass,
                );
                pipeline_descriptor
                    .blender
                    .targets
                    .push(hal::pso::ColorBlendDesc(
                        hal::pso::ColorMask::ALL,
                        blend_state,
                    ));

                pipeline_descriptor.depth_stencil = hal::pso::DepthStencilDesc {
                    depth: depth_test,
                    depth_bounds: false,
                    stencil: hal::pso::StencilTest::Off,
                };

                pipeline_descriptor.vertex_buffers =
                    pipeline_requirements.vertex_buffer_descriptors.clone();
                pipeline_descriptor.attributes =
                    pipeline_requirements.attribute_descriptors.clone();
                pipeline_descriptor
            };

            let pipelines_descriptors = pipeline_states.clone().map(|ps| create_desc(*ps));

            let pipelines =
                unsafe { device.create_graphics_pipelines(pipelines_descriptors, pipeline_cache) }
                    .into_iter();

            let mut states = pipeline_states
                .cloned()
                .zip(pipelines.map(|pipeline| pipeline.expect("Pipeline creation failed")))
                .collect::<FastHashMap<(hal::pso::BlendState, hal::pso::DepthTest), B::GraphicsPipeline>>();

            if features.contains(&"DEBUG_OVERDRAW") {
                let pipeline_state = (OVERDRAW, LESS_EQUAL_TEST);
                let pipeline_descriptor = create_desc(pipeline_state);
                let pipeline = unsafe {
                    device.create_graphics_pipeline(&pipeline_descriptor, pipeline_cache)
                }
                .expect("Pipeline creation failed");
                states.insert(pipeline_state, pipeline);
            }

            states
        };

        let vertex_buffer_stride = match shader_kind {
            ShaderKind::DebugColor => mem::size_of::<vertex_types::DebugColorVertex>(),
            ShaderKind::DebugFont => mem::size_of::<vertex_types::DebugFontVertex>(),
            _ => mem::size_of::<vertex_types::Vertex>(),
        };

        let instance_buffer_stride = match shader_kind {
            ShaderKind::Primitive
            | ShaderKind::Brush
            | ShaderKind::Text
            | ShaderKind::Cache(VertexArrayKind::Primitive) => {
                mem::size_of::<vertex_types::PrimitiveInstanceData>()
            }
            ShaderKind::ClipCache | ShaderKind::Cache(VertexArrayKind::Clip) => {
                mem::size_of::<vertex_types::ClipMaskInstance>()
            }
            ShaderKind::Cache(VertexArrayKind::Blur) => {
                mem::size_of::<vertex_types::BlurInstance>()
            }
            ShaderKind::Cache(VertexArrayKind::Border) => {
                mem::size_of::<vertex_types::BorderInstance>()
            }
            ShaderKind::Cache(VertexArrayKind::Scale) => {
                mem::size_of::<vertex_types::ScalingInstance>()
            }
            ShaderKind::Cache(VertexArrayKind::LineDecoration) => {
                mem::size_of::<vertex_types::LineDecorationInstance>()
            }
            sk if sk.is_debug() => 1,
            _ => unreachable!(),
        };

        let mut vertex_buffer = SmallVec::new();
        let mut instance_buffer = SmallVec::new();
        let mut locals_buffer = SmallVec::new();
        let mut index_buffer = if shader_kind.is_debug() {
            Some(SmallVec::new())
        } else {
            None
        };
        for _ in 0 .. frame_count {
            vertex_buffer.push(VertexBufferHandler::new(
                device,
                heaps,
                hal::buffer::Usage::VERTEX,
                &QUAD,
                vertex_buffer_stride,
                (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                (limits.non_coherent_atom_size - 1) as usize,
            ));
            instance_buffer.push(InstanceBufferHandler::new(
                device,
                heaps,
                instance_buffer_stride,
                (limits.non_coherent_atom_size - 1) as usize,
                (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
            ));
            locals_buffer.push(UniformBufferHandler::new(
                hal::buffer::Usage::UNIFORM,
                mem::size_of::<Locals>(),
                (limits.min_uniform_buffer_offset_alignment - 1) as usize,
                (limits.non_coherent_atom_size - 1) as usize,
            ));
            if let Some(ref mut index_buffer) = index_buffer {
                index_buffer.push(VertexBufferHandler::new(
                    device,
                    heaps,
                    hal::buffer::Usage::INDEX,
                    &vec![0u32; MAX_INDEX_COUNT],
                    mem::size_of::<u32>(),
                    (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                    (limits.non_coherent_atom_size - 1) as usize,
                ));
            }
        }

        let bindings_map = pipeline_requirements.bindings_map;

        Program {
            bindings_map,
            pipelines,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            locals_buffer,
            shader_name: String::from(shader_name),
            shader_kind,
            bound_textures: [0; 16],
        }
    }

    pub(super) fn bind_instances<T: Copy>(
        &mut self,
        device: &B::Device,
        heaps: &mut Heaps<B>,
        instances: &[T],
        buffer_id: usize,
    ) {
        assert!(!instances.is_empty());
        self.instance_buffer[buffer_id].add(device, instances, heaps);
    }

    pub(super) fn bind_locals(
        &mut self,
        device: &B::Device,
        heaps: &mut Heaps<B>,
        set: &B::DescriptorSet,
        projection: &Transform3D<f32>,
        u_mode: i32,
        buffer_id: usize,
    ) {
        let locals_data = Locals {
            uTransform: projection.to_row_arrays(),
            uMode: u_mode,
        };
        self.locals_buffer[buffer_id].add(device, &[locals_data], heaps);
        unsafe {
            device.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                set,
                binding: self.bindings_map["Locals"],
                array_offset: 0,
                descriptors: Some(hal::pso::Descriptor::Buffer(
                    &self.locals_buffer[buffer_id].buffer().buffer,
                    Some(0) .. None,
                )),
            }));
        }
    }

    pub(super) fn bind_texture(
        &self,
        device: &B::Device,
        set: &B::DescriptorSet,
        image: &ImageCore<B>,
        binding: &'static str,
        cmd_buffer: &mut hal::command::CommandBuffer<B, hal::Graphics>,
    ) {
        if let Some(binding) = self.bindings_map.get(&("t".to_owned() + binding)) {
            unsafe {
                let mut src_stage = Some(hal::pso::PipelineStage::empty());
                if let Some(barrier) = image.transit(
                    hal::image::Access::SHADER_READ,
                    hal::image::Layout::ShaderReadOnlyOptimal,
                    image.subresource_range.clone(),
                    src_stage.as_mut(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        src_stage.unwrap()
                            .. hal::pso::PipelineStage::FRAGMENT_SHADER,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                device.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                    set,
                    binding: *binding,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Image(
                        &image.view,
                        hal::image::Layout::ShaderReadOnlyOptimal,
                    )),
                }));
            }
        }
    }

    pub(super) fn bind_sampler(
        &self,
        device: &B::Device,
        set: &B::DescriptorSet,
        sampler: &B::Sampler,
        binding: &'static str,
    ) {
        if let Some(binding) = self.bindings_map.get(&("s".to_owned() + binding)) {
            unsafe {
                device.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                    set,
                    binding: *binding,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Sampler(sampler)),
                }));
            }
        }
    }

    pub(super) fn submit(
        &self,
        cmd_buffer: &mut hal::command::CommandBuffer<B, hal::Graphics>,
        viewport: hal::pso::Viewport,
        render_pass: &B::RenderPass,
        frame_buffer: &B::Framebuffer,
        desc_pools: &mut DescriptorPools<B>,
        desc_pools_global: &mut DescriptorPools<B>,
        desc_pools_sampler: &mut DescriptorPools<B>,
        clear_values: &[hal::command::ClearValue],
        blend_state: hal::pso::BlendState,
        blend_color: ColorF,
        depth_test: hal::pso::DepthTest,
        scissor_rect: Option<DeviceIntRect>,
        next_id: usize,
        pipeline_layouts: &FastHashMap<ShaderKind, B::PipelineLayout>,
        pipeline_requirements: &FastHashMap<String, PipelineRequirements>,
        device: &B::Device,
    ) {
        let vertex_buffer = &self.vertex_buffer[next_id];
        let instance_buffer = &self.instance_buffer[next_id];

        unsafe {
            cmd_buffer.set_viewports(0, &[viewport.clone()]);
            match scissor_rect {
                Some(r) => cmd_buffer.set_scissors(
                    0,
                    &[hal::pso::Rect {
                        x: r.origin.x as _,
                        y: r.origin.y as _,
                        w: r.size.width as _,
                        h: r.size.height as _,
                    }],
                ),
                None => cmd_buffer.set_scissors(0, &[viewport.rect]),
            }
            cmd_buffer.bind_graphics_pipeline(
                &self
                    .pipelines
                    .get(&(blend_state, depth_test))
                    .expect(&format!(
                        "The blend state {:?} with depth test {:?} not found for {} program!",
                        blend_state, depth_test, self.shader_name
                    )),
            );

            cmd_buffer.bind_graphics_descriptor_sets(
                &pipeline_layouts[&self.shader_kind],
                0,
                Some(desc_pools.get(&self.shader_kind))
                    .into_iter()
                    .chain(Some(desc_pools_global.get(&self.shader_kind)))
                    .chain(Some(desc_pools_sampler.get(&self.shader_kind))),
                &[],
            );
            desc_pools.next(&self.shader_kind, device, pipeline_requirements);
            desc_pools_global.next(&self.shader_kind, device, pipeline_requirements);
            desc_pools_sampler.next(&self.shader_kind, device, pipeline_requirements);

            if blend_state == SUBPIXEL_CONSTANT_TEXT_COLOR {
                cmd_buffer.set_blend_constants(blend_color.to_array());
            }

            if let Some(ref index_buffer) = self.index_buffer {
                cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer.buffer().buffer, 0)));
                cmd_buffer.bind_index_buffer(hal::buffer::IndexBufferView {
                    buffer: &index_buffer[next_id].buffer().buffer,
                    offset: 0,
                    index_type: hal::IndexType::U32,
                });

                {
                    let mut encoder = cmd_buffer.begin_render_pass_inline(
                        render_pass,
                        frame_buffer,
                        viewport.rect,
                        clear_values,
                    );

                    encoder.draw_indexed(
                        0 .. index_buffer[next_id].buffer().buffer_len as u32,
                        0,
                        0 .. 1,
                    );
                }
            } else {
                for i in 0 ..= instance_buffer.current_buffer_index {
                    cmd_buffer.bind_vertex_buffers(
                        0,
                        Some((&vertex_buffer.buffer().buffer, 0))
                            .into_iter()
                            .chain(Some((&instance_buffer.buffers[i].buffer.buffer, 0))),
                    );

                    {
                        let mut encoder = cmd_buffer.begin_render_pass_inline(
                            render_pass,
                            frame_buffer,
                            viewport.rect,
                            clear_values,
                        );
                        let offset = instance_buffer.buffers[i].offset;
                        let size = instance_buffer.buffers[i].last_update_size;
                        encoder.draw(
                            0 .. vertex_buffer.buffer_len as _,
                            (offset - size) as u32 .. offset as u32,
                        );
                    }
                }
            }
        }
    }

    pub(super) fn deinit(mut self, device: &B::Device, heaps: &mut Heaps<B>) {
        for mut vertex_buffer in self.vertex_buffer {
            vertex_buffer.deinit(device, heaps);
        }
        if let Some(index_buffer) = self.index_buffer {
            for mut index_buffer in index_buffer {
                index_buffer.deinit(device, heaps);
            }
        }
        for mut instance_buffer in self.instance_buffer {
            instance_buffer.deinit(device, heaps);
        }
        for mut locals_buffer in self.locals_buffer {
            locals_buffer.deinit(device, heaps);
        }
        for pipeline in self.pipelines.drain() {
            unsafe { device.destroy_graphics_pipeline(pipeline.1) };
        }
    }
}
