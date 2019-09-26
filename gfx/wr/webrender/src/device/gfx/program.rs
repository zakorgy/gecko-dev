/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, DeviceIntRect, ImageFormat};
use hal::{self, Device as BackendDevice};
use hal::command::{RawCommandBuffer, SubpassContents};
use internal_types::FastHashMap;
use smallvec::SmallVec;
use rendy_memory::Heaps;
use std::borrow::Cow::{Borrowed};

use super::buffer::{InstanceBufferHandler, VertexBufferHandler};
use super::blend_state::SUBPIXEL_CONSTANT_TEXT_COLOR;
use super::render_pass::RenderPass;
use super::PipelineRequirements;
use super::super::{ShaderKind, VertexArrayKind};
use super::super::super::shader_source;

const ENTRY_NAME: &str = "main";
// The size of the push constant block is 68 bytes, and we upload it with u32 data (4 bytes).
pub(super) const PUSH_CONSTANT_BLOCK_SIZE: usize = 17; // 68 / 4
// The number of specialization constants in each shader.
const SPECIALIZATION_CONSTANT_COUNT: usize = 6;
// Size of a specialization constant variable in bytes.
const SPECIALIZATION_CONSTANT_SIZE: usize = 4;
const SPECIALIZATION_FEATURES: &'static [&'static str] = &[
    "ALPHA_PASS",
    "COLOR_TARGET",
    "GLYPH_TRANSFORM",
    "DITHERING",
    "DEBUG_OVERDRAW",
];


pub(crate) struct Program<B: hal::Backend> {
    pipelines: FastHashMap<(Option<hal::pso::BlendState>, Option<hal::pso::DepthTest>), B::GraphicsPipeline>,
    pub(super) vertex_buffer: Option<SmallVec<[VertexBufferHandler<B>; 1]>>,
    pub(super) index_buffer: Option<SmallVec<[VertexBufferHandler<B>; 1]>>,
    pub(super) shader_name: String,
    pub(super) shader_kind: ShaderKind,
    pub(super) constants: [u32; PUSH_CONSTANT_BLOCK_SIZE],
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
        use_push_consts: bool,
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

        let mut specialization_data =
            vec![0; (SPECIALIZATION_CONSTANT_COUNT - 1) * SPECIALIZATION_CONSTANT_SIZE];
        let mut constants = SPECIALIZATION_FEATURES
            .iter()
            .zip(specialization_data.chunks_mut(SPECIALIZATION_CONSTANT_SIZE))
            .enumerate()
            .map(|(i, (feature, out_data))| {
                out_data[0] = features.contains(feature) as u8;
                hal::pso::SpecializationConstant {
                    id: i as _,
                    range: (SPECIALIZATION_CONSTANT_SIZE * i) as _
                        .. (SPECIALIZATION_CONSTANT_SIZE * (i + 1)) as _,
                }
            })
            .collect::<Vec<_>>();
        constants.push(hal::pso::SpecializationConstant {
            id: (SPECIALIZATION_CONSTANT_COUNT - 1) as _,
            range: {
                let from = (SPECIALIZATION_CONSTANT_COUNT - 1) * SPECIALIZATION_CONSTANT_SIZE;
                let to = from + SPECIALIZATION_CONSTANT_SIZE;
                from as _ .. to as _
            },
        });
        specialization_data.extend_from_slice(&[use_push_consts as u8, 0, 0, 0]);

        let pipelines = {
            let (vs_entry, fs_entry) = (
                hal::pso::EntryPoint::<B> {
                    entry: ENTRY_NAME,
                    module: &vs_module,
                    specialization: hal::pso::Specialization {
                        constants: Borrowed(&constants),
                        data: Borrowed(&specialization_data.as_slice()),
                    },
                },
                hal::pso::EntryPoint::<B> {
                    entry: ENTRY_NAME,
                    module: &fs_module,
                    specialization: hal::pso::Specialization {
                        constants: Borrowed(&constants),
                        data: Borrowed(&specialization_data.as_slice()),
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

            use hal::pso::{BlendState};
            use super::blend_state::*;
            use super::{LESS_EQUAL_TEST, LESS_EQUAL_WRITE};

            let pipeline_states = match shader_kind {
                ShaderKind::Cache(VertexArrayKind::Scale) => [
                    (None, None),
                    (Some(BlendState::MULTIPLY), None),
                ]
                .into_iter(),
                ShaderKind::Cache(VertexArrayKind::Blur) => [
                    (None, None),
                    (None, Some(LESS_EQUAL_TEST)),
                ]
                .into_iter(),
                ShaderKind::Cache(VertexArrayKind::Border)
                | ShaderKind::Cache(VertexArrayKind::LineDecoration) => {
                    [(Some(BlendState::PREMULTIPLIED_ALPHA), None)].into_iter()
                }
                ShaderKind::ClipCache => [(Some(BlendState::MULTIPLY), None)].into_iter(),
                ShaderKind::Text => {
                    if features.contains(&"DUAL_SOURCE_BLENDING") {
                        [
                            (Some(BlendState::PREMULTIPLIED_ALPHA), None),
                            (Some(BlendState::PREMULTIPLIED_ALPHA), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_CONSTANT_TEXT_COLOR), None),
                            (Some(SUBPIXEL_CONSTANT_TEXT_COLOR), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_PASS0), None),
                            (Some(SUBPIXEL_PASS0), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_PASS1), None),
                            (Some(SUBPIXEL_PASS1), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS0), None),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS0), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS1), None),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS1), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS2), None),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS2), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_DUAL_SOURCE), None),
                            (Some(SUBPIXEL_DUAL_SOURCE), Some(LESS_EQUAL_TEST)),
                        ]
                        .into_iter()
                    } else {
                        [
                            (Some(BlendState::PREMULTIPLIED_ALPHA), None),
                            (Some(BlendState::PREMULTIPLIED_ALPHA), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_CONSTANT_TEXT_COLOR), None),
                            (Some(SUBPIXEL_CONSTANT_TEXT_COLOR), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_PASS0), None),
                            (Some(SUBPIXEL_PASS0), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_PASS1), None),
                            (Some(SUBPIXEL_PASS1), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS0), None),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS0), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS1), None),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS1), Some(LESS_EQUAL_TEST)),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS2), None),
                            (Some(SUBPIXEL_WITH_BG_COLOR_PASS2), Some(LESS_EQUAL_TEST)),
                        ]
                        .into_iter()
                    }
                }
                ShaderKind::DebugColor | ShaderKind::DebugFont => {
                    [(Some(BlendState::PREMULTIPLIED_ALPHA), None)].into_iter()
                }
                _ => [
                    (None, None),
                    (None, Some(LESS_EQUAL_TEST)),
                    (None, Some(LESS_EQUAL_WRITE)),
                    (Some(ALPHA), None),
                    (Some(ALPHA), Some(LESS_EQUAL_TEST)),
                    (Some(ALPHA), Some(LESS_EQUAL_WRITE)),
                    (Some(BlendState::PREMULTIPLIED_ALPHA), None),
                    (Some(BlendState::PREMULTIPLIED_ALPHA), Some(LESS_EQUAL_TEST)),
                    (Some(BlendState::PREMULTIPLIED_ALPHA), Some(LESS_EQUAL_WRITE)),
                    (Some(PREMULTIPLIED_DEST_OUT), None),
                    (Some(PREMULTIPLIED_DEST_OUT), Some(LESS_EQUAL_TEST)),
                    (Some(PREMULTIPLIED_DEST_OUT), Some(LESS_EQUAL_WRITE)),
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
                        .get_render_pass(format, depth_test != None),
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
                    .push(hal::pso::ColorBlendDesc{
                        mask: hal::pso::ColorMask::ALL,
                        blend: blend_state,
                    });

                pipeline_descriptor.depth_stencil = hal::pso::DepthStencilDesc {
                    depth: depth_test,
                    depth_bounds: false,
                    stencil: None,
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
                .collect::<FastHashMap<(Option<hal::pso::BlendState>, Option<hal::pso::DepthTest>), B::GraphicsPipeline>>();

            if features.contains(&"DEBUG_OVERDRAW") {
                let pipeline_state = (Some(OVERDRAW), Some(LESS_EQUAL_TEST));
                let pipeline_descriptor = create_desc(pipeline_state);
                let pipeline = unsafe {
                    device.create_graphics_pipeline(&pipeline_descriptor, pipeline_cache)
                }
                .expect("Pipeline creation failed");
                states.insert(pipeline_state, pipeline);
            }

            states
        };

        let (mut vertex_buffer, mut index_buffer) = if shader_kind.is_debug() {
            (Some(SmallVec::new()), Some(SmallVec::new()))
        } else {
            (None, None)
        };
        for _ in 0 .. frame_count {
            if let Some(ref mut vertex_buffer) = vertex_buffer {
                vertex_buffer.push(VertexBufferHandler::new(
                    device,
                    heaps,
                    hal::buffer::Usage::VERTEX,
                    &[0u8],
                    (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                    (limits.non_coherent_atom_size - 1) as usize,
                ));
            }
            if let Some(ref mut index_buffer) = index_buffer {
                index_buffer.push(VertexBufferHandler::new(
                    device,
                    heaps,
                    hal::buffer::Usage::INDEX,
                    &[0u8],
                    (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                    (limits.non_coherent_atom_size - 1) as usize,
                ));
            }
        }

        Program {
            pipelines,
            vertex_buffer,
            index_buffer,
            shader_name: String::from(shader_name),
            shader_kind,
            constants: [0; PUSH_CONSTANT_BLOCK_SIZE],
        }
    }

    pub(super) fn submit(
        &mut self,
        cmd_buffer: &mut B::CommandBuffer,
        viewport: hal::pso::Viewport,
        render_pass: &B::RenderPass,
        frame_buffer: &B::Framebuffer,
        desc_set_per_draw: &B::DescriptorSet,
        desc_set_per_pass: Option<&B::DescriptorSet>,
        desc_set_per_frame: &B::DescriptorSet,
        desc_set_locals: Option<&B::DescriptorSet>,
        clear_values: &[hal::command::ClearValueRaw],
        blend_state: Option<hal::pso::BlendState>,
        blend_color: ColorF,
        depth_test: Option<hal::pso::DepthTest>,
        scissor_rect: Option<DeviceIntRect>,
        next_id: usize,
        pipeline_layout: &B::PipelineLayout,
        use_push_consts: bool,
        vertex_buffer: &VertexBufferHandler<B>,
        instance_buffer: &InstanceBufferHandler<B>,
        instance_range: std::ops::Range<usize>,
    ) {
        let vertex_buffer = match &self.vertex_buffer {
            Some(ref vb) => vb.get(next_id).unwrap(),
            None => vertex_buffer
        };
        unsafe {
            if use_push_consts {
                cmd_buffer.push_graphics_constants(
                    pipeline_layout,
                    hal::pso::ShaderStageFlags::VERTEX,
                    0,
                    &self.constants,
                );
            }
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

            if !use_push_consts {
                assert!(desc_set_locals.is_some());
            }
            use std::iter;
            cmd_buffer.bind_graphics_descriptor_sets(
                pipeline_layout,
                if desc_set_per_pass.is_some() { 0 } else { 1 },
                desc_set_per_pass.into_iter()
                    .chain(iter::once(desc_set_per_frame))
                    .chain(iter::once(desc_set_per_draw))
                    .chain(desc_set_locals),
                &[],
            );

            if blend_state == Some(SUBPIXEL_CONSTANT_TEXT_COLOR) {
                cmd_buffer.set_blend_constants(blend_color.to_array());
            }

            if let Some(ref index_buffer) = self.index_buffer {
                cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer.buffer().buffer, 0)));
                cmd_buffer.bind_index_buffer(hal::buffer::IndexBufferView {
                    buffer: &index_buffer[next_id].buffer().buffer,
                    offset: 0,
                    index_type: hal::IndexType::U32,
                });

                cmd_buffer.begin_render_pass(
                    render_pass,
                    frame_buffer,
                    viewport.rect,
                    clear_values,
                    SubpassContents::Inline,
                );

                cmd_buffer.draw_indexed(
                    0 .. index_buffer[next_id].buffer().buffer_len as u32,
                    0,
                    0 .. 1,
                );

                cmd_buffer.end_render_pass();
            } else {
                for i in instance_range.into_iter() {
                    cmd_buffer.bind_vertex_buffers(
                        0,
                        Some((&vertex_buffer.buffer().buffer, 0))
                            .into_iter()
                            .chain(Some((&instance_buffer.buffers[i].buffer.buffer, 0))),
                    );

                    cmd_buffer.begin_render_pass(
                        render_pass,
                        frame_buffer,
                        viewport.rect,
                        clear_values,
                        SubpassContents::Inline,
                    );

                    let data_stride = instance_buffer.buffers[i].last_data_stride;
                    let end = instance_buffer.buffers[i].offset / data_stride;
                    let start = end - instance_buffer.buffers[i].last_update_size / data_stride;
                    cmd_buffer.draw(
                        0 .. vertex_buffer.buffer_len as _,
                        start as u32 .. end as u32,
                    );

                    cmd_buffer.end_render_pass();
                }
            }
        }
    }

    pub(super) fn deinit(mut self, device: &B::Device, heaps: &mut Heaps<B>) {
        if let Some(vertex_buffer) = self.vertex_buffer {
            for vertex_buffer in vertex_buffer {
                vertex_buffer.deinit(device, heaps);
            }
        }
        if let Some(index_buffer) = self.index_buffer {
            for index_buffer in index_buffer {
                index_buffer.deinit(device, heaps);
            }
        }
        for pipeline in self.pipelines.drain() {
            unsafe { device.destroy_graphics_pipeline(pipeline.1) };
        }
    }
}
