/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ImageFormat;
use arrayvec::ArrayVec;
use hal::{self, device::Device as BackendDevice};
use hal::command::CommandBuffer;
use crate::internal_types::FastHashMap;
use smallvec::SmallVec;
use rendy_memory::Heaps;
use std::borrow::Cow::{Borrowed};

use super::buffer::{InstanceBufferHandler, VertexBufferHandler};
use super::render_pass::HalRenderPasses;
use super::PipelineRequirements;
use super::super::{ShaderKind, VertexArrayKind};
use super::super::super::shader_source;

const ENTRY_NAME: &str = "main";
// The number of specialization constants in each shader.
const SPECIALIZATION_CONSTANT_COUNT: usize = 8;
// Size of a specialization constant variable in bytes.
const SPECIALIZATION_CONSTANT_SIZE: usize = 4;
const SPECIALIZATION_FEATURES: &'static [&'static str] = &[
    "ALPHA_PASS",
    "COLOR_TARGET",
    "GLYPH_TRANSFORM",
    "DITHERING",
    "DEBUG_OVERDRAW",
    "REPETITION",
    "ANTIALIASING",
];

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub(super) enum RenderPassDepthState {
    Enabled,
    Disabled,
}

type PipelineKey = (
    ImageFormat,
    Option<hal::pso::BlendState>,
    RenderPassDepthState,
    Option<hal::pso::DepthTest>,
);

pub(crate) struct Program<B: hal::Backend> {
    pub(super) pipelines: FastHashMap<PipelineKey, B::GraphicsPipeline>,
    pub(super) vertex_buffer: Option<SmallVec<[VertexBufferHandler<B>; 1]>>,
    pub(super) index_buffer: Option<SmallVec<[VertexBufferHandler<B>; 1]>>,
    pub(super) shader_name: String,
    pub(super) shader_kind: ShaderKind,
    last_frame_used: usize,
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
        render_passes: &HalRenderPasses<B>,
        frame_count: usize,
        shader_modules: &mut FastHashMap<String, (B::ShaderModule, B::ShaderModule)>,
        pipeline_cache: Option<&B::PipelineCache>,
        surface_format: ImageFormat,
    ) -> Program<B> {
        use hal::pso::{BlendState, EntryPoint, GraphicsShaderSet, Specialization, SpecializationConstant};
        use super::blend_state::*;
        use super::{LESS_EQUAL_TEST, LESS_EQUAL_WRITE};
        use self::RenderPassDepthState as RPDS;

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

        let mut specialization_data = vec![];
        let program_modes = if let ShaderKind::Text = shader_kind {
            // 0 default uMode
            // 3 uMode for ShaderColorMode::SubpixelWithBgColorPass0
            // 4 uMode for ShaderColorMode::SubpixelWithBgColorPass1
            // 5 uMode for ShaderColorMode::SubpixelWithBgColorPass2
            [0, 3, 4, 5].into_iter()
        } else {
            [0].into_iter()
        };
        let constant_setups: ArrayVec<[ArrayVec<[_; SPECIALIZATION_CONSTANT_COUNT]>; 4]> =
            program_modes
            .enumerate()
            .map(|(i, mode)| {
                specialization_data.push(vec![0; (SPECIALIZATION_CONSTANT_COUNT - 1) * SPECIALIZATION_CONSTANT_SIZE]);
                let mut constants: ArrayVec<[_; SPECIALIZATION_CONSTANT_COUNT]> = SPECIALIZATION_FEATURES
                    .iter()
                    .zip(specialization_data[i].chunks_mut(SPECIALIZATION_CONSTANT_SIZE))
                    .enumerate()
                    .map(|(j, (feature, out_data))| {
                        out_data[0] = features.contains(feature) as u8;
                        SpecializationConstant {
                            id: j as _,
                            range: (SPECIALIZATION_CONSTANT_SIZE * j) as _
                                ..(SPECIALIZATION_CONSTANT_SIZE * (j + 1)) as _,
                        }
                    })
                    .collect();
                if *mode != 0 {
                    constants.push(SpecializationConstant {
                        id: (SPECIALIZATION_CONSTANT_COUNT - 1) as _,
                        range: {
                            let from = (SPECIALIZATION_CONSTANT_COUNT - 1) * SPECIALIZATION_CONSTANT_SIZE;
                            let to = from + SPECIALIZATION_CONSTANT_SIZE;
                            from as _..to as _
                        },
                    });
                    specialization_data[i].extend_from_slice(&[*mode as u8, 0, 0, 0]);
                }
                constants
            }).collect();

        let pipelines = {
            let shader_entries: ArrayVec<[GraphicsShaderSet<B>; 4]> =
                constant_setups
                .iter()
                .zip(specialization_data.iter())
                .map(|(constants, spec_data)| {
                    let (vs_entry, fs_entry) = (
                        EntryPoint::<B> {
                            entry: ENTRY_NAME,
                            module: &vs_module,
                            specialization: Specialization {
                                constants: Borrowed(&constants),
                                data: Borrowed(&spec_data.as_slice()),
                            },
                        },
                        EntryPoint::<B> {
                            entry: ENTRY_NAME,
                            module: &fs_module,
                            specialization: Specialization {
                                constants: Borrowed(&constants),
                                data: Borrowed(&spec_data.as_slice()),
                            },
                        },
                    );

                    GraphicsShaderSet {
                        vertex: vs_entry,
                        hull: None,
                        domain: None,
                        geometry: None,
                        fragment: Some(fs_entry),
                    }
                })
                .collect();

            let pipeline_states = match shader_kind {
                ShaderKind::Cache(VertexArrayKind::Gradient) => {
                    vec![(surface_format, None, RPDS::Disabled, None)]
                }
                ShaderKind::Cache(VertexArrayKind::SvgFilter) => vec![
                    (surface_format, None, RPDS::Disabled, None),
                    (surface_format, None, RPDS::Enabled, Some(LESS_EQUAL_TEST)),
                ],
                ShaderKind::Cache(VertexArrayKind::Scale) => vec![
                    (ImageFormat::R8, None, RPDS::Enabled, None),
                    (ImageFormat::R8, None, RPDS::Disabled, None),
                    (
                        ImageFormat::R8,
                        Some(BlendState::MULTIPLY),
                        RPDS::Enabled,
                        None,
                    ),
                    (
                        ImageFormat::R8,
                        Some(BlendState::MULTIPLY),
                        RPDS::Disabled,
                        None,
                    ),
                    (surface_format, None, RPDS::Enabled, None),
                    (surface_format, None, RPDS::Disabled, None),
                    (
                        surface_format,
                        Some(BlendState::MULTIPLY),
                        RPDS::Enabled,
                        None,
                    ),
                    (
                        surface_format,
                        Some(BlendState::MULTIPLY),
                        RPDS::Disabled,
                        None,
                    ),
                ],
                ShaderKind::Cache(VertexArrayKind::Blur) => {
                    if features.contains(&"ALPHA_TARGET") {
                        vec![
                            (ImageFormat::R8, None, RPDS::Enabled, None),
                            (ImageFormat::R8, None, RPDS::Disabled, None),
                            (ImageFormat::R8, None, RPDS::Enabled, Some(LESS_EQUAL_TEST)),
                        ]
                    } else {
                        vec![
                            (surface_format, None, RPDS::Enabled, None),
                            (surface_format, None, RPDS::Disabled, None),
                            (surface_format, None, RPDS::Enabled, Some(LESS_EQUAL_TEST)),
                        ]
                    }
                }
                ShaderKind::Cache(VertexArrayKind::Border)
                | ShaderKind::Cache(VertexArrayKind::LineDecoration) => vec![(
                    surface_format,
                    Some(BlendState::PREMULTIPLIED_ALPHA),
                    RPDS::Disabled,
                    None,
                )],
                ShaderKind::ClipCache => vec![
                    (ImageFormat::R8, None, RPDS::Disabled, None),
                    (
                        ImageFormat::R8,
                        Some(BlendState::MULTIPLY),
                        RPDS::Enabled,
                        None,
                    ),
                    (
                        ImageFormat::R8,
                        Some(BlendState::MULTIPLY),
                        RPDS::Disabled,
                        None,
                    ),
                ],
                ShaderKind::Text => {
                    if features.contains(&"DUAL_SOURCE_BLENDING") {
                        vec![
                            (
                                surface_format,
                                Some(BlendState::PREMULTIPLIED_ALPHA),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(BlendState::PREMULTIPLIED_ALPHA),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(BlendState::PREMULTIPLIED_ALPHA),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_CONSTANT_TEXT_COLOR),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_CONSTANT_TEXT_COLOR),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_CONSTANT_TEXT_COLOR),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS0),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS0),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS0),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS1),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS1),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS1),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS2),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS2),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS2),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_DUAL_SOURCE),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_DUAL_SOURCE),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_DUAL_SOURCE),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                        ]
                    } else {
                        vec![
                            (
                                surface_format,
                                Some(BlendState::PREMULTIPLIED_ALPHA),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(BlendState::PREMULTIPLIED_ALPHA),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(BlendState::PREMULTIPLIED_ALPHA),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_CONSTANT_TEXT_COLOR),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_CONSTANT_TEXT_COLOR),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_CONSTANT_TEXT_COLOR),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS0),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS0),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS0),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS1),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS1),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS1),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS2),
                                RPDS::Enabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS2),
                                RPDS::Disabled,
                                None,
                            ),
                            (
                                surface_format,
                                Some(SUBPIXEL_WITH_BG_COLOR_PASS2),
                                RPDS::Enabled,
                                Some(LESS_EQUAL_TEST),
                            ),
                        ]
                    }
                }
                ShaderKind::DebugColor | ShaderKind::DebugFont => vec![(
                    surface_format,
                    Some(BlendState::PREMULTIPLIED_ALPHA),
                    RPDS::Enabled,
                    None,
                )],
                ShaderKind::Service => vec![
                    (surface_format, None, RPDS::Enabled, None),
                    (surface_format, None, RPDS::Disabled, None),
                ],
                _ => vec![
                    (surface_format, None, RPDS::Enabled, Some(LESS_EQUAL_WRITE)),
                    (surface_format, Some(ALPHA), RPDS::Enabled, None),
                    (surface_format, Some(ALPHA), RPDS::Disabled, None),
                    (
                        surface_format,
                        Some(ALPHA),
                        RPDS::Enabled,
                        Some(LESS_EQUAL_TEST),
                    ),
                    (
                        surface_format,
                        Some(BlendState::PREMULTIPLIED_ALPHA),
                        RPDS::Enabled,
                        None,
                    ),
                    (
                        surface_format,
                        Some(BlendState::PREMULTIPLIED_ALPHA),
                        RPDS::Disabled,
                        None,
                    ),
                    (
                        surface_format,
                        Some(BlendState::PREMULTIPLIED_ALPHA),
                        RPDS::Enabled,
                        Some(LESS_EQUAL_TEST),
                    ),
                    (
                        surface_format,
                        Some(PREMULTIPLIED_DEST_OUT),
                        RPDS::Enabled,
                        None,
                    ),
                    (
                        surface_format,
                        Some(PREMULTIPLIED_DEST_OUT),
                        RPDS::Disabled,
                        None,
                    ),
                    (
                        surface_format,
                        Some(PREMULTIPLIED_DEST_OUT),
                        RPDS::Enabled,
                        Some(LESS_EQUAL_TEST),
                    ),
                ],
            };

            let create_desc = |(format, blend_state, render_pass_depth_state, depth_test)| {
                let depth_enabled = match depth_test {
                    Some(_) => true,
                    None => match render_pass_depth_state {
                        RenderPassDepthState::Enabled => true,
                        RenderPassDepthState::Disabled => false,
                    },
                };
                let subpass = hal::pass::Subpass {
                    index: 0,
                    main_pass: render_passes.render_pass(format, depth_enabled, false),
                };
                let mut pipeline_descriptor = hal::pso::GraphicsPipelineDesc::new(
                    match blend_state {
                        Some(SUBPIXEL_WITH_BG_COLOR_PASS0) => shader_entries[1].clone(),
                        Some(SUBPIXEL_WITH_BG_COLOR_PASS1) => shader_entries[2].clone(),
                        Some(SUBPIXEL_WITH_BG_COLOR_PASS2) => shader_entries[3].clone(),
                        _ => shader_entries[0].clone(),
                    },
                    hal::pso::Primitive::TriangleList,
                    hal::pso::Rasterizer::FILL,
                    &pipeline_layout,
                    subpass,
                );
                pipeline_descriptor
                    .blender
                    .targets
                    .push(hal::pso::ColorBlendDesc {
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

            let cloned_states = pipeline_states.clone();
            let pipelines_descriptors = cloned_states.into_iter().map(|ps| create_desc(ps));

            let pipelines =
                unsafe { device.create_graphics_pipelines(pipelines_descriptors, pipeline_cache) }
                    .into_iter();

            let mut states = pipeline_states
                .into_iter()
                .zip(pipelines.map(|pipeline| pipeline.expect("Pipeline creation failed")))
                .collect::<FastHashMap<PipelineKey, B::GraphicsPipeline>>();

            if features.contains(&"DEBUG_OVERDRAW") {
                let pipeline_state = (
                    surface_format,
                    Some(OVERDRAW),
                    RPDS::Enabled,
                    Some(LESS_EQUAL_TEST),
                );
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
        for _ in 0..frame_count {
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
            last_frame_used: 0,
        }
    }

    pub(super) fn submit(
        &mut self,
        cmd_buffer: &mut B::CommandBuffer,
        next_id: usize,
        vertex_buffer: &VertexBufferHandler<B>,
        instance_buffer: &InstanceBufferHandler<B>,
        instance_buffer_range: std::ops::Range<usize>,
    ) {
        if self.shader_kind.is_debug() {
            if self.last_frame_used != next_id {
                self.vertex_buffer.as_mut().unwrap()[next_id].reset();
                self.index_buffer.as_mut().unwrap()[next_id].reset();
                self.last_frame_used = next_id;
            }
        }
        let vertex_buffer = match &self.vertex_buffer {
            Some(ref vb) => vb.get(next_id).unwrap(),
            None => vertex_buffer,
        };
        unsafe {
            match &self.index_buffer {
                // Debug shaders
                Some(ref index_buffer) => {
                    cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer.buffer().buffer, 0)));
                    cmd_buffer.bind_index_buffer(hal::buffer::IndexBufferView {
                        buffer: &index_buffer[next_id].buffer().buffer,
                        offset: 0,
                        index_type: hal::IndexType::U32,
                    });

                    cmd_buffer.draw_indexed(
                        0..index_buffer[next_id].buffer().buffer_len as u32,
                        0,
                        0..1,
                    );
                }
                // Default WR shaders
                None => {
                    let number_of_vertices = match self.shader_kind {
                        ShaderKind::Service => 3,
                        _ => vertex_buffer.buffer_len,
                    };
                    for i in instance_buffer_range.into_iter() {
                        cmd_buffer.bind_vertex_buffers(
                            0,
                            Some((&vertex_buffer.buffer().buffer, 0))
                                .into_iter()
                                .chain(Some((&instance_buffer.buffers[i].buffer.buffer, 0))),
                        );

                        let data_stride = instance_buffer.buffers[i].last_data_stride;
                        let end = instance_buffer.buffers[i].offset / data_stride;
                        let start = end - instance_buffer.buffers[i].last_update_size / data_stride;
                        cmd_buffer.draw(0..number_of_vertices as u32, start as u32..end as u32);
                    }
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
