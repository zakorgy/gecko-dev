/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

mod blend_state;
mod buffer;
mod command;
mod descriptor;
mod device;
mod image;
mod program;
mod render_pass;
pub(crate) mod vertex_types;

pub use self::device::*;
pub use self::buffer::{BufferMemorySlice, GpuCacheBuffer, PersistentlyMappedBuffer};

use hal;
use crate::internal_types::FastHashMap;

pub type TextureId = u32;

pub const LESS_EQUAL_TEST: hal::pso::DepthTest = hal::pso::DepthTest {
    fun: hal::pso::Comparison::LessEqual,
    write: false,
};

pub const LESS_EQUAL_WRITE: hal::pso::DepthTest = hal::pso::DepthTest {
    fun: hal::pso::Comparison::LessEqual,
    write: true,
};

#[derive(Clone, Deserialize)]
pub struct PipelineRequirements {
    pub attribute_descriptors: Vec<hal::pso::AttributeDesc>,
    pub bindings_map: FastHashMap<String, u32>,
    pub vertex_buffer_descriptors: Vec<hal::pso::VertexBufferDesc>,
}

pub trait PrimitiveType {
    type Primitive: Clone + Copy;
    fn to_primitive_type(&self) -> Self::Primitive;
}

impl PrimitiveType for crate::gpu_types::BlurInstance {
    type Primitive = vertex_types::BlurInstance;
    fn to_primitive_type(&self) -> vertex_types::BlurInstance {
        vertex_types::BlurInstance {
            aData: [0, 0, 0, 0],
            aBlurRenderTaskAddress: self.task_address.0 as i32,
            aBlurSourceTaskAddress: self.src_task_address.0 as i32,
            aBlurDirection: self.blur_direction as i32,
        }
    }
}

impl PrimitiveType for crate::gpu_types::BorderInstance {
    type Primitive = vertex_types::BorderInstance;
    fn to_primitive_type(&self) -> vertex_types::BorderInstance {
        vertex_types::BorderInstance {
            aTaskOrigin: [self.task_origin.x, self.task_origin.y],
            aRect: [
                self.local_rect.origin.x,
                self.local_rect.origin.y,
                self.local_rect.size.width,
                self.local_rect.size.height,
            ],
            aColor0: self.color0.to_array(),
            aColor1: self.color1.to_array(),
            aFlags: self.flags,
            aWidths: [self.widths.width, self.widths.height],
            aRadii: [self.radius.width, self.radius.height],
            aClipParams1: [
                self.clip_params[0],
                self.clip_params[1],
                self.clip_params[2],
                self.clip_params[3],
            ],
            aClipParams2: [
                self.clip_params[4],
                self.clip_params[5],
                self.clip_params[6],
                self.clip_params[7],
            ],
        }
    }
}

impl PrimitiveType for crate::gpu_types::ClipMaskInstance {
    type Primitive = vertex_types::ClipMaskInstance;
    fn to_primitive_type(&self) -> vertex_types::ClipMaskInstance {
        vertex_types::ClipMaskInstance {
            aTransformIds: [
                self.clip_transform_id.0 as i32,
                self.prim_transform_id.0 as i32,
            ],
            aClipDataResourceAddress: [
                self.clip_data_address.u as i32,
                self.clip_data_address.v as i32,
                self.resource_address.u as i32,
                self.resource_address.v as i32,
            ],
            aClipLocalPos: [
                self.local_pos.x,
                self.local_pos.y,
            ],
            aClipTileRect: [
                self.tile_rect.origin.x,
                self.tile_rect.origin.y,
                self.tile_rect.size.width,
                self.tile_rect.size.height,
            ],
            aClipDeviceArea: [
                self.sub_rect.origin.x,
                self.sub_rect.origin.y,
                self.sub_rect.size.width,
                self.sub_rect.size.height,
            ],
            aClipOrigins: [
                self.task_origin.x,
                self.task_origin.y,
                self.screen_origin.x,
                self.screen_origin.y,
            ],
            aDevicePixelScale: self.device_pixel_scale,
        }
    }
}

impl PrimitiveType for crate::gpu_types::PrimitiveInstanceData {
    type Primitive = vertex_types::PrimitiveInstanceData;
    fn to_primitive_type(&self) -> vertex_types::PrimitiveInstanceData {
        vertex_types::PrimitiveInstanceData { aData: self.data }
    }
}

impl PrimitiveType for crate::gpu_types::ScalingInstance {
    type Primitive = vertex_types::ScalingInstance;
    fn to_primitive_type(&self) -> vertex_types::ScalingInstance {
        vertex_types::ScalingInstance {
            aData: [0, 0, 0, 0],
            aScaleTargetRect: [
                self.target_rect.origin.x,
                self.target_rect.origin.y,
                self.target_rect.size.width,
                self.target_rect.size.height,
            ],
            aScaleSourceRect: [
                self.source_rect.origin.x,
                self.source_rect.origin.y,
                self.source_rect.size.width,
                self.source_rect.size.height,
            ],
            aScaleSourceLayer: self.source_layer,
        }
    }
}

impl PrimitiveType for crate::render_target::LineDecorationJob {
    type Primitive = vertex_types::LineDecorationInstance;
    fn to_primitive_type(&self) -> vertex_types::LineDecorationInstance {
        vertex_types::LineDecorationInstance {
            aTaskRect: [
                self.task_rect.origin.x,
                self.task_rect.origin.y,
                self.task_rect.size.width,
                self.task_rect.size.height,
            ],
            aLocalSize: [self.local_size.width, self.local_size.height],
            aStyle: self.style,
            aOrientation: self.orientation,
            aWavyLineThickness: self.wavy_line_thickness,
        }
    }
}

impl PrimitiveType for crate::render_target::GradientJob {
    type Primitive = vertex_types::GradientInstance;
    fn to_primitive_type(&self) -> vertex_types::GradientInstance {
        vertex_types::GradientInstance {
            aTaskRect: [
                self.task_rect.origin.x,
                self.task_rect.origin.y,
                self.task_rect.size.width,
                self.task_rect.size.height,
            ],
            aAxisSelect: self.axis_select,
            aStops: self.stops,
            aColor0: self.colors[0].to_array(),
            aColor1: self.colors[1].to_array(),
            aColor2: self.colors[2].to_array(),
            aColor3: self.colors[3].to_array(),
            aStartStop: self.start_stop,
        }
    }
}

impl PrimitiveType for crate::gpu_types::CompositeInstance {
    type Primitive = vertex_types::CompositeInstance;
    fn to_primitive_type(&self) -> vertex_types::CompositeInstance {
        vertex_types::CompositeInstance {
            aDeviceRect: [
                self.rect.origin.x,
                self.rect.origin.y,
                self.rect.size.width,
                self.rect.size.height,
            ],
            aDeviceClipRect: [
                self.clip_rect.origin.x,
                self.clip_rect.origin.y,
                self.clip_rect.size.width,
                self.clip_rect.size.height,
            ],
            aColor: self.color.to_array(),
            aLayer: self.layer,
            aZId: self.z_id,
        }
    }
}

impl PrimitiveType for crate::gpu_types::SvgFilterInstance {
    type Primitive = vertex_types::SvgFilterInstance;
    fn to_primitive_type(&self) -> vertex_types::SvgFilterInstance {
        vertex_types::SvgFilterInstance {
            aData: [0, 0, 0, 0],
            aFilterRenderTaskAddress: self.task_address.0 as i32,
            aFilterInput1TaskAddress: self.input_1_task_address.0 as i32,
            aFilterInput2TaskAddress: self.input_2_task_address.0 as i32,
            aFilterKind: self.kind as i32,
            aFilterInputCount: self.input_count as i32,
            aFilterGenericInt: self.generic_int as i32,
            aFilterExtraDataAddress: [
                self.extra_data_address.u as i32,
                self.extra_data_address.v as i32,
            ],
        }
    }
}
