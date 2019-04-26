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

use gpu_types;
use hal;
use internal_types::FastHashMap;
use tiling;

pub type TextureId = u32;

pub const LESS_EQUAL_TEST: hal::pso::DepthTest = hal::pso::DepthTest::On {
    fun: hal::pso::Comparison::LessEqual,
    write: false,
};

pub const LESS_EQUAL_WRITE: hal::pso::DepthTest = hal::pso::DepthTest::On {
    fun: hal::pso::Comparison::LessEqual,
    write: true,
};

#[derive(Clone, Deserialize)]
pub struct PipelineRequirements {
    pub attribute_descriptors: Vec<hal::pso::AttributeDesc>,
    pub bindings_map: FastHashMap<String, u32>,
    pub descriptor_range_descriptors: Vec<Vec<hal::pso::DescriptorRangeDesc>>,
    pub descriptor_set_layout_bindings: Vec<Vec<hal::pso::DescriptorSetLayoutBinding>>,
    pub vertex_buffer_descriptors: Vec<hal::pso::VertexBufferDesc>,
}

pub trait PrimitiveType {
    type Primitive: Clone + Copy;
    fn to_primitive_type(&self) -> Self::Primitive;
}

impl PrimitiveType for gpu_types::BlurInstance {
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

impl PrimitiveType for gpu_types::BorderInstance {
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

impl PrimitiveType for gpu_types::ClipMaskInstance {
    type Primitive = vertex_types::ClipMaskInstance;
    fn to_primitive_type(&self) -> vertex_types::ClipMaskInstance {
        vertex_types::ClipMaskInstance {
            aClipRenderTaskAddress: self.render_task_address.0 as i32,
            aClipTransformId: self.clip_transform_id.0 as i32,
            aPrimTransformId: self.prim_transform_id.0 as i32,
            aClipDataResourceAddress: [
                self.clip_data_address.u as i32,
                self.clip_data_address.v as i32,
                self.resource_address.u as i32,
                self.resource_address.v as i32,
            ],
            aClipLocalPos: [self.local_pos.x, self.local_pos.y],
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
        }
    }
}

impl PrimitiveType for gpu_types::PrimitiveInstanceData {
    type Primitive = vertex_types::PrimitiveInstanceData;
    fn to_primitive_type(&self) -> vertex_types::PrimitiveInstanceData {
        vertex_types::PrimitiveInstanceData { aData: self.data }
    }
}

impl PrimitiveType for gpu_types::ScalingInstance {
    type Primitive = vertex_types::ScalingInstance;
    fn to_primitive_type(&self) -> vertex_types::ScalingInstance {
        vertex_types::ScalingInstance {
            aData: [0, 0, 0, 0],
            aScaleRenderTaskAddress: self.task_address.0 as i32,
            aScaleSourceTaskAddress: self.src_task_address.0 as i32,
        }
    }
}

impl PrimitiveType for tiling::LineDecorationJob {
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
