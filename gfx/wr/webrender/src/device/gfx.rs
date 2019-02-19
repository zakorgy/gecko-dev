/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, ImageFormat, MemoryReport};
use api::{DeviceIntPoint, DeviceIntRect, DeviceIntSize};
use api::TextureTarget;
#[cfg(any(feature = "debug_renderer", feature = "capture"))]
use api::ImageDescriptor;
use euclid::Transform3D;
use gpu_types;
use internal_types::{FastHashMap, RenderTargetInfo};
use rand::{self, Rng};
use ron::de::from_str;
use smallvec::SmallVec;
use std::cell::Cell;
use std::convert::Into;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::mem;
use std::path::PathBuf;
use std::rc::Rc;
use std::slice;
use super::Capabilities;
use super::{ShaderKind,VertexArrayKind, ExternalTexture, GpuFrameId, TextureSlot, TextureFilter};
use super::{VertexDescriptor, UploadMethod, Texel, ReadPixelsFormat, TextureFlags};
use super::{Texture, DrawTarget, ReadTarget, FBOId, RBOId, VertexUsageHint, ShaderError, ShaderPrecacheFlags, SharedDepthTarget, ProgramCache};
use super::{depth_target_size_in_bytes, record_gpu_alloc, record_gpu_free};
use super::super::shader_source;
use tiling::LineDecorationJob;
use vertex_types::*;

use hal;

// gfx-hal
use hal::pso::{AttributeDesc, DescriptorRangeDesc, DescriptorSetLayoutBinding, VertexBufferDesc};
use hal::pso::{BlendState, BlendOp, Comparison, DepthTest, Factor};
use hal::{Device as BackendDevice, PhysicalDevice, Surface, Swapchain};
use hal::{Backbuffer, DescriptorPool, FrameSync, Primitive, SwapchainConfig};
use hal::pass::Subpass;
use hal::pso::PipelineStage;
use hal::queue::Submission;

pub type TextureId = u32;

pub const MAX_INDEX_COUNT: usize = 4096;
pub const MAX_INSTANCE_COUNT: usize = 8192;
pub const TEXTURE_CACHE_SIZE: usize = 128 << 20; // 128MB
pub const INVALID_TEXTURE_ID: TextureId = 0;
pub const INVALID_PROGRAM_ID: ProgramId = ProgramId(0);
pub const DEFAULT_READ_FBO: FBOId = FBOId(0);
pub const DEFAULT_DRAW_FBO: FBOId = FBOId(1);
pub const DEBUG_READ_FBO: FBOId = FBOId(2);

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::COLOR,
    levels: 0 .. 1,
    layers: 0 .. 1,
};
const DEPTH_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::DEPTH,
    levels: 0 .. 1,
    layers: 0 .. 1,
};

const ENTRY_NAME: &str = "main";

pub struct DeviceInit<B: hal::Backend> {
    pub adapter: hal::Adapter<B>,
    pub surface: B::Surface,
    pub window_size: (i32, i32),
    pub frame_count: Option<usize>,
    pub descriptor_count: Option<usize>,
    pub cache_path: Option<PathBuf>,
    pub save_cache: bool,
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct Locals {
    uTransform: [[f32; 4]; 4],
    uMode: i32,
}

const MAX_FRAME_COUNT: usize = 2;
const DESCRIPTOR_COUNT: usize = 400;
const DEBUG_DESCRIPTOR_COUNT: usize = 5;
const DESCRIPTOR_SET_PER_DRAW: usize = 0;
const DESCRIPTOR_SET_PER_INSTANCE: usize = 1;
const DESCRIPTOR_SET_SAMPLER: usize = 2;
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

const NON_SPECIALIZATION_FEATURES: &'static [&'static str] = &[
    "TEXTURE_RECT", "TEXTURE_2D", "DUAL_SOURCE_BLENDING"
];

const SAMPLERS: [(usize, &'static str); 11] = [
    (0, "Color0"),
    (1, "Color1"),
    (2, "Color2"),
    (3, "PrevPassAlpha"),
    (4, "PrevPassColor"),
    (5, "GpuCache"),
    (6, "TransformPalette"),
    (7, "RenderTasks"),
    (8, "Dither"),
    (9, "PrimitiveHeadersF"),
    (10, "PrimitiveHeadersI"),
];

#[derive(Clone, Deserialize)]
pub struct PipelineRequirements {
    pub attribute_descriptors: Vec<AttributeDesc>,
    pub bindings_map: HashMap<String, u32>,
    pub descriptor_range_descriptors: Vec<Vec<DescriptorRangeDesc>>,
    pub descriptor_set_layout_bindings: Vec<Vec<DescriptorSetLayoutBinding>>,
    pub vertex_buffer_descriptors: Vec<VertexBufferDesc>,
}

const QUAD: [Vertex; 6] = [
    Vertex {
        aPosition: [0.0, 0.0, 0.0],
    },
    Vertex {
        aPosition: [1.0, 0.0, 0.0],
    },
    Vertex {
        aPosition: [0.0, 1.0, 0.0],
    },
    Vertex {
        aPosition: [0.0, 1.0, 0.0],
    },
    Vertex {
        aPosition: [1.0, 0.0, 0.0],
    },
    Vertex {
        aPosition: [1.0, 1.0, 0.0],
    },
];

#[repr(u32)]
pub enum DepthFunction {
    Less,
    LessEqual,
}

pub trait PrimitiveType {
    type Primitive: Clone + Copy;
    fn to_primitive_type(&self) -> Self::Primitive;
}

impl Texture {
    pub fn still_in_flight(&self, frame_id: GpuFrameId, frame_count: usize) -> bool {
        for i in 0 .. frame_count {
            if self.bound_in_frame.get() == GpuFrameId(frame_id.0 - i) {
                return true;
            }
        }
        false
    }
}

impl PrimitiveType for gpu_types::BlurInstance {
    type Primitive = BlurInstance;
    fn to_primitive_type(&self) -> BlurInstance {
        BlurInstance {
            aData: [0, 0, 0, 0],
            aBlurRenderTaskAddress: self.task_address.0 as i32,
            aBlurSourceTaskAddress: self.src_task_address.0 as i32,
            aBlurDirection: self.blur_direction as i32,
        }
    }
}

impl PrimitiveType for gpu_types::BorderInstance {
    type Primitive = BorderInstance;
    fn to_primitive_type(&self) -> BorderInstance {
        BorderInstance {
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
    type Primitive = ClipMaskInstance;
    fn to_primitive_type(&self) -> ClipMaskInstance {
        ClipMaskInstance {
            aClipRenderTaskAddress: self.render_task_address.0 as i32,
            aClipTransformId: self.clip_transform_id.0 as i32,
            aPrimTransformId: self.prim_transform_id.0 as i32,
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
        }
    }
}

impl PrimitiveType for gpu_types::PrimitiveInstanceData {
    type Primitive = PrimitiveInstanceData;
    fn to_primitive_type(&self) -> PrimitiveInstanceData {
        PrimitiveInstanceData { aData: self.data }
    }
}

impl PrimitiveType for gpu_types::ScalingInstance {
    type Primitive = ScalingInstance;
    fn to_primitive_type(&self) -> ScalingInstance {
        ScalingInstance {
            aData: [0, 0, 0, 0],
            aScaleRenderTaskAddress: self.task_address.0 as i32,
            aScaleSourceTaskAddress: self.src_task_address.0 as i32,
        }
    }
}

impl PrimitiveType for LineDecorationJob {
    type Primitive = LineDecorationInstance;
    fn to_primitive_type(&self) -> LineDecorationInstance {
        LineDecorationInstance {
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

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct ProgramId(u32);

pub struct PBO;
pub struct VAO;

impl ShaderKind {
    fn is_debug(&self) -> bool {
        match *self {
            ShaderKind::DebugFont | ShaderKind::DebugColor => true,
            _ => false,
        }
    }
}

const ALPHA: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::SrcAlpha,
        dst: Factor::OneMinusSrcAlpha,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
};

const PREMULTIPLIED_DEST_OUT: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcAlpha,
    },
    alpha: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcAlpha,
    },
};

const MAX: BlendState = BlendState::On {
    color: BlendOp::Max,
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
};

const MIN: BlendState = BlendState::On {
    color: BlendOp::Min,
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
};

const SUBPIXEL_PASS0: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcColor,
    },
    alpha: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcAlpha,
    },
};

const SUBPIXEL_PASS1: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
};

const SUBPIXEL_WITH_BG_COLOR_PASS0: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcColor,
    },
    alpha: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::One,
    },
};

const SUBPIXEL_WITH_BG_COLOR_PASS1: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::OneMinusDstAlpha,
        dst: Factor::One,
    },
    alpha: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::One,
    },
};

const SUBPIXEL_WITH_BG_COLOR_PASS2: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrcAlpha,
    },
};

// This requires blend color to be set
const SUBPIXEL_CONSTANT_TEXT_COLOR: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::ConstColor,
        dst: Factor::OneMinusSrcColor,
    },
    alpha: BlendOp::Add {
        src: Factor::ConstAlpha,
        dst: Factor::OneMinusSrcAlpha,
    },
};

const SUBPIXEL_DUAL_SOURCE: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrc1Color,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrc1Alpha,
    },
};

const OVERDRAW: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrcAlpha,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrcAlpha,
    },
};

const LESS_EQUAL_TEST: DepthTest = DepthTest::On {
    fun: Comparison::LessEqual,
    write: false,
};

const LESS_EQUAL_WRITE: DepthTest = DepthTest::On {
    fun: Comparison::LessEqual,
    write: true,
};

pub struct ImageCore<B: hal::Backend> {
    pub image: B::Image,
    pub memory: Option<B::Memory>,
    pub view: B::ImageView,
    pub subresource_range: hal::image::SubresourceRange,
    pub state: Cell<hal::image::State>,
}

impl<B: hal::Backend> ImageCore<B> {
    fn from_image(
        device: &B::Device,
        image: B::Image,
        view_kind: hal::image::ViewKind,
        format: hal::format::Format,
        subresource_range: hal::image::SubresourceRange,
    ) -> Self {
        let view = unsafe {
            device.create_image_view(
                &image,
                view_kind,
                format,
                hal::format::Swizzle::NO,
                subresource_range.clone(),
            )
        }
        .expect("create_image_view failed");
        ImageCore {
            image,
            memory: None,
            view,
            subresource_range,
            state: Cell::new((hal::image::Access::empty(), hal::image::Layout::Undefined)),
        }
    }

    fn create(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        kind: hal::image::Kind,
        view_kind: hal::image::ViewKind,
        mip_levels: hal::image::Level,
        format: hal::format::Format,
        usage: hal::image::Usage,
        subresource_range: hal::image::SubresourceRange,
    ) -> Self {
        let mut image = unsafe {
            device.create_image(
                kind,
                mip_levels,
                format,
                hal::image::Tiling::Optimal,
                usage,
                hal::image::ViewCapabilities::empty(),
            )
        }
        .expect("create_image failed");
        let requirements = unsafe { device.get_image_requirements(&image) };

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                requirements.type_mask & (1 << id) != 0
                    && mem_type
                        .properties
                        .contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .expect("No memory type found")
            .into();

        let memory = unsafe {
            device
                .allocate_memory(mem_type, requirements.size)
                .expect("Allocate memory failed")
        };
        unsafe {
            device
                .bind_image_memory(&memory, 0, &mut image)
                .expect("Bind image memory failed")
        };

        ImageCore {
            memory: Some(memory),
            ..Self::from_image(device, image, view_kind, format, subresource_range)
        }
    }

    fn _reset(&self) {
        self.state
            .set((hal::image::Access::empty(), hal::image::Layout::Undefined));
    }

    fn deinit(self, device: &B::Device) {
        unsafe { device.destroy_image_view(self.view) };
        if let Some(memory) = self.memory {
            unsafe {
                device.destroy_image(self.image);
                device.free_memory(memory);
            }
        }
    }

    fn transit(
        &self,
        access: hal::image::Access,
        layout: hal::image::Layout,
        range: hal::image::SubresourceRange,
    ) -> Option<hal::memory::Barrier<B>> {
        let src_state = self.state.get();
        if src_state == (access, layout) {
            None
        } else {
            self.state.set((access, layout));
            Some(hal::memory::Barrier::Image {
                states: src_state .. (access, layout),
                target: &self.image,
                families: None,
                range,
            })
        }
    }
}

pub struct Image<B: hal::Backend> {
    pub core: ImageCore<B>,
    pub kind: hal::image::Kind,
    pub format: ImageFormat,
}

impl<B: hal::Backend> Image<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        image_format: ImageFormat,
        image_width: i32,
        image_height: i32,
        image_depth: i32,
        view_kind: hal::image::ViewKind,
        mip_levels: hal::image::Level,
        usage: hal::image::Usage,
    ) -> Self {
        let format = match image_format {
            ImageFormat::R8 => hal::format::Format::R8Unorm,
            ImageFormat::R16 => hal::format::Format::R16Unorm,
            ImageFormat::RG8 => hal::format::Format::Rg8Unorm,
            ImageFormat::RGBA8 => hal::format::Format::Rgba8Unorm,
            ImageFormat::BGRA8 => hal::format::Format::Bgra8Unorm,
            ImageFormat::RGBAF32 => hal::format::Format::Rgba32Float,
            ImageFormat::RGBAI32 => hal::format::Format::Rgba32Int,
        };
        let kind = hal::image::Kind::D2(image_width as _, image_height as _, image_depth as _, 1);

        let core = ImageCore::create(
            device,
            memory_types,
            kind,
            view_kind,
            mip_levels,
            format,
            usage,
            hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. mip_levels,
                layers: 0 .. image_depth as _,
            },
        );

        Image {
            core,
            kind,
            format: image_format,
        }
    }

    pub fn update(
        &self,
        device: &B::Device,
        cmd_pool: &mut CommandPool<B>,
        staging_buffer_pool: &mut BufferPool<B>,
        rect: DeviceIntRect,
        layer_index: i32,
        image_data: &[u8],
    ) {
        let pos = rect.origin;
        let size = rect.size;
        staging_buffer_pool.add(device, image_data);
        let buffer = staging_buffer_pool.buffer();
        let cmd_buffer = cmd_pool.acquire_command_buffer();

        unsafe {
            cmd_buffer.begin();

            let range = hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
                layers: layer_index as _ .. (layer_index + 1) as _,
            };
            let barriers = buffer
                .transit(hal::buffer::Access::TRANSFER_READ)
                .into_iter()
                .chain(self.core.transit(
                    hal::image::Access::TRANSFER_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    range.clone(),
                ));

            cmd_buffer.pipeline_barrier(
                PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );
            cmd_buffer.copy_buffer_to_image(
                &buffer.buffer,
                &self.core.image,
                hal::image::Layout::TransferDstOptimal,
                &[hal::command::BufferImageCopy {
                    buffer_offset: staging_buffer_pool.buffer_offset as _,
                    buffer_width: size.width as _,
                    buffer_height: size.height as _,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: layer_index as _ .. (layer_index + 1) as _,
                    },
                    image_offset: hal::image::Offset {
                        x: pos.x as i32,
                        y: pos.y as i32,
                        z: 0,
                    },
                    image_extent: hal::image::Extent {
                        width: size.width as u32,
                        height: size.height as u32,
                        depth: 1,
                    },
                }],
            );

            if let Some(barrier) = self.core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ
                    | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                range,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            cmd_buffer.finish();
        }
    }

    pub fn deinit(self, device: &B::Device) {
        self.core.deinit(device);
    }
}

pub struct Buffer<B: hal::Backend> {
    pub memory: B::Memory,
    pub buffer: B::Buffer,
    pub buffer_size: usize,
    pub buffer_len: usize,
    pub stride: usize,
    state: Cell<hal::buffer::State>,
}

impl<B: hal::Backend> Buffer<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        usage: hal::buffer::Usage,
        pitch_alignment_mask: usize,
        data_len: usize,
        stride: usize,
    ) -> Self {
        let buffer_size = (data_len * stride + pitch_alignment_mask) & !pitch_alignment_mask;
        let mut buffer = unsafe {
            device
                .create_buffer(buffer_size as u64, usage)
                .expect("create_buffer failed")
        };
        let requirements = unsafe { device.get_buffer_requirements(&buffer) };
        let memory_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                requirements.type_mask & (1 << id) != 0
                    && mem_type
                        .properties
                        .contains(hal::memory::Properties::CPU_VISIBLE)
            })
            .expect("No memory type found")
            .into();
        let memory = unsafe { device.allocate_memory(memory_type, requirements.size) }
            .expect("Allocate memory failed");
        unsafe { device.bind_buffer_memory(&memory, 0, &mut buffer) }
            .expect("Bind buffer memory failed");

        let buffer = Buffer {
            memory,
            buffer,
            buffer_size: requirements.size as _,
            buffer_len: data_len,
            stride,
            state: Cell::new(hal::buffer::Access::empty()),
        };
        buffer
    }

    pub fn update_all<T: Copy>(&self, device: &B::Device, data: &[T]) {
        let mut upload_data = unsafe {
            device.acquire_mapping_writer::<T>(&self.memory, 0 .. self.buffer_size as u64)
        }
        .expect("acquire_mapping_writer failed");
        upload_data[0 .. data.len()].copy_from_slice(&data);
        unsafe { device.release_mapping_writer(upload_data) }
            .expect("release_mapping_writer failed");
    }

    pub fn update<T: Copy>(
        &self,
        device: &B::Device,
        data: &[T],
        offset: usize,
        alignment_mask: usize,
    ) -> usize {
        let size_aligned = (data.len() * self.stride + alignment_mask) & !alignment_mask;
        let mut upload_data = unsafe {
            device.acquire_mapping_writer::<T>(
                &self.memory,
                (offset * self.stride) as u64 .. (size_aligned + (offset * self.stride)) as u64,
            )
        }
        .expect("acquire_mapping_writer failed");
        upload_data[0 .. data.len()].copy_from_slice(&data);
        unsafe { device.release_mapping_writer(upload_data) }
            .expect("release_mapping_writer failed");
        size_aligned
    }

    fn transit(&self, access: hal::buffer::Access) -> Option<hal::memory::Barrier<B>> {
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

    pub fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_buffer(self.buffer);
            device.free_memory(self.memory);
        }
    }
}

pub struct BufferPool<B: hal::Backend> {
    pub buffer: Buffer<B>,
    pub data_stride: usize,
    non_coherent_atom_size: usize,
    copy_alignment_mask: usize,
    offset: usize,
    size: usize,
    pub buffer_offset: usize,
}

impl<B: hal::Backend> BufferPool<B> {
    fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        usage: hal::buffer::Usage,
        data_stride: usize,
        non_coherent_atom_size: usize,
        pitch_alignment_mask: usize,
        copy_alignment_mask: usize,
    ) -> Self {
        let buffer = Buffer::new(
            device,
            memory_types,
            usage,
            pitch_alignment_mask,
            TEXTURE_CACHE_SIZE,
            data_stride,
        );
        BufferPool {
            buffer,
            data_stride,
            non_coherent_atom_size,
            copy_alignment_mask,
            offset: 0,
            size: 0,
            buffer_offset: 0,
        }
    }

    pub fn add<T: Copy>(&mut self, device: &B::Device, data: &[T]) {
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
        self.size = self
            .buffer
            .update(device, data, self.offset, self.non_coherent_atom_size);
        self.buffer_offset = self.offset;
        self.offset += (self.size + self.copy_alignment_mask) & !self.copy_alignment_mask;
    }

    fn buffer(&self) -> &Buffer<B> {
        &self.buffer
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.size = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        self.buffer.deinit(device);
    }
}

struct InstancePoolBuffer<B: hal::Backend> {
    buffer: Buffer<B>,
    pub offset: usize,
    pub last_update_size: usize,
}

impl<B: hal::Backend> InstancePoolBuffer<B> {
    fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        usage: hal::buffer::Usage,
        data_stride: usize,
        pitch_alignment_mask: usize,
    ) -> Self {
        let buffer = Buffer::new(
            device,
            memory_types,
            usage,
            pitch_alignment_mask,
            MAX_INSTANCE_COUNT,
            data_stride,
        );
        InstancePoolBuffer {
            buffer,
            offset: 0,
            last_update_size: 0,
        }
    }

    fn update<T: Copy>(&mut self, device: &B::Device, data: &[T], non_coherent_atom_size: usize) {
        self.buffer
            .update(device, data, self.offset, non_coherent_atom_size);
        self.last_update_size = data.len();
        self.offset += self.last_update_size;
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.last_update_size = 0;
    }

    fn deinit(self, device: &B::Device) {
        self.buffer.deinit(device);
    }
}

pub struct InstanceBufferHandler<B: hal::Backend> {
    buffers: Vec<InstancePoolBuffer<B>>,
    memory_types: Vec<hal::MemoryType>,
    data_stride: usize,
    non_coherent_atom_size: usize,
    pitch_alignment_mask: usize,
    current_buffer_index: usize,
}

impl<B: hal::Backend> InstanceBufferHandler<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        usage: hal::buffer::Usage,
        data_stride: usize,
        non_coherent_atom_size: usize,
        pitch_alignment_mask: usize,
    ) -> Self {
        let buffers = vec![InstancePoolBuffer::new(
            device,
            memory_types,
            usage,
            data_stride,
            pitch_alignment_mask,
        )];

        InstanceBufferHandler {
            buffers,
            memory_types: memory_types.to_vec(),
            non_coherent_atom_size,
            pitch_alignment_mask,
            data_stride,
            current_buffer_index: 0,
        }
    }

    pub fn add<T: Copy>(&mut self, device: &B::Device, mut data: &[T]) {
        assert_eq!(self.data_stride, mem::size_of::<T>());
        while !data.is_empty() {
            if self.current_buffer().buffer.buffer_size
                == self.current_buffer().offset * self.data_stride
            {
                self.current_buffer_index += 1;
                if self.buffers.len() <= self.current_buffer_index {
                    self.buffers.push(InstancePoolBuffer::new(
                        device,
                        &self.memory_types,
                        hal::buffer::Usage::VERTEX,
                        self.data_stride,
                        self.pitch_alignment_mask,
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

            self.buffers[self.current_buffer_index].update(
                device,
                &data[0 .. update_size],
                self.non_coherent_atom_size,
            );

            data = &data[update_size ..]
        }
    }

    fn current_buffer(&self) -> &InstancePoolBuffer<B> {
        &self.buffers[self.current_buffer_index]
    }

    fn reset(&mut self) {
        for buffer in &mut self.buffers {
            buffer.reset();
        }
        self.current_buffer_index = 0;
    }

    fn deinit(self, device: &B::Device) {
        for buffer in self.buffers {
            buffer.deinit(device);
        }
    }
}

pub struct UniformBufferHandler<B: hal::Backend> {
    buffers: Vec<Buffer<B>>,
    offset: usize,
    memory_types: Vec<hal::MemoryType>,
    usage: hal::buffer::Usage,
    data_stride: usize,
    pitch_alignment_mask: usize,
}

impl<B: hal::Backend> UniformBufferHandler<B> {
    pub fn new(
        memory_types: &Vec<hal::MemoryType>,
        usage: hal::buffer::Usage,
        data_stride: usize,
        pitch_alignment_mask: usize,
    ) -> Self {
        UniformBufferHandler {
            buffers: vec![],
            offset: 0,
            memory_types: memory_types.clone(),
            usage,
            data_stride,
            pitch_alignment_mask,
        }
    }

    pub fn add<T: Copy>(&mut self, device: &B::Device, data: &[T]) {
        if self.buffers.len() == self.offset {
            self.buffers.push(Buffer::new(
                device,
                &self.memory_types,
                self.usage,
                self.pitch_alignment_mask,
                data.len(),
                self.data_stride,
            ));
        }
        self.buffers[self.offset].update_all(device, data);
        self.offset += 1;
    }

    pub fn buffer(&self) -> &Buffer<B> {
        &self.buffers[self.offset - 1]
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        for buffer in self.buffers {
            buffer.deinit(device);
        }
    }
}

pub struct VertexBufferHandler<B: hal::Backend> {
    buffer: Buffer<B>,
    memory_types: Vec<hal::MemoryType>,
    usage: hal::buffer::Usage,
    data_stride: usize,
    pitch_alignment_mask: usize,
    pub buffer_len: usize,
}

impl<B: hal::Backend> VertexBufferHandler<B> {
    pub fn new<T: Copy>(
        device: &B::Device,
        memory_types: &Vec<hal::MemoryType>,
        usage: hal::buffer::Usage,
        data: &[T],
        data_stride: usize,
        pitch_alignment_mask: usize,
    ) -> Self {
        let buffer = Buffer::new(
            device,
            memory_types,
            usage,
            pitch_alignment_mask,
            data.len(),
            data_stride,
        );
        buffer.update_all(device, data);
        VertexBufferHandler {
            buffer_len: buffer.buffer_len,
            buffer,
            memory_types: memory_types.clone(),
            usage,
            data_stride,
            pitch_alignment_mask,
        }
    }

    pub fn update<T: Copy>(&mut self, device: &B::Device, data: &[T]) {
        let buffer_len = data.len() * self.data_stride;
        if self.buffer.buffer_len < buffer_len {
            let old_buffer = mem::replace(
                &mut self.buffer,
                Buffer::new(
                    device,
                    &self.memory_types,
                    self.usage,
                    self.pitch_alignment_mask,
                    data.len(),
                    self.data_stride,
                ),
            );
            old_buffer.deinit(device);
        }
        self.buffer.update_all(device, data);
        self.buffer_len = buffer_len;
    }

    pub fn buffer(&self) -> &Buffer<B> {
        &self.buffer
    }

    pub fn reset(&mut self) {
        self.buffer_len = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        self.buffer.deinit(device);
    }
}

pub(crate) struct Program<B: hal::Backend> {
    pub bindings_map: HashMap<String, u32>,
    pub pipelines: HashMap<(BlendState, DepthTest), B::GraphicsPipeline>,
    pub vertex_buffer: SmallVec<[VertexBufferHandler<B>; 1]>,
    pub index_buffer: Option<SmallVec<[VertexBufferHandler<B>; 1]>>,
    pub instance_buffer: SmallVec<[InstanceBufferHandler<B>; 1]>,
    pub locals_buffer: SmallVec<[UniformBufferHandler<B>; 1]>,
    shader_name: String,
    shader_kind: ShaderKind,
    bound_textures: [u32; 16],
}

impl<B: hal::Backend> Program<B> {
    pub fn create(
        pipeline_requirements: PipelineRequirements,
        device: &B::Device,
        pipeline_layout: &B::PipelineLayout,
        memory_types: &Vec<hal::MemoryType>,
        limits: &hal::Limits,
        shader_name: &str,
        features: &[&str],
        shader_kind: ShaderKind,
        render_pass: &RenderPass<B>,
        frame_count: usize,
        shader_modules: &mut FastHashMap<String, (B::ShaderModule, B::ShaderModule)>,
        pipeline_cache: Option<&B::PipelineCache>,
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

        let (vs_module, fs_module) = shader_modules.get(shader_name).expect(&format!("No shader module for shader names: {:?}", shader_name));

        let mut constants = Vec::with_capacity(SPECIALIZATION_CONSTANT_COUNT);
        let mut specialization_data = vec![0; SPECIALIZATION_CONSTANT_COUNT * SPECIALIZATION_CONSTANT_SIZE];
        for i in 0 .. SPECIALIZATION_CONSTANT_COUNT {
            constants.push(hal::pso::SpecializationConstant {
                id: i as _,
                range: (SPECIALIZATION_CONSTANT_SIZE * i) as _ .. (SPECIALIZATION_CONSTANT_SIZE * (i + 1)) as _,
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

            let mut pipeline_states = match shader_kind {
                ShaderKind::Cache(VertexArrayKind::Scale) => vec![
                    (BlendState::Off, DepthTest::Off),
                    (BlendState::MULTIPLY, DepthTest::Off),
                ],
                ShaderKind::Cache(VertexArrayKind::Blur) => vec![
                    (BlendState::Off, DepthTest::Off),
                    (BlendState::Off, LESS_EQUAL_TEST),
                ],
                ShaderKind::Cache(VertexArrayKind::Border)
                | ShaderKind::Cache(VertexArrayKind::LineDecoration) => {
                    vec![(BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off)]
                }
                ShaderKind::ClipCache => vec![(BlendState::MULTIPLY, DepthTest::Off)],
                ShaderKind::Text => {
                    let mut states = vec![
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
                    ];
                    if features.contains(&"DUAL_SOURCE_BLENDING") {
                        states.push((SUBPIXEL_DUAL_SOURCE, DepthTest::Off));
                        states.push((SUBPIXEL_DUAL_SOURCE, LESS_EQUAL_TEST));
                    }
                    states
                }
                ShaderKind::DebugColor | ShaderKind::DebugFont => {
                    vec![(BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off)]
                }
                _ => vec![
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
                ],
            };

            if features.contains(&"DEBUG_OVERDRAW") {
                pipeline_states.push((OVERDRAW, LESS_EQUAL_TEST));
            }

            let format = match shader_kind {
                ShaderKind::ClipCache => ImageFormat::R8,
                ShaderKind::Cache(VertexArrayKind::Blur) if features.contains(&"ALPHA_TARGET") => {
                    ImageFormat::R8
                }
                ShaderKind::Cache(VertexArrayKind::Scale) if features.contains(&"ALPHA_TARGET") => {
                    ImageFormat::R8
                }
                _ => ImageFormat::BGRA8,
            };

            let pipelines_descriptors = pipeline_states
                .iter()
                .map(|&(blend_state, depth_test)| {
                    let subpass = Subpass {
                        index: 0,
                        main_pass: render_pass
                            .get_render_pass(format, depth_test != DepthTest::Off),
                    };
                    let mut pipeline_descriptor = hal::pso::GraphicsPipelineDesc::new(
                        shader_entries.clone(),
                        Primitive::TriangleList,
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
                })
                .collect::<Vec<_>>();

            let pipelines =
                unsafe { device.create_graphics_pipelines(pipelines_descriptors.as_slice(), pipeline_cache) }
                    .into_iter();

            pipeline_states
                .iter()
                .cloned()
                .zip(pipelines.map(|pipeline| pipeline.expect("Graphic pipeline creation failed")))
                .collect::<HashMap<(BlendState, DepthTest), B::GraphicsPipeline>>()
        };

        let vertex_buffer_stride = match shader_kind {
            ShaderKind::DebugColor => mem::size_of::<DebugColorVertex>(),
            ShaderKind::DebugFont => mem::size_of::<DebugFontVertex>(),
            _ => mem::size_of::<Vertex>(),
        };

        let instance_buffer_stride = match shader_kind {
            ShaderKind::Primitive
            | ShaderKind::Brush
            | ShaderKind::Text
            | ShaderKind::Cache(VertexArrayKind::Primitive) => {
                mem::size_of::<PrimitiveInstanceData>()
            }
            ShaderKind::ClipCache | ShaderKind::Cache(VertexArrayKind::Clip) => {
                mem::size_of::<ClipMaskInstance>()
            }
            ShaderKind::Cache(VertexArrayKind::Blur) => mem::size_of::<BlurInstance>(),
            ShaderKind::Cache(VertexArrayKind::Border) => mem::size_of::<BorderInstance>(),
            ShaderKind::Cache(VertexArrayKind::Scale) => mem::size_of::<ScalingInstance>(),
            ShaderKind::Cache(VertexArrayKind::LineDecoration) => {
                mem::size_of::<LineDecorationInstance>()
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
                memory_types,
                hal::buffer::Usage::VERTEX,
                &QUAD,
                vertex_buffer_stride,
                (limits.min_buffer_copy_pitch_alignment - 1) as usize,
            ));
            instance_buffer.push(InstanceBufferHandler::new(
                device,
                memory_types,
                hal::buffer::Usage::VERTEX,
                instance_buffer_stride,
                (limits.non_coherent_atom_size - 1) as usize,
                (limits.min_buffer_copy_pitch_alignment - 1) as usize,
            ));
            locals_buffer.push(UniformBufferHandler::new(
                memory_types,
                hal::buffer::Usage::UNIFORM,
                mem::size_of::<Locals>(),
                (limits.min_uniform_buffer_offset_alignment - 1) as usize,
            ));
            if let Some(ref mut index_buffer) = index_buffer {
                index_buffer.push(VertexBufferHandler::new(
                    device,
                    memory_types,
                    hal::buffer::Usage::INDEX,
                    &vec![0u32; MAX_INDEX_COUNT],
                    mem::size_of::<u32>(),
                    (limits.min_buffer_copy_pitch_alignment - 1) as usize,
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

    pub fn bind_instances<T: Copy>(
        &mut self,
        device: &B::Device,
        instances: &[T],
        buffer_id: usize,
    ) {
        assert!(!instances.is_empty());
        self.instance_buffer[buffer_id].add(device, instances);
    }

    fn bind_locals(
        &mut self,
        device: &B::Device,
        set: &B::DescriptorSet,
        projection: &Transform3D<f32>,
        u_mode: i32,
        buffer_id: usize,
    ) {
        let locals_data = vec![Locals {
            uTransform: projection.to_row_arrays(),
            uMode: u_mode,
        }];
        self.locals_buffer[buffer_id].add(device, &locals_data);
        unsafe {
            device.write_descriptor_sets(vec![hal::pso::DescriptorSetWrite {
                set,
                binding: self.bindings_map["Locals"],
                array_offset: 0,
                descriptors: Some(hal::pso::Descriptor::Buffer(
                    &self.locals_buffer[buffer_id].buffer().buffer,
                    Some(0) .. None,
                )),
            }]);
        }
    }

    fn bind_texture(
        &self,
        device: &B::Device,
        set: &B::DescriptorSet,
        image: &ImageCore<B>,
        binding: &'static str,
    ) {
        if let Some(binding) = self.bindings_map.get(&("t".to_owned() + binding)) {
            unsafe {
                device.write_descriptor_sets(vec![hal::pso::DescriptorSetWrite {
                    set,
                    binding: *binding,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Image(
                        &image.view,
                        image.state.get().1,
                    )),
                }]);
            }
        }
    }

    fn bind_sampler(
        &self,
        device: &B::Device,
        set: &B::DescriptorSet,
        sampler: &B::Sampler,
        binding: &'static str,
    ) {
        if let Some(binding) = self.bindings_map.get(&("s".to_owned() + binding)) {
            unsafe {
                device.write_descriptor_sets(vec![hal::pso::DescriptorSetWrite {
                    set,
                    binding: *binding,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Sampler(sampler)),
                }]);
            }
        }
    }

    pub fn submit(
        &self,
        cmd_pool: &mut CommandPool<B>,
        viewport: hal::pso::Viewport,
        render_pass: &B::RenderPass,
        frame_buffer: &B::Framebuffer,
        desc_pools: &mut DescriptorPools<B>,
        desc_pools_global: &DescriptorPools<B>,
        desc_pools_sampler: &DescriptorPools<B>,
        clear_values: &[hal::command::ClearValue],
        blend_state: BlendState,
        blend_color: ColorF,
        depth_test: DepthTest,
        scissor_rect: Option<DeviceIntRect>,
        next_id: usize,
        pipeline_layouts: &FastHashMap<ShaderKind, B::PipelineLayout>,
    ) {
        let cmd_buffer = cmd_pool.acquire_command_buffer();
        let vertex_buffer = &self.vertex_buffer[next_id];
        let instance_buffer = &self.instance_buffer[next_id];

        unsafe {
            cmd_buffer.begin();
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
                vec![
                    desc_pools.get(&self.shader_kind),
                    desc_pools_global.get(&self.shader_kind),
                    desc_pools_sampler.get(&self.shader_kind),
                ],
                &[],
            );
            desc_pools.next(&self.shader_kind);

            if blend_state == SUBPIXEL_CONSTANT_TEXT_COLOR {
                cmd_buffer.set_blend_constants(blend_color.to_array());
            }

            if let Some(ref index_buffer) = self.index_buffer {
                cmd_buffer.bind_vertex_buffers(0, vec![(&vertex_buffer.buffer().buffer, 0)]);
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
                        vec![
                            (&vertex_buffer.buffer().buffer, 0),
                            (&instance_buffer.buffers[i].buffer.buffer, 0),
                        ],
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

            cmd_buffer.finish();
        }
    }

    pub fn deinit(mut self, device: &B::Device) {
        for mut vertex_buffer in self.vertex_buffer {
            vertex_buffer.deinit(device);
        }
        if let Some(index_buffer) = self.index_buffer {
            for mut index_buffer in index_buffer {
                index_buffer.deinit(device);
            }
        }
        for mut instance_buffer in self.instance_buffer {
            instance_buffer.deinit(device);
        }
        for mut locals_buffer in self.locals_buffer {
            locals_buffer.deinit(device);
        }
        for pipeline in self.pipelines.drain() {
            unsafe { device.destroy_graphics_pipeline(pipeline.1) };
        }
    }
}

pub struct Framebuffer<B: hal::Backend> {
    pub texture: TextureId,
    pub layer_index: u16,
    pub format: ImageFormat,
    pub image_view: B::ImageView,
    pub fbo: B::Framebuffer,
    pub rbo: RBOId,
}

impl<B: hal::Backend> Framebuffer<B> {
    pub fn new(
        device: &B::Device,
        texture: &Texture,
        image: &Image<B>,
        layer_index: u16,
        render_pass: &RenderPass<B>,
        rbo: RBOId,
        depth: Option<&B::ImageView>,
    ) -> Self {
        let extent = hal::image::Extent {
            width: texture.size.width as _,
            height: texture.size.height as _,
            depth: 1,
        };
        let format = match texture.format {
            ImageFormat::R8 => hal::format::Format::R8Unorm,
            ImageFormat::BGRA8 => hal::format::Format::Bgra8Unorm,
            f => unimplemented!("TODO image format missing {:?}", f),
        };
        let image_view = unsafe {
            device.create_image_view(
                &image.core.image,
                hal::image::ViewKind::D2Array,
                format,
                hal::format::Swizzle::NO,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: 0 .. 1,
                    layers: layer_index .. layer_index + 1,
                },
            )
        }
        .expect("create_image_view failed");
        let fbo = unsafe {
            if rbo != RBOId(0) {
                device.create_framebuffer(
                    render_pass.get_render_pass(texture.format, true),
                    vec![&image_view, depth.expect("Depth value is None")],
                    extent,
                )
            } else {
                device.create_framebuffer(
                    render_pass.get_render_pass(texture.format, false),
                    Some(&image_view),
                    extent,
                )
            }
        }
        .expect("create_framebuffer failed");

        Framebuffer {
            texture: texture.id,
            layer_index,
            format: texture.format,
            image_view,
            fbo,
            rbo,
        }
    }

    pub fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_framebuffer(self.fbo);
            device.destroy_image_view(self.image_view);
        }
    }
}

pub struct DepthBuffer<B: hal::Backend> {
    pub core: ImageCore<B>,
}

impl<B: hal::Backend> DepthBuffer<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        pixel_width: u32,
        pixel_height: u32,
        depth_format: hal::format::Format,
    ) -> Self {
        let core = ImageCore::create(
            device,
            memory_types,
            hal::image::Kind::D2(pixel_width, pixel_height, 1, 1),
            hal::image::ViewKind::D2,
            1,
            depth_format,
            hal::image::Usage::TRANSFER_DST | hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
            DEPTH_RANGE,
        );
        DepthBuffer { core }
    }

    pub fn deinit(self, device: &B::Device) {
        self.core.deinit(device);
    }
}

pub struct RenderPass<B: hal::Backend> {
    pub r8: B::RenderPass,
    pub r8_depth: B::RenderPass,
    pub bgra8: B::RenderPass,
    pub bgra8_depth: B::RenderPass,
}

impl<B: hal::Backend> RenderPass<B> {
    pub fn get_render_pass(&self, format: ImageFormat, depth_enabled: bool) -> &B::RenderPass {
        match format {
            ImageFormat::R8 if depth_enabled => &self.r8_depth,
            ImageFormat::R8 => &self.r8,
            ImageFormat::BGRA8 if depth_enabled => &self.bgra8_depth,
            ImageFormat::BGRA8 => &self.bgra8,
            f => unimplemented!("No render pass for image format {:?}", f),
        }
    }

    pub fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_render_pass(self.r8);
            device.destroy_render_pass(self.r8_depth);
            device.destroy_render_pass(self.bgra8);
            device.destroy_render_pass(self.bgra8_depth);
        }
    }
}

pub struct DescPool<B: hal::Backend> {
    descriptor_pool: B::DescriptorPool,
    descriptor_set: Vec<B::DescriptorSet>,
    descriptor_set_layout: B::DescriptorSetLayout,
    current_descriptor_set_id: usize,
    max_descriptor_set_size: usize,
}

impl<B: hal::Backend> DescPool<B> {
    pub fn new(
        device: &B::Device,
        max_size: usize,
        descriptor_range_descriptors: Vec<DescriptorRangeDesc>,
        descriptor_set_layout: Vec<DescriptorSetLayoutBinding>,
    ) -> Self {
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(max_size, descriptor_range_descriptors.as_slice())
        }
        .expect("create_descriptor_pool failed");
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_set_layout, &[]) }
                .expect("create_descriptor_set_layout failed");
        let mut dp = DescPool {
            descriptor_pool,
            descriptor_set: vec![],
            descriptor_set_layout,
            current_descriptor_set_id: 0,
            max_descriptor_set_size: max_size,
        };
        dp.allocate();
        dp
    }

    pub fn descriptor_set(&self) -> &B::DescriptorSet {
        &self.descriptor_set[self.current_descriptor_set_id]
    }

    pub fn descriptor_set_layout(&self) -> &B::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    pub fn next(&mut self) {
        self.current_descriptor_set_id += 1;
        assert!(
            self.current_descriptor_set_id < self.max_descriptor_set_size,
            "Maximum descriptor set size({}) exceeded!",
            self.max_descriptor_set_size
        );
        if self.current_descriptor_set_id == self.descriptor_set.len() {
            self.allocate();
        }
    }

    fn allocate(&mut self) {
        let desc_set = unsafe {
            self.descriptor_pool
                .allocate_set(&self.descriptor_set_layout)
        }
        .expect(&format!(
            "Failed to allocate set with layout: {:?}",
            self.descriptor_set_layout
        ));
        self.descriptor_set.push(desc_set);
    }

    pub fn reset(&mut self) {
        self.current_descriptor_set_id = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout);
            device.destroy_descriptor_pool(self.descriptor_pool);
        }
    }
}

pub struct DescriptorPools<B: hal::Backend> {
    pub debug_pool: DescPool<B>,
    pub cache_clip_pool: DescPool<B>,
    pub default_pool: DescPool<B>,
}

impl<B: hal::Backend> DescriptorPools<B> {
    pub fn new(
        device: &B::Device,
        descriptor_count: usize,
        pipeline_requirements: &HashMap<String, PipelineRequirements>,
        set: usize,
    ) -> Self {
        fn increase_range_count(range: &mut Vec<DescriptorRangeDesc>, count: usize) {
            for r in range {
                r.count *= count;
            }
        }
        fn get_layout_and_range(
            pipeline: &PipelineRequirements,
            set: usize,
        ) -> (Vec<DescriptorSetLayoutBinding>, Vec<DescriptorRangeDesc>) {
            (
                pipeline.descriptor_set_layout_bindings[set].clone(),
                pipeline.descriptor_range_descriptors[set].clone(),
            )
        }

        let (debug_layout, mut debug_layout_range) = get_layout_and_range(
            pipeline_requirements
                .get("debug_color")
                .expect("debug_color missing"),
            set,
        );
        increase_range_count(&mut debug_layout_range, DEBUG_DESCRIPTOR_COUNT);

        let (cache_clip_layout, mut cache_clip_layout_range) = get_layout_and_range(
            pipeline_requirements
                .get("cs_clip_rectangle")
                .expect("cs_clip_rectangle missing"),
            set,
        );
        increase_range_count(&mut cache_clip_layout_range, descriptor_count);

        let (default_layout, mut default_layout_range) = get_layout_and_range(
            pipeline_requirements
                .get("brush_solid")
                .expect("brush_solid missing"),
            set,
        );
        increase_range_count(&mut default_layout_range, descriptor_count);

        DescriptorPools {
            debug_pool: DescPool::new(
                device,
                DEBUG_DESCRIPTOR_COUNT,
                debug_layout_range,
                debug_layout,
            ),
            cache_clip_pool: DescPool::new(
                device,
                descriptor_count,
                cache_clip_layout_range,
                cache_clip_layout,
            ),
            default_pool: DescPool::new(
                device,
                descriptor_count,
                default_layout_range,
                default_layout,
            ),
        }
    }

    fn get_pool(&self, shader_kind: &ShaderKind) -> &DescPool<B> {
        match *shader_kind {
            ShaderKind::DebugColor | ShaderKind::DebugFont => &self.debug_pool,
            ShaderKind::ClipCache => &self.cache_clip_pool,
            _ => &self.default_pool,
        }
    }

    fn get_pool_mut(&mut self, shader_kind: &ShaderKind) -> &mut DescPool<B> {
        match *shader_kind {
            ShaderKind::DebugColor | ShaderKind::DebugFont => &mut self.debug_pool,
            ShaderKind::ClipCache => &mut self.cache_clip_pool,
            _ => &mut self.default_pool,
        }
    }

    pub fn get(&self, shader_kind: &ShaderKind) -> &B::DescriptorSet {
        self.get_pool(shader_kind).descriptor_set()
    }

    pub fn get_layout(&self, shader_kind: &ShaderKind) -> &B::DescriptorSetLayout {
        self.get_pool(shader_kind).descriptor_set_layout()
    }

    pub fn next(&mut self, shader_kind: &ShaderKind) {
        self.get_pool_mut(shader_kind).next()
    }

    pub fn reset(&mut self) {
        self.debug_pool.reset();
        self.cache_clip_pool.reset();
        self.default_pool.reset();
    }

    pub fn deinit(self, device: &B::Device) {
        self.debug_pool.deinit(device);
        self.cache_clip_pool.deinit(device);
        self.default_pool.deinit(device);
    }
}

struct Fence<B: hal::Backend> {
    inner: B::Fence,
    is_submitted: bool,
}

pub struct CommandPool<B: hal::Backend> {
    command_pool: hal::CommandPool<B, hal::Graphics>,
    command_buffers: Vec<hal::command::CommandBuffer<B, hal::Graphics>>,
    size: usize,
}

impl<B: hal::Backend> CommandPool<B> {
    fn new(mut command_pool: hal::CommandPool<B, hal::Graphics>) -> Self {
        let command_buffer = command_pool.acquire_command_buffer::<hal::command::OneShot>();
        CommandPool {
            command_pool,
            command_buffers: vec![command_buffer],
            size: 0,
        }
    }

    fn acquire_command_buffer(&mut self) -> &mut hal::command::CommandBuffer<B, hal::Graphics> {
        if self.size >= self.command_buffers.len() {
            let command_buffer = self
                .command_pool
                .acquire_command_buffer::<hal::command::OneShot>();
            self.command_buffers.push(command_buffer);
        }
        self.size += 1;
        &mut self.command_buffers[self.size - 1]
    }

    fn command_buffers(&self) -> &[hal::command::CommandBuffer<B, hal::Graphics>] {
        &self.command_buffers[0 .. self.size]
    }

    unsafe fn reset(&mut self) {
        self.command_pool.reset();
        self.size = 0;
    }

    unsafe fn destroy(self, device: &B::Device) {
        device.destroy_command_pool(self.command_pool.into_raw());
    }
}

pub struct Device<B: hal::Backend> {
    pub device: B::Device,
    pub memory_types: Vec<hal::MemoryType>,
    pub limits: hal::Limits,
    adapter: hal::Adapter<B>,
    surface: B::Surface,
    pub surface_format: hal::format::Format,
    pub depth_format: hal::format::Format,
    pub queue_group: hal::QueueGroup<B, hal::Graphics>,
    pub command_pool: SmallVec<[CommandPool<B>; 1]>,
    pub staging_buffer_pool: SmallVec<[BufferPool<B>; 1]>,
    pub swap_chain: Option<B::Swapchain>,
    pub render_pass: Option<RenderPass<B>>,
    pub framebuffers: Vec<B::Framebuffer>,
    pub framebuffers_depth: Vec<B::Framebuffer>,
    pub frame_images: Vec<ImageCore<B>>,
    pub frame_depths: Vec<DepthBuffer<B>>,
    pub frame_count: usize,
    pub viewport: hal::pso::Viewport,
    pub sampler_linear: B::Sampler,
    pub sampler_nearest: B::Sampler,
    pub current_frame_id: usize,
    current_blend_state: Cell<BlendState>,
    blend_color: Cell<ColorF>,
    current_depth_test: DepthTest,
    // device state
    programs: FastHashMap<ProgramId, Program<B>>,
    shader_modules: FastHashMap<String, (B::ShaderModule, B::ShaderModule)>,
    images: FastHashMap<TextureId, Image<B>>,
    retained_textures: Vec<Texture>,
    fbos: FastHashMap<FBOId, Framebuffer<B>>,
    rbos: FastHashMap<RBOId, DepthBuffer<B>>,
    descriptor_pools: SmallVec<[DescriptorPools<B>; 1]>,
    descriptor_pools_global: DescriptorPools<B>,
    descriptor_pools_sampler: DescriptorPools<B>,
    // device state
    bound_textures: [u32; 16],
    bound_program: ProgramId,
    bound_sampler: [TextureFilter; 16],
    bound_read_texture: (TextureId, i32),
    bound_read_fbo: FBOId,
    bound_draw_fbo: FBOId,
    program_mode_id: i32,
    scissor_rect: Option<DeviceIntRect>,
    //default_read_fbo: FBOId,
    //default_draw_fbo: FBOId,
    device_pixel_ratio: f32,
    depth_available: bool,
    upload_method: UploadMethod,

    // HW or API capabilities
    capabilities: Capabilities,

    /// Map from texture dimensions to shared depth buffers for render targets.
    ///
    /// Render targets often have the same width/height, so we can save memory
    /// by sharing these across targets.
    depth_targets: FastHashMap<DeviceIntSize, SharedDepthTarget>,

    // debug
    inside_frame: bool,

    // resources
    _resource_override_path: Option<PathBuf>,

    max_texture_size: i32,
    max_texture_layers: u32,
    _renderer_name: String,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    frame_id: GpuFrameId,

    // Supported features
    features: hal::Features,

    next_id: usize,
    frame_fence: SmallVec<[Fence<B>; 1]>,
    image_available_semaphore: B::Semaphore,
    render_finished_semaphore: B::Semaphore,
    pipeline_requirements: HashMap<String, PipelineRequirements>,
    pipeline_layouts: FastHashMap<ShaderKind, B::PipelineLayout>,
    pipeline_cache: Option<B::PipelineCache>,
    cache_path: Option<PathBuf>,
    save_cache: bool,
}

impl<B: hal::Backend> Device<B> {
    pub fn new(
        init: DeviceInit<B>,
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _cached_programs: Option<Rc<ProgramCache>>,
    ) -> Self {
        let DeviceInit {
            adapter,
            mut surface,
            window_size,
            frame_count,
            descriptor_count,
            cache_path,
            save_cache,
        } = init;
        let renderer_name = "TODO renderer name".to_owned();
        let features = adapter.physical_device.features();

        let memory_types = adapter.physical_device.memory_properties().memory_types;
        println!("memory_types: {:?}", memory_types);

        let limits = adapter.physical_device.limits();
        let max_texture_size = 4400i32; // TODO use limits after it points to the correct texture size

        let (device, queue_group) = {
            use hal::Capability;
            use hal::queue::QueueFamily;

            let family = adapter
                .queue_families
                .iter()
                .find(|family| {
                    hal::Graphics::supported_by(family.queue_type())
                        && surface.supports_queue_family(&family)
                })
                .expect("No queue family found");

            let priorities = vec![1.0];
            let (id, families) = (family.id(), [(family, priorities.as_slice())]);
            let hal::Gpu { device, mut queues } = unsafe {
                adapter
                    .physical_device
                    .open(&families, hal::Features::DUAL_SRC_BLENDING)
                    .unwrap_or_else( |_| {
                        adapter.physical_device.open(&families, hal::Features::empty()).expect("Unable to open physical_device")
                    }
                )
            };
            (device, queues.take(id).expect("queue_group doesn't exist"))
        };

        let (
            swap_chain,
            surface_format,
            depth_format,
            render_pass,
            framebuffers,
            framebuffers_depth,
            frame_depths,
            frame_images,
            viewport,
        ) = Device::init_swapchain_resources(
            &device,
            &memory_types,
            &adapter,
            &mut surface,
            window_size,
        );

        // Samplers

        let sampler_linear = unsafe {
            device.create_sampler(hal::image::SamplerInfo::new(
                hal::image::Filter::Linear,
                hal::image::WrapMode::Clamp,
            ))
        }
        .expect("sampler_linear failed");

        let sampler_nearest = unsafe {
            device.create_sampler(hal::image::SamplerInfo::new(
                hal::image::Filter::Nearest,
                hal::image::WrapMode::Clamp,
            ))
        }
        .expect("sampler_linear failed");

        let pipeline_requirements: HashMap<String, PipelineRequirements> =
            from_str(&shader_source::PIPELINES).expect("Failed to load pipeline requirements");

        let mut descriptor_pools = SmallVec::new();
        let mut frame_fence = SmallVec::new();
        let mut command_pool = SmallVec::new();
        let mut staging_buffer_pool = SmallVec::new();
        let frame_count = frame_count.unwrap_or(MAX_FRAME_COUNT);
        for _ in 0 .. frame_count {
            descriptor_pools.push(DescriptorPools::new(
                &device,
                descriptor_count.unwrap_or(DESCRIPTOR_COUNT),
                &pipeline_requirements,
                DESCRIPTOR_SET_PER_DRAW,
            ));

            let fence = device.create_fence(false).expect("create_fence failed");
            frame_fence.push(Fence {
                inner: fence,
                is_submitted: false,
            });

            let mut hal_cp = unsafe {
                device.create_command_pool_typed(
                    &queue_group,
                    hal::pool::CommandPoolCreateFlags::empty(),
                )
            }
            .expect("create_command_pool_typed failed");
            unsafe { hal_cp.reset() };
            let cp = CommandPool::new(hal_cp);
            command_pool.push(cp);
            staging_buffer_pool.push(BufferPool::new(
                &device,
                &memory_types,
                hal::buffer::Usage::TRANSFER_SRC,
                1,
                (limits.non_coherent_atom_size - 1) as usize,
                (limits.min_buffer_copy_pitch_alignment - 1) as usize,
                (limits.min_buffer_copy_offset_alignment - 1) as usize,
            ));
        }
        let descriptor_pools_global = DescriptorPools::new(
            &device,
            1,
            &pipeline_requirements,
            DESCRIPTOR_SET_PER_INSTANCE,
        );
        let descriptor_pools_sampler =
            DescriptorPools::new(&device, 1, &pipeline_requirements, DESCRIPTOR_SET_SAMPLER);

        let image_available_semaphore = device.create_semaphore().expect("create_semaphore failed");
        let render_finished_semaphore = device.create_semaphore().expect("create_semaphore failed");

        let pipeline_cache = if let Some(ref path) = cache_path {
            Self::load_pipeline_cache(
                &device,
                &path,
                &adapter.physical_device,
            )
        } else {
            None
        };

        Device {
            device,
            limits,
            memory_types,
            surface_format,
            adapter,
            surface,
            depth_format,
            queue_group,
            command_pool,
            staging_buffer_pool,
            swap_chain: Some(swap_chain),
            render_pass: Some(render_pass),
            framebuffers,
            framebuffers_depth,
            frame_images,
            frame_depths,
            frame_count,
            viewport,
            sampler_linear,
            sampler_nearest,
            current_frame_id: 0,
            current_blend_state: Cell::new(BlendState::Off),
            current_depth_test: DepthTest::Off,
            blend_color: Cell::new(ColorF::new(0.0, 0.0, 0.0, 0.0)),
            _resource_override_path: resource_override_path,
            // This is initialized to 1 by default, but it is reset
            // at the beginning of each frame in `Renderer::bind_frame_data`.
            device_pixel_ratio: 1.0,
            depth_available: true,
            upload_method,
            inside_frame: false,

            capabilities: Capabilities {
                supports_multisampling: false, //TODO
            },
            depth_targets: FastHashMap::default(),

            programs: FastHashMap::default(),
            shader_modules: FastHashMap::default(),
            images: FastHashMap::default(),
            retained_textures: Vec::new(),
            fbos: FastHashMap::default(),
            rbos: FastHashMap::default(),
            descriptor_pools,
            descriptor_pools_global,
            descriptor_pools_sampler,
            bound_textures: [0; 16],
            bound_program: INVALID_PROGRAM_ID,
            bound_sampler: [TextureFilter::Linear; 16],
            bound_read_fbo: DEFAULT_READ_FBO,
            bound_read_texture: (INVALID_TEXTURE_ID, 0),
            bound_draw_fbo: DEFAULT_DRAW_FBO,
            program_mode_id: 0,
            scissor_rect: None,

            max_texture_size,
            //TODO: Add to the limits in gfx-rs
            max_texture_layers: 4,
            _renderer_name: renderer_name,
            frame_id: GpuFrameId(0),
            features,

            next_id: 0,
            frame_fence,
            image_available_semaphore,
            render_finished_semaphore,
            pipeline_requirements,
            pipeline_layouts: FastHashMap::default(),
            pipeline_cache,
            cache_path,
            save_cache,
        }
    }

    fn load_pipeline_cache(device: &B::Device, path: &PathBuf, physical_device: &B::PhysicalDevice) -> Option<B::PipelineCache> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(_) => {
                warn!("File not found: {:?}", path);
                return None;
            }
        };
        let mut bytes = Vec::new();
        match file.read_to_end(&mut bytes) {
            Err(_) => {
                warn!("Failed to read file: {:?}", path);
                return None;
            }
            _ => {}
        };

        if physical_device.is_valid_cache(&bytes) {
            let cache = unsafe {
                device.create_pipeline_cache(Some(&bytes))
            }.expect(&format!("Failed to create pipeline cache from file: {:?}", path));
            Some(cache)
        } else {
            None
        }
    }

    pub(crate) fn recreate_swapchain(&mut self, window_size: Option<(i32, i32)>) -> DeviceIntSize {
        self.device.wait_idle().expect("Wait idle failed");

        for (_id, program) in self.programs.drain() {
            program.deinit(&self.device);
        }

        for image in self.frame_images.drain(..) {
            image.deinit(&self.device);
        }

        for depth in self.frame_depths.drain(..) {
            depth.deinit(&self.device);
        }

        self.render_pass.take().unwrap().deinit(&self.device);

        self.descriptor_pools_global.reset();
        self.descriptor_pools_sampler.reset();

        unsafe {
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer);
            }
            for framebuffer_depth in self.framebuffers_depth.drain(..) {
                self.device.destroy_framebuffer(framebuffer_depth);
            }
            self.device
                .destroy_swapchain(self.swap_chain.take().expect("Destroy swapchain failed"));
        }

        let window_size =
            window_size.unwrap_or((self.viewport.rect.w.into(), self.viewport.rect.h.into()));

        let (
            swap_chain,
            surface_format,
            depth_format,
            render_pass,
            framebuffers,
            framebuffers_depth,
            frame_depths,
            frame_images,
            viewport,
        ) = Device::init_swapchain_resources(
            &self.device,
            &self.memory_types,
            &self.adapter,
            &mut self.surface,
            window_size,
        );

        self.swap_chain = Some(swap_chain);
        self.render_pass = Some(render_pass);
        self.framebuffers = framebuffers;
        self.framebuffers_depth = framebuffers_depth;
        self.frame_depths = frame_depths;
        self.frame_images = frame_images;
        self.viewport = viewport;
        self.surface_format = surface_format;
        self.depth_format = depth_format;

        let pipeline_cache = unsafe { self.device.create_pipeline_cache(None) }.expect("Failed to create pipeline cache");
        if let Some(ref cache) = self.pipeline_cache {
            unsafe {
                self.device.merge_pipeline_caches(&cache, Some(&pipeline_cache)).expect("merge_pipeline_caches failed");;
                self.device.destroy_pipeline_cache(pipeline_cache);
            }
        } else {
            self.pipeline_cache = Some(pipeline_cache);
        }
        DeviceIntSize::new(self.viewport.rect.w.into(), self.viewport.rect.h.into())
    }

    fn init_swapchain_resources(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        adapter: &hal::Adapter<B>,
        surface: &mut B::Surface,
        window_size: (i32, i32),
    ) -> (
        B::Swapchain,
        hal::format::Format,
        hal::format::Format,
        RenderPass<B>,
        Vec<B::Framebuffer>,
        Vec<B::Framebuffer>,
        Vec<DepthBuffer<B>>,
        Vec<ImageCore<B>>,
        hal::pso::Viewport,
    ) {
        let (caps, formats, _present_modes, _composite_alphas) =
            surface.compatibility(&adapter.physical_device);
        let surface_format = formats.map_or(hal::format::Format::Bgra8Unorm, |formats| {
            formats
                .into_iter()
                .find(|format| format == &hal::format::Format::Bgra8Unorm)
                .expect("No surface format found")
        });

        let mut extent = caps.current_extent.unwrap_or(hal::window::Extent2D {
            width: (window_size.0 as u32)
                .max(caps.extents.start.width)
                .min(caps.extents.end.width),
            height: (window_size.1 as u32)
                .max(caps.extents.start.height)
                .min(caps.extents.end.height),
        });

        if extent.width == 0 {
            extent.width = 1;
        }
        if extent.height == 0 {
            extent.height = 1;
        }

        let swap_config = SwapchainConfig::new(
            extent.width,
            extent.height,
            surface_format,
            caps.image_count.start,
        )
        .with_image_usage(
            hal::image::Usage::TRANSFER_SRC
                | hal::image::Usage::TRANSFER_DST
                | hal::image::Usage::COLOR_ATTACHMENT,
        );

        let (swap_chain, backbuffer) =
            unsafe { device.create_swapchain(surface, swap_config, None) }
                .expect("create_swapchain failed");
        println!("backbuffer={:?}", backbuffer);
        let depth_format = hal::format::Format::D32Float; //maybe d24s8?
        let render_pass = {
            let attachment_r8 = hal::pass::Attachment {
                format: Some(hal::format::Format::R8Unorm),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::ColorAttachmentOptimal
                    .. hal::image::Layout::ColorAttachmentOptimal,
            };

            let attachment_bgra8 = hal::pass::Attachment {
                format: Some(hal::format::Format::Bgra8Unorm),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::ColorAttachmentOptimal
                    .. hal::image::Layout::ColorAttachmentOptimal,
            };

            let attachment_depth = hal::pass::Attachment {
                format: Some(depth_format),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::DepthStencilAttachmentOptimal
                    .. hal::image::Layout::DepthStencilAttachmentOptimal,
            };

            let subpass_r8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let subpass_depth_r8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let subpass_bgra8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let subpass_depth_bgra8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let dependency = hal::pass::SubpassDependency {
                passes: hal::pass::SubpassRef::External .. hal::pass::SubpassRef::Pass(0),
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
                    .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: hal::image::Access::empty()
                    .. (hal::image::Access::COLOR_ATTACHMENT_READ
                        | hal::image::Access::COLOR_ATTACHMENT_WRITE),
            };

            let depth_dependency = hal::pass::SubpassDependency {
                passes: hal::pass::SubpassRef::External .. hal::pass::SubpassRef::Pass(0),
                stages: PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::LATE_FRAGMENT_TESTS,
                accesses: hal::image::Access::empty()
                    .. (hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                        | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE),
            };

            RenderPass {
                r8: unsafe {
                    device.create_render_pass(
                        &[attachment_r8.clone()],
                        &[subpass_r8],
                        &[dependency.clone()],
                    )
                }
                .expect("create_render_pass failed"),
                r8_depth: unsafe {
                    device.create_render_pass(
                        &[attachment_r8, attachment_depth.clone()],
                        &[subpass_depth_r8],
                        &[dependency.clone(), depth_dependency.clone()],
                    )
                }
                .expect("create_render_pass failed"),
                bgra8: unsafe {
                    device.create_render_pass(
                        &[attachment_bgra8.clone()],
                        &[subpass_bgra8],
                        &[dependency.clone()],
                    )
                }
                .expect("create_render_pass failed"),
                bgra8_depth: unsafe {
                    device.create_render_pass(
                        &[attachment_bgra8, attachment_depth],
                        &[subpass_depth_bgra8],
                        &[dependency, depth_dependency],
                    )
                }
                .expect("create_render_pass failed"),
            }
        };

        let mut frame_depths = Vec::new();

        // Framebuffer and render target creation
        let (frame_images, framebuffers, framebuffers_depth) = match backbuffer {
            Backbuffer::Images(images) => {
                let extent = hal::image::Extent {
                    width: extent.width as _,
                    height: extent.height as _,
                    depth: 1,
                };
                let cores = images
                    .into_iter()
                    .map(|image| {
                        frame_depths.push(DepthBuffer::new(
                            device,
                            &memory_types,
                            extent.width,
                            extent.height,
                            depth_format,
                        ));
                        ImageCore::from_image(
                            device,
                            image,
                            hal::image::ViewKind::D2Array,
                            surface_format,
                            COLOR_RANGE.clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                let fbos = cores
                    .iter()
                    .map(|core| {
                        unsafe {
                            device.create_framebuffer(&render_pass.bgra8, Some(&core.view), extent)
                        }
                        .expect("create_framebuffer failed")
                    })
                    .collect();
                let fbos_depth = cores
                    .iter()
                    .zip(frame_depths.iter())
                    .map(|(core, depth)| {
                        unsafe {
                            device.create_framebuffer(
                                &render_pass.bgra8_depth,
                                vec![&core.view, &depth.core.view],
                                extent,
                            )
                        }
                        .expect("create_framebuffer failed")
                    })
                    .collect();
                (cores, fbos, fbos_depth)
            }
            // TODO fix depth fbos
            Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo], vec![]),
        };

        // Rendering setup
        let viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0 .. 1.0,
        };
        (
            swap_chain,
            surface_format,
            depth_format,
            render_pass,
            framebuffers,
            framebuffers_depth,
            frame_depths,
            frame_images,
            viewport,
        )
    }

    pub(crate) fn bound_program(&self) -> ProgramId {
        self.bound_program
    }

    pub(crate) fn viewport_size(&self) -> DeviceIntSize {
        DeviceIntSize::new(self.viewport.rect.w as _, self.viewport.rect.h as _)
    }

    pub fn set_device_pixel_ratio(&mut self, ratio: f32) {
        self.device_pixel_ratio = ratio;
    }

    pub fn update_program_cache(&mut self, _cached_programs: Rc<ProgramCache>) {
        warn!("Program cache is not supported!");
    }

    /// Ensures that the maximum texture size is less than or equal to the
    /// provided value. If the provided value is less than the value supported
    /// by the driver, the latter is used.
    pub fn clamp_max_texture_size(&mut self, size: i32) {
        self.max_texture_size = self.max_texture_size.min(size);
    }

    /// Returns the limit on texture dimensions (width or height).
    pub fn max_texture_size(&self) -> i32 {
        self.max_texture_size
    }

    /// Returns the limit on texture array layers.
    pub fn max_texture_layers(&self) -> usize {
        self.max_texture_layers as usize
    }

    pub fn get_capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub fn reset_state(&mut self) {
        self.bound_textures = [0; 16];
        self.bound_program = INVALID_PROGRAM_ID;
        self.bound_sampler = [TextureFilter::Linear; 16];
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;
    }

    fn reset_program_buffer_offsets(&mut self) {
        for program in self.programs.values_mut() {
            program.instance_buffer[self.next_id].reset();
            program.locals_buffer[self.next_id].reset();
            if let Some(ref mut index_buffer) = program.index_buffer {
                index_buffer[self.next_id].reset();
                program.vertex_buffer[self.next_id].reset();
            }
        }
    }

    pub fn delete_program(&mut self, mut _program: ProgramId) {
        // TODO delete program
        _program = INVALID_PROGRAM_ID;
    }

    pub fn create_program_linked(
        &mut self,
        base_filename: &str,
        features: &str,
        descriptor: &VertexDescriptor,
        shader_kind: &ShaderKind,
    ) -> Result<ProgramId, ShaderError> {
        let program = self.create_program(base_filename, shader_kind, &[features])?;
        self.link_program(program, descriptor)?;
        Ok(program)
    }

    pub fn link_program(
        &mut self,
        _program: ProgramId,
        _descriptor: &VertexDescriptor,
    ) -> Result<(), ShaderError> {
        warn!("link_program is not implemented with gfx backend");
        Ok(())
    }

    pub fn create_program(
        &mut self,
        shader_name: &str,
        shader_kind: &ShaderKind,
        features: &[&str],
    ) -> Result<ProgramId, ShaderError> {
        let mut name = String::from(shader_name);
        for feature_names in features {
            for feature in feature_names.split(',') {
                if NON_SPECIALIZATION_FEATURES.iter().any(|f| *f == feature) {
                    name.push_str(&format!("_{}", feature.to_lowercase()));
                }
            }
        }

        if !self.pipeline_layouts.contains_key(shader_kind) {
            let layout = unsafe {
                self.device.create_pipeline_layout(
                    vec![
                        self.descriptor_pools[self.next_id].get_layout(shader_kind),
                        self.descriptor_pools_global.get_layout(shader_kind),
                        self.descriptor_pools_sampler.get_layout(shader_kind),
                    ],
                    &[],
                )
            }
            .expect("create_pipeline_layout failed");
            self.pipeline_layouts.insert(shader_kind.clone(), layout);
        };
        let program = Program::create(
            self.pipeline_requirements
                .get(&name)
                .expect(&format!("Can't load pipeline data for: {}!", name))
                .clone(),
            &self.device,
            &self.pipeline_layouts[shader_kind],
            &self.memory_types,
            &self.limits,
            &name,
            features,
            shader_kind.clone(),
            self.render_pass.as_ref().unwrap(),
            self.frame_count,
            &mut self.shader_modules,
            self.pipeline_cache.as_ref(),
        );
        self.bind_samplers(&program);

        let id = self.generate_program_id();
        self.programs.insert(id, program);
        Ok(id)
    }

    pub fn create_program_with_kind(
        &mut self,
        shader_name: &str,
        shader_kind: &ShaderKind,
        features: &[&str],
        _precache_flags: ShaderPrecacheFlags,
    ) -> Result<ProgramId, ShaderError> {
        self.create_program(shader_name, shader_kind, features)
    }

    pub fn bind_program(&mut self, program_id: &ProgramId) {
        debug_assert!(self.inside_frame);

        if self.bound_program != *program_id {
            self.bound_program = *program_id;
        }
    }

    pub fn bind_textures(&mut self) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self
            .programs
            .get_mut(&self.bound_program)
            .expect("Program not found.");
        let desc_set = self.descriptor_pools[self.next_id].get(&program.shader_kind);
        for &(index, sampler_name) in SAMPLERS[0 .. 5].iter() {
            program.bind_texture(
                &self.device,
                desc_set,
                &self.images[&self.bound_textures[index]].core,
                sampler_name,
            );
        }
        let desc_set = self.descriptor_pools_global.get(&program.shader_kind);
        for &(index, sampler_name) in SAMPLERS[5 .. 11].iter() {
            if self.bound_textures[index] != program.bound_textures[index] {
                program.bind_texture(
                    &self.device,
                    desc_set,
                    &self.images[&self.bound_textures[index]].core,
                    sampler_name,
                );
                program.bound_textures[index] = self.bound_textures[index];
            }
        }
    }

    fn bind_samplers(&self, program: &Program<B>) {
        let desc_set = self.descriptor_pools_sampler.get(&program.shader_kind);
        for &(index, sampler_name) in SAMPLERS.iter() {
            let sampler = match self.bound_sampler[index] {
                TextureFilter::Linear | TextureFilter::Trilinear => &self.sampler_linear,
                TextureFilter::Nearest => &self.sampler_nearest,
            };
            program.bind_sampler(&self.device, desc_set, &sampler, sampler_name);
        }
    }

    pub fn update_indices<I: Copy>(&mut self, indices: &[I]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self
            .programs
            .get_mut(&self.bound_program)
            .expect("Program not found.");

        if let Some(ref mut index_buffer) = program.index_buffer {
            index_buffer[self.next_id].update(&self.device, indices);
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    pub fn update_vertices<T: Copy>(&mut self, vertices: &[T]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self
            .programs
            .get_mut(&self.bound_program)
            .expect("Program not found.");

        if program.shader_name.contains("debug") {
            program.vertex_buffer[self.next_id].update(&self.device, vertices);
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    pub fn set_uniforms(&mut self, program: &ProgramId, transform: &Transform3D<f32>) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        assert_eq!(*program, self.bound_program);
        let program = self
            .programs
            .get_mut(&self.bound_program)
            .expect("Program not found.");
        let desc_set = &self.descriptor_pools[self.next_id].get(&program.shader_kind);
        program.bind_locals(
            &self.device,
            desc_set,
            transform,
            self.program_mode_id,
            self.next_id,
        );
    }

    fn update_instances<T: Copy>(&mut self, instances: &[T]) {
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        self.programs
            .get_mut(&self.bound_program)
            .expect("Program not found.")
            .bind_instances(&self.device, instances, self.next_id);
    }

    fn draw(&mut self) {
        let (fb, format) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            (
                &self.fbos[&self.bound_draw_fbo].fbo,
                self.fbos[&self.bound_draw_fbo].format,
            )
        } else {
            if self.current_depth_test == DepthTest::Off {
                (
                    &self.framebuffers[self.current_frame_id],
                    ImageFormat::BGRA8,
                )
            } else {
                (
                    &self.framebuffers_depth[self.current_frame_id],
                    ImageFormat::BGRA8,
                )
            }
        };
        let rp = self
            .render_pass
            .as_ref()
            .unwrap()
            .get_render_pass(format, self.current_depth_test != DepthTest::Off);

        self.programs
            .get_mut(&self.bound_program)
            .expect("Program not found")
            .submit(
                &mut self.command_pool[self.next_id],
                self.viewport.clone(),
                rp,
                &fb,
                &mut self.descriptor_pools[self.next_id],
                &self.descriptor_pools_global,
                &self.descriptor_pools_sampler,
                &vec![],
                self.current_blend_state.get(),
                self.blend_color.get(),
                self.current_depth_test,
                self.scissor_rect,
                self.next_id,
                &self.pipeline_layouts,
            );
    }

    pub fn begin_frame(&mut self) -> GpuFrameId {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;

        self.bound_textures = [0; 16];
        self.bound_sampler = [TextureFilter::Linear; 16];
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;
        self.program_mode_id = 0;

        self.frame_id
    }

    fn bind_texture_impl(&mut self, slot: TextureSlot, id: u32, sampler: TextureFilter) {
        debug_assert!(self.inside_frame);

        if self.bound_textures[slot.0] != id {
            self.bound_textures[slot.0] = id;
            self.bound_sampler[slot.0] = sampler;
        }
    }

    pub fn bind_texture<S>(&mut self, sampler: S, texture: &Texture)
    where
        S: Into<TextureSlot>,
    {
        self.bind_texture_impl(sampler.into(), texture.id, texture.filter);
        texture.bound_in_frame.set(self.frame_id);
    }

    pub fn bind_external_texture<S>(&mut self, sampler: S, external_texture: &ExternalTexture)
    where
        S: Into<TextureSlot>,
    {
        self.bind_texture_impl(sampler.into(), external_texture.id, TextureFilter::Linear);
    }

    pub fn bind_read_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);
        if self.bound_read_fbo != fbo_id {
            self.bound_read_fbo = fbo_id;
        }
    }

    pub fn bind_read_target(&mut self, read_target: ReadTarget) {
        let fbo_id = match read_target {
            ReadTarget::Default => DEFAULT_READ_FBO,
            ReadTarget::Texture { texture, layer } => texture.fbos[layer],
        };
        self.bind_read_target_impl(fbo_id)
    }

    #[cfg(feature = "capture")]
    fn bind_read_texture(&mut self, texture_id: TextureId, layer_id: i32) {
        self.bound_read_texture = (texture_id, layer_id);
    }

    fn bind_draw_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
        }
    }

    pub fn reset_read_target(&mut self) {
        self.bind_read_target_impl(DEFAULT_READ_FBO);
    }


    pub fn reset_draw_target(&mut self) {
        self.bind_draw_target_impl(DEFAULT_DRAW_FBO);
        self.depth_available = true;
    }

    pub fn bind_draw_target(
        &mut self,
        texture_target: DrawTarget,
    ) {
        let (fbo_id, dimensions, depth_available) = match texture_target {
            DrawTarget::Default(dim) => (DEFAULT_DRAW_FBO, dim, true),
            DrawTarget::Texture { texture, layer, with_depth } => {
                texture.bound_in_frame.set(self.frame_id);
                let fbo_id = if with_depth {
                    texture.fbos_with_depth[layer]
                } else {
                    texture.fbos[layer]
                };
                let cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer();
                let rbos = &self.rbos;

                unsafe {
                    cmd_buffer.begin();
                    if let Some(barrier) = self.images[&texture.id].core.transit(
                        hal::image::Access::COLOR_ATTACHMENT_READ
                            | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                        hal::image::Layout::ColorAttachmentOptimal,
                        hal::image::SubresourceRange {
                            aspects: hal::format::Aspects::COLOR,
                            levels: 0 .. 1,
                            layers: layer as _ .. (layer + 1) as _,
                        },
                    ) {
                        cmd_buffer.pipeline_barrier(
                            PipelineStage::COLOR_ATTACHMENT_OUTPUT
                                .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                            hal::memory::Dependencies::empty(),
                            &[barrier],
                        );
                    }

                    if with_depth {
                        let rbo_id = self.fbos[&fbo_id].rbo;
                        if let Some(barrier) = rbos[&rbo_id].core.transit(
                            hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                                | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                            hal::image::Layout::DepthStencilAttachmentOptimal,
                            rbos[&rbo_id].core.subresource_range.clone(),
                        ) {
                            cmd_buffer.pipeline_barrier(
                                PipelineStage::EARLY_FRAGMENT_TESTS
                                    .. PipelineStage::LATE_FRAGMENT_TESTS,
                                hal::memory::Dependencies::empty(),
                                &[barrier],
                            );
                        }
                    }
                    cmd_buffer.finish();
                }
                (fbo_id, texture.get_dimensions(), with_depth)
            },
        };
        self.depth_available = depth_available;
        self.bind_draw_target_impl(fbo_id);
        self.viewport.rect = hal::pso::Rect {
            x: 0,
            y: 0,
            w: dimensions.width as _,
            h: dimensions.height as _,
        };
    }

    pub fn create_fbo_for_external_texture(&mut self, _texture_id: u32) -> FBOId {
        warn!("External texture creation is missing");
        FBOId(0)
    }

    pub fn create_fbo(&mut self) -> FBOId {
        DEBUG_READ_FBO
    }

    pub fn delete_fbo(&mut self, _fbo: FBOId) {
        warn!("delete fbo is missing");
    }

    pub fn bind_external_draw_target(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
        }
    }

    fn generate_texture_id(&self) -> TextureId {
        let mut rng = rand::thread_rng();
        let mut texture_id = INVALID_TEXTURE_ID + 1;
        while self.images.contains_key(&texture_id) {
            texture_id = rng.gen_range::<u32>(INVALID_TEXTURE_ID + 1, u32::max_value());
        }
        texture_id
    }

    fn generate_program_id(&self) -> ProgramId {
        let mut rng = rand::thread_rng();
        let mut program_id = ProgramId(INVALID_PROGRAM_ID.0 + 1);
        while self.programs.contains_key(&program_id) {
            program_id =
                ProgramId(rng.gen_range::<u32>(INVALID_PROGRAM_ID.0 + 1, u32::max_value()));
        }
        program_id
    }

    fn generate_fbo_ids(&mut self, count: i32) -> Vec<FBOId> {
        let mut rng = rand::thread_rng();
        let mut fboids = vec![];
        let mut fbo_id = FBOId(DEFAULT_DRAW_FBO.0 + 1);
        for _ in 0 .. count {
            while self.fbos.contains_key(&fbo_id) || fboids.contains(&fbo_id) {
                fbo_id = FBOId(rng.gen_range::<u32>(DEFAULT_DRAW_FBO.0 + 1, u32::max_value()));
            }
            fboids.push(fbo_id);
        }
        fboids
    }

    fn generate_rbo_id(&mut self) -> RBOId {
        let mut rng = rand::thread_rng();
        let mut rbo_id = RBOId(1); // 0 is used for invalid
        while self.rbos.contains_key(&rbo_id) {
            rbo_id = RBOId(rng.gen_range::<u32>(1, u32::max_value()));
        }
        rbo_id
    }

    pub fn create_texture(
        &mut self,
        target: TextureTarget,
        format: ImageFormat,
        mut width: i32,
        mut height: i32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
        layer_count: i32,
    ) -> Texture {
        debug_assert!(self.inside_frame);
        assert!(!(width == 0 || height == 0 || layer_count == 0));

        if width > self.max_texture_size || height > self.max_texture_size {
            error!("Attempting to allocate a texture of size {}x{} above the limit, trimming", width, height);
            width = width.min(self.max_texture_size);
            height = height.min(self.max_texture_size);
        }

        // Set up the texture book-keeping.
        let mut texture = Texture {
            id: self.generate_texture_id(),
            target: target as _,
            size: DeviceIntSize::new(width, height),
            layer_count,
            format,
            filter,
            fbos: vec![],
            fbos_with_depth: vec![],
            last_frame_used: self.frame_id,
            bound_in_frame: Cell::new(GpuFrameId(0)),
            flags: TextureFlags::default(),
        };

        assert!(!self.images.contains_key(&texture.id));
        let usage_base = hal::image::Usage::TRANSFER_SRC
            | hal::image::Usage::TRANSFER_DST
            | hal::image::Usage::SAMPLED;
        let (view_kind, mip_levels, usage) = match texture.filter {
            TextureFilter::Nearest => (
                hal::image::ViewKind::D2,
                1,
                usage_base | hal::image::Usage::COLOR_ATTACHMENT,
            ),
            TextureFilter::Linear => (
                hal::image::ViewKind::D2Array,
                1,
                usage_base | hal::image::Usage::COLOR_ATTACHMENT,
            ),
            TextureFilter::Trilinear => (
                hal::image::ViewKind::D2Array,
                (width as f32).max(height as f32).log2().floor() as u8 + 1,
                usage_base | hal::image::Usage::COLOR_ATTACHMENT,
            ),
        };
        let img = Image::new(
            &self.device,
            &self.memory_types,
            texture.format,
            texture.size.width,
            texture.size.height,
            texture.layer_count,
            view_kind,
            mip_levels,
            usage,
        );

        unsafe {
            let cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer();
            cmd_buffer.begin();

            if let Some(barrier) = img.core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ
                    | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                img.core.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT
                        .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.finish();
        }

        self.images.insert(texture.id, img);

        // Set up FBOs, if required.
        if let Some(rt_info) = render_target {
            self.init_fbos(&mut texture, false);
            if rt_info.has_depth {
                self.init_fbos(&mut texture, true);
            }
        }

        record_gpu_alloc(texture.size_in_bytes());

        texture
    }

    fn init_fbos(&mut self, texture: &mut Texture, with_depth: bool) {
        let new_fbos = self.generate_fbo_ids(texture.layer_count);
        let (rbo_id, depth) = if with_depth {
            let rbo_id = self.acquire_depth_target(texture.get_dimensions());
            (rbo_id, Some(&self.rbos[&rbo_id].core.view))
        } else {
            (RBOId(0), None)
        };

        for i in 0 .. texture.layer_count as u16 {
            let fbo = Framebuffer::new(
                &self.device,
                &texture,
                &self.images.get(&texture.id).expect(&format!("No image found with id {:?}", texture.id)),
                i,
                self.render_pass.as_ref().unwrap(),
                rbo_id.clone(),
                depth,
            );
            self.fbos.insert(new_fbos[i as usize], fbo);

            if with_depth {
                texture.fbos_with_depth.push(new_fbos[i as usize])
            } else {
                texture.fbos.push(new_fbos[i as usize])
            }
        }
    }

    /// Copies the contents from one renderable texture to another.
    pub fn blit_renderable_texture(&mut self, dst: &mut Texture, src: &Texture) {
        dst.bound_in_frame.set(self.frame_id);
        src.bound_in_frame.set(self.frame_id);

        debug_assert!(self.inside_frame);
        debug_assert!(dst.size.width >= src.size.width);
        debug_assert!(dst.size.height >= src.size.height);
        assert!(dst.layer_count >= src.layer_count);

        let rect = DeviceIntRect::new(DeviceIntPoint::zero(), src.get_dimensions().to_i32());
        let (src_img, layers) = (&self.images[&src.id].core, src.layer_count);
        let dst_img = &self.images[&dst.id].core;
        let cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer();

        let range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: 0 .. layers as _,
        };

        unsafe {
            cmd_buffer.begin();
            let barriers = src_img
                .transit(
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                    range.clone(),
                )
                .into_iter()
                .chain(dst_img.transit(
                    hal::image::Access::TRANSFER_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    range.clone(),
                ));

            cmd_buffer.pipeline_barrier(
                PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );

            cmd_buffer.copy_image(
                &src_img.image,
                hal::image::Layout::TransferSrcOptimal,
                &dst_img.image,
                hal::image::Layout::TransferDstOptimal,
                &[hal::command::ImageCopy {
                    src_subresource: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: 0 .. layers as _,
                    },
                    src_offset: hal::image::Offset {
                        x: rect.origin.x as i32,
                        y: rect.origin.y as i32,
                        z: 0,
                    },
                    dst_subresource: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: 0 .. layers as _,
                    },
                    dst_offset: hal::image::Offset {
                        x: rect.origin.x as i32,
                        y: rect.origin.y as i32,
                        z: 0,
                    },
                    extent: hal::image::Extent {
                        width: rect.size.width as u32,
                        height: rect.size.height as u32,
                        depth: 1,
                    },
                }],
            );

            // the blit caller code expects to be able to render to the target
            let barriers = src_img
                .transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ
                        | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::ColorAttachmentOptimal,
                    range.clone(),
                )
                .into_iter()
                .chain(dst_img.transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ
                        | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::ColorAttachmentOptimal,
                    range,
                ));

            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                hal::memory::Dependencies::empty(),
                barriers,
            );
            cmd_buffer.finish();
        }
    }

    fn generate_mipmaps(&mut self, texture: &Texture) {
        texture.bound_in_frame.set(self.frame_id);
        let cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer();

        let image = self
            .images
            .get_mut(&texture.id)
            .expect("Texture not found.");

        let mut mip_width = texture.size.width;
        let mut mip_height = texture.size.height;

        let mut half_mip_width = mip_width / 2;
        let mut half_mip_height = mip_height / 2;

        unsafe {
            cmd_buffer.begin();
            if let Some(barrier) = image.core.transit(
                hal::image::Access::TRANSFER_WRITE,
                hal::image::Layout::TransferDstOptimal,
                image.core.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            for index in 1 .. image.kind.num_levels() {
                if let Some(barrier) = image.core.transit(
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                    hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: index - 1 .. index,
                        layers: 0 .. 1,
                    },
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                cmd_buffer.blit_image(
                    &image.core.image,
                    hal::image::Layout::TransferSrcOptimal,
                    &image.core.image,
                    hal::image::Layout::TransferDstOptimal,
                    hal::image::Filter::Linear,
                    &[hal::command::ImageBlit {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: index - 1,
                            layers: 0 .. 1,
                        },
                        src_bounds: hal::image::Offset { x: 0, y: 0, z: 0 } .. hal::image::Offset {
                            x: mip_width as i32,
                            y: mip_height as i32,
                            z: 1,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: index,
                            layers: 0 .. 1,
                        },
                        dst_bounds: hal::image::Offset {
                            x: 0 as i32,
                            y: 0 as i32,
                            z: 0,
                        } .. hal::image::Offset {
                            x: half_mip_width as i32,
                            y: half_mip_height as i32,
                            z: 1,
                        },
                    }],
                );
                if let Some(barrier) = image.core.transit(
                    hal::image::Access::TRANSFER_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: index - 1 .. index,
                        layers: 0 .. 1,
                    },
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                mip_width = half_mip_width;
                if half_mip_width > 1 {
                    half_mip_width /= 2;
                }
                mip_height = half_mip_height;
                if half_mip_height > 1 {
                    half_mip_height /= 2;
                }
            }

            if let Some(barrier) = image.core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ
                    | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                image.core.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.finish();
        }
    }

    fn acquire_depth_target(&mut self, dimensions: DeviceIntSize) -> RBOId {
        if self.depth_targets.contains_key(&dimensions) {
            let target = self.depth_targets.get_mut(&dimensions).expect(&format!("No depth target found with dimensions {:?}", dimensions));
            target.refcount += 1;
            target.rbo_id
        } else {
            let rbo_id = self.generate_rbo_id();
            let rbo = DepthBuffer::new(
                &self.device,
                &self.memory_types,
                dimensions.width as _,
                dimensions.height as _,
                self.depth_format,
            );
            self.rbos.insert(rbo_id, rbo);
            let target = SharedDepthTarget {
                rbo_id,
                refcount: 1,
            };
            record_gpu_alloc(depth_target_size_in_bytes(&dimensions));
            self.depth_targets.insert(dimensions, target);
            rbo_id
        }
    }

    fn release_depth_target(&mut self, dimensions: DeviceIntSize) {
        let mut entry = match self.depth_targets.entry(dimensions) {
            Entry::Occupied(x) => x,
            Entry::Vacant(..) => panic!("Releasing unknown depth target"),
        };
        debug_assert!(entry.get().refcount != 0);
        entry.get_mut().refcount -= 1;
        if entry.get().refcount == 0 {
            let t = entry.remove();
            let old_rbo = self.rbos.remove(&t.rbo_id).expect(&format!("No rbo found with id {:?}", t.rbo_id));
            old_rbo.deinit(&self.device);
            record_gpu_free(depth_target_size_in_bytes(&dimensions));
        }
    }

    pub fn blit_render_target(&mut self, src_rect: DeviceIntRect, dest_rect: DeviceIntRect) {
        debug_assert!(self.inside_frame);

        let (src_img, src_layer) = if self.bound_read_fbo != DEFAULT_READ_FBO {
            let fbo = &self.fbos[&self.bound_read_fbo];
            let img = &self.images[&fbo.texture];
            let layer = fbo.layer_index;
            (&img.core, layer)
        } else {
            (&self.frame_images[self.current_frame_id], 0)
        };

        let (dest_img, dest_layer) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let fbo = &self.fbos[&self.bound_draw_fbo];
            let img = &self.images[&fbo.texture];
            let layer = fbo.layer_index;
            (&img.core, layer)
        } else {
            (&self.frame_images[self.current_frame_id], 0)
        };

        let cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer();

        let src_range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: src_layer .. src_layer + 1,
        };
        let dest_range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: dest_layer .. dest_layer + 1,
        };

        unsafe {
            cmd_buffer.begin();
            let barriers = src_img
                .transit(
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                    src_range.clone(),
                )
                .into_iter()
                .chain(dest_img.transit(
                    hal::image::Access::TRANSFER_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    dest_range.clone(),
                ));
            cmd_buffer.pipeline_barrier(
                PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );

            if src_rect.size != dest_rect.size {
                cmd_buffer.blit_image(
                    &src_img.image,
                    hal::image::Layout::TransferSrcOptimal,
                    &dest_img.image,
                    hal::image::Layout::TransferDstOptimal,
                    hal::image::Filter::Linear,
                    &[hal::command::ImageBlit {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: src_layer .. src_layer + 1,
                        },
                        src_bounds: hal::image::Offset {
                            x: src_rect.origin.x as i32,
                            y: src_rect.origin.y as i32,
                            z: 0,
                        } .. hal::image::Offset {
                            x: src_rect.origin.x as i32 + src_rect.size.width as i32,
                            y: src_rect.origin.y as i32 + src_rect.size.height as i32,
                            z: 1,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: dest_layer .. dest_layer + 1,
                        },
                        dst_bounds: hal::image::Offset {
                            x: dest_rect.origin.x as i32,
                            y: dest_rect.origin.y as i32,
                            z: 0,
                        } .. hal::image::Offset {
                            x: dest_rect.origin.x as i32 + dest_rect.size.width as i32,
                            y: dest_rect.origin.y as i32 + dest_rect.size.height as i32,
                            z: 1,
                        },
                    }],
                );
            } else {
                cmd_buffer.copy_image(
                    &src_img.image,
                    hal::image::Layout::TransferSrcOptimal,
                    &dest_img.image,
                    hal::image::Layout::TransferDstOptimal,
                    &[hal::command::ImageCopy {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: src_layer .. src_layer + 1,
                        },
                        src_offset: hal::image::Offset {
                            x: src_rect.origin.x as i32,
                            y: src_rect.origin.y as i32,
                            z: 0,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: dest_layer as _ .. (dest_layer + 1) as _,
                        },
                        dst_offset: hal::image::Offset {
                            x: dest_rect.origin.x as i32,
                            y: dest_rect.origin.y as i32,
                            z: 0,
                        },
                        extent: hal::image::Extent {
                            width: src_rect.size.width as u32,
                            height: src_rect.size.height as u32,
                            depth: 1,
                        },
                    }],
                );
            }

            // the blit caller code expects to be able to render to the target
            let barriers = src_img
                .transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ
                        | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::ColorAttachmentOptimal,
                    src_range,
                )
                .into_iter()
                .chain(dest_img.transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ
                        | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::ColorAttachmentOptimal,
                    dest_range,
                ));

            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                hal::memory::Dependencies::empty(),
                barriers,
            );
            cmd_buffer.finish();
        }
    }

    /// Performs a blit while flipping vertically. Useful for blitting textures
    /// (which use origin-bottom-left) to the main framebuffer (which uses
    /// origin-top-left).
    pub fn blit_render_target_invert_y(
        &mut self,
        _src_rect: DeviceIntRect,
        _dest_rect: DeviceIntRect,
    ) {
        debug_assert!(self.inside_frame);
        println!("blit_render_target_invert_y is not implemented yet!");
    }

    /// Notifies the device that the contents of a render target are no longer
    /// needed.
    ///
    /// FIXME(bholley): We could/should invalidate the depth targets earlier
    /// than the color targets, i.e. immediately after each pass.
    pub fn invalidate_render_target(&mut self, _texture: &Texture) {
        warn!("invalidate_render_target not implemented!");
    }

    /// Notifies the device that a render target is about to be reused.
    ///
    /// This method adds or removes a depth target as necessary.
    pub fn reuse_render_target<T: Texel>(
        &mut self,
        texture: &mut Texture,
        rt_info: RenderTargetInfo,
    ) {
        texture.last_frame_used = self.frame_id;

        // Add depth support if needed.
        if rt_info.has_depth && !texture.supports_depth() {
            self.init_fbos(texture, true);
        }
    }

    pub fn free_image(&mut self, texture: &mut Texture) {
        if texture.bound_in_frame.get() == self.frame_id {
            self.retained_textures.push(texture.clone());
            return;
        }

        if texture.still_in_flight(self.frame_id, self.frame_count) {
            self.wait_for_resources();
        }

        if texture.supports_depth() {
            self.release_depth_target(texture.get_dimensions());
        }

        if !texture.fbos_with_depth.is_empty() {
            for old in texture.fbos_with_depth.drain(..) {
                debug_assert!(self.bound_draw_fbo != old || self.bound_read_fbo != old);
                let old_fbo = self.fbos.remove(&old).expect(&format!("No fbo found with id {:?}", old));
                old_fbo.deinit(&self.device);
            }
        }

        if !texture.fbos.is_empty() {
            for old in texture.fbos.drain(..) {
                debug_assert!(self.bound_draw_fbo != old || self.bound_read_fbo != old);
                let old_fbo = self.fbos.remove(&old).expect(&format!("No fbo found with id {:?}", old));
                old_fbo.deinit(&self.device);
            }
        }

        let image = self.images.remove(&texture.id).expect("Texture not found.");
        record_gpu_free(texture.size_in_bytes());
        image.deinit(&self.device);
    }

    fn delete_retained_textures(&mut self) {
        let textures: Vec<_> = self.retained_textures.drain(..).collect();
        for ref mut texture in textures {
            self.free_image(texture);
            texture.id = 0;
        }
    }

    pub fn delete_texture(&mut self, mut texture: Texture) {
        debug_assert!(self.inside_frame);
        if texture.size.width + texture.size.height == 0 {
            return;
        }

        self.free_image(&mut texture);

        texture.size = DeviceIntSize::zero();
        texture.layer_count = 0;
        texture.id = 0;
    }

    #[cfg(feature = "replay")]
    pub fn delete_external_texture(&mut self, mut external: ExternalTexture) {
        warn!("delete external texture is missing");
        external.id = 0;
    }

    pub fn switch_mode(&mut self, mode: i32) {
        debug_assert!(self.inside_frame);
        self.program_mode_id = mode;
    }

    pub fn create_pbo(&mut self) -> PBO {
        PBO {}
    }

    pub fn delete_pbo(&mut self, _pbo: PBO) {}

    pub fn upload_texture<'a>(
        &'a mut self,
        texture: &'a Texture,
        _pbo: &PBO,
        _upload_count: usize,
    ) -> TextureUploader<'a, B> {
        debug_assert!(self.inside_frame);

        match self.upload_method {
            UploadMethod::Immediate => unimplemented!("UploadMethod::Immediate is not supported"),
            UploadMethod::PixelBuffer(..) => TextureUploader {
                device: self,
                texture,
            },
        }
    }

    pub fn upload_texture_immediate<T: Texel>(&mut self, texture: &Texture, pixels: &[T]) {
        texture.bound_in_frame.set(self.frame_id);
        let len = pixels.len() / texture.layer_count as usize;
        for i in 0 .. texture.layer_count {
            let start = len * i as usize;

            self.images
                .get_mut(&texture.id)
                .expect("Texture not found.")
                .update(
                    &self.device,
                    &mut self.command_pool[self.next_id],
                    &mut self.staging_buffer_pool[self.next_id],
                    DeviceIntRect::new(
                        DeviceIntPoint::new(0, 0),
                        texture.size,
                    ),
                    i,
                    texels_to_u8_slice(&pixels[start .. (start + len)]),
                );
        }
        if texture.filter == TextureFilter::Trilinear {
            self.generate_mipmaps(texture);
        }
    }

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    pub fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8> {
        let mut pixels = vec![0; (img_desc.size.width * img_desc.size.height * 4) as usize];
        self.read_pixels_into(
            DeviceIntRect::new(
                DeviceIntPoint::zero(),
                DeviceIntSize::new(img_desc.size.width, img_desc.size.height),
            ),
            ReadPixelsFormat::Rgba8,
            &mut pixels,
        );
        pixels
    }

    /// Read rectangle of pixels into the specified output slice.
    pub fn read_pixels_into(
        &mut self,
        rect: DeviceIntRect,
        format: ReadPixelsFormat,
        output: &mut [u8],
    ) {
        self.wait_for_resources();

        let bytes_per_pixel = match format {
            ReadPixelsFormat::Standard(imf) => imf.bytes_per_pixel(),
            ReadPixelsFormat::Rgba8 => 4,
        };
        let size_in_bytes = (bytes_per_pixel * rect.size.width * rect.size.height) as usize;
        assert_eq!(output.len(), size_in_bytes);
        let capture_read =
            cfg!(feature = "capture") && self.bound_read_texture.0 != INVALID_TEXTURE_ID;

        let (image, layer) = if capture_read {
            let img = &self.images[&self.bound_read_texture.0];
            (&img.core, self.bound_read_texture.1 as u16)
        } else if self.bound_read_fbo != DEFAULT_READ_FBO {
            let fbo = &self.fbos[&self.bound_read_fbo];
            let img = &self.images[&fbo.texture];
            let layer = fbo.layer_index;
            (&img.core, layer)
        } else {
            (&self.frame_images[self.current_frame_id], 0)
        };
        let download_buffer: Buffer<B> = Buffer::new(
            &self.device,
            &self.memory_types,
            hal::buffer::Usage::TRANSFER_DST,
            (self.limits.min_buffer_copy_pitch_alignment - 1) as usize,
            output.len(),
            1,
        );

        let mut command_pool = unsafe {
            self.device.create_command_pool_typed(
                &self.queue_group,
                hal::pool::CommandPoolCreateFlags::empty(),
            )
        }
        .expect("create_command_pool_typed failed");
        unsafe { command_pool.reset() };

        let mut cmd_buffer = command_pool.acquire_command_buffer::<hal::command::OneShot>();
        unsafe {
            cmd_buffer.begin();
            let range = hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
                layers: layer .. layer + 1,
            };
            let barriers = download_buffer
                .transit(hal::buffer::Access::TRANSFER_WRITE)
                .into_iter()
                .chain(image.transit(
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                    range.clone(),
                ));

            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );

            cmd_buffer.copy_image_to_buffer(
                &image.image,
                hal::image::Layout::TransferSrcOptimal,
                &download_buffer.buffer,
                &[hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: rect.size.width as u32,
                    buffer_height: rect.size.height as u32,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: layer .. layer + 1,
                    },
                    image_offset: hal::image::Offset {
                        x: rect.origin.x as i32,
                        y: rect.origin.y as i32,
                        z: 0,
                    },
                    image_extent: hal::image::Extent {
                        width: rect.size.width as _,
                        height: rect.size.height as _,
                        depth: 1 as _,
                    },
                }],
            );
            if let Some(barrier) = image.transit(
                hal::image::Access::empty(),
                hal::image::Layout::ColorAttachmentOptimal,
                range,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.finish();
        }

        let mut copy_fence = self
            .device
            .create_fence(false)
            .expect("create_fence failed");

        unsafe {
            self.device
                .reset_fence(&copy_fence)
                .expect("reset_fence failed");
            self.queue_group.queues[0]
                .submit_nosemaphores(Some(&cmd_buffer), Some(&mut copy_fence));
            self.device
                .wait_for_fence(&copy_fence, !0)
                .expect("wait_for_fence failed");
            self.device.destroy_fence(copy_fence);
        }

        let mut data = vec![0; download_buffer.buffer_size];
        if let Ok(reader) = unsafe {
            self.device.acquire_mapping_reader::<u8>(
                &download_buffer.memory,
                0 .. download_buffer.buffer_size as u64,
            )
        } {
            data[0 .. reader.len()].copy_from_slice(&reader);
            unsafe { self.device.release_mapping_reader(reader) };
        } else {
            panic!("Fail to read the download buffer!");
        }
        data.truncate(output.len());
        if !capture_read
            && self.surface_format.base_format().0 == hal::format::SurfaceType::B8_G8_R8_A8
        {
            let width = rect.size.width as usize;
            let height = rect.size.height as usize;
            let row_pitch: usize = bytes_per_pixel as usize * width;
            // Vertical flip the result and convert to RGBA
            for y in 0 .. height as usize {
                for x in 0 .. width as usize {
                    let offset: usize = y * row_pitch + x * 4;
                    let rev_offset: usize = (height - 1 - y) * row_pitch + x * 4;
                    output[offset + 0] = data[rev_offset + 2];
                    output[offset + 1] = data[rev_offset + 1];
                    output[offset + 2] = data[rev_offset + 0];
                    output[offset + 3] = data[rev_offset + 3];
                }
            }
        } else {
            output.swap_with_slice(&mut data);
        }

        download_buffer.deinit(&self.device);
        unsafe {
            command_pool.reset();
            self.device.destroy_command_pool(command_pool.into_raw());
        }
    }

    /// Get texels of a texture into the specified output slice.
    pub fn get_tex_image_into(
        &mut self,
        _texture: &Texture,
        _format: ImageFormat,
        _output: &mut [u8],
    ) {
        println!("get_tex_image_into is missing");
    }

    /// Attaches the provided texture to the current Read FBO binding.
    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    fn attach_read_texture_raw(&mut self, texture_id: u32, _target: TextureTarget, layer_id: i32) {
        self.bind_read_texture(texture_id, layer_id);
    }

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    pub fn attach_read_texture_external(
        &mut self,
        _texture_id: u32,
        _target: TextureTarget,
        _layer_id: i32,
    ) {
        println!("attach_read_texture_external is missing");
    }

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    pub fn attach_read_texture(&mut self, texture: &Texture, layer_id: i32) {
        self.attach_read_texture_raw(texture.id, texture.target.into(), layer_id)
    }

    pub fn invalidate_read_texture(&mut self) {
        self.bound_read_texture = (INVALID_TEXTURE_ID, 0);
    }

    pub fn bind_vao(&mut self, _vao: &VAO) {}

    pub fn create_vao(&mut self, _descriptor: &VertexDescriptor) -> VAO {
        VAO {}
    }

    pub fn delete_vao(&mut self, _vao: VAO) {}

    pub fn create_vao_with_new_instances(
        &mut self,
        _descriptor: &VertexDescriptor,
        _base_vao: &VAO,
    ) -> VAO {
        VAO {}
    }

    pub fn update_vao_main_vertices<V: Copy + PrimitiveType>(
        &mut self,
        _vao: &VAO,
        _vertices: &[V],
        _usage_hint: VertexUsageHint,
    ) {
        if cfg!(feature = "debug_renderer") && self.bound_program != INVALID_PROGRAM_ID {
            self.update_vertices(
                &_vertices
                    .iter()
                    .map(|v| v.to_primitive_type())
                    .collect::<Vec<_>>(),
            );
        }
    }

    pub fn update_vao_instances<V: PrimitiveType>(
        &mut self,
        _vao: &VAO,
        instances: &[V],
        _usage_hint: VertexUsageHint,
    ) {
        let data = instances
            .iter()
            .map(|pi| pi.to_primitive_type())
            .collect::<Vec<V::Primitive>>();
        self.update_instances(&data);
    }

    pub fn update_vao_indices<I: Copy>(
        &mut self,
        _vao: &VAO,
        _indices: &[I],
        _usage_hint: VertexUsageHint,
    ) {
        if self.bound_program != INVALID_PROGRAM_ID {
            self.update_indices(_indices);
        }
    }

    pub fn draw_triangles_u16(&mut self, _first_vertex: i32, _index_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn draw_triangles_u32(&mut self, _first_vertex: i32, _index_count: i32) {
        debug_assert!(self.inside_frame);
        self.bind_textures();
        self.draw();
    }

    pub fn draw_nonindexed_lines(&mut self, _first_vertex: i32, _vertex_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn draw_indexed_triangles_instanced_u16(
        &mut self,
        _index_count: i32,
        _instance_count: i32,
    ) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn end_frame(&mut self) {
        self.reset_draw_target();
        self.reset_read_target();

        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        self.frame_id.0 += 1;
    }

    fn clear_target_rect(
        &mut self,
        rect: DeviceIntRect,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
    ) {
        if color.is_none() && depth.is_none() {
            return;
        }
        let mut clears = Vec::new();
        let mut rects = Vec::new();
        if let Some(color) = color {
            clears.push(hal::command::AttachmentClear::Color {
                index: 0,
                value: hal::command::ClearColor::Float(color),
            });
            rects.push(hal::pso::ClearRect {
                rect: hal::pso::Rect {
                    x: rect.origin.x as i16,
                    y: rect.origin.y as i16,
                    w: rect.size.width as i16,
                    h: rect.size.height as i16,
                },
                layers: 0 .. 1,
            });
        }

        if let Some(depth) = depth {
            assert!(self.current_depth_test != DepthTest::Off);
            clears.push(hal::command::AttachmentClear::DepthStencil {
                depth: Some(depth),
                stencil: None,
            });
            rects.push(hal::pso::ClearRect {
                rect: hal::pso::Rect {
                    x: rect.origin.x as i16,
                    y: rect.origin.y as i16,
                    w: rect.size.width as i16,
                    h: rect.size.height as i16,
                },
                layers: 0 .. 1,
            });
        }

        let (frame_buffer, format, has_depth) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            (
                &self.fbos[&self.bound_draw_fbo].fbo,
                self.fbos[&self.bound_draw_fbo].format,
                self.fbos[&self.bound_draw_fbo].rbo != RBOId(0),
            )
        } else {
            if self.current_depth_test == DepthTest::Off {
                (
                    &self.framebuffers[self.current_frame_id],
                    ImageFormat::BGRA8,
                    false,
                )
            } else {
                (
                    &self.framebuffers_depth[self.current_frame_id],
                    ImageFormat::BGRA8,
                    true,
                )
            }
        };

        let render_pass = self
            .render_pass
            .as_ref()
            .unwrap()
            .get_render_pass(format, has_depth);

        let cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer();
        unsafe {
            cmd_buffer.begin();
            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    render_pass,
                    frame_buffer,
                    self.viewport.rect,
                    &vec![],
                );

                encoder.clear_attachments(clears, rects);
            }
            cmd_buffer.finish();
        }
    }

    fn clear_target_image(&mut self, color: Option<[f32; 4]>, depth: Option<f32>) {
        let (img, layer, dimg) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let fbo = &self.fbos[&self.bound_draw_fbo];
            let img = &self.images[&fbo.texture];
            let dimg = if depth.is_some() {
                Some(&self.rbos[&fbo.rbo].core)
            } else {
                None
            };
            (&img.core, fbo.layer_index, dimg)
        } else {
            (
                &self.frame_images[self.current_frame_id],
                0,
                Some(&self.frame_depths[self.current_frame_id].core),
            )
        };

        let cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer();
        //Note: this function is assumed to be called within an active FBO
        // thus, we bring back the targets into renderable state

        unsafe {
            cmd_buffer.begin();
            if let Some(color) = color {
                let color_range = hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: 0 .. 1,
                    layers: layer .. layer + 1,
                };
                if let Some(barrier) = img.transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ
                        | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    color_range.clone(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::COLOR_ATTACHMENT_OUTPUT
                            .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                cmd_buffer.clear_image(
                    &img.image,
                    hal::image::Layout::TransferDstOptimal,
                    hal::command::ClearColor::Float([color[0], color[1], color[2], color[3]]),
                    hal::command::ClearDepthStencil(0.0, 0),
                    Some(color_range.clone()),
                );
                if let Some(barrier) = img.transit(
                    hal::image::Access::empty(),
                    hal::image::Layout::ColorAttachmentOptimal,
                    color_range.clone(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::COLOR_ATTACHMENT_OUTPUT
                            .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }

            if let (Some(depth), Some(dimg)) = (depth, dimg) {
                assert_ne!(self.current_depth_test, DepthTest::Off);
                if let Some(barrier) = dimg.transit(
                    hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    dimg.subresource_range.clone(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::LATE_FRAGMENT_TESTS,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                cmd_buffer.clear_image(
                    &dimg.image,
                    hal::image::Layout::TransferDstOptimal,
                    hal::command::ClearColor::Float([0.0; 4]),
                    hal::command::ClearDepthStencil(depth, 0),
                    Some(dimg.subresource_range.clone()),
                );
                if let Some(barrier) = dimg.transit(
                    hal::image::Access::empty(),
                    hal::image::Layout::DepthStencilAttachmentOptimal,
                    dimg.subresource_range.clone(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::LATE_FRAGMENT_TESTS,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }
            cmd_buffer.finish();
        }
    }

    pub fn clear_target(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<DeviceIntRect>,
    ) {
        if let Some(rect) = rect {
            let target_rect = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
                let extent = &self.images[&self.fbos[&self.bound_draw_fbo].texture]
                    .kind
                    .extent();
                DeviceIntRect::new(
                    DeviceIntPoint::zero(),
                    DeviceIntSize::new(extent.width as _, extent.height as _),
                )
            } else {
                DeviceIntRect::new(
                    DeviceIntPoint::zero(),
                    DeviceIntSize::new(self.viewport.rect.w as _, self.viewport.rect.h as _),
                )
            };
            if rect.size.width > target_rect.size.width
                || rect.size.height > target_rect.size.height
            {
                // This can happen, when we resize
                self.clear_target_image(color, depth);
            } else if rect == target_rect {
                self.clear_target_image(color, depth);
            } else {
                self.clear_target_rect(rect, color, depth);
            }
        } else {
            self.clear_target_image(color, depth);
        }
    }

    pub fn enable_depth(&mut self) {
        assert!(self.depth_available, "Enabling depth test without depth target");
        self.current_depth_test = LESS_EQUAL_TEST;
    }

    pub fn disable_depth(&mut self) {
        self.current_depth_test = DepthTest::Off;
    }

    pub fn set_depth_func(&mut self, _depth_func: DepthFunction) {
        // TODO add Less depth function
        //self.current_depth_test = depth_func;
    }

    pub fn enable_depth_write(&mut self) {
        assert!(self.depth_available, "Enabling depth test without depth target");
        self.current_depth_test = LESS_EQUAL_WRITE;
    }

    pub fn disable_depth_write(&mut self) {
        if self.current_depth_test != DepthTest::Off {
            self.current_depth_test = LESS_EQUAL_TEST;
        }
    }

    pub fn disable_stencil(&self) {
        warn!("disable stencil is missing")
    }

    pub fn set_scissor_rect(&mut self, rect: DeviceIntRect) {
        self.scissor_rect = Some(rect);
    }

    pub fn enable_scissor(&self) {}

    pub fn disable_scissor(&mut self) {
        self.scissor_rect = None;
    }

    pub fn set_blend(&self, enable: bool) {
        if !enable {
            self.current_blend_state.set(BlendState::Off)
        }
    }

    pub fn set_blend_mode_alpha(&self) { self.current_blend_state.set(ALPHA); }

    pub fn set_blend_mode_premultiplied_alpha(&self) { self.current_blend_state.set(BlendState::PREMULTIPLIED_ALPHA); }

    pub fn set_blend_mode_premultiplied_dest_out(&self) { self.current_blend_state.set(PREMULTIPLIED_DEST_OUT); }

    pub fn set_blend_mode_multiply(&self) { self.current_blend_state.set(BlendState::MULTIPLY); }

    pub fn set_blend_mode_max(&self) { self.current_blend_state.set(MAX); }

    pub fn set_blend_mode_min(&self) { self.current_blend_state.set(MIN); }

    pub fn set_blend_mode_subpixel_pass0(&self) { self.current_blend_state.set(SUBPIXEL_PASS0); }

    pub fn set_blend_mode_subpixel_pass1(&self) { self.current_blend_state.set(SUBPIXEL_PASS1); }

    pub fn set_blend_mode_subpixel_with_bg_color_pass0(&self) { self.current_blend_state.set(SUBPIXEL_WITH_BG_COLOR_PASS0); }

    pub fn set_blend_mode_subpixel_with_bg_color_pass1(&self) { self.current_blend_state.set(SUBPIXEL_WITH_BG_COLOR_PASS1); }

    pub fn set_blend_mode_subpixel_with_bg_color_pass2(&self) { self.current_blend_state.set(SUBPIXEL_WITH_BG_COLOR_PASS2); }

    pub fn set_blend_mode_subpixel_constant_text_color(&self, color: ColorF) {
        self.current_blend_state.set(SUBPIXEL_CONSTANT_TEXT_COLOR);
        // color is an unpremultiplied color.
        self.blend_color.set(ColorF::new(color.r, color.g, color.b, 1.0));
    }

    pub fn set_blend_mode_subpixel_dual_source(&self) { self.current_blend_state.set(SUBPIXEL_DUAL_SOURCE); }

    pub fn set_blend_mode_show_overdraw(&self) { self.current_blend_state.set(OVERDRAW); }

    pub fn supports_features(&self, features: hal::Features) -> bool {
        self.features.contains(features)
    }

    pub fn echo_driver_messages(&self) {
        warn!("echo_driver_messages is unimplemeneted");
    }

    pub fn set_next_frame_id(&mut self) -> bool {
        unsafe {
            match self.swap_chain.as_mut().unwrap().acquire_image(
                !0,
                FrameSync::Semaphore(&mut self.image_available_semaphore),
            ) {
                Ok(id) => {
                    self.current_frame_id = id as _;
                    true
                }
                Err(_) => false,
            }
        }
    }

    pub fn submit_to_gpu(&mut self) -> bool {
        if self.frame_id.0 / 30 == 0 {
            println!("#### Gfx submit");            
        }
        {
            let cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer();
            let image = &self.frame_images[self.current_frame_id];
            unsafe {
                cmd_buffer.begin();
                if let Some(barrier) = image.transit(
                    hal::image::Access::empty(),
                    hal::image::Layout::Present,
                    image.subresource_range.clone(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::COLOR_ATTACHMENT_OUTPUT
                            .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                cmd_buffer.finish();
            }
        }

        let present_error = unsafe {
            let submission = Submission {
                command_buffers: self.command_pool[self.next_id].command_buffers(),
                wait_semaphores: Some((
                    &self.image_available_semaphore,
                    PipelineStage::BOTTOM_OF_PIPE,
                )),
                signal_semaphores: Some(&self.render_finished_semaphore),
            };
            self.queue_group.queues[0]
                .submit(submission, Some(&mut self.frame_fence[self.next_id].inner));
            self.frame_fence[self.next_id].is_submitted = true;

            // present frame
            self.swap_chain
                .as_mut()
                .unwrap()
                .present(
                    &mut self.queue_group.queues[0],
                    self.current_frame_id as _,
                    Some(&self.render_finished_semaphore),
                )
                .is_err()
        };
        self.next_id = (self.next_id + 1) % self.frame_count;
        self.reset_state();
        if self.frame_fence[self.next_id].is_submitted {
            unsafe {
                self.device
                    .wait_for_fence(&self.frame_fence[self.next_id].inner, !0)
            }
            .expect("wait_for_fence failed");
            unsafe {
                self.device
                    .reset_fence(&self.frame_fence[self.next_id].inner)
            }
            .expect("reset_fence failed");
            self.frame_fence[self.next_id].is_submitted = false;
        }
        unsafe {
            self.command_pool[self.next_id].reset();
        }
        self.staging_buffer_pool[self.next_id].reset();
        self.descriptor_pools[self.next_id].reset();
        self.reset_program_buffer_offsets();
        self.delete_retained_textures();
        return !present_error;
    }

    pub fn wait_for_resources_and_reset(&mut self) {
        self.wait_for_resources();
        self.reset_command_pools();
    }

    fn wait_for_resources(&mut self) {
        for fence in &mut self.frame_fence {
            if fence.is_submitted {
                unsafe { self.device.wait_for_fence(&fence.inner, !0) }
                    .expect("wait_for_fence failed");
                unsafe { self.device.reset_fence(&fence.inner) }.expect("reset_fence failed");
                fence.is_submitted = false;
            }
        }
    }

    fn reset_command_pools(&mut self) {
        for command_pool in &mut self.command_pool {
            unsafe { command_pool.reset() };
        }
    }

    /// Generates a memory report for the resources managed by the device layer.
    pub fn report_memory(&self) -> MemoryReport {
        let mut report = MemoryReport::default();
        for dim in self.depth_targets.keys() {
            report.depth_target_textures += depth_target_size_in_bytes(dim);
        }
        report
    }

    pub fn deinit(self) {
        for mut texture in self.retained_textures {
            texture.id = 0;
        }
        unsafe {
            if self.save_cache && self.cache_path.is_some() {
                let pipeline_cache = self.device.create_pipeline_cache(None).expect("Failed to create pipeline cache");
                let data;
                if let Some(cache) = self.pipeline_cache {
                    self.device.merge_pipeline_caches(&cache, Some(&pipeline_cache)).expect("merge_pipeline_caches failed");
                    data = self.device.get_pipeline_cache_data(&cache).expect("get_pipeline_cache_data failed");
                    self.device.destroy_pipeline_cache(cache);
                } else {
                    data = self.device.get_pipeline_cache_data(&pipeline_cache).expect("get_pipeline_cache_data failed");
                };
                self.device.destroy_pipeline_cache(pipeline_cache);

                if data.len() > 0 {
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .open(&self.cache_path.as_ref().expect("Cache path not set"))
                        .expect("File open/creation failed");

                    file.write(&data).expect("File write failed");
                }
            } else {
                if let Some(cache) = self.pipeline_cache {
                    self.device.destroy_pipeline_cache(cache);
                }
            }

            for command_pool in self.command_pool {
                command_pool.destroy(&self.device);
            }
            for mut staging_buffer_pool in self.staging_buffer_pool {
                staging_buffer_pool.deinit(&self.device);
            }
            for image in self.frame_images {
                image.deinit(&self.device);
            }
            for depth in self.frame_depths {
                depth.deinit(&self.device);
            }
            for (_, image) in self.images {
                image.deinit(&self.device);
            }
            for (_, rbo) in self.fbos {
                rbo.deinit(&self.device);
            }
            for (_, rbo) in self.rbos {
                rbo.deinit(&self.device);
            }
            for framebuffer in self.framebuffers {
                self.device.destroy_framebuffer(framebuffer);
            }
            for framebuffer_depth in self.framebuffers_depth {
                self.device.destroy_framebuffer(framebuffer_depth);
            }
            self.device.destroy_sampler(self.sampler_linear);
            self.device.destroy_sampler(self.sampler_nearest);
            for descriptor_pool in self.descriptor_pools {
                descriptor_pool.deinit(&self.device);
            }
            self.descriptor_pools_global.deinit(&self.device);
            self.descriptor_pools_sampler.deinit(&self.device);
            for (_, program) in self.programs {
                program.deinit(&self.device)
            }
            for (_, (vs_module, fs_module)) in self.shader_modules {
                self.device.destroy_shader_module(vs_module);
                self.device.destroy_shader_module(fs_module);
            }
            for (_, layout) in self.pipeline_layouts {
                self.device.destroy_pipeline_layout(layout);
            }
            self.render_pass.unwrap().deinit(&self.device);
            for fence in self.frame_fence {
                self.device.destroy_fence(fence.inner);
            }
            self.device
                .destroy_semaphore(self.image_available_semaphore);
            self.device
                .destroy_semaphore(self.render_finished_semaphore);
            self.device.destroy_swapchain(self.swap_chain.unwrap());
        }
    }
}

pub struct TextureUploader<'a, B: hal::Backend> {
    device: &'a mut Device<B>,
    texture: &'a Texture,
}

impl<'a, B: hal::Backend> TextureUploader<'a, B> {
    pub fn upload<T>(
        &mut self,
        rect: DeviceIntRect,
        layer_index: i32,
        stride: Option<i32>,
        data: &[T],
    ) -> usize {
        let data = unsafe {
            slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * mem::size_of::<T>())
        };
        let data_stride: usize = self.texture.format.bytes_per_pixel() as usize;
        let width = rect.size.width as usize;
        let height = rect.size.height as usize;
        let size = width * height * data_stride;
        let mut new_data = vec![0u8; size];
        let data = if stride.is_some() {
            let row_length = (stride.unwrap()) as usize;

            if data_stride == 4 {
                for j in 0 .. height {
                    for i in 0 .. width {
                        let offset = i * data_stride + j * data_stride * width;
                        let src = &data[j * row_length + i * data_stride ..];
                        assert!(
                            offset + 3 < new_data.len(),
                            "offset={:?}, data len={:?}, data stride={:?}",
                            offset,
                            new_data.len(),
                            data_stride,
                        ); // optimization
                           // convert from BGRA
                        new_data[offset + 0] = src[0];
                        new_data[offset + 1] = src[1];
                        new_data[offset + 2] = src[2];
                        new_data[offset + 3] = src[3];
                    }
                }
            } else {
                for j in 0 .. height {
                    for i in 0 .. width {
                        let offset = i * data_stride + j * data_stride * width;
                        let src = &data[j * row_length + i * data_stride ..];
                        for i in 0 .. data_stride {
                            new_data[offset + i] = src[i];
                        }
                    }
                }
            }

            new_data.as_slice()
        } else {
            data
        };
        assert_eq!(
            data.len(),
            width * height * data_stride,
            "data len = {}, width = {}, height = {}, data stride = {}",
            data.len(),
            width,
            height,
            data_stride
        );

        self.texture.bound_in_frame.set(self.device.frame_id);

        self.device
            .images
            .get_mut(&self.texture.id)
            .expect("Texture not found.")
            .update(
                &self.device.device,
                &mut self.device.command_pool[self.device.next_id],
                &mut self.device.staging_buffer_pool[self.device.next_id],
                rect,
                layer_index,
                data,
            );

        if self.texture.filter == TextureFilter::Trilinear {
            self.device.generate_mipmaps(self.texture);
        }
        size
    }
}

fn texels_to_u8_slice<T: Texel>(texels: &[T]) -> &[u8] {
    unsafe {
        slice::from_raw_parts(
            texels.as_ptr() as *const u8,
            texels.len() * mem::size_of::<T>(),
        )
    }
}
