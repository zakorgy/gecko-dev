/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, ImageFormat};
use api::{DeviceIntPoint, DeviceIntRect, DeviceIntSize, DeviceUintPoint, DeviceUintRect, DeviceUintSize};
use api::TextureTarget;
#[cfg(any(feature = "debug_renderer", feature="capture"))]
use api::ImageDescriptor;
use euclid::Transform3D;
use gpu_types;
use internal_types::{FastHashMap, RenderTargetInfo};
use rand::{self, Rng};
use ron::de::from_reader;
use smallvec::SmallVec;
use std::cell::Cell;
#[cfg(feature = "debug_renderer")]
use std::convert::Into;
use std::collections::HashMap;
use std::fs::File;
use std::mem;
use std::path::PathBuf;
use std::rc::Rc;
use std::slice;
use std::sync::Arc;
#[cfg(feature = "debug_renderer")]
use super::Capabilities;
use super::{ShaderKind,VertexArrayKind, ExternalTexture, FrameId, TextureSlot, TextureFilter};
use super::{VertexDescriptor, UploadMethod, Texel, ReadPixelsFormat, FileWatcherHandler};
use super::{Texture, FBOId, RBOId, VertexUsageHint, ShaderError};
use vertex_types::*;

use hal;

// gfx-hal
use hal::pso::{AttributeDesc, DescriptorRangeDesc, DescriptorSetLayoutBinding, VertexBufferDesc};
use hal::pso::{BlendState, BlendOp, Comparison, DepthTest, Factor};
use hal::{Device as BackendDevice, PhysicalDevice, QueueFamily, Surface, Swapchain};
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
pub const MAX_FRAME_COUNT: usize = 1;

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
    pub window_size: (u32, u32),
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct Locals {
    uTransform: [[f32; 4]; 4],
    uDevicePixelRatio: f32,
    uMode: i32,
}

#[derive(Clone, Deserialize)]
pub struct PipelineRequirements {
    pub attribute_descriptors: Vec<AttributeDesc>,
    pub bindings_map: HashMap<String, u32>,
    pub descriptor_range_descriptors: Vec<DescriptorRangeDesc>,
    pub descriptor_set_layouts: Vec<DescriptorSetLayoutBinding>,
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

fn get_shader_source(filename: &str, extension: &str) -> Vec<u8> {
    use std::io::Read;
    let path_str = format!("{}/{}{}", env!("OUT_DIR"), filename, extension);
    let mut file = File::open(path_str).expect(&format!("Unable to open shader file: {}", filename));
    let mut shader = Vec::new();
    file.read_to_end(&mut shader).unwrap();
    shader
}

#[repr(u32)]
pub enum DepthFunction {
    #[cfg(feature = "debug_renderer")]
    Less,
    LessEqual,
}

pub trait PrimitiveType {
    type Primitive: Clone + Copy;
    fn to_primitive_type(&self) -> Self::Primitive;
}

impl Texture {
    pub fn still_in_flight(&self, frame_id: FrameId) -> bool {
        for i in 0..MAX_FRAME_COUNT {
            if self.bound_in_frame.get() == FrameId(frame_id.0 - i) {
                return true
            }
        }
        false
    }
}

impl PrimitiveType for gpu_types::BlurInstance {
    type Primitive = BlurInstance;
    fn to_primitive_type(&self) -> BlurInstance {
        BlurInstance {
            aData: [0,0,0,0],
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
            aRect: [self.local_rect.origin.x, self.local_rect.origin.y, self.local_rect.size.width, self.local_rect.size.height],
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
            aClipSegment: self.segment,
            aClipDataResourceAddress: [
                self.clip_data_address.u as i32,
                self.clip_data_address.v as i32,
                self.resource_address.u as i32,
                self.resource_address.v as i32,
            ],
        }
    }
}

impl PrimitiveType for gpu_types::PrimitiveInstanceData {
    type Primitive = PrimitiveInstanceData;
    fn to_primitive_type(&self) -> PrimitiveInstanceData {
        PrimitiveInstanceData {
            aData: self.data,
        }
    }
}

impl PrimitiveType for gpu_types::ScalingInstance {
    type Primitive = ScalingInstance;
    fn to_primitive_type(&self) -> ScalingInstance {
        ScalingInstance {
            aData: [0,0,0,0],
            aScaleRenderTaskAddress: self.task_address.0 as i32,
            aScaleSourceTaskAddress: self.src_task_address.0 as i32,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct ProgramId(u32);

pub struct PBO;
pub struct VAO;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramSources;

#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramBinary;

pub trait ProgramCacheObserver {
    fn notify_binary_added(&self, program_binary: &Arc<ProgramBinary>);
    fn notify_program_binary_failed(&self, program_binary: &Arc<ProgramBinary>);
}

impl ShaderKind {
    fn is_debug(&self) -> bool {
        match *self {
            #[cfg(feature = "debug_renderer")]
            ShaderKind::DebugFont | ShaderKind::DebugColor => true,
            _ => false,
        }
    }
}

pub struct ProgramCache;

impl ProgramCache {
    pub fn new() -> Rc<Self> {
        Rc::new(ProgramCache {})
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

#[cfg(feature = "debug_renderer")]
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
        let view = device.create_image_view(
            &image,
            view_kind,
            format,
            hal::format::Swizzle::NO,
            subresource_range.clone(),
        ).unwrap();
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
        let image_unbound = device
            .create_image(kind, mip_levels, format, hal::image::Tiling::Optimal, usage, hal::image::ViewCapabilities::empty())
            .unwrap();
        let requirements = device.get_image_requirements(&image_unbound);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                requirements.type_mask & (1 << id) != 0 &&
                    mem_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let memory = device.allocate_memory(mem_type, requirements.size).unwrap();
        let image = device.bind_image_memory(&memory, 0, image_unbound).unwrap();

        ImageCore {
            memory: Some(memory),
            .. Self::from_image(device, image, view_kind, format, subresource_range)
        }
    }

    fn _reset(&self) {
        self.state.set((hal::image::Access::empty(), hal::image::Layout::Undefined));
    }

    fn deinit(self, device: &B::Device) {
        device.destroy_image_view(self.view);
        if let Some(memory) = self.memory {
            device.destroy_image(self.image);
            device.free_memory(memory);
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
        image_width: u32,
        image_height: u32,
        image_depth: i32,
        view_kind: hal::image::ViewKind,
        mip_levels: hal::image::Level,
        usage: hal::image::Usage,
    ) -> Self {
        let format = match image_format {
            ImageFormat::R8 => hal::format::Format::R8Unorm,
            ImageFormat::R16 => hal::format::Format::R16Unorm,
            ImageFormat::RG8 => hal::format::Format::Rg8Unorm,
            ImageFormat::BGRA8 => hal::format::Format::Bgra8Unorm,
            ImageFormat::RGBAF32 => hal::format::Format::Rgba32Float,
            ImageFormat::RGBAI32 => hal::format::Format::Rgba32Int,
        };
        let kind = hal::image::Kind::D2(
            image_width as _,
            image_height as _,
            image_depth as _,
            1,
        );

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
        &mut self,
        device: &mut B::Device,
        cmd_pool: &mut hal::CommandPool<B, hal::queue::Graphics>,
        staging_buffer_pool: &mut BufferPool<B>,
        rect: DeviceUintRect,
        layer_index: i32,
        image_data: &[u8],
    ) -> hal::command::Submit<B, hal::Graphics, hal::command::OneShot, hal::command::Primary>
    {
        let pos = rect.origin;
        let size = rect.size;
        staging_buffer_pool.add(device, image_data);
        let buffer = staging_buffer_pool.buffer();
        let mut cmd_buffer = cmd_pool.acquire_command_buffer(false);

        let range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: layer_index as _ .. (layer_index + 1) as _,
        };
        let barriers = buffer.transit(hal::buffer::Access::TRANSFER_READ)
            .into_iter()
            .chain(self.core.transit(
                hal::image::Access::TRANSFER_WRITE,
                hal::image::Layout::TransferDstOptimal,
                range.clone()));

        cmd_buffer.pipeline_barrier(
            PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
            hal::memory::Dependencies::empty(),
            barriers,
        );
        cmd_buffer.copy_buffer_to_image(
            &buffer.buffer,
            &self.core.image,
            hal::image::Layout::TransferDstOptimal,
            &[
                hal::command::BufferImageCopy {
                    buffer_offset: staging_buffer_pool.buffer_offset as _,
                    buffer_width: size.width,
                    buffer_height: size.height,
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
                },
            ],
        );

        if let Some(barrier) = self.core.transit(
            hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
            hal::image::Layout::ColorAttachmentOptimal,
            range,
        ) {
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                hal::memory::Dependencies::empty(),
                &[barrier],
            );
        }

        cmd_buffer.finish()
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
        let unbound_buffer = device.create_buffer(buffer_size as u64, usage).unwrap();
        let requirements = device.get_buffer_requirements(&unbound_buffer);
        let memory_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                requirements.type_mask & (1 << id) != 0 &&
                    mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();
        let memory = device
            .allocate_memory(memory_type, requirements.size)
            .unwrap();
        let buffer = device
            .bind_buffer_memory(&memory, 0, unbound_buffer)
            .unwrap();
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
        let mut upload_data = device
            .acquire_mapping_writer::<T>(
                &self.memory,
                0..self.buffer_size as u64,
            )
            .unwrap();
        upload_data[0..data.len()].copy_from_slice(&data);
        device.release_mapping_writer(upload_data);
    }

    pub fn update<T: Copy>(&self, device: &B::Device, data: &[T], offset: usize, alignment_mask: usize) -> usize {
        let size_aligned = (data.len() * self.stride + alignment_mask) & !alignment_mask;
        let mut upload_data = device
            .acquire_mapping_writer::<T>(
                &self.memory,
                (offset * self.stride) as u64 .. (size_aligned + (offset * self.stride))  as u64,
            )
            .unwrap();
        upload_data[0..data.len()].copy_from_slice(&data);
        device.release_mapping_writer(upload_data);
        size_aligned
    }

    fn transit(
        &self,
        access: hal::buffer::Access,
    ) -> Option<hal::memory::Barrier<B>> {
        let src_state = self.state.get();
        if src_state == access {
            None
        } else {
            self.state.set(access);
            Some(hal::memory::Barrier::Buffer {
                states: src_state .. access,
                target: &self.buffer,
            })
        }
    }

    pub fn deinit(self, device: &B::Device) {
        device.destroy_buffer(self.buffer);
        device.free_memory(self.memory);
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
            mem::size_of::<T>(), self.data_stride
        );
        let buffer_len = data.len() * self.data_stride;
        assert!(
            self.offset * self.data_stride + buffer_len < self.buffer.buffer_size,
            "offset({:?}) * data_stride({:?}) + buffer_len({:?}) < buffer_size({:?})",
            self.offset, self.data_stride, buffer_len, self.buffer.buffer_size
        );
        self.size = self.buffer.update(
            device,
            data,
            self.offset,
            self.non_coherent_atom_size,
        );
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

pub struct InstanceBufferHandler<B: hal::Backend> {
    buffer: Buffer<B>,
    data_stride: usize,
    non_coherent_atom_size: usize,
    pub offset: usize,
    pub size: usize,
}

impl<B: hal::Backend> InstanceBufferHandler<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &Vec<hal::MemoryType>,
        usage: hal::buffer::Usage,
        data_stride: usize,
        non_coherent_atom_size: usize,
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
        InstanceBufferHandler {
            offset: 0,
            size: 0,
            non_coherent_atom_size,
            buffer,
            data_stride,
        }
    }

    pub fn add<T: Copy>(&mut self, device: &B::Device, data: &[T]) {
        let buffer_len = data.len() * self.data_stride;
        assert!(
            self.offset * self.data_stride + buffer_len < self.buffer.buffer_size,
            "offset({:?}) * data_stride({:?}) + buffer_len({:?}) < buffer_size({:?})",
            self.offset, self.data_stride, buffer_len, self.buffer.buffer_size
        );
        self.buffer.update(
            device,
            data,
            self.offset,
            self.non_coherent_atom_size,
        );
        self.size = data.len();
        self.offset += self.size;
    }

    pub fn buffer(&self) -> &Buffer<B> {
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
            buffers: vec!(),
            offset: 0,
            memory_types: memory_types.clone(),
            usage,
            data_stride,
            pitch_alignment_mask,
        }
    }

    pub fn add<T: Copy>(&mut self, device: &B::Device, data: &[T]) {
        if self.buffers.len() == self.offset {
            self.buffers.push(
                Buffer::new(
                    device,
                    &self.memory_types,
                    self.usage,
                    self.pitch_alignment_mask,
                    data.len(),
                    self.data_stride,
                )
            );
        }
        self.buffers[self.offset].update_all(
            device,
            data,
        );
        self.offset+=1;
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
                )
            );
            old_buffer.deinit(device);
        }
        self.buffer.update_all(
            device,
            data,
        );
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
    pub pipeline_layout: B::PipelineLayout,
    pub pipelines: HashMap<(BlendState, DepthTest), B::GraphicsPipeline>,
    pub vertex_buffer: SmallVec<[VertexBufferHandler<B>; 1]>,
    pub index_buffer: Option<SmallVec<[VertexBufferHandler<B>; 1]>>,
    pub instance_buffer: SmallVec<[InstanceBufferHandler<B>; 1]>,
    pub locals_buffer: SmallVec<[UniformBufferHandler<B>; 1]>,
    shader_name: String,
    shader_kind: ShaderKind,
}

impl<B: hal::Backend> Program<B> {
    pub fn create(
        pipeline_requirements: PipelineRequirements,
        device: &B::Device,
        pipeline_layout: B::PipelineLayout,
        memory_types: &Vec<hal::MemoryType>,
        limits: &hal::Limits,
        shader_name: &str,
        shader_kind: ShaderKind,
        render_pass: &RenderPass<B>,
    ) -> Program<B> {
        let vs_module = device
            .create_shader_module(get_shader_source(shader_name, ".vert.spv").as_slice())
            .unwrap();
        let fs_module = device
            .create_shader_module(get_shader_source(shader_name, ".frag.spv").as_slice())
            .unwrap();

        let pipelines = {
            let (vs_entry, fs_entry) = (
                hal::pso::EntryPoint::<B> {
                    entry: ENTRY_NAME,
                    module: &vs_module,
                    specialization: hal::pso::Specialization::default(),
                },
                hal::pso::EntryPoint::<B> {
                    entry: ENTRY_NAME,
                    module: &fs_module,
                    specialization: hal::pso::Specialization::default(),
                },
            );

            let shader_entries = hal::pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let pipeline_states = match shader_kind {
                ShaderKind::Cache(VertexArrayKind::Scale) => {
                    vec![
                        (BlendState::Off, DepthTest::Off),
                        (BlendState::MULTIPLY, DepthTest::Off),
                    ]
                },
                ShaderKind::Cache(VertexArrayKind::Blur) => vec![(BlendState::Off, DepthTest::Off)],
                ShaderKind::Cache(VertexArrayKind::Border) => vec![(BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off)],
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
                    if shader_name.contains("dual_source_blending") {
                        states.push((SUBPIXEL_DUAL_SOURCE, DepthTest::Off));
                        states.push((SUBPIXEL_DUAL_SOURCE, LESS_EQUAL_TEST));
                    }
                    states
                },
                #[cfg(feature = "debug_renderer")]
                ShaderKind::DebugColor | ShaderKind::DebugFont => vec![
                    (BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off),
                ],
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
            let format = match shader_kind {
                ShaderKind::ClipCache => ImageFormat::R8,
                ShaderKind::Cache(VertexArrayKind::Blur) if shader_name.contains("_alpha_target") => ImageFormat::R8,
                ShaderKind::Cache(VertexArrayKind::Scale) if shader_name.contains("_alpha_target") => ImageFormat::R8,
                _ => ImageFormat::BGRA8,
            };

            let pipelines_descriptors = pipeline_states.iter().map(|&(blend_state, depth_test)| {
                let subpass = Subpass {
                    index: 0,
                    main_pass: render_pass.get_render_pass(format, depth_test != DepthTest::Off),
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

                pipeline_descriptor.vertex_buffers = pipeline_requirements.vertex_buffer_descriptors.clone();
                pipeline_descriptor.attributes = pipeline_requirements.attribute_descriptors.clone();
                pipeline_descriptor
            }).collect::<Vec<_>>();

            let pipelines = device
                .create_graphics_pipelines(pipelines_descriptors.as_slice(), None)
                .into_iter();

            pipeline_states.iter()
                .cloned()
                .zip(pipelines.map(|pipeline| pipeline.unwrap()))
                .collect::<HashMap<(BlendState, DepthTest), B::GraphicsPipeline>>()
        };

        device.destroy_shader_module(vs_module);
        device.destroy_shader_module(fs_module);

        let vertex_buffer_stride = match shader_kind {
            #[cfg(feature = "debug_renderer")]
            ShaderKind::DebugColor => mem::size_of::<DebugColorVertex>(),
            #[cfg(feature = "debug_renderer")]
            ShaderKind::DebugFont => mem::size_of::<DebugFontVertex>(),
            _ => mem::size_of::<Vertex>(),
        };

        let instance_buffer_stride = match shader_kind {
            ShaderKind::Primitive |
            ShaderKind::Brush |
            ShaderKind::Text |
            ShaderKind::Cache(VertexArrayKind::Primitive) => mem::size_of::<PrimitiveInstanceData>(),
            ShaderKind::ClipCache | ShaderKind::Cache(VertexArrayKind::Clip) => mem::size_of::<ClipMaskInstance>(),
            ShaderKind::Cache(VertexArrayKind::Blur) => mem::size_of::<BlurInstance>(),
            ShaderKind::Cache(VertexArrayKind::Border) => mem::size_of::<BorderInstance>(),
            ShaderKind::Cache(VertexArrayKind::Scale) => mem::size_of::<ScalingInstance>(),
            sk if sk.is_debug() => 1,
            _ => unreachable!()
        };

        let mut vertex_buffer = SmallVec::new();
        let mut instance_buffer = SmallVec::new();
        let mut locals_buffer = SmallVec::new();
        let mut index_buffer = if shader_kind.is_debug() {
            Some(SmallVec::new())
        } else {
            None
        };
        for _ in 0..MAX_FRAME_COUNT {
            vertex_buffer.push(
                VertexBufferHandler::new(
                    device,
                    memory_types,
                    hal::buffer::Usage::VERTEX,
                    &QUAD,
                    vertex_buffer_stride,
                    (limits.min_buffer_copy_pitch_alignment - 1) as usize,
                )
            );
            instance_buffer.push(
                InstanceBufferHandler::new(
                    device,
                    memory_types,
                    hal::buffer::Usage::VERTEX,
                    instance_buffer_stride,
                    (limits.non_coherent_atom_size - 1) as usize,
                    (limits.min_buffer_copy_pitch_alignment - 1) as usize,
                )
            );
            locals_buffer.push(
                UniformBufferHandler::new(
                    memory_types,
                    hal::buffer::Usage::UNIFORM,
                    mem::size_of::<Locals>(),
                    (limits.min_uniform_buffer_offset_alignment - 1) as usize,
                )
            );
            if let Some(ref mut index_buffer) = index_buffer {
                index_buffer.push(
                    VertexBufferHandler::new(
                        device,
                        memory_types,
                        hal::buffer::Usage::INDEX,
                        &vec![0u32; MAX_INDEX_COUNT],
                        mem::size_of::<u32>(),
                        (limits.min_buffer_copy_pitch_alignment - 1) as usize,
                    )
                );
            }
        }

        let bindings_map = pipeline_requirements.bindings_map;

        Program {
            bindings_map,
            pipeline_layout,
            pipelines,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            locals_buffer,
            shader_name: String::from(shader_name),
            shader_kind,
        }
    }


    pub fn bind_instances<T: Copy>(
        &mut self,
        device: &B::Device,
        instances: &[T],
        buffer_id: usize,
    ) {
        assert!(!instances.is_empty());
        self.instance_buffer[buffer_id].add(
            device,
            instances,
        );
    }

    fn bind_locals(
        &mut self,
        device: &B::Device,
        set: &B::DescriptorSet,
        projection: &Transform3D<f32>,
        device_pixel_ratio: f32,
        u_mode: i32,
        buffer_id: usize,
    ) {
        let locals_data = vec![
            Locals {
                uTransform: projection.to_row_arrays(),
                uDevicePixelRatio: device_pixel_ratio,
                uMode: u_mode,
            },
        ];
        self.locals_buffer[buffer_id].add(
            device,
            &locals_data,
        );
        device.write_descriptor_sets(vec![
            hal::pso::DescriptorSetWrite {
                set,
                binding: self.bindings_map["Locals"],
                array_offset: 0,
                descriptors: Some(
                    hal::pso::Descriptor::Buffer(&self.locals_buffer[buffer_id].buffer().buffer, Some(0)..None),
                ),
            },
        ]);
    }

    fn bind_texture(&mut self, device: &B::Device, set: &B::DescriptorSet, image: &ImageCore<B>, sampler: &B::Sampler, binding: &'static str) {
        if self.bindings_map.contains_key(&("t".to_owned() + binding)) {
            device.write_descriptor_sets(vec![
                hal::pso::DescriptorSetWrite {
                    set,
                    binding: self.bindings_map[&("t".to_owned() + binding)],
                    array_offset: 0,
                    descriptors: Some(
                        hal::pso::Descriptor::Image(&image.view, image.state.get().1)
                    ),
                },
                hal::pso::DescriptorSetWrite {
                    set,
                    binding: self.bindings_map[&("s".to_owned() + binding)],
                    array_offset: 0,
                    descriptors: Some(
                        hal::pso::Descriptor::Sampler(sampler)
                    )
                },
            ]);
        }
    }

    pub fn submit(
        &mut self,
        cmd_pool: &mut hal::CommandPool<B, hal::queue::Graphics>,
        viewport: hal::pso::Viewport,
        render_pass: &B::RenderPass,
        frame_buffer: &B::Framebuffer,
        desc_pools: &mut DescriptorPools<B>,
        clear_values: &[hal::command::ClearValue],
        blend_state: BlendState,
        blend_color: ColorF,
        depth_test: DepthTest,
        scissor_rect: Option<DeviceIntRect>,
        next_id: usize,
    ) -> hal::command::Submit<B, hal::Graphics, hal::command::OneShot, hal::command::Primary> {
        let mut cmd_buffer = cmd_pool.acquire_command_buffer(false);
        let vertex_buffer = &self.vertex_buffer[next_id];
        let instance_buffer = &self.instance_buffer[next_id];

        cmd_buffer.set_viewports(0, &[viewport.clone()]);
        match scissor_rect {
            Some(r) => cmd_buffer.set_scissors(
                0,
                &[hal::pso::Rect {x: r.origin.x as _, y: r.origin.y as _, w: r.size.width as _, h: r.size.height as _}],
            ),
            None => cmd_buffer.set_scissors(0, &[viewport.rect]),
        }
        cmd_buffer.bind_graphics_pipeline(
            &self.pipelines.get(&(blend_state, depth_test)).expect(&format!("The blend state {:?} with depth test {:?} not found for {} program!", blend_state, depth_test, self.shader_name)));


        if let Some(ref index_buffer) = self.index_buffer {
            cmd_buffer.bind_vertex_buffers(
                0,
                vec![
                    (&vertex_buffer.buffer().buffer, 0),
                ],
            );
            cmd_buffer.bind_index_buffer(
                hal::buffer::IndexBufferView {
                    buffer: &index_buffer[next_id].buffer().buffer,
                    offset: 0,
                    index_type: hal::IndexType::U32,
                }
            );
        } else {
            cmd_buffer.bind_vertex_buffers(
                0,
                vec![
                    (&vertex_buffer.buffer().buffer, 0),
                    (&instance_buffer.buffer().buffer, 0),
                ],
            );
        }

        cmd_buffer.bind_graphics_descriptor_sets(
            &self.pipeline_layout,
            0,
            Some(desc_pools.get(&self.shader_kind)),
            &[],
        );
        desc_pools.next(&self.shader_kind);

        if blend_state == SUBPIXEL_CONSTANT_TEXT_COLOR {
            cmd_buffer.set_blend_constants(blend_color.to_array());
        }

        {
            let mut encoder = cmd_buffer.begin_render_pass_inline(
                render_pass,
                frame_buffer,
                viewport.rect,
                clear_values,
            );

            if let Some(ref index_buffer) = self.index_buffer {
                encoder.draw_indexed(
                    0 .. index_buffer[next_id].buffer().buffer_len as u32,
                    0,
                    0 .. 1,
                );
            } else {
                let offset = instance_buffer.offset as u32;
                let size = instance_buffer.size as u32;
                encoder.draw(
                    0 .. vertex_buffer.buffer_len as _,
                    (offset - size) .. offset,
                );
            }
        }

        cmd_buffer.finish()
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
        device.destroy_pipeline_layout(self.pipeline_layout);
        for pipeline in self.pipelines.drain() {
            device.destroy_graphics_pipeline(pipeline.1);
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
        depth: Option<&B::ImageView>
    ) -> Self {
        let extent = hal::image::Extent {
            width: texture.width as _,
            height: texture.height as _,
            depth: 1,
        };
        let format = match texture.format {
            ImageFormat::R8 => hal::format::Format::R8Unorm,
            ImageFormat::BGRA8 => hal::format::Format::Bgra8Unorm,
            _ => unimplemented!("TODO image format missing"),
        };
        let image_view = device
            .create_image_view(
                &image.core.image,
                hal::image::ViewKind::D2Array,
                format,
                hal::format::Swizzle::NO,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: 0 .. 1,
                    layers: layer_index .. layer_index+1,
                },
            )
            .unwrap();
        let fbo = if rbo != RBOId(0) {
            device
                .create_framebuffer(render_pass.get_render_pass(texture.format, true), vec![&image_view, depth.unwrap()], extent)
                .unwrap()
        } else {
            device
                .create_framebuffer(render_pass.get_render_pass(texture.format, false), Some(&image_view), extent)
                .unwrap()
        };

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
        device.destroy_framebuffer(self.fbo);
        device.destroy_image_view(self.image_view);
    }
}

pub struct DepthBuffer<B: hal::Backend> {
    pub core: ImageCore<B>
}

impl<B: hal::Backend> DepthBuffer<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        pixel_width: u32,
        pixel_height: u32,
        depth_format: hal::format::Format
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
        DepthBuffer {
            core
        }
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
            _ => unimplemented!(),
        }
    }

    pub fn deinit(self, device: &B::Device) {
        device.destroy_render_pass(self.r8);
        device.destroy_render_pass(self.r8_depth);
        device.destroy_render_pass(self.bgra8);
        device.destroy_render_pass(self.bgra8_depth);
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
        let descriptor_pool =
            device.create_descriptor_pool(
                max_size,
                descriptor_range_descriptors.as_slice(),
            );
        let descriptor_set_layout = device.create_descriptor_set_layout(&descriptor_set_layout, &[]);
        let mut dp = DescPool {
            descriptor_pool,
            descriptor_set: vec!(),
            descriptor_set_layout,
            current_descriptor_set_id: 0,
            max_descriptor_set_size: max_size,
        };
        dp.allocate();
        dp
    }

    pub fn descriptor_set(&mut self) -> &B::DescriptorSet {
        &self.descriptor_set[self.current_descriptor_set_id]
    }

    pub fn descriptor_set_layout(&mut self) -> &B::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    pub fn next(&mut self) {
        self.current_descriptor_set_id += 1;
        assert!(self.current_descriptor_set_id < self.max_descriptor_set_size);
        if self.current_descriptor_set_id == self.descriptor_set.len() {
            self.allocate();
        }
    }

    fn allocate(&mut self) {
        let desc_set =
            self.descriptor_pool.allocate_set(&self.descriptor_set_layout)
                .expect(&format!("Failed to allocate set with layout: {:?}", self.descriptor_set_layout));
        self.descriptor_set.push(desc_set);
    }

    pub fn reset(&mut self) {
        self.current_descriptor_set_id = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        device.destroy_descriptor_set_layout(self.descriptor_set_layout);
        device.destroy_descriptor_pool(self.descriptor_pool);
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
        debug_layout: Vec<DescriptorSetLayoutBinding>,
        cache_clip_layout: Vec<DescriptorSetLayoutBinding>,
        default_layout: Vec<DescriptorSetLayoutBinding>,
    ) -> Self {
        let debug_range = vec![
            DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::SampledImage,
                count: 20,
            },
            DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::Sampler,
                count: 20,
            },
            DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::UniformBuffer,
                count: 5,
            }
        ];

        let count = 400;
        let cache_clip_range = vec![
            hal::pso::DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::SampledImage,
                count: count * 10,
            },
            DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::Sampler,
                count: count * 10,
            },
            DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::UniformBuffer,
                count: count,
            }
        ];

        let default_range = vec![
            hal::pso::DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::SampledImage,
                count: count * 10,
            },
            DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::Sampler,
                count: count * 10,
            },
            DescriptorRangeDesc {
                ty: hal::pso::DescriptorType::UniformBuffer,
                count: count,
            }
        ];

        DescriptorPools {
            debug_pool: DescPool::new(device, 5, debug_range, debug_layout),
            cache_clip_pool: DescPool::new(device, count, cache_clip_range, cache_clip_layout),
            default_pool: DescPool::new(device, count, default_range, default_layout),
        }
    }

    fn get_pool(&mut self, shader_kind: &ShaderKind) -> &mut DescPool<B> {
        match *shader_kind {
            #[cfg(feature = "debug_renderer")]
            ShaderKind::DebugColor | ShaderKind::DebugFont => &mut self.debug_pool,
            ShaderKind::ClipCache => &mut self.cache_clip_pool,
            _ => &mut self.default_pool,
        }
    }

    pub fn get(&mut self, shader_kind: &ShaderKind) -> &B::DescriptorSet {
        self.get_pool(shader_kind).descriptor_set()
    }

    pub fn get_layout(&mut self, shader_kind: &ShaderKind) -> &B::DescriptorSetLayout {
        self.get_pool(shader_kind).descriptor_set_layout()
    }

    pub fn next(&mut self, shader_kind: &ShaderKind) {
        self.get_pool(shader_kind).next()
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

pub struct Device<B: hal::Backend> {
    pub device: B::Device,
    pub memory_types: Vec<hal::MemoryType>,
    pub limits: hal::Limits,
    adapter: hal::Adapter<B>,
    surface: B::Surface,
    pub surface_format: hal::format::Format,
    pub depth_format: hal::format::Format,
    pub queue_group: hal::QueueGroup<B, hal::Graphics>,
    pub command_pool: SmallVec<[hal::CommandPool<B, hal::Graphics>; 1]>,
    pub staging_buffer_pool: SmallVec<[BufferPool<B>; 1]>,
    pub swap_chain: Option<B::Swapchain>,
    pub render_pass: Option<RenderPass<B>>,
    pub framebuffers: Vec<B::Framebuffer>,
    pub framebuffers_depth: Vec<B::Framebuffer>,
    pub frame_images: Vec<ImageCore<B>>,
    pub frame_depths: Vec<DepthBuffer<B>>,
    pub viewport: hal::pso::Viewport,
    pub sampler_linear: B::Sampler,
    pub sampler_nearest: B::Sampler,
    pub upload_queue: Vec<hal::command::Submit<B, hal::Graphics, hal::command::OneShot, hal::command::Primary>>,
    pub current_frame_id: usize,
    current_blend_state: BlendState,
    blend_color: ColorF,
    current_depth_test: DepthTest,
    // device state
    programs: FastHashMap<ProgramId, Program<B>>,
    images: FastHashMap<TextureId, Image<B>>,
    fbos: FastHashMap<FBOId, Framebuffer<B>>,
    rbos: FastHashMap<RBOId, DepthBuffer<B>>,
    descriptor_pools: SmallVec<[DescriptorPools<B>; 1]>,
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
    upload_method: UploadMethod,

    // HW or API capabilities
    #[cfg(feature = "debug_renderer")]
    capabilities: Capabilities,

    // debug
    inside_frame: bool,

    // resources
    _resource_override_path: Option<PathBuf>,

    max_texture_size: u32,
    _renderer_name: String,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    frame_id: FrameId,

    // Supported features
    features: hal::Features,

    next_id: usize,
    frame_fence: SmallVec<[Fence<B>; 1]>,
    image_available_semaphore: B::Semaphore,
    render_finished_semaphore: B::Semaphore,
    pipeline_requirements: HashMap<String, PipelineRequirements>,
}

impl<B: hal::Backend> Device<B> {
    pub fn new(
        init: DeviceInit<B>,
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _file_changed_handler: Box<FileWatcherHandler>,
        _cached_programs: Option<Rc<ProgramCache>>,
    ) -> Self {
        let (adapter, mut surface, window_size) = (init.adapter, init.surface, init.window_size);
        let renderer_name = "gfx-rs".to_owned();
        let features = adapter.physical_device.features();

        let memory_types = adapter
            .physical_device
            .memory_properties()
            .memory_types;
        println!("memory_types: {:?}", memory_types);

        let limits = adapter
            .physical_device
            .limits();
        let max_texture_size = 4400u32; // TODO use limits after it points to the correct texture size

        let (device, queue_group) = {
            let queue_family = adapter.queue_families
                .iter()
                .find(|family| surface.supports_queue_family(family))
                .expect("No queue family is able to render to the surface!");
            let mut gpu = adapter.physical_device
                .open(&[(queue_family, &[1.0])])
                .unwrap();
            let queue_group = gpu.queues
                .take(queue_family.id())
                .unwrap();
            (gpu.device, queue_group)
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
            viewport
        ) = Device::init_swapchain_resources(
            &device,
            &memory_types,
            &adapter,
            &mut surface,
            window_size
        );

        // Samplers

        let sampler_linear = device.create_sampler(hal::image::SamplerInfo::new(
            hal::image::Filter::Linear,
            hal::image::WrapMode::Clamp,
        ));

        let sampler_nearest = device.create_sampler(hal::image::SamplerInfo::new(
            hal::image::Filter::Nearest,
            hal::image::WrapMode::Clamp,
        ));

        let file =
            File::open(concat!(env!("OUT_DIR"), "/shader_bindings.ron")).expect("Unable to open the file");
        let pipeline_requirements: HashMap<String, PipelineRequirements> =
            from_reader(file).expect("Failed to load shader_bindings.ron");

        let mut descriptor_pools = SmallVec::new();
        let mut frame_fence = SmallVec::new();
        let mut command_pool = SmallVec::new();
        let mut staging_buffer_pool = SmallVec::new();
        for _ in 0..MAX_FRAME_COUNT {
            descriptor_pools.push(
                DescriptorPools::new(
                    &device,
                    pipeline_requirements.get("debug_color").expect("debug_color missing").descriptor_set_layouts.clone(),
                    pipeline_requirements.get("cs_clip_rectangle").expect("cs_clip_rectangle missing").descriptor_set_layouts.clone(),
                    pipeline_requirements.get("brush_solid").expect("brush_solid missing").descriptor_set_layouts.clone(),
                )
            );

            let fence = device.create_fence(false);
            frame_fence.push(
                Fence {
                    inner: fence,
                    is_submitted: false,
                }
            );

            let mut cp = device.create_command_pool_typed(
                &queue_group,
                hal::pool::CommandPoolCreateFlags::empty(),
                64,
            );
            cp.reset();
            command_pool.push(
                cp
            );
            staging_buffer_pool.push(
                BufferPool::new(
                    &device,
                    &memory_types,
                    hal::buffer::Usage::TRANSFER_SRC,
                    1,
                    (limits.non_coherent_atom_size - 1) as usize,
                    (limits.min_buffer_copy_pitch_alignment - 1) as usize,
                    (limits.min_buffer_copy_offset_alignment - 1) as usize,
                )
            );
        }

        let image_available_semaphore = device.create_semaphore();
        let render_finished_semaphore = device.create_semaphore();

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
            viewport,
            sampler_linear,
            sampler_nearest,
            upload_queue: Vec::new(),
            current_frame_id: 0,
            current_blend_state: BlendState::Off,
            current_depth_test: DepthTest::Off,
            blend_color: ColorF::new(0.0, 0.0, 0.0, 0.0),
            _resource_override_path: resource_override_path,
            // This is initialized to 1 by default, but it is reset
            // at the beginning of each frame in `Renderer::bind_frame_data`.
            device_pixel_ratio: 1.0,
            upload_method,
            inside_frame: false,

            #[cfg(feature = "debug_renderer")]
            capabilities: Capabilities {
                supports_multisampling: false, //TODO
            },

            programs: FastHashMap::default(),
            images: FastHashMap::default(),
            fbos: FastHashMap::default(),
            rbos: FastHashMap::default(),
            descriptor_pools,
            bound_textures: [0; 16],
            bound_program: INVALID_PROGRAM_ID,
            bound_sampler: [TextureFilter::Linear; 16],
            bound_read_fbo: DEFAULT_READ_FBO,
            bound_read_texture: (INVALID_TEXTURE_ID, 0),
            bound_draw_fbo: DEFAULT_DRAW_FBO,
            program_mode_id: 0,
            scissor_rect: None,

            max_texture_size,
            _renderer_name: renderer_name,
            frame_id: FrameId(0),
            features,

            next_id: 0,
            frame_fence,
            image_available_semaphore,
            render_finished_semaphore,
            pipeline_requirements,
        }
    }

    pub(crate) fn recreate_swapchain(&mut self, window_size: Option<(u32, u32)>) -> DeviceUintSize {
        println!("recreate_swapchain");
        self.device.wait_idle().unwrap();

        for (_id, program) in self.programs.drain() {
            program.deinit(&self.device);
        }

        for image in self.frame_images.drain(..) {
            image.deinit(&self.device);
        }

        for depth in self.frame_depths.drain(..) {
            depth.deinit(&self.device);
        }

        for framebuffer in self.framebuffers.drain(..) {
            self.device.destroy_framebuffer(framebuffer);
        }
        for framebuffer_depth in self.framebuffers_depth.drain(..) {
            self.device.destroy_framebuffer(framebuffer_depth);
        }

        self.render_pass.take().unwrap().deinit(&self.device);

        self.device.destroy_swapchain(self.swap_chain.take().unwrap());

        let window_size = window_size.unwrap_or((self.viewport.rect.w as u32, self.viewport.rect.h as u32));

        let (
            swap_chain,
            surface_format,
            depth_format,
            render_pass,
            framebuffers,
            framebuffers_depth,
            frame_depths,
            frame_images,
            viewport
        ) = Device::init_swapchain_resources(
            &self.device,
            &self.memory_types,
            &self.adapter,
            &mut self.surface,
            window_size
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
        DeviceUintSize::new(self.viewport.rect.w as u32, self.viewport.rect.h as u32)
    }

    fn init_swapchain_resources(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        adapter: &hal::Adapter<B>,
        surface: &mut B::Surface,
        window_size: (u32, u32),
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
        let (caps, formats, _present_modes) = surface.compatibility(&adapter.physical_device);
        let surface_format = formats
            .map_or(
                hal::format::Format::Bgra8Unorm,
                |formats| {
                    formats
                        .into_iter()
                        .find(|format| {
                            format == &hal::format::Format::Bgra8Unorm
                        })
                        .unwrap()
                },
            );

        let mut extent = caps.current_extent.unwrap_or(
            hal::window::Extent2D {
                width: window_size.0.max(caps.extents.start.width).min(caps.extents.end.width),
                height: window_size.1.max(caps.extents.start.height).min(caps.extents.end.height),
            }
        );

        if extent.width == 0 { extent.width = 1; }
        if extent.height == 0 { extent.height = 1; }

        let swap_config = SwapchainConfig::from_caps(&caps, surface_format)
            .with_image_usage(
                hal::image::Usage::TRANSFER_SRC | hal::image::Usage::TRANSFER_DST | hal::image::Usage::COLOR_ATTACHMENT
            );

        let (swap_chain, backbuffer) = device.create_swapchain(surface, swap_config, None);
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
                layouts: hal::image::Layout::ColorAttachmentOptimal .. hal::image::Layout::ColorAttachmentOptimal,
            };

            let attachment_bgra8 = hal::pass::Attachment {
                format: Some(hal::format::Format::Bgra8Unorm),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::ColorAttachmentOptimal .. hal::image::Layout::ColorAttachmentOptimal,
            };

            let attachment_depth = hal::pass::Attachment {
                format: Some(depth_format),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::DepthStencilAttachmentOptimal .. hal::image::Layout::DepthStencilAttachmentOptimal,
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
                stages: PipelineStage::EARLY_FRAGMENT_TESTS
                    .. PipelineStage::LATE_FRAGMENT_TESTS,
                accesses: hal::image::Access::empty()
                    .. (hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE),
            };

            RenderPass {
                r8: device.create_render_pass(&[attachment_r8.clone()], &[subpass_r8], &[dependency.clone()]),
                r8_depth: device.create_render_pass(&[attachment_r8, attachment_depth.clone()], &[subpass_depth_r8], &[dependency.clone(), depth_dependency.clone()]),
                bgra8: device.create_render_pass(&[attachment_bgra8.clone()], &[subpass_bgra8], &[dependency.clone()]),
                bgra8_depth: device.create_render_pass(&[attachment_bgra8, attachment_depth], &[subpass_depth_bgra8], &[dependency, depth_dependency]),
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
                        frame_depths.push(DepthBuffer::new(device, &memory_types, extent.width, extent.height, depth_format));
                        ImageCore::from_image(device, image, hal::image::ViewKind::D2Array, surface_format, COLOR_RANGE.clone())
                    })
                    .collect::<Vec<_>>();
                let fbos = cores
                    .iter()
                    .map(|core| {
                        device
                            .create_framebuffer(
                                &render_pass.bgra8,
                                Some(&core.view),
                                extent,
                            )
                            .unwrap()
                    })
                    .collect();
                let fbos_depth = cores
                    .iter()
                    .zip(frame_depths.iter())
                    .map(|(core, depth)| {
                        device
                            .create_framebuffer(
                                &render_pass.bgra8_depth,
                                vec![&core.view, &depth.core.view],
                                extent,
                            )
                            .unwrap()
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
        (swap_chain, surface_format, depth_format, render_pass,
         framebuffers, framebuffers_depth, frame_depths, frame_images, viewport)
    }

    pub(crate) fn bound_program(&self) -> ProgramId { self.bound_program }

    pub fn set_device_pixel_ratio(&mut self, ratio: f32) {
        self.device_pixel_ratio = ratio;
    }

    pub fn update_program_cache(&mut self, _cached_programs: Rc<ProgramCache>) {
        unimplemented!();
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    #[cfg(feature = "debug_renderer")]
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

    pub fn create_program(
        &mut self,
        shader_name: &str,
        shader_kind: &ShaderKind,
        features: &[&str],
    ) -> Result<ProgramId, ShaderError> {
        let mut name = String::from(shader_name);
        for feature_names in features {
            for feature in feature_names.split(',') {
                if !feature.is_empty() {
                    name.push_str(&format!("_{}", feature.to_lowercase()));
                }
            }
        }
        let program = Program::create(
            self.pipeline_requirements.get(&name)
                    .expect(&format!("Can't load pipeline data for: {}!", name)).clone(),
            &self.device,
            self.device.create_pipeline_layout(Some(self.descriptor_pools[self.next_id].get_layout(shader_kind)), &[]),
            &self.memory_types,
            &self.limits,
            &name,
            shader_kind.clone(),
            self.render_pass.as_ref().unwrap(),
        );

        let id = self.generate_program_id();
        self.programs.insert(id, program);
        Ok(id)
    }

    pub fn create_program_with_kind(
        &mut self,
        shader_name: &str,
        shader_kind: &ShaderKind,
        features: &[&str],
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
        const SAMPLERS: [(usize, &'static str); 12] = [
            (0, "Color0"),
            (1, "Color1"),
            (2, "Color2"),
            (3, "CacheA8"),
            (4, "CacheRGBA8"),
            (5, "ResourceCache"),
            (6, "TransformPalette"),
            (7, "RenderTasks"),
            (8, "Dither"),
            (9, "SharedCacheA8"),
            (10, "PrimitiveHeadersF"),
            (11, "PrimitiveHeadersI"),
        ];
        let program = self.programs.get_mut(&self.bound_program).expect("Program not found.");
        let desc_set = self.descriptor_pools[self.next_id].get(&program.shader_kind);
        for &(index, sampler_name) in SAMPLERS.iter() {
            if self.bound_textures[index] != 0 {
                let sampler = match self.bound_sampler[index] {
                    TextureFilter::Linear | TextureFilter::Trilinear => &self.sampler_linear,
                    TextureFilter::Nearest => &self.sampler_nearest,
                };
                program.bind_texture(&self.device, desc_set, &self.images[&self.bound_textures[index]].core, &sampler, sampler_name);
            }
        }
    }

    #[cfg(feature = "debug_renderer")]
    pub fn update_indices<I: Copy>(&mut self, indices: &[I]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self.programs.get_mut(&self.bound_program).expect("Program not found.");

        if let Some(ref mut index_buffer) = program.index_buffer {
            index_buffer[self.next_id].update(
                &self.device,
                indices,
            );
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    pub fn update_vertices<T: Copy>(&mut self, vertices: &[T]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self.programs.get_mut(&self.bound_program).expect("Program not found.");

        if program.shader_name.contains("debug") {
            program.vertex_buffer[self.next_id].update(
                &self.device,
                vertices,
            );
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    pub fn set_uniforms(
        &mut self,
        program: &ProgramId,
        transform: &Transform3D<f32>,
    ) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        assert_eq!(*program, self.bound_program);
        let program = self.programs.get_mut(&self.bound_program).expect("Program not found.");
        let desc_set = &self.descriptor_pools[self.next_id].get(&program.shader_kind);
        program.bind_locals(&self.device, desc_set, transform, self.device_pixel_ratio, self.program_mode_id, self.next_id);
    }

    fn update_instances<T: Copy>(
        &mut self,
        instances: &[T],
    ) {
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        self.programs.get_mut(&self.bound_program).expect("Program not found.").bind_instances(&self.device, instances, self.next_id);
    }

    fn draw(
        &mut self,
    ) {
        println!("draw");
        let submit = {
            let (fb, format) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
                (&self.fbos[&self.bound_draw_fbo].fbo, self.fbos[&self.bound_draw_fbo].format)
            } else {
                if self.current_depth_test == DepthTest::Off {
                    (&self.framebuffers[self.current_frame_id], ImageFormat::BGRA8)
                } else {
                    (&self.framebuffers_depth[self.current_frame_id], ImageFormat::BGRA8)
                }
            };
            let rp = self.render_pass.as_ref().unwrap().get_render_pass(format, self.current_depth_test != DepthTest::Off);
            self.programs.get_mut(&self.bound_program).expect("Program not found").submit(
                &mut self.command_pool[self.next_id],
                self.viewport.clone(),
                rp,
                &fb,
                &mut self.descriptor_pools[self.next_id],
                &vec![],
                self.current_blend_state,
                self.blend_color,
                self.current_depth_test,
                self.scissor_rect,
                self.next_id,
            )
        };

        self.upload_queue.push(submit);
    }

    pub fn begin_frame(&mut self) -> FrameId {
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

    pub fn bind_read_target(&mut self, texture_and_layer: Option<(&Texture, i32)>) {
        let fbo_id = texture_and_layer.map_or(DEFAULT_READ_FBO, |texture_and_layer| {
            texture_and_layer.0.fbo_ids[texture_and_layer.1 as usize]
        });

        self.bind_read_target_impl(fbo_id)
    }

    #[cfg(feature = "capture")]
    fn bind_read_texture(&mut self, texture_id: TextureId, layer_id: i32 ) {
        self.bound_read_texture = (texture_id, layer_id);
    }

    fn bind_draw_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
        }
    }

    pub fn bind_draw_target(
        &mut self,
        texture_and_layer: Option<(&Texture, i32)>,
        dimensions: Option<DeviceUintSize>,
    ) {
        let fbo_id = texture_and_layer.map_or(DEFAULT_DRAW_FBO, |texture_and_layer| {
            texture_and_layer.0.fbo_ids[texture_and_layer.1 as usize]
        });

        if let Some((texture, layer_index)) = texture_and_layer {
            let mut cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer(false);
            let rbos = &self.rbos;
            if let Some(barrier) = self.images[&texture.id].core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: 0 .. 1,
                    layers: layer_index as _ .. (layer_index + 1) as _,
                },
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            if let Some(barrier) = texture.depth_rb.and_then(|rbo| rbos[&rbo].core.transit(
                hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                hal::image::Layout::DepthStencilAttachmentOptimal,
                rbos[&rbo].core.subresource_range.clone(),
            )) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::LATE_FRAGMENT_TESTS,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            self.upload_queue.push(cmd_buffer.finish())
        }

        self.bind_draw_target_impl(fbo_id);

        if let Some(dimensions) = dimensions {
            self.viewport.rect = hal::pso::Rect {
                x: 0,
                y: 0,
                w: dimensions.width as _,
                h: dimensions.height as _,
            };
        }
    }

    pub fn create_fbo_for_external_texture(&mut self, _texture_id: u32) -> FBOId {
        warn!("External texture creation is missing");
        FBOId(0)
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

    fn generate_texture_id(&mut self) -> TextureId {
        let mut rng = rand::thread_rng();
        let mut texture_id = INVALID_TEXTURE_ID + 1;
        while self.images.contains_key(&texture_id) {
            texture_id = rng.gen_range::<u32>(INVALID_TEXTURE_ID + 1, u32::max_value());
        }
        texture_id
    }

    fn generate_program_id(&mut self) -> ProgramId {
        let mut rng = rand::thread_rng();
        let mut program_id = ProgramId(INVALID_PROGRAM_ID.0 + 1);
        while self.programs.contains_key(&program_id) {
            program_id = ProgramId(rng.gen_range::<u32>(INVALID_PROGRAM_ID.0 + 1, u32::max_value()));
        }
        program_id
    }

    fn generate_fbo_ids(&mut self, count: i32) -> Vec<FBOId> {
        let mut rng = rand::thread_rng();
        let mut fboids = vec!();
        let mut fbo_id = FBOId(DEFAULT_DRAW_FBO.0 + 1);
        for _ in 0..count {
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
    ) -> Texture {
        Texture {
            id: 0,
            target: target as _,
            width: 0,
            height: 0,
            layer_count: 0,
            format,
            filter: TextureFilter::Nearest,
            render_target: None,
            fbo_ids: vec![],
            depth_rb: None,
            last_frame_used: self.frame_id,
            bound_in_frame: Cell::new(FrameId(0)),
        }
    }

    /// Resizes a texture with enabled render target views,
    /// preserves the data by blitting the old texture contents over.
    pub fn resize_renderable_texture(
        &mut self,
        _texture: &mut Texture,
        _new_size: DeviceUintSize,
    ) {
        unimplemented!();
    }

    pub fn init_texture<T: Texel>(
        &mut self,
        texture: &mut Texture,
        mut width: u32,
        mut height: u32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
        layer_count: i32,
        pixels: Option<&[T]>,
    ) {
        debug_assert!(self.inside_frame);
        if width == 0 || height == 0 || layer_count == 0 {
            return;
        }

        if width > self.max_texture_size || height > self.max_texture_size {
            error!("Attempting to allocate a texture of size {}x{} above the limit, trimming", width, height);
            width = width.min(self.max_texture_size);
            height = height.min(self.max_texture_size);
        }

        let is_resized = texture.width != width || texture.height != height;

        if texture.id == 0 {
            let id = self.generate_texture_id();
            texture.id = id;
        } else {
            self.free_image(texture);
        }

        texture.width = width;
        texture.height = height;
        texture.filter = filter;
        texture.layer_count = layer_count;
        texture.render_target = render_target;
        texture.last_frame_used = self.frame_id;

        assert_eq!(self.images.contains_key(&texture.id), false);
        let usage_base = hal::image::Usage::TRANSFER_SRC | hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED;
        let (view_kind, mip_levels, usage) = match texture.filter {
            TextureFilter::Nearest => (hal::image::ViewKind::D2, 1, usage_base | hal::image::Usage::COLOR_ATTACHMENT),
            TextureFilter::Linear => (hal::image::ViewKind::D2Array, 1, usage_base | hal::image::Usage::COLOR_ATTACHMENT),
            TextureFilter::Trilinear => (hal::image::ViewKind::D2Array, (width as f32).max(height as f32).log2().floor() as u8 + 1, usage_base),
        };
        let img = Image::new(
            &self.device,
            &self.memory_types,
            texture.format,
            texture.width,
            texture.height,
            texture.layer_count,
            view_kind,
            mip_levels,
            usage,
        );

        assert_eq!(texture.fbo_ids.len(), 0);

        {
            let mut cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer(false);

            if let Some(barrier) = img.core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                img.core.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            self.upload_queue.push(cmd_buffer.finish());
        }

        if let Some(rt_info) = render_target {
            let (depth_rb, allocate_depth) = match texture.depth_rb {
                Some(rbo) => (rbo, is_resized || !rt_info.has_depth),
                None if rt_info.has_depth => {
                    let depth_rb = self.generate_rbo_id();
                    texture.depth_rb = Some(depth_rb);
                    (depth_rb, true)
                },
                None => (RBOId(0), false),
            };

            if allocate_depth {
                if self.rbos.contains_key(&depth_rb) {
                    let old_rbo = self.rbos.remove(&depth_rb).unwrap();
                    old_rbo.deinit(&self.device);
                }
                if rt_info.has_depth {
                    let rbo = DepthBuffer::new(
                        &self.device,
                        &self.memory_types,
                        texture.width,
                        texture.height,
                        self.depth_format
                    );
                    self.rbos.insert(depth_rb, rbo);
                } else {
                    texture.depth_rb = None;
                }
            }

            let new_fbos = self.generate_fbo_ids(texture.layer_count);

            for i in 0..texture.layer_count as u16 {
                let (rbo_id, depth) = match texture.depth_rb {
                    Some(rbo_id) => (rbo_id.clone(), Some(&self.rbos[&rbo_id].core.view)),
                    None => (RBOId(0), None)
                };
                let fbo = Framebuffer::new(&self.device, &texture, &img, i, self.render_pass.as_ref().unwrap(), rbo_id.clone(), depth);
                self.fbos.insert(new_fbos[i as usize], fbo);
                texture.fbo_ids.push(new_fbos[i as usize]);
            }
        }

        self.images.insert(texture.id, img);


        if let Some(data) = pixels {
            let len = data.len() / layer_count as usize;
            for i in 0..layer_count {
                let start = len * i as usize;
                self.upload_queue
                    .push(
                        self.images
                            .get_mut(&texture.id)
                            .expect("Texture not found.")
                            .update(
                                &mut self.device,
                                &mut self.command_pool[self.next_id],
                                &mut self.staging_buffer_pool[self.next_id],
                                DeviceUintRect::new(
                                    DeviceUintPoint::new(0, 0),
                                    DeviceUintSize::new(texture.width, texture.height),
                                ),
                                i,
                                texels_to_u8_slice(&data[start .. (start + len)]),
                            )
                    );
            }
            if texture.filter == TextureFilter::Trilinear {
                self.generate_mipmaps(texture);
            }
        }
    }

    fn generate_mipmaps(&mut self, texture: &Texture) {
        let mut cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer(false);

        let image = self.images
            .get_mut(&texture.id)
            .expect("Texture not found.");

        let mut mip_width = texture.width;
        let mut mip_height = texture.height;

        let mut half_mip_width =  mip_width / 2;
        let mut half_mip_height =  mip_height / 2;

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
                &[
                    hal::command::ImageBlit {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: index - 1,
                            layers: 0 .. 1,
                        },
                        src_bounds: hal::image::Offset {
                            x: 0,
                            y: 0,
                            z: 0,
                        } .. hal::image::Offset {
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
                    }
                ],
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
            hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
            hal::image::Layout::ColorAttachmentOptimal,
            image.core.subresource_range.clone(),
        ) {
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                hal::memory::Dependencies::empty(),
                &[barrier],
            );
        }

        self.upload_queue.push(cmd_buffer.finish());
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

        let mut cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer(false);

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

        let barriers = src_img.transit(
            hal::image::Access::TRANSFER_READ,
            hal::image::Layout::TransferSrcOptimal,
            src_range.clone(),
        ).into_iter().chain(
            dest_img.transit(
                hal::image::Access::TRANSFER_WRITE,
                hal::image::Layout::TransferDstOptimal,
                dest_range.clone(),
            )
        );
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
                &[
                    hal::command::ImageBlit {
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
                    }
                ],
            );
        } else {
            cmd_buffer.copy_image(
                &src_img.image,
                hal::image::Layout::TransferSrcOptimal,
                &dest_img.image,
                hal::image::Layout::TransferDstOptimal,
                &[
                    hal::command::ImageCopy {
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
                    }
                ],
            );
        }

        // the blit caller code expects to be able to render to the target
        let barriers = src_img.transit(
            hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
            hal::image::Layout::ColorAttachmentOptimal,
            src_range,
        ).into_iter().chain(dest_img.transit(
            hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
            hal::image::Layout::ColorAttachmentOptimal,
            dest_range,
        ));

        cmd_buffer.pipeline_barrier(
            PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            hal::memory::Dependencies::empty(),
            barriers,
        );

        self.upload_queue.push(cmd_buffer.finish());
    }

    pub fn free_texture_storage(&mut self, texture: &mut Texture) {
        debug_assert!(self.inside_frame);
        if texture.width + texture.height == 0 {
            return;
        }

        self.free_image(texture);

        texture.width = 0;
        texture.height = 0;
        texture.layer_count = 0;
        texture.id = 0;
    }

    pub fn free_image(&mut self, texture: &mut Texture) {
        // Note: this is a very rare case, but if it becomes a problem
        // we need to handle this in renderer.rs
        if texture.still_in_flight(self.frame_id) && !self.upload_queue.is_empty() {
            println!("free_image");
            self.wait_for_resources();
            /*let fence = self.device.create_fence(false);
            self.device.reset_fence(&fence);
            {
                println!("self.upload_queue.len={:?}", self.upload_queue.len());
                let submission = Submission::new()
                    .submit(self.upload_queue.drain(..));
                self.queue_group.queues[0].submit(submission, Some(&fence));
            }
            self.device.wait_for_fence(&fence, !0);
            self.device.destroy_fence(fence);*/
            self.reset_command_pools();
        }
        if let Some(depth_rb) = texture.depth_rb.take() {
            let old_rbo = self.rbos.remove(&depth_rb).unwrap();
            old_rbo.deinit(&self.device);
        }

        if !texture.fbo_ids.is_empty() {
            for old in texture.fbo_ids.drain(..) {
                let old_fbo = self.fbos.remove(&old).unwrap();
                old_fbo.deinit(&self.device);
            }
        }

        let image = self.images.remove(&texture.id).expect("Texture not found.");
        image.deinit(&self.device);
    }

    pub fn delete_texture(&mut self, mut texture: Texture) {
        self.free_texture_storage(&mut texture);
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
        PBO { }
    }

    pub fn delete_pbo(&mut self, _pbo: PBO) {
    }

    pub fn upload_texture<'a>(
        &'a mut self,
        texture: &'a Texture,
        _pbo: &PBO,
        _upload_count: usize,
    ) -> TextureUploader<'a, B> {
        debug_assert!(self.inside_frame);

        match self.upload_method {
            UploadMethod::Immediate => unimplemented!(),
            UploadMethod::PixelBuffer(..) => {
                TextureUploader {
                    device: self,
                    texture,
                }
            },
        }

    }

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    pub fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8> {
        let mut pixels = vec![0; (img_desc.size.width * img_desc.size.height * 4) as usize];
        self.read_pixels_into(
            DeviceUintRect::new(DeviceUintPoint::zero(), DeviceUintSize::new(img_desc.size.width, img_desc.size.height)),
            ReadPixelsFormat::Rgba8,
            &mut pixels,
        );
        pixels
    }

    /// Read rectangle of pixels into the specified output slice.
    pub fn read_pixels_into(
        &mut self,
        rect: DeviceUintRect,
        format: ReadPixelsFormat,
        output: &mut [u8],
    ) {
        println!("read_pixels_into");
        self.wait_for_resources();

        let bytes_per_pixel = match format {
            ReadPixelsFormat::Standard(imf) => imf.bytes_per_pixel(),
            ReadPixelsFormat::Rgba8 => 4,
        };
        let size_in_bytes = (bytes_per_pixel * rect.size.width * rect.size.height) as usize;
        assert_eq!(output.len(), size_in_bytes);
        let capture_read = cfg!(feature = "capture") && self.bound_read_texture.0 != INVALID_TEXTURE_ID;

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

        let mut command_pool = self.device.create_command_pool_typed(
            &self.queue_group,
            hal::pool::CommandPoolCreateFlags::empty(),
            4,
        );
        command_pool.reset();

        let copy_submit = {
            let mut cmd_buffer = command_pool.acquire_command_buffer(false);
            let range = hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
                layers: layer .. layer + 1,
            };
            let barriers = download_buffer.transit(hal::buffer::Access::TRANSFER_WRITE)
                .into_iter()
                .chain(image.transit(
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                    range.clone()));

            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers
            );

            cmd_buffer.copy_image_to_buffer(
                &image.image,
                hal::image::Layout::TransferSrcOptimal,
                &download_buffer.buffer,
                &[hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: rect.size.width,
                    buffer_height: rect.size.height,
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
                }]);
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
            cmd_buffer.finish()
        };

        let copy_fence = self.device.create_fence(false);
        self.device.reset_fence(&copy_fence);
        let submission = hal::queue::Submission::new()
            .submit(Some(copy_submit));
        self.queue_group.queues[0].submit(submission, Some(&copy_fence));
        self.device.wait_for_fence(&copy_fence, !0);
        self.device.destroy_fence(copy_fence);

        let mut data = vec![0; download_buffer.buffer_size];
        if let Ok(reader) = self.device
            .acquire_mapping_reader::<u8>(
                &download_buffer.memory,
                0 .. download_buffer.buffer_size as u64,
            )
        {
            data[0..reader.len()].copy_from_slice(&reader);
            self.device.release_mapping_reader(reader);
        } else {
            panic!("Fail to read the download buffer!");
        }
        data.truncate(output.len());
        if !capture_read && self.surface_format.base_format().0 == hal::format::SurfaceType::B8_G8_R8_A8 {
            let mut offset = 0;
            for _ in 0..output.len()/4 {
                output[offset + 0] = data[offset + 2];
                output[offset + 1] = data[offset + 1];
                output[offset + 2] = data[offset + 0];
                output[offset + 3] = data[offset + 3];
                offset += 4;
            }
        } else {
            output.swap_with_slice(&mut data);
        }

        download_buffer.deinit(&self.device);
        command_pool.reset();
        self.device.destroy_command_pool(command_pool.into_raw());
    }

    /// Get texels of a texture into the specified output slice.
    #[cfg(feature = "debug_renderer")]
    pub fn get_tex_image_into(
        &mut self,
        _texture: &Texture,
        _format: ImageFormat,
        _output: &mut [u8],
    ) {
        unimplemented!();
    }

    /// Attaches the provided texture to the current Read FBO binding.
    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    fn attach_read_texture_raw(
        &mut self, texture_id: u32, _target: TextureTarget, layer_id: i32
    ) {
        self.bind_read_texture(texture_id, layer_id);
    }

    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    pub fn attach_read_texture_external(
        &mut self, _texture_id: u32, _target: TextureTarget, _layer_id: i32
    ) {
        unimplemented!();
    }

    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    pub fn attach_read_texture(&mut self, texture: &Texture, layer_id: i32) {
        self.attach_read_texture_raw(texture.id, texture.target.into(), layer_id)
    }

    pub fn invalidate_read_texture(&mut self) {
        self.bound_read_texture = (INVALID_TEXTURE_ID, 0);
    }

    pub fn bind_vao(&mut self, _vao: &VAO) { }

    pub fn create_vao(&mut self, _descriptor: &VertexDescriptor) -> VAO {
        VAO { }
    }

    pub fn delete_vao(&mut self, _vao: VAO) { }

    pub fn create_vao_with_new_instances(
        &mut self,
        _descriptor: &VertexDescriptor,
        _base_vao: &VAO,
    ) -> VAO {
        VAO { }
    }

    pub fn update_vao_main_vertices<V: Copy + PrimitiveType>(
        &mut self,
        _vao: &VAO,
        _vertices: &[V],
        _usage_hint: VertexUsageHint,
    ) {
        if cfg!(feature = "debug_renderer") && self.bound_program != INVALID_PROGRAM_ID {
            self.update_vertices(&_vertices.iter().map(|v| v.to_primitive_type()).collect::<Vec<_>>());
        }
    }

    pub fn update_vao_instances<V: PrimitiveType>(
        &mut self,
        _vao: &VAO,
        instances: &[V],
        _usage_hint: VertexUsageHint,
    ) {
        let data = instances.iter().map(|pi| pi.to_primitive_type()).collect::<Vec<V::Primitive>>();
        self.update_instances(&data);
    }

    pub fn update_vao_indices<I: Copy>(
        &mut self,
        _vao: &VAO,
        _indices: &[I],
        _usage_hint: VertexUsageHint
    ) {
        if self.bound_program != INVALID_PROGRAM_ID {
            #[cfg(feature = "debug_renderer")]
            self.update_indices(_indices);
        }
    }

    pub fn draw_triangles_u16(&mut self, _first_vertex: i32, _index_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    #[cfg(feature = "debug_renderer")]
    pub fn draw_triangles_u32(&mut self, _first_vertex: i32, _index_count: i32) {
        debug_assert!(self.inside_frame);
        self.bind_textures();
        self.draw();
    }

    #[cfg(feature = "debug_renderer")]
    pub fn draw_nonindexed_lines(&mut self, _first_vertex: i32, _vertex_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn draw_indexed_triangles_instanced_u16(&mut self, _index_count: i32, _instance_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn end_frame(&mut self) {
        self.bind_draw_target(None, None);
        self.bind_read_target(None);

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
            return
        }
        let mut clears = Vec::new();
        let mut rects = Vec::new();
        if let Some(color) =  color {
            clears.push(
                hal::command::AttachmentClear::Color {
                    index: 0,
                    value: hal::command::ClearColor::Float(color),
                }
            );
            rects.push(
                hal::pso::ClearRect {
                    rect: hal::pso::Rect {
                        x: rect.origin.x as i16,
                        y: rect.origin.y as i16,
                        w: rect.size.width as i16,
                        h: rect.size.height as i16,
                    },
                    layers: 0 .. 1,
                }
            );
        }

        if let Some(depth) = depth {
            assert!(self.current_depth_test != DepthTest::Off);
            clears.push(
                hal::command::AttachmentClear::DepthStencil {
                    depth: Some(depth),
                    stencil: None,
                },
            );
            rects.push(
                hal::pso::ClearRect {
                    rect: hal::pso::Rect {
                        x: rect.origin.x as i16,
                        y: rect.origin.y as i16,
                        w: rect.size.width as i16,
                        h: rect.size.height as i16,
                    },
                    layers: 0 .. 1,
                }
            );
        }

        let submit = {
            let (frame_buffer, format, has_depth) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
                (&self.fbos[&self.bound_draw_fbo].fbo, self.fbos[&self.bound_draw_fbo].format, self.fbos[&self.bound_draw_fbo].rbo != RBOId(0))
            } else {
                if self.current_depth_test == DepthTest::Off {
                    (&self.framebuffers[self.current_frame_id], ImageFormat::BGRA8, false)
                } else {
                    (&self.framebuffers_depth[self.current_frame_id], ImageFormat::BGRA8, true)
                }
            };

            let render_pass = self.render_pass.as_ref().unwrap().get_render_pass(format, has_depth);
            let mut cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer(false);
            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    render_pass,
                    frame_buffer,
                    self.viewport.rect,
                    &vec![],
                );

                encoder.clear_attachments(clears, rects);
            }
            cmd_buffer.finish()
        };

        self.upload_queue.push(submit);
    }

    fn clear_target_image(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
    ) {
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
            (&self.frame_images[self.current_frame_id], 0, Some(&self.frame_depths[self.current_frame_id].core))
        };

        let mut cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer(false);
        //Note: this function is assumed to be called within an active FBO
        // thus, we bring back the targets into renderable state

        if let Some(color) = color {
            let color_range = hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
                layers: layer .. layer + 1,
            };
            if let Some(barrier) = img.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::TransferDstOptimal,
                color_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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
        self.upload_queue.push(cmd_buffer.finish());
    }

    pub fn clear_target(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<DeviceIntRect>,
    ) {
        if let Some(rect) = rect {
            let target_rect = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
                let extent = &self.images[&self.fbos[&self.bound_draw_fbo].texture].kind.extent();
                DeviceIntRect::new(DeviceIntPoint::zero(), DeviceIntSize::new(extent.width as _, extent.height as _))
            } else {
                DeviceIntRect::new(DeviceIntPoint::zero(), DeviceIntSize::new(self.viewport.rect.w as _, self.viewport.rect.h as _))
            };
            if rect.size.width > target_rect.size.width || rect.size.height > target_rect.size.height {
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

    pub fn set_blend(&mut self, enable: bool) {
        if !enable {
            self.current_blend_state = BlendState::Off
        }
    }

    pub fn set_blend_mode_alpha(&mut self) {
        self.current_blend_state = ALPHA;
    }

    pub fn set_blend_mode_premultiplied_alpha(&mut self) {
        self.current_blend_state = BlendState::PREMULTIPLIED_ALPHA;
    }

    pub fn set_blend_mode_premultiplied_dest_out(&mut self) {
        self.current_blend_state = PREMULTIPLIED_DEST_OUT;
    }

    pub fn set_blend_mode_multiply(&mut self) {
        self.current_blend_state = BlendState::MULTIPLY;
    }
    pub fn set_blend_mode_max(&mut self) {
        self.current_blend_state = MAX;
    }
    #[cfg(feature = "debug_renderer")]
    pub fn set_blend_mode_min(&mut self) {
        self.current_blend_state = MIN;
    }
    pub fn set_blend_mode_subpixel_pass0(&mut self) {
        self.current_blend_state = SUBPIXEL_PASS0;
    }
    pub fn set_blend_mode_subpixel_pass1(&mut self) {
        self.current_blend_state = SUBPIXEL_PASS1;
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass0(&mut self) {
        self.current_blend_state = SUBPIXEL_WITH_BG_COLOR_PASS0;
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass1(&mut self) {
        self.current_blend_state = SUBPIXEL_WITH_BG_COLOR_PASS1;
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass2(&mut self) {
        self.current_blend_state = SUBPIXEL_WITH_BG_COLOR_PASS2;
    }
    pub fn set_blend_mode_subpixel_constant_text_color(&mut self, color: ColorF) {
        self.current_blend_state = SUBPIXEL_CONSTANT_TEXT_COLOR;
        // color is an unpremultiplied color.
        self.blend_color = ColorF::new(color.r, color.g, color.b, 1.0);
    }
    pub fn set_blend_mode_subpixel_dual_source(&mut self) {
        self.current_blend_state = SUBPIXEL_DUAL_SOURCE;
    }

    pub fn set_blend_mode_show_overdraw(&mut self) {
        self.current_blend_state = OVERDRAW;
    }

    pub fn supports_features(&self, features: hal::Features) -> bool {
        self.features.contains(features)
    }

    pub fn echo_driver_messages(&self) {
        warn!("echo_driver_messages is unimplemeneted");
    }

    pub fn set_next_frame_id(&mut self) -> bool {
        match self.swap_chain.as_mut().unwrap()
            .acquire_image(!0, FrameSync::Semaphore(&mut self.image_available_semaphore))
        {
            Ok(id) => {
                self.current_frame_id = id as _;
                true
            },
            Err(_) => false,
        }
    }

    pub fn submit_to_gpu(&mut self) -> bool {
        {
            let mut cmd_buffer = self.command_pool[self.next_id].acquire_command_buffer(false);
            let image = &self.frame_images[self.current_frame_id];
            if let Some(barrier) = image.transit(
                hal::image::Access::empty(),
                hal::image::Layout::Present,
                image.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            self.upload_queue.push(cmd_buffer.finish());
        }

        let present_error = {
            let submission = Submission::new()
                .wait_on(&[(&self.image_available_semaphore, PipelineStage::BOTTOM_OF_PIPE)])
                .signal(Some(&self.render_finished_semaphore))
                .submit(self.upload_queue.drain(..));
            self.queue_group.queues[0].submit(submission, Some(&mut self.frame_fence[self.next_id].inner));
            self.frame_fence[self.next_id].is_submitted = true;

            // present frame
            self.swap_chain.as_mut().unwrap()
                .present(
                    &mut self.queue_group.queues[0],
                    self.current_frame_id as _,
                    Some(&self.render_finished_semaphore)).is_err()
        };
        self.next_id = (self.next_id + 1) % MAX_FRAME_COUNT;
        self.reset_state();
        if self.frame_fence[self.next_id].is_submitted {
            self.device.wait_for_fence(&self.frame_fence[self.next_id].inner, 1000);
            self.device.reset_fence(&self.frame_fence[self.next_id].inner);
            self.frame_fence[self.next_id].is_submitted = false;
        }
        self.command_pool[self.next_id].reset();
        self.staging_buffer_pool[self.next_id].reset();
        self.descriptor_pools[self.next_id].reset();
        self.reset_program_buffer_offsets();
        return !present_error;
    }

    pub fn wait_for_resources_and_reset(&mut self) {
        self.wait_for_resources();
        self.reset_command_pools();
    }

    fn wait_for_resources(&mut self) {
        for fence in &mut self.frame_fence {
            if fence.is_submitted {
                self.device.wait_for_fence(&fence.inner, 1000);
                self.device.reset_fence(&fence.inner);
                fence.is_submitted = false;
            }
        }
    }

    fn reset_command_pools(&mut self) {
        for command_pool in &mut self.command_pool {
            command_pool.reset();
        }
    }

    pub fn deinit(self) {
        for command_pool in self.command_pool {
            self.device.destroy_command_pool(command_pool.into_raw());
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
        for (_, program) in self.programs {
            program.deinit(&self.device)
        }
        self.render_pass.unwrap().deinit(&self.device);
        for fence in self.frame_fence {
            self.device.destroy_fence(fence.inner);
        }
        self.device.destroy_semaphore(self.image_available_semaphore);
        self.device.destroy_semaphore(self.render_finished_semaphore);
        self.device.destroy_swapchain(self.swap_chain.unwrap());
    }
}

pub struct TextureUploader<'a, B: hal::Backend> {
    device: &'a mut Device<B>,
    texture: &'a Texture,
}

impl<'a, B: hal::Backend> TextureUploader<'a, B> {
    pub fn upload<T>(
        &mut self,
        rect: DeviceUintRect,
        layer_index: i32,
        stride: Option<u32>,
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
        let data= if stride.is_some() {
            let row_length = (stride.unwrap()) as usize;

            if data_stride == 4 {
                for j in 0..height {
                    for i in 0..width {
                        let offset = i * data_stride + j * data_stride * width;
                        let src = &data[j * row_length + i * data_stride ..];
                        assert!(
                            offset + 3 < new_data.len(),
                            "offset={:?}, data len={:?}, data stride={:?}",
                            offset, new_data.len(), data_stride,
                        ); // optimization
                        // convert from BGRA
                        new_data[offset + 0] = src[0];
                        new_data[offset + 1] = src[1];
                        new_data[offset + 2] = src[2];
                        new_data[offset + 3] = src[3];
                    }
                }
            } else {
                for j in 0..height {
                    for i in 0..width {
                        let offset = i * data_stride + j * data_stride * width;
                        let src = &data[j * row_length + i * data_stride ..];
                        for i in 0..data_stride {
                            new_data[offset + i] = src[i];
                        }

                    }
                }
            }

            new_data.as_slice()
        } else {
            data
        };
        assert_eq!(data.len(), width * height * data_stride,
                   "data len = {}, width = {}, height = {}, data stride = {}",
                   data.len(), width, height, data_stride);
        self.device.upload_queue
            .push(
                self.device.images
                    .get_mut(&self.texture.id)
                    .expect("Texture not found.")
                    .update(
                        &mut self.device.device,
                        &mut self.device.command_pool[self.device.next_id],
                        &mut self.device.staging_buffer_pool[self.device.next_id],
                        rect,
                        layer_index,
                        data,
                    )
            );

        if self.texture.filter == TextureFilter::Trilinear {
            self.device.generate_mipmaps(self.texture);
        }
        size
    }
}

fn texels_to_u8_slice<T: Texel>(texels: &[T]) -> &[u8] {
    unsafe {
        slice::from_raw_parts(texels.as_ptr() as *const u8, texels.len() * mem::size_of::<T>())
    }
}
