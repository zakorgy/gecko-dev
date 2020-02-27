/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, ImageFormat, MemoryReport, MixBlendMode};
use api::round_to_int;
use api::units::{
    DeviceIntPoint, DeviceIntRect, DeviceIntSize, FramebufferIntPoint, FramebufferIntRect,
    FramebufferIntSize,
};
use api::TextureTarget;
#[cfg(feature = "capture")]
use api::ImageDescriptor;
use arrayvec::ArrayVec;
use euclid::default;
use crate::internal_types::{FastHashMap, RenderTargetInfo, Swizzle, SwizzleSettings};
use rand::{self, Rng};
use rendy_memory::{Block, DynamicConfig, Heaps, HeapsConfig, LinearConfig, MemoryUsageValue};
use rendy_descriptor::{DescriptorAllocator, DescriptorRanges, DescriptorSet};
use ron::de::from_str;
use smallvec::SmallVec;
use std::borrow::Borrow;
use std::cell::Cell;
use std::convert::Into;
use std::collections::hash_map::Entry;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::mem;
use std::path::PathBuf;
use std::num::NonZeroUsize;
use std::rc::Rc;
use std::slice;
use std::sync::{Arc, Mutex};

use super::blend_state::*;
use super::buffer::*;
use super::command::*;
use super::descriptor::*;
use super::image::*;
use super::program::{Program, RenderPassDepthState};
use super::render_pass::*;
use super::{PipelineRequirements, PrimitiveType, TextureId};
use super::{LESS_EQUAL_TEST, LESS_EQUAL_WRITE, MAX_FRAME_COUNT};
use super::vertex_types;

use super::super::{BoundPBO, Capabilities};
use super::super::{ShaderKind, ExternalTexture, GpuFrameId, TextureSlot, TextureFilter};
use super::super::{VertexDescriptor, UploadMethod, Texel, TextureFlags, TextureFormatPair};
use super::super::{
    Texture, DrawTarget, ReadTarget, FBOId, RBOId, PBO, VertexUsageHint, ShaderError,
    ShaderPrecacheFlags, SharedDepthTarget, ProgramCache,
};
use super::super::{depth_target_size_in_bytes, record_gpu_alloc, record_gpu_free};
use super::super::super::shader_source;

use hal;
use hal::adapter::PhysicalDevice;
use hal::pso::{BlendState, DepthTest};
use hal::device::Device as _;
use hal::window::{Surface, SwapchainConfig, PresentationSurface};
use hal::pso::PipelineStage;
use hal::queue::CommandQueue;
use hal::command::{
    ClearColor, ClearDepthStencil, ClearValue, CommandBufferFlags, CommandBufferInheritanceInfo,
    CommandBuffer, Level,
};
use hal::pool::{CommandPool as HalCommandPool};
use hal::queue::{QueueFamilyId};

pub const INVALID_TEXTURE_ID: TextureId = 0;
pub const INVALID_PROGRAM_ID: ProgramId = ProgramId(0, DescriptorGroup::Invalid);
pub const DEFAULT_READ_FBO: FBOId = FBOId(0);
pub const DEFAULT_DRAW_FBO: FBOId = FBOId(1);
pub const DEBUG_READ_FBO: FBOId = FBOId(2);

const HEADLESS_FRAME_COUNT: usize = 1;
const SURFACE_FORMAT: hal::format::Format = hal::format::Format::Bgra8Unorm;
const DEPTH_FORMAT: hal::format::Format = hal::format::Format::D32Sfloat;
const DITHER_BINDING: usize = 8;

#[derive(PartialEq)]
pub enum BackendApiType {
    Vulkan,
    Dx12,
    Metal,
}

#[derive(Debug, Eq, PartialEq)]
pub enum DrawTargetUsage {
    // Only a blit or copy operation is applied to the target
    CopyOnly,
    // The target will be drawn or cleared and copy is also possible
    Draw,
}

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

pub struct DeviceInit<B: hal::Backend> {
    pub instance: B::Instance,
    pub adapter: hal::adapter::Adapter<B>,
    pub surface: Option<B::Surface>,
    pub dimensions: (i32, i32),
    pub descriptor_count: Option<u32>,
    pub cache_path: Option<PathBuf>,
    pub save_cache: bool,
    pub backend_api: BackendApiType,
}

const NON_SPECIALIZATION_FEATURES: &'static [&'static str] = &[
    "TEXTURE_RECT",
    "TEXTURE_2D",
    "DUAL_SOURCE_BLENDING",
    "FAST_PATH",
];

#[repr(u32)]
pub enum DepthFunction {
    Less,
    LessEqual,
}

impl ShaderKind {
    pub(super) fn is_debug(&self) -> bool {
        match *self {
            ShaderKind::DebugFont | ShaderKind::DebugColor => true,
            _ => false,
        }
    }
}

impl Texture {
    pub fn still_in_flight(&self, frame_id: GpuFrameId, frame_count: usize) -> bool {
        for i in 0..frame_count {
            if self.bound_in_frame.get() == GpuFrameId(frame_id.0 - i) {
                return true;
            }
        }
        false
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct ProgramId(u32, DescriptorGroup);

pub struct VAO;

struct Fence<B: hal::Backend> {
    inner: B::Fence,
    is_submitted: bool,
}

#[derive(Debug)]
struct Frame<B: hal::Backend> {
    swapchain_image: <B::Surface as PresentationSurface<B>>::SwapchainImage,
    framebuffers: FastHashMap<(hal::image::Layout, hal::image::Layout, bool), B::Framebuffer>,
}

impl<B: hal::Backend> Frame<B> {
    fn new(swapchain_image: <B::Surface as PresentationSurface<B>>::SwapchainImage) -> Self {
        Frame {
            swapchain_image,
            framebuffers: FastHashMap::default(),
        }
    }

    fn get_or_create_fbo(
        &mut self,
        device: &B::Device,
        old_layout: hal::image::Layout,
        new_layout: hal::image::Layout,
        clear: bool,
        depth: &B::ImageView,
        viewport_rect: hal::pso::Rect,
        render_passe: &B::RenderPass,
    ) -> &B::Framebuffer {
        let key = (old_layout, new_layout, clear);
        if let Entry::Vacant(v) = self.framebuffers.entry(key) {
            let image_extent = hal::image::Extent {
                width: viewport_rect.w as _,
                height: viewport_rect.h as _,
                depth: 1,
            };
            let framebuffer = unsafe {
                device.create_framebuffer(
                    &render_passe,
                    std::iter::once(self.swapchain_image.borrow()).chain(std::iter::once(depth)),
                    image_extent,
                )
            }
            .expect("Failed to create Framebuffer");
            v.insert(framebuffer);
        }
        &self.framebuffers[&key]
    }

    fn deinit(self, device: &B::Device) {
        for (_, fb) in self.framebuffers {
            unsafe {
                device.destroy_framebuffer(fb);
            }
        }
    }
}

struct ClearValues {
    color: ClearValue,
    depth: Option<ClearValue>,
}

pub struct Device<B: hal::Backend> {
    pub device: Arc<B::Device>,
    pub heaps: Arc<Mutex<Heaps<B>>>,
    pub limits: hal::Limits,
    adapter: hal::adapter::Adapter<B>,
    surface: Option<B::Surface>,
    _instance: B::Instance,
    pub surface_format: ImageFormat,
    pub depth_format: hal::format::Format,
    pub queue_group_family: QueueFamilyId,
    pub queue_group_queues: Vec<B::CommandQueue>,
    command_pools: ArrayVec<[CommandPool<B>; MAX_FRAME_COUNT]>,
    command_buffer: B::CommandBuffer,
    staging_buffer_pool: ArrayVec<[BufferPool<B>; MAX_FRAME_COUNT]>,
    frame: Option<Frame<B>>,
    frame_depth: DepthBuffer<B>,
    swapchain_image_layouts: ArrayVec<[hal::image::Layout; MAX_FRAME_COUNT]>,
    readback_texture: Option<Texture>,
    render_passes: HalRenderPasses<B>,
    pub frame_count: usize,
    pub viewport: hal::pso::Viewport,
    scissor_rect: hal::pso::Rect,
    scissor_enabled: bool,
    pub dimensions: (i32, i32),
    pub sampler_linear: B::Sampler,
    pub sampler_nearest: B::Sampler,
    current_blend_state: Option<BlendState>,
    blend_color: ColorF,
    current_depth_test: Option<DepthTest>,
    clear_values: FastHashMap<FBOId, ClearValues>,
    // device state
    programs: FastHashMap<ProgramId, Program<B>>,
    blit_programs: FastHashMap<hal::image::ViewKind, (ProgramId, Program<B>)>,
    next_program_id: u32,
    shader_modules: FastHashMap<String, (B::ShaderModule, B::ShaderModule)>,
    images: FastHashMap<TextureId, Image<B>>,
    pub(crate) gpu_cache_buffer: Option<GpuCacheBuffer<B>>,
    pub(crate) gpu_cache_buffers: FastHashMap<TextureId, PersistentlyMappedBuffer<B>>,
    retained_textures: Vec<Texture>,
    fbos: FastHashMap<FBOId, Framebuffer<B>>,
    rbos: FastHashMap<RBOId, DepthBuffer<B>>,

    desc_allocator: DescriptorAllocator<B>,

    per_draw_descriptors: DescriptorSetHandler<PerDrawBindings, B, Vec<DescriptorSet<B>>>,
    bound_per_draw_textures: PerDrawBindings,
    pub(crate) bound_per_draw_descriptor: Option<DescriptorSet<B>>,

    per_pass_descriptors: DescriptorSetHandler<PerPassBindings, B, Vec<DescriptorSet<B>>>,
    bound_per_pass_textures: PerPassBindings,
    bound_per_pass_descriptor: Option<DescriptorSet<B>>,

    per_group_descriptors: DescriptorSetHandler<
        (DescriptorGroup, PerGroupBindings),
        B,
        FastHashMap<DescriptorGroup, Vec<DescriptorSet<B>>>,
    >,
    bound_per_group_textures: PerGroupBindings,
    bound_per_group_descriptors: [Option<DescriptorSet<B>>; 3],

    // Locals related things
    uniform_buffer_handler: UniformBufferHandler<B>,
    bound_projection: default::Transform3D<f32>,

    descriptor_data: DescriptorData<B>,
    bound_textures: [u32; RENDERER_TEXTURE_COUNT],
    bound_program: ProgramId,
    bound_sampler: [TextureFilter; RENDERER_TEXTURE_COUNT],
    bound_read_texture: (TextureId, i32),
    bound_read_fbo: FBOId,
    bound_draw_fbo: FBOId,
    draw_target_usage: DrawTargetUsage,
    //default_read_fbo: FBOId,
    //default_draw_fbo: FBOId,
    device_pixel_ratio: f32,
    depth_available: bool,
    upload_method: UploadMethod,
    quad_buffer: VertexBufferHandler<B>,
    instance_buffers: ArrayVec<[InstanceBufferHandler<B>; MAX_FRAME_COUNT]>,
    free_instance_buffers: Vec<InstancePoolBuffer<B>>,
    download_buffer: Option<Buffer<B>>,
    instance_buffer_range: std::ops::Range<usize>,

    // HW or API capabilities
    capabilities: Capabilities,

    /// Map from texture dimensions to shared depth buffers for render targets.
    ///
    /// Render targets often have the same width/height, so we can save memory
    /// by sharing these across targets.
    depth_targets: FastHashMap<DeviceIntSize, SharedDepthTarget>,

    // debug
    inside_frame: bool,
    inside_render_pass: bool,
    render_pass_depth_state: RenderPassDepthState,

    // resources
    _resource_override_path: Option<PathBuf>,

    max_texture_size: i32,
    _renderer_name: String,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    pub frame_id: GpuFrameId,

    // Supported features
    features: hal::Features,

    next_id: usize,
    frame_fence: ArrayVec<[Fence<B>; MAX_FRAME_COUNT]>,
    render_finished_semaphores: ArrayVec<[B::Semaphore; MAX_FRAME_COUNT]>,
    pipeline_requirements: FastHashMap<String, PipelineRequirements>,
    pipeline_cache: Option<B::PipelineCache>,
    cache_path: Option<PathBuf>,
    save_cache: bool,

    swizzle_settings: SwizzleSettings,
    color_formats: TextureFormatPair<ImageFormat>,
    optimal_pbo_stride: NonZeroUsize,
    last_rp_in_frame_reached: bool,
    pub readback_supported: bool,
    #[cfg(debug_assertions)]
    shader_is_ready: bool,
    rebind_descriptors: bool,
}

impl<B: hal::Backend> Device<B> {
    pub fn new(
        init: DeviceInit<B>,
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _cached_programs: Option<Rc<ProgramCache>>,
        _allow_pixel_local_storage_support: bool,
        _allow_texture_storage_support: bool,
        _allow_texture_swizzling: bool,
        _dump_shader_source: Option<String>,
        heaps_config: HeapsConfig,
        instance_buffer_size: usize,
        texture_cache_size: usize,
        readback_supported: bool,
    ) -> Self {
        let DeviceInit {
            instance,
            adapter,
            mut surface,
            dimensions,
            descriptor_count,
            cache_path,
            save_cache,
            backend_api: _backend_api,
        } = init;
        let renderer_name = "TODO renderer name".to_owned();
        let features = adapter.physical_device.features();

        let memory_properties = adapter.physical_device.memory_properties();
        let mut heaps = {
            let types = memory_properties.memory_types.iter().map(|ref mt| {
                let mut config = heaps_config;
                if !mt.properties.contains(hal::memory::Properties::CPU_VISIBLE) {
                    config.linear = None;
                } else if config.linear.is_none() {
                    config.linear = Some(LinearConfig {
                        linear_size: (memory_properties.memory_heaps[mt.heap_index as usize] / 8
                            - 1)
                        .next_power_of_two(),
                    });
                }
                if config.dynamic.is_none() {
                    config.dynamic = Some(DynamicConfig {
                        min_device_allocation: (memory_properties.memory_heaps
                            [mt.heap_index as usize]
                            / 1024
                            - 1)
                        .next_power_of_two(),
                        block_size_granularity: (memory_properties.memory_heaps
                            [mt.heap_index as usize]
                            / 1024
                            - 1)
                        .next_power_of_two(),
                        max_chunk_size: (memory_properties.memory_heaps[mt.heap_index as usize]
                            / 8
                            - 1)
                        .next_power_of_two(),
                    })
                }
                (mt.properties, mt.heap_index as u32, config)
            });

            let heaps = memory_properties.memory_heaps.iter().cloned();
            unsafe { Heaps::new(types, heaps) }
        };

        let limits = adapter.physical_device.limits();
        let max_texture_size = limits.max_image_2d_size as i32;

        let (device, queue_group_family, queue_group_queues) = {
            use hal::queue::QueueFamily;

            let family = adapter
                .queue_families
                .iter()
                .find(|family| {
                    family.queue_type().supports_graphics()
                        && match &surface {
                            Some(surface) => surface.supports_queue_family(family),
                            None => true,
                        }
                })
                .unwrap();

            let priorities = vec![1.0];
            let (id, families) = (family.id(), [(family, priorities.as_slice())]);
            let hal::adapter::Gpu {
                device,
                mut queue_groups,
            } = unsafe {
                adapter
                    .physical_device
                    .open(&families, hal::Features::DUAL_SRC_BLENDING)
                    .unwrap_or_else(|_| {
                        adapter
                            .physical_device
                            .open(&families, hal::Features::empty())
                            .unwrap()
                    })
            };
            (device, id, queue_groups.remove(id.0 as _).queues)
        };

        let render_passes =
            HalRenderPasses::create_render_passes(&device, SURFACE_FORMAT, DEPTH_FORMAT);

        let (frame_depth, surface_format, dimensions, frame_count) =
            Self::init_drawables(&device, &mut heaps, &adapter, surface.as_mut(), dimensions);

        let viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: dimensions.0 as _,
                h: dimensions.1 as _,
            },
            depth: 0.0..1.0,
        };

        // Samplers
        let sampler_linear = unsafe {
            device.create_sampler(&hal::image::SamplerDesc::new(
                hal::image::Filter::Linear,
                hal::image::WrapMode::Clamp,
            ))
        }
        .expect("sampler_linear failed");

        let sampler_nearest = unsafe {
            device.create_sampler(&hal::image::SamplerDesc::new(
                hal::image::Filter::Nearest,
                hal::image::WrapMode::Clamp,
            ))
        }
        .expect("sampler_linear failed");

        let pipeline_requirements: FastHashMap<String, PipelineRequirements> =
            from_str(&shader_source::PIPELINES).expect("Failed to load pipeline requirements");

        let mut desc_allocator = DescriptorAllocator::new();

        let mut frame_fence = ArrayVec::new();
        let mut render_finished_semaphores = ArrayVec::new();
        let mut command_pools: ArrayVec<[CommandPool<B>; MAX_FRAME_COUNT]> = ArrayVec::new();
        let mut staging_buffer_pool = ArrayVec::new();
        let mut instance_buffers = ArrayVec::new();
        let mut swapchain_image_layouts = ArrayVec::new();
        for _ in 0..frame_count {
            let fence = device.create_fence(false).expect("create_fence failed");
            frame_fence.push(Fence {
                inner: fence,
                is_submitted: false,
            });

            render_finished_semaphores
                .push(device.create_semaphore().expect("create_semaphore failed"));

            let mut hal_cp = unsafe {
                device.create_command_pool(
                    queue_group_family,
                    hal::pool::CommandPoolCreateFlags::empty(),
                )
            }
            .expect("create_command_pool failed");
            unsafe { hal_cp.reset(false) };
            let mut cp = CommandPool::new(hal_cp);
            cp.create_command_buffer();
            command_pools.push(cp);
            staging_buffer_pool.push(BufferPool::new(
                &device,
                &mut heaps,
                hal::buffer::Usage::TRANSFER_SRC,
                1,
                (limits.non_coherent_atom_size - 1) as usize,
                (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                (limits.optimal_buffer_copy_offset_alignment - 1) as usize,
                texture_cache_size,
            ));
            instance_buffers.push(InstanceBufferHandler::new(
                (limits.non_coherent_atom_size - 1) as usize,
                (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                instance_buffer_size,
            ));
            swapchain_image_layouts.push(hal::image::Layout::Undefined);
        }

        let mut command_buffer = command_pools[0].remove_cmd_buffer();
        // Start recording for the 1st frame
        unsafe { Self::begin_cmd_buffer(&mut command_buffer) };

        let quad_buffer = VertexBufferHandler::new(
            &device,
            &mut heaps,
            hal::buffer::Usage::VERTEX,
            &QUAD,
            (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
            (limits.non_coherent_atom_size - 1) as usize,
        );

        let mut per_group_descriptor_sets = FastHashMap::default();
        let descriptor_data: FastHashMap<DescriptorGroup, DescriptorGroupData<B>> = [
            DescriptorGroup::Default,
            DescriptorGroup::Clip,
            DescriptorGroup::Primitive,
        ]
        .iter()
        .map(|g| {
            let layouts_and_samplers = match g {
                DescriptorGroup::Default => [
                    (EMPTY_SET_0, vec![]),
                    (DEFAULT_SET_1, vec![&sampler_nearest]),
                    (COMMON_SET_2, vec![]),
                    (COMMON_SET_3, vec![]),
                ],
                DescriptorGroup::Clip => [
                    (EMPTY_SET_0, vec![]),
                    (
                        CLIP_SET_1,
                        vec![
                            &sampler_nearest,
                            &sampler_nearest,
                            &sampler_nearest,
                            &sampler_nearest,
                        ],
                    ),
                    (COMMON_SET_2, vec![]),
                    (COMMON_SET_3, vec![]),
                ],
                DescriptorGroup::Primitive => [
                    (PRIMITIVE_SET_0, vec![&sampler_linear, &sampler_linear]),
                    (
                        PRIMITIVE_SET_1,
                        vec![
                            &sampler_nearest,
                            &sampler_nearest,
                            &sampler_nearest,
                            &sampler_nearest,
                            &sampler_nearest,
                            &sampler_nearest,
                        ],
                    ),
                    (COMMON_SET_2, vec![]),
                    (COMMON_SET_3, vec![]),
                ],
                DescriptorGroup::Invalid => unimplemented!(),
            };

            let ranges = layouts_and_samplers
                .iter()
                .map(|(bindings, _)| DescriptorRanges::from_bindings(bindings))
                .collect::<ArrayVec<_>>();

            let set_layouts: ArrayVec<[B::DescriptorSetLayout; MAX_DESCRIPTOR_SET_COUNT]> =
                layouts_and_samplers
                    .iter()
                    .enumerate()
                    .map(|(index, (bindings, immutable_samplers))| {
                        let layout = unsafe {
                            device.create_descriptor_set_layout(
                                *bindings,
                                immutable_samplers.iter().map(|s| *s),
                            )
                        }
                        .expect("create_descriptor_set_layout failed");
                        if index == DESCRIPTOR_SET_PER_GROUP {
                            let mut descriptor_sets = Vec::new();
                            unsafe {
                                desc_allocator.allocate(
                                    &device,
                                    &layout,
                                    ranges[index],
                                    descriptor_count.unwrap_or(DESCRIPTOR_COUNT),
                                    &mut descriptor_sets,
                                )
                            }
                            .expect("Allocate descriptor sets failed");
                            per_group_descriptor_sets.insert(*g, descriptor_sets);
                        }
                        layout
                    })
                    .collect();

            let pipeline_layout = unsafe {
                device.create_pipeline_layout(
                    &set_layouts,
                    &[],
                )
            }
            .expect("create_pipeline_layout failed");
            (
                *g,
                DescriptorGroupData {
                    set_layouts,
                    ranges,
                    pipeline_layout,
                },
            )
        })
        .collect();

        let descriptor_data = DescriptorData(descriptor_data);

        let per_pass_descriptors = DescriptorSetHandler::new(
            &device,
            &mut desc_allocator,
            &descriptor_data,
            &DescriptorGroup::Primitive,
            DESCRIPTOR_SET_PER_PASS,
            descriptor_count.unwrap_or(DESCRIPTOR_COUNT),
            Vec::new(),
        );

        let per_draw_descriptors = DescriptorSetHandler::new(
            &device,
            &mut desc_allocator,
            &descriptor_data,
            &DescriptorGroup::Default,
            DESCRIPTOR_SET_PER_DRAW,
            descriptor_count.unwrap_or(DESCRIPTOR_COUNT),
            Vec::new(),
        );

        let uniform_buffer_handler = UniformBufferHandler::new(
            mem::size_of::<default::Transform3D<f32>>(),
            (limits.min_uniform_buffer_offset_alignment - 1) as usize,
            frame_count,
            &device,
            &mut heaps,
            &mut desc_allocator,
            &descriptor_data,
            &DescriptorGroup::Default,
            DESCRIPTOR_SET_PER_TARGET,
        );

        let pipeline_cache = if let Some(ref path) = cache_path {
            Self::load_pipeline_cache(&device, &path, &adapter.physical_device)
        } else {
            None
        };

        let mut device = Device {
            device: Arc::new(device),
            heaps: Arc::new(Mutex::new(heaps)),
            limits,
            surface_format,
            adapter,
            surface,
            _instance: instance,
            depth_format: DEPTH_FORMAT,
            queue_group_family,
            queue_group_queues,
            command_pools,
            command_buffer,
            staging_buffer_pool,
            render_passes,
            readback_texture: None,
            frame: None,
            frame_depth,
            swapchain_image_layouts,
            frame_count,
            scissor_rect: viewport.rect,
            viewport,
            scissor_enabled: false,
            dimensions,
            sampler_linear,
            sampler_nearest,
            current_blend_state: None,
            current_depth_test: None,
            clear_values: FastHashMap::default(),
            blend_color: ColorF::new(0.0, 0.0, 0.0, 0.0),
            _resource_override_path: resource_override_path,
            // This is initialized to 1 by default, but it is reset
            // at the beginning of each frame in `Renderer::bind_frame_data`.
            device_pixel_ratio: 1.0,
            depth_available: true,
            upload_method,
            inside_frame: false,
            inside_render_pass: false,
            render_pass_depth_state: RenderPassDepthState::Disabled,

            capabilities: Capabilities {
                supports_multisampling: false,
                supports_copy_image_sub_data: false,
                supports_blit_to_texture_array: false,
                supports_pixel_local_storage: false,
                supports_advanced_blend_equation: false,
                supports_khr_debug: false,
                supports_texture_swizzle: false,
            },
            depth_targets: FastHashMap::default(),

            programs: FastHashMap::default(),
            blit_programs: FastHashMap::default(),
            next_program_id: INVALID_PROGRAM_ID.0 + 1,
            shader_modules: FastHashMap::default(),
            images: FastHashMap::default(),
            gpu_cache_buffer: None,
            gpu_cache_buffers: FastHashMap::default(),
            retained_textures: Vec::new(),
            fbos: FastHashMap::default(),
            rbos: FastHashMap::default(),
            desc_allocator,

            per_draw_descriptors,
            bound_per_draw_textures: PerDrawBindings::default(),
            bound_per_draw_descriptor: None,

            per_pass_descriptors,
            bound_per_pass_textures: PerPassBindings::default(),
            bound_per_pass_descriptor: None,

            per_group_descriptors: DescriptorSetHandler::from_existing(per_group_descriptor_sets),
            bound_per_group_textures: PerGroupBindings::default(),
            bound_per_group_descriptors: [None, None, None],

            bound_projection: Default::default(),
            descriptor_data,

            bound_textures: [0; RENDERER_TEXTURE_COUNT],
            bound_program: INVALID_PROGRAM_ID,
            bound_sampler: [TextureFilter::Linear; RENDERER_TEXTURE_COUNT],
            bound_read_fbo: DEFAULT_READ_FBO,
            bound_read_texture: (INVALID_TEXTURE_ID, 0),
            bound_draw_fbo: DEFAULT_DRAW_FBO,
            draw_target_usage: DrawTargetUsage::Draw,

            max_texture_size,
            _renderer_name: renderer_name,
            frame_id: GpuFrameId(0),
            features,

            next_id: 0,
            frame_fence,
            render_finished_semaphores,
            pipeline_requirements,
            pipeline_cache,
            cache_path,
            save_cache,

            uniform_buffer_handler,
            quad_buffer,
            instance_buffers,
            free_instance_buffers: Vec::new(),
            download_buffer: None,
            instance_buffer_range: 0..0,

            swizzle_settings: SwizzleSettings {
                bgra8_sampling_swizzle: Swizzle::Rgba,
            },
            color_formats: TextureFormatPair::from(ImageFormat::BGRA8),
            optimal_pbo_stride: NonZeroUsize::new(4).unwrap(),
            last_rp_in_frame_reached: false,
            readback_supported,

            #[cfg(debug_assertions)]
            shader_is_ready: false,
            rebind_descriptors: true,
        };

        if readback_supported || device.headless_mode() {
            device.inside_frame = true;
            let texture = device.create_texture(
                TextureTarget::Default,
                ImageFormat::BGRA8,
                device.dimensions.0 as i32,
                device.dimensions.1 as i32,
                TextureFilter::Nearest,
                Some(RenderTargetInfo { has_depth: true }),
                1,
            );
            device.readback_texture = Some(texture);
            device.inside_frame = false;
        }
        device
    }

    pub fn supports_extension(&self, _extension: &str) -> bool {
        false
    }

    pub fn enable_pixel_local_storage(&mut self, _enable: bool) {
        warn!("Pixel local storage not supported");
    }

    pub fn swizzle_settings(&self) -> Option<SwizzleSettings> {
        if self.capabilities.supports_texture_swizzle {
            Some(self.swizzle_settings)
        } else {
            None
        }
    }

    pub fn preferred_color_formats(&self) -> TextureFormatPair<ImageFormat> {
        self.color_formats.clone()
    }

    pub fn optimal_pbo_stride(&self) -> NonZeroUsize {
        self.optimal_pbo_stride
    }

    pub fn map_pbo_for_readback<'a>(&'a mut self, _pbo: &'a PBO) -> Option<BoundPBO<'a, B>> {
        warn!("map_pbo_for_readback is not implemented");
        None
    }

    pub fn read_pixels_into_pbo(
        &mut self,
        _read_target: ReadTarget,
        _rect: DeviceIntRect,
        _format: ImageFormat,
        _pbo: &PBO,
    ) {
        warn!("read_pixels_into_pbo not implemented");
    }

    fn ensure_blit_program(&mut self, kind: hal::image::ViewKind) {
        if !self.blit_programs.contains_key(&kind) {
            let program = self.create_program_inner(
                "blit",
                &ShaderKind::Service,
                match kind {
                    hal::image::ViewKind::D2 => &["TEXTURE_2D"],
                    hal::image::ViewKind::D2Array => &[""],
                    _ => unimplemented!(),
                },
            );
            let id = self.generate_program_id(DescriptorGroup::from(ShaderKind::Service));
            self.blit_programs.insert(kind, (id, program));
        }
    }

    fn load_pipeline_cache(
        device: &B::Device,
        path: &PathBuf,
        physical_device: &B::PhysicalDevice,
    ) -> Option<B::PipelineCache> {
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
            let cache = unsafe { device.create_pipeline_cache(Some(&bytes)) }.expect(&format!(
                "Failed to create pipeline cache from file: {:?}",
                path
            ));
            Some(cache)
        } else {
            None
        }
    }

    fn replace_active_descriptor_set<K, F>(
        descriptor: &mut Option<DescriptorSet<B>>,
        new_descriptor: Option<DescriptorSet<B>>,
        key: &K,
        handler: &mut DescriptorSetHandler<K, B, F>,
     ) where
        K: Copy + Clone + std::fmt::Debug + Eq + std::hash::Hash + DescGroupKey,
        F: FreeSets<B>,
    {
        let old_descriptor = match new_descriptor {
            Some(desc) => descriptor.replace(desc),
            None => descriptor.take(),
        };

        if let Some(descriptor) = old_descriptor {
            handler.push_back_descriptor_set(*key, descriptor);
        }
    }

    fn reset_descriptor_key_and_handler<D, K, F>(
        key: &mut D,
        handler: &mut DescriptorSetHandler<K, B, F>,
    ) where
        D: Default,
        K: Copy + Clone + std::fmt::Debug + Eq + std::hash::Hash + DescGroupKey,
        F: FreeSets<B>,
    {
        *key = D::default();
        handler.reset();
    }

    fn replace_active_descriptor_set_and_reset_handler<K, F>(
        descriptor: &mut Option<DescriptorSet<B>>,
        new_descriptor: Option<DescriptorSet<B>>,
        key: &mut K,
        handler: &mut DescriptorSetHandler<K, B, F>,
    ) where
        K: Copy + Clone + std::fmt::Debug + Eq + std::hash::Hash + DescGroupKey + Default,
        F: FreeSets<B>,
    {
        Self::replace_active_descriptor_set(descriptor, new_descriptor, key, handler);
        Self::reset_descriptor_key_and_handler(key, handler);
    }

    pub(crate) fn recreate_swapchain(
        &mut self,
        window_size: Option<(i32, i32)>,
    ) -> (bool, DeviceIntSize) {
        if let Some(dimensions) = window_size {
            if self.dimensions == dimensions {
                return (
                    false,
                    DeviceIntSize::new(self.dimensions.0, self.dimensions.1),
                );
            }
        }
        {
            let ref mut heaps = *self.heaps.lock().unwrap();

            Self::replace_active_descriptor_set_and_reset_handler(
                &mut self.bound_per_draw_descriptor,
                None,
                &mut self.bound_per_draw_textures,
                &mut self.per_draw_descriptors
            );

            Self::replace_active_descriptor_set_and_reset_handler(
                &mut self.bound_per_pass_descriptor,
                None,
                &mut self.bound_per_pass_textures,
                &mut self.per_pass_descriptors
            );

            for descriptor_group in [DescriptorGroup::Default, DescriptorGroup::Clip, DescriptorGroup::Primitive].iter() {
                Self::replace_active_descriptor_set(
                    &mut self.bound_per_group_descriptors[*descriptor_group as usize],
                    None,
                    &(*descriptor_group, self.bound_per_group_textures),
                    &mut self.per_group_descriptors,
                );
            }
            Self::reset_descriptor_key_and_handler(&mut self.bound_per_group_textures, &mut self.per_group_descriptors);

            self.uniform_buffer_handler.reset(self.next_id);

            let (frame_depth, surface_format, dimensions, _frame_count) = Self::init_drawables(
                self.device.as_ref(),
                heaps,
                &self.adapter,
                self.surface.as_mut(),
                window_size.unwrap_or(self.dimensions),
            );

            if let Some(old_frame) = self.frame.take() {
                old_frame.deinit(self.device.as_ref());
            }
            let old_depth = mem::replace(&mut self.frame_depth, frame_depth);
            old_depth.deinit(self.device.as_ref(), heaps);
            self.dimensions = dimensions;
            self.surface_format = surface_format;
        }

        for layout in self.swapchain_image_layouts.iter_mut() {
            *layout = hal::image::Layout::Undefined;
        }

        let old_readback_texture = self.readback_texture.take();
        if let Some(texture) = old_readback_texture {
            self.free_texture_unconditionaly(texture);
            let texture = self.create_texture(
                TextureTarget::Default,
                ImageFormat::BGRA8,
                self.dimensions.0 as i32,
                self.dimensions.1 as i32,
                TextureFilter::Nearest,
                Some(RenderTargetInfo { has_depth: true }),
                1,
            );
            self.readback_texture = Some(texture);
        }

        (
            true,
            DeviceIntSize::new(self.dimensions.0, self.dimensions.1),
        )
    }

    fn init_drawables(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        adapter: &hal::adapter::Adapter<B>,
        mut surface: Option<&mut B::Surface>,
        dimensions: (i32, i32),
    ) -> (DepthBuffer<B>, ImageFormat, (i32, i32), usize) {
        let (surface_format, extent, frame_count) = match surface.as_mut() {
            Some(ref mut surface) => {
                let caps = surface.capabilities(&adapter.physical_device);
                let formats = surface.supported_formats(&adapter.physical_device);
                let available_surface_format = formats.map_or(SURFACE_FORMAT, |formats| {
                    formats
                        .into_iter()
                        .find(|format| format == &SURFACE_FORMAT)
                        .expect(&format!("{:?} surface is not supported!", SURFACE_FORMAT))
                });
                let surface_format = match available_surface_format {
                    SURFACE_FORMAT => ImageFormat::BGRA8,
                    f => unimplemented!("Unsupported surface format: {:?}", f),
                };

                let window_extent = hal::window::Extent2D {
                    width: (dimensions.0 as u32)
                        .min(caps.extents.end().width)
                        .max(caps.extents.start().width)
                        .max(1),
                    height: (dimensions.1 as u32)
                        .min(caps.extents.end().height)
                        .max(caps.extents.start().height)
                        .max(1),
                };

                let mut swap_config =
                    SwapchainConfig::from_caps(&caps, available_surface_format, window_extent);
                swap_config.image_count = 3;

                let frame_cunt = swap_config.image_count as usize;
                unsafe {
                    surface
                        .configure_swapchain(&device, swap_config)
                        .expect("Can't configure swapchain");
                };
                (surface_format, window_extent, frame_cunt)
            }
            None => {
                let extent = hal::window::Extent2D {
                    width: dimensions.0 as u32,
                    height: dimensions.1 as u32,
                };
                (ImageFormat::BGRA8, extent, HEADLESS_FRAME_COUNT)
            }
        };
        info!("Frame count {}", frame_count);

        let depth = DepthBuffer::new(device, heaps, extent.width, extent.height, DEPTH_FORMAT);

        (
            depth,
            surface_format,
            (extent.width as i32, extent.height as i32),
            frame_count,
        )
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
        self.limits.max_image_array_layers as usize
    }

    pub fn get_capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn headless_mode(&self) -> bool {
        self.surface.is_none()
    }

    fn reset_next_frame_resources(&mut self) {
        let prev_id = self.next_id;
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
            self.command_pools[self.next_id].reset();
            if self.frame_count != 1 {
                let old_buffer = mem::replace(
                    &mut self.command_buffer,
                    self.command_pools[self.next_id].remove_cmd_buffer(),
                );
                self.command_pools[prev_id].return_cmd_buffer(old_buffer);
            }
            Self::begin_cmd_buffer(&mut self.command_buffer);
        }
        self.bound_projection = Default::default();
        self.uniform_buffer_handler.reset(self.next_id);
        self.staging_buffer_pool[self.next_id].reset();
        self.instance_buffers[self.next_id].reset(&mut self.free_instance_buffers);
        self.delete_retained_textures();
    }

    pub fn reset_state(&mut self) {
        for i in 0..RENDERER_TEXTURE_COUNT {
            // Don't reset the dither texture and sampler, those are set at the start only
            if i != DITHER_BINDING {
                self.bound_textures[i] = INVALID_TEXTURE_ID;
                self.bound_sampler[i] = TextureFilter::Linear;
            }
        }
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;
        self.draw_target_usage = DrawTargetUsage::Draw;
        self.bound_program = INVALID_PROGRAM_ID;
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

    fn create_program_inner(
        &mut self,
        shader_name: &str,
        shader_kind: &ShaderKind,
        features: &[&str],
    ) -> Program<B> {
        let mut name = String::from(shader_name);
        for feature_names in features {
            for feature in feature_names.split(',') {
                if NON_SPECIALIZATION_FEATURES.iter().any(|f| *f == feature) {
                    name.push_str(&format!("_{}", feature.to_lowercase()));
                }
            }
        }

        let desc_group = DescriptorGroup::from(*shader_kind);
        Program::create(
            self.pipeline_requirements
                .get(&name)
                .expect(&format!("Can't load pipeline data for: {}!", name))
                .clone(),
            self.device.as_ref(),
            self.descriptor_data.pipeline_layout(&desc_group),
            &mut *self.heaps.lock().unwrap(),
            &self.limits,
            &name,
            features,
            shader_kind.clone(),
            &self.render_passes,
            self.frame_count,
            &mut self.shader_modules,
            self.pipeline_cache.as_ref(),
            self.surface_format,
        )
    }

    pub fn create_program(
        &mut self,
        shader_name: &str,
        shader_kind: &ShaderKind,
        features: &[&str],
    ) -> Result<ProgramId, ShaderError> {
        let program = self.create_program_inner(shader_name, shader_kind, features);
        let id = self.generate_program_id(DescriptorGroup::from(*shader_kind));
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
        let pipeline_layout_changed = self.bound_program.1 != program_id.1;
        self.rebind_descriptors |= pipeline_layout_changed;
        self.bound_program = *program_id;
        let program = self.programs
            .get(&self.bound_program)
            .expect("Program not found");

        let format = self.fbos
            .get(&self.bound_draw_fbo)
            .map_or(self.surface_format, |fbo| fbo.format);

        let blend_state = self.current_blend_state;
        unsafe {
            self.command_buffer.bind_graphics_pipeline(
                &program
                    .pipelines
                    .get(&(format, blend_state, self.render_pass_depth_state, self.current_depth_test))
                    .unwrap_or_else(|| panic!(
                        "The blend state {:?} with depth test {:?} and render_pass_depth_state {:?} and format {:?} not found for {} program!",
                            blend_state, self.current_depth_test, self.render_pass_depth_state, format, program.shader_name
                    ))
            );
            if let Some(SUBPIXEL_CONSTANT_TEXT_COLOR) = blend_state  {
                self.command_buffer.set_blend_constants(self.blend_color.to_array());
            }
            if self.rebind_descriptors {
                use std::iter;
                let descriptor_group = self.bound_program.1;

                let desc_set_per_frame = self
                    .bound_per_group_descriptors[descriptor_group as usize]
                    .as_ref()
                    .unwrap()
                    .raw();
                let (desc_set_per_pass, first_ds_slot) = match descriptor_group {
                    DescriptorGroup::Primitive => (
                        self.bound_per_pass_descriptor.as_ref().map(|d| d.raw()),
                        DESCRIPTOR_SET_PER_PASS,
                    ),
                    _ => (None, DESCRIPTOR_SET_PER_GROUP),
                };

                let(desc_set_per_target, dynamic_offset) = self.uniform_buffer_handler.buffer_info(self.next_id);

                self.command_buffer.bind_graphics_descriptor_sets(
                    self.descriptor_data.pipeline_layout(&descriptor_group),
                    first_ds_slot,
                    desc_set_per_pass
                        .into_iter()
                        .chain(iter::once(desc_set_per_frame))
                        .chain(iter::once(desc_set_per_target)),
                    &[dynamic_offset],
                );

                // Force per draw descriptor rebind by resetting the bound descriptor
                Self::replace_active_descriptor_set(
                    &mut self.bound_per_draw_descriptor,
                    None,
                    &self.bound_per_draw_textures,
                    &mut self.per_draw_descriptors,
                );
                self.bound_per_draw_textures = PerDrawBindings::default();
                self.rebind_descriptors = false;
            }
        }
        #[cfg(debug_assertions)]
        {
            self.shader_is_ready = true;
        }
    }

    pub fn set_uniforms(
        &mut self,
        _program_id: &ProgramId,
        projection: &default::Transform3D<f32>,
    ) {
        assert_eq!(self.bound_projection, *projection);
    }

    unsafe fn begin_cmd_buffer(cmd_buffer: &mut B::CommandBuffer) {
        let flags = CommandBufferFlags::ONE_TIME_SUBMIT;
        cmd_buffer.begin(flags, CommandBufferInheritanceInfo::default());
    }

    fn bind_per_draw_textures_impl(
        &mut self,
        descriptor_group: DescriptorGroup,
        per_draw_bindings: PerDrawBindings,
    ) {
        let mut bound_textures = self.bound_textures;
        bound_textures[0] = per_draw_bindings.0[0];
        bound_textures[1] = per_draw_bindings.0[1];
        bound_textures[2] = per_draw_bindings.0[2];
        let mut bound_sampler = self.bound_sampler;
        bound_sampler[0] = per_draw_bindings.1;
        bound_sampler[1] = per_draw_bindings.1;
        bound_sampler[2] = per_draw_bindings.1;

        let descriptor = self.per_draw_descriptors.bind_textures(
            &bound_textures,
            &bound_sampler,
            per_draw_bindings,
            &self.images,
            None,
            &mut self.desc_allocator,
            self.device.as_ref(),
            &self.descriptor_data,
            &descriptor_group,
            DESCRIPTOR_SET_PER_DRAW,
            0..PER_DRAW_TEXTURE_COUNT,
            &self.sampler_linear,
            &self.sampler_nearest,
        );
        Self::replace_active_descriptor_set(
            &mut self.bound_per_draw_descriptor,
            Some(descriptor),
            &self.bound_per_draw_textures,
            &mut self.per_draw_descriptors,
        );
        self.bound_per_draw_textures = per_draw_bindings;
    }

    pub fn bind_per_pass_textures(&mut self) {
        let per_pass_bindings = PerPassBindings([self.bound_textures[3], self.bound_textures[4]]);
        if per_pass_bindings == self.bound_per_pass_textures {
            return;
        }
        let descriptor = self.per_pass_descriptors.bind_textures(
            &self.bound_textures,
            &self.bound_sampler,
            per_pass_bindings,
            &self.images,
            None,
            &mut self.desc_allocator,
            self.device.as_ref(),
            &self.descriptor_data,
            &DescriptorGroup::Primitive,
            DESCRIPTOR_SET_PER_PASS,
            PER_DRAW_TEXTURE_COUNT..PER_DRAW_TEXTURE_COUNT + PER_PASS_TEXTURE_COUNT,
            &self.sampler_linear,
            &self.sampler_nearest,
        );

        Self::replace_active_descriptor_set(
            &mut self.bound_per_pass_descriptor,
            Some(descriptor),
            &self.bound_per_pass_textures,
            &mut self.per_pass_descriptors,
        );
        self.bound_per_pass_textures = per_pass_bindings;
        self.rebind_descriptors = true;
    }

    fn bind_per_group_textures_impl(&mut self, descriptor_group: DescriptorGroup, per_group_bindings: PerGroupBindings) {
        let descriptor = self.per_group_descriptors.bind_textures(
            &self.bound_textures,
            &self.bound_sampler,
            (descriptor_group, per_group_bindings),
            &self.images,
            match descriptor_group {
                DescriptorGroup::Default => None,
                _ => self.gpu_cache_buffer.as_ref().map(|b| b.buffer.as_ref()),
            },
            &mut self.desc_allocator,
            self.device.as_ref(),
            &self.descriptor_data,
            &descriptor_group,
            DESCRIPTOR_SET_PER_GROUP,
            match descriptor_group {
                DescriptorGroup::Default => PER_GROUP_RANGE_DEFAULT,
                DescriptorGroup::Clip => PER_GROUP_RANGE_CLIP,
                DescriptorGroup::Primitive => PER_GROUP_RANGE_PRIMITIVE,
                DescriptorGroup::Invalid => unimplemented!(),
            },
            &self.sampler_linear,
            &self.sampler_nearest,
        );

        Self::replace_active_descriptor_set(
            &mut self.bound_per_group_descriptors[descriptor_group as usize],
            Some(descriptor),
            &(descriptor_group, self.bound_per_group_textures),
            &mut self.per_group_descriptors,
        );
    }

    pub fn bind_per_draw_textures(&mut self, can_skip_bind: bool) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        if can_skip_bind && self.bound_per_draw_descriptor.is_some() {
            return;
        }
        let descriptor_group = self.bound_program.1;
        let per_draw_bindings = PerDrawBindings(
            [
                self.bound_textures[0],
                self.bound_textures[1],
                self.bound_textures[2],
            ],
             self.bound_sampler[0],
        );

        if self.bound_per_draw_textures != per_draw_bindings {
            self.bind_per_draw_textures_impl(descriptor_group, per_draw_bindings);
        }
        unsafe {
            self.command_buffer.bind_graphics_descriptor_sets(
                self.descriptor_data.pipeline_layout(&descriptor_group),
                DESCRIPTOR_SET_PER_DRAW,
                self.bound_per_draw_descriptor.as_ref().map(|descriptor| descriptor.raw()),
                &[],
            );
        }
    }

    pub fn bind_per_group_textures(&mut self) {
        let per_group_bindings = PerGroupBindings([
            self.bound_textures[5],
            self.bound_textures[6],
            self.bound_textures[7],
            self.bound_textures[8],
            self.bound_textures[9],
            self.bound_textures[10],
        ]);
        if per_group_bindings == self.bound_per_group_textures {
            return;
        }
        for descriptor_group in [DescriptorGroup::Default, DescriptorGroup::Clip, DescriptorGroup::Primitive].iter() {
            self.bind_per_group_textures_impl(*descriptor_group, per_group_bindings);
        }
        self.bound_per_group_textures = per_group_bindings;
        self.rebind_descriptors = true;
    }

    pub fn update_indices<I: Copy>(&mut self, indices: &[I]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self
            .programs
            .get_mut(&self.bound_program)
            .expect("Program not found.");

        if let Some(ref mut index_buffer) = program.index_buffer {
            index_buffer[self.next_id].update(
                self.device.as_ref(),
                indices,
                &mut *self.heaps.lock().unwrap(),
            );
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

        if program.shader_kind.is_debug() {
            program.vertex_buffer.as_mut().unwrap()[self.next_id].update(
                self.device.as_ref(),
                vertices,
                &mut *self.heaps.lock().unwrap(),
            );
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    fn update_instances<T: Copy>(&mut self, instances: &[T]) {
        self.instance_buffer_range = self.instance_buffers[self.next_id].add(
            self.device.as_ref(),
            instances,
            &mut *self.heaps.lock().unwrap(),
            &mut self.free_instance_buffers,
        );
    }

    fn draw(&mut self) {
        assert!(self.inside_render_pass);

        assert_eq!(self.draw_target_usage, DrawTargetUsage::Draw);
        self.programs
            .get_mut(&self.bound_program)
            .expect("Program not found")
            .submit(
                &mut self.command_buffer,
                self.next_id,
                &self.quad_buffer,
                &self.instance_buffers[self.next_id],
                self.instance_buffer_range.clone(),
            );
    }

    pub fn begin_frame(&mut self) -> GpuFrameId {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;
        #[cfg(debug_assertions)]
        {
            self.shader_is_ready = false;
        }
        self.reset_state();
        self.frame_id
    }

    fn bind_texture_impl(
        &mut self,
        slot: TextureSlot,
        id: TextureId,
        sampler: TextureFilter,
        _set_swizzle: Option<Swizzle>,
    ) {
        if self.bound_textures[slot.0] != id {
            self.bound_textures[slot.0] = id;
            self.bound_sampler[slot.0] = sampler;
        }
    }

    pub fn bind_texture<S>(&mut self, slot: S, texture: &Texture, swizzle: Swizzle)
    where
        S: Into<TextureSlot>,
    {
        let old_swizzle = texture.active_swizzle.replace(swizzle);
        let set_swizzle = if old_swizzle != swizzle {
            Some(swizzle)
        } else {
            None
        };
        self.bind_texture_impl(slot.into(), texture.id, texture.filter, set_swizzle);
        texture.bound_in_frame.set(self.frame_id);
    }

    pub fn bind_external_texture<S>(&mut self, slot: S, external_texture: &ExternalTexture)
    where
        S: Into<TextureSlot>,
    {
        self.bind_texture_impl(
            slot.into(),
            external_texture.id,
            TextureFilter::Linear,
            None,
        );
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
            ReadTarget::Texture { fbo_id } => fbo_id,
            ReadTarget::External { .. } => unimplemented!("Extrenal FBO id not supported"),
        };
        self.bind_read_target_impl(fbo_id)
    }

    #[cfg(feature = "capture")]
    fn bind_read_texture(&mut self, texture_id: TextureId, layer_id: i32) {
        self.bound_read_texture = (texture_id, layer_id);
    }

    fn bind_draw_target_impl(
        &mut self,
        fbo_id: FBOId,
        usage: DrawTargetUsage,
        projection: Option<&default::Transform3D<f32>>,
    ) {
        debug_assert!(self.inside_frame);

        if let Some(projection) = projection {
            // This check here can save us a bunch of buffer update
            if self.bound_projection != *projection {
                self.bound_projection = *projection;
                self.uniform_buffer_handler.add(
                    self.device.as_ref(),
                    &[self.bound_projection],
                    self.next_id
                );
                self.rebind_descriptors = true;
            }
        }

        let fbo_id = if fbo_id == DEFAULT_DRAW_FBO && self.headless_mode() {
            self.readback_texture.as_ref().unwrap().fbos_with_depth[0]
        } else {
            fbo_id
        };
        self.draw_target_usage = usage;
        if self.bound_draw_fbo != fbo_id {
            let old_fbo_id = mem::replace(&mut self.bound_draw_fbo, fbo_id);
            let transit_back_old_image = match (
                self.fbos.get(&old_fbo_id),
                self.fbos.get(&self.bound_draw_fbo),
            ) {
                (None, _) => false,
                (Some(_), None) => true,
                (Some(old_fbo), Some(bound_fbo)) => old_fbo.texture_id != bound_fbo.texture_id,
            };
            if transit_back_old_image {
                let texture_id = self.fbos[&old_fbo_id].texture_id;
                if let Some((barrier, pipeline_stages)) = self.images[&texture_id].core.transit(
                    (
                        hal::image::Access::SHADER_READ,
                        hal::image::Layout::ShaderReadOnlyOptimal,
                    ),
                    self.images[&texture_id].core.subresource_range.clone(),
                ) {
                    unsafe {
                        self.command_buffer.pipeline_barrier(
                            pipeline_stages,
                            hal::memory::Dependencies::empty(),
                            &[barrier],
                        );
                    }
                }
            }
        }
    }

    pub fn reset_read_target(&mut self) {
        self.bind_read_target_impl(DEFAULT_READ_FBO);
    }

    pub fn reset_draw_target(&mut self, projection: Option<&default::Transform3D<f32>>) {
        self.bind_draw_target_impl(DEFAULT_DRAW_FBO, DrawTargetUsage::Draw, projection);
        self.depth_available = true;
        self.render_pass_depth_state = RenderPassDepthState::Enabled;
    }

    pub fn bind_draw_target(
        &mut self,
        texture_target: DrawTarget,
        usage: DrawTargetUsage,
        projection: Option<&default::Transform3D<f32>>,
    ) {
        let (fbo_id, rect, depth_available) = match texture_target {
            DrawTarget::Default { rect, .. } => {
                if let DrawTargetUsage::CopyOnly = usage {
                    panic!("We should not have default target with CopyOnly usage!");
                }
                (DEFAULT_DRAW_FBO, rect, true)
            }
            DrawTarget::ReadBack { rect, .. } => {
                let texture = &self.readback_texture.as_ref().unwrap();
                let fbo_id = texture.fbos_with_depth[0];
                if self.bound_draw_fbo != fbo_id {
                    let image = &self.images[&texture.id].core;
                    let image_state = match usage {
                        DrawTargetUsage::CopyOnly => (
                            hal::image::Access::TRANSFER_WRITE,
                            hal::image::Layout::TransferDstOptimal,
                        ),
                        DrawTargetUsage::Draw => (
                            hal::image::Access::COLOR_ATTACHMENT_WRITE,
                            hal::image::Layout::ColorAttachmentOptimal,
                        ),
                    };
                    if let Some((barrier, pipeline_stages)) =
                        image.transit(image_state, image.subresource_range.clone())
                    {
                        unsafe {
                            self.command_buffer.pipeline_barrier(
                                pipeline_stages,
                                hal::memory::Dependencies::empty(),
                                &[barrier],
                            );
                        }
                    }
                }
                (fbo_id, rect, true)
            }
            DrawTarget::Texture {
                dimensions,
                layer,
                with_depth,
                fbo_id,
                id,
                ..
            } => {
                self.fbos.get_mut(&fbo_id).unwrap().layer_index = layer as u16;
                if self.bound_draw_fbo != fbo_id {
                    let image = &self.images[&id].core;
                    let image_state = match usage {
                        DrawTargetUsage::CopyOnly => (
                            hal::image::Access::TRANSFER_WRITE,
                            hal::image::Layout::TransferDstOptimal,
                        ),
                        DrawTargetUsage::Draw => (
                            hal::image::Access::COLOR_ATTACHMENT_WRITE,
                            hal::image::Layout::ColorAttachmentOptimal,
                        ),
                    };
                    if let Some((barrier, pipeline_stages)) =
                        image.transit(image_state, image.subresource_range.clone())
                    {
                        unsafe {
                            self.command_buffer.pipeline_barrier(
                                pipeline_stages,
                                hal::memory::Dependencies::empty(),
                                &[barrier],
                            );
                        }
                    }
                }
                let rect = FramebufferIntRect::new(
                    FramebufferIntPoint::zero(),
                    FramebufferIntSize::from_untyped(dimensions.to_untyped()),
                );

                (fbo_id, rect, with_depth)
            }
            DrawTarget::External { .. } => {
                unimplemented!("External draw targets are not supported")
            }
        };

        self.depth_available = depth_available;
        self.render_pass_depth_state = match self.depth_available {
            true => RenderPassDepthState::Enabled,
            false => RenderPassDepthState::Disabled,
        };
        self.bind_draw_target_impl(fbo_id, usage, projection);
        self.viewport.rect = hal::pso::Rect {
            x: rect.origin.x as i16,
            y: rect.origin.y as i16,
            w: rect.size.width as i16,
            h: rect.size.height as i16,
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

    fn generate_program_id(&mut self, descriptor_group: DescriptorGroup) -> ProgramId {
        let program_id = ProgramId(self.next_program_id, descriptor_group);
        self.next_program_id += 1;
        program_id
    }

    fn generate_fbo_ids(&mut self, count: i32) -> SmallVec<[FBOId; 16]> {
        let mut rng = rand::thread_rng();
        let mut fboids = SmallVec::new();
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

    pub fn create_dummy_gpu_cache_texture(&self) -> Texture {
        debug_assert!(self.inside_frame);
        let texture = Texture {
            id: self.generate_texture_id(),
            target: TextureTarget::Default as _,
            size: DeviceIntSize::new(1, 1),
            layer_count: 1,
            format: ImageFormat::RGBAF32,
            filter: Default::default(),
            fbos: vec![],
            fbos_with_depth: vec![],
            last_frame_used: self.frame_id,
            bound_in_frame: Cell::new(GpuFrameId(0)),
            flags: TextureFlags::default(),
            is_buffer: true,
            active_swizzle: Cell::default(),
            blit_workaround_buffer: None,
        };
        record_gpu_alloc(texture.size_in_bytes());
        texture
    }

    pub fn set_gpu_cache_buffer(&mut self, buffer: GpuCacheBuffer<B>) {
        if let Some(barrier) = buffer.transit(
            hal::buffer::Access::HOST_WRITE | hal::buffer::Access::HOST_READ,
            false,
        ) {
            unsafe {
                self.command_buffer.pipeline_barrier(
                    PipelineStage::VERTEX_SHADER | PipelineStage::FRAGMENT_SHADER
                        ..PipelineStage::HOST,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
        self.gpu_cache_buffer = Some(buffer);
    }

    pub fn transit_gpu_cache_buffer(&mut self) {
        use hal::pso::PipelineStage as PS;
        let ref buffer = self.gpu_cache_buffer.as_ref().unwrap();
        if let Some(barrier) = buffer.transit(hal::buffer::Access::SHADER_READ, true) {
            unsafe {
                self.command_buffer.pipeline_barrier(
                    PS::HOST..PS::VERTEX_SHADER | PS::FRAGMENT_SHADER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
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
            error!(
                "Attempting to allocate a texture of size {}x{} above the limit, trimming",
                width, height
            );
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
            is_buffer: false,
            active_swizzle: Cell::default(),
            blit_workaround_buffer: None,
        };

        assert!(!self.images.contains_key(&texture.id));
        let usage_base = hal::image::Usage::TRANSFER_SRC
            | hal::image::Usage::TRANSFER_DST
            | hal::image::Usage::SAMPLED;

        let view_kind = match target {
            TextureTarget::Array => hal::image::ViewKind::D2Array,
            _ => hal::image::ViewKind::D2,
        };

        let (mip_levels, usage) = match texture.filter {
            TextureFilter::Nearest => (1, usage_base | hal::image::Usage::COLOR_ATTACHMENT),
            TextureFilter::Linear => (1, usage_base | hal::image::Usage::COLOR_ATTACHMENT),
            TextureFilter::Trilinear => (
                (width as f32).max(height as f32).log2().floor() as u8 + 1,
                usage_base | hal::image::Usage::COLOR_ATTACHMENT,
            ),
        };
        let img = Image::new(
            self.device.as_ref(),
            &mut *self.heaps.lock().unwrap(),
            texture.format,
            texture.size.width,
            texture.size.height,
            texture.layer_count,
            view_kind,
            mip_levels,
            usage,
        );

        unsafe {
            if let Some((barrier, pipeline_stages)) = img.core.transit(
                (
                    hal::image::Access::SHADER_READ,
                    hal::image::Layout::ShaderReadOnlyOptimal,
                ),
                img.core.subresource_range.clone(),
            ) {
                self.command_buffer.pipeline_barrier(
                    pipeline_stages,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
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

        for i in 0..texture.layer_count as u16 {
            let fbo = Framebuffer::new(
                self.device.as_ref(),
                &texture,
                &self.images.get(&texture.id).unwrap(),
                i,
                &self.render_passes,
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

    fn execute_transitions<'a>(
        command_buffer: &mut B::CommandBuffer,
        transitions: impl IntoIterator<
            Item = (hal::memory::Barrier<'a, B>, std::ops::Range<PipelineStage>),
        >,
    ) {
        let mut barriers: SmallVec<[hal::memory::Barrier<B>; 2]> = SmallVec::new();
        let mut pipeline_stages = PipelineStage::empty()..PipelineStage::empty();
        for (barrier, ps) in transitions {
            barriers.push(barrier);
            pipeline_stages.start |= ps.start;
            pipeline_stages.end |= ps.end;
        }
        if !barriers.is_empty() {
            unsafe {
                command_buffer.pipeline_barrier(
                    pipeline_stages,
                    hal::memory::Dependencies::empty(),
                    barriers,
                );
            }
        }
    }

    /// Copies the contents from one renderable texture to another.
    pub fn blit_renderable_texture(&mut self, dst: &mut Texture, src: &Texture) {
        assert!(!self.inside_render_pass);
        dst.bound_in_frame.set(self.frame_id);
        src.bound_in_frame.set(self.frame_id);
        debug_assert!(self.inside_frame);
        debug_assert!(dst.size.width >= src.size.width);
        debug_assert!(dst.size.height >= src.size.height);
        assert!(dst.layer_count >= src.layer_count);

        let rect = DeviceIntRect::new(DeviceIntPoint::zero(), src.get_dimensions().to_i32());
        let (src_img, layers) = (&self.images[&src.id].core, src.layer_count);
        let dst_img = &self.images[&dst.id].core;

        // let range = hal::image::SubresourceRange {
        //     aspects: hal::format::Aspects::COLOR,
        //     levels: 0 .. 1,
        //     layers: 0 .. layers as _,
        // };

        unsafe {
            assert_eq!(
                src_img.state.get().1,
                hal::image::Layout::ShaderReadOnlyOptimal
            );
            let (src_image_prev_state, dst_image_prev_state) =
                (src_img.state.get(), dst_img.state.get());
            let transitions = src_img
                .transit(
                    (
                        hal::image::Access::TRANSFER_READ,
                        hal::image::Layout::TransferSrcOptimal,
                    ),
                    src_img.subresource_range.clone(),
                )
                .into_iter()
                .chain(dst_img.transit(
                    (
                        hal::image::Access::TRANSFER_WRITE,
                        hal::image::Layout::TransferDstOptimal,
                    ),
                    dst_img.subresource_range.clone(),
                ));

            Self::execute_transitions(&mut self.command_buffer, transitions);

            self.command_buffer.copy_image(
                &src_img.image,
                hal::image::Layout::TransferSrcOptimal,
                &dst_img.image,
                hal::image::Layout::TransferDstOptimal,
                &[hal::command::ImageCopy {
                    src_subresource: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: 0..layers as _,
                    },
                    src_offset: hal::image::Offset {
                        x: rect.origin.x as i32,
                        y: rect.origin.y as i32,
                        z: 0,
                    },
                    dst_subresource: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: 0..layers as _,
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

            let transitions = src_img
                .transit(src_image_prev_state, src_img.subresource_range.clone())
                .into_iter()
                .chain(dst_img.transit(dst_image_prev_state, dst_img.subresource_range.clone()));
            Self::execute_transitions(&mut self.command_buffer, transitions);
        }
    }

    fn generate_mipmaps(&mut self, texture: &Texture) {
        assert!(!self.inside_render_pass);
        texture.bound_in_frame.set(self.frame_id);

        let image = self
            .images
            .get_mut(&texture.id)
            .expect("Texture not found.");

        let mut mip_width = texture.size.width;
        let mut mip_height = texture.size.height;

        let mut half_mip_width = mip_width / 2;
        let mut half_mip_height = mip_height / 2;

        unsafe {
            assert_eq!(
                image.core.state.get().1,
                hal::image::Layout::ShaderReadOnlyOptimal
            );
            if let Some((barrier, pipeline_stages)) = image.core.transit(
                (
                    hal::image::Access::TRANSFER_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                ),
                image.core.subresource_range.clone(),
            ) {
                self.command_buffer.pipeline_barrier(
                    pipeline_stages,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            for index in 1..image.kind.num_levels() {
                if let Some((barrier, pipeline_stages)) = image.core.transit(
                    (
                        hal::image::Access::TRANSFER_READ,
                        hal::image::Layout::TransferSrcOptimal,
                    ),
                    hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: index - 1..index,
                        layers: 0..1,
                    },
                ) {
                    self.command_buffer.pipeline_barrier(
                        pipeline_stages,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                self.command_buffer.blit_image(
                    &image.core.image,
                    hal::image::Layout::TransferSrcOptimal,
                    &image.core.image,
                    hal::image::Layout::TransferDstOptimal,
                    hal::image::Filter::Linear,
                    &[hal::command::ImageBlit {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: index - 1,
                            layers: 0..1,
                        },
                        src_bounds: hal::image::Offset { x: 0, y: 0, z: 0 }..hal::image::Offset {
                            x: mip_width as i32,
                            y: mip_height as i32,
                            z: 1,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: index,
                            layers: 0..1,
                        },
                        dst_bounds: hal::image::Offset {
                            x: 0 as i32,
                            y: 0 as i32,
                            z: 0,
                        }..hal::image::Offset {
                            x: half_mip_width as i32,
                            y: half_mip_height as i32,
                            z: 1,
                        },
                    }],
                );
                if let Some((barrier, pipeline_stages)) = image.core.transit(
                    (
                        hal::image::Access::TRANSFER_WRITE,
                        hal::image::Layout::TransferDstOptimal,
                    ),
                    hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: index - 1..index,
                        layers: 0..1,
                    },
                ) {
                    self.command_buffer.pipeline_barrier(
                        pipeline_stages,
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

            if let Some((barrier, pipeline_stages)) = image.core.transit(
                (
                    hal::image::Access::SHADER_READ,
                    hal::image::Layout::ShaderReadOnlyOptimal,
                ),
                image.core.subresource_range.clone(),
            ) {
                self.command_buffer.pipeline_barrier(
                    pipeline_stages,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
    }

    fn acquire_depth_target(&mut self, dimensions: DeviceIntSize) -> RBOId {
        if let Entry::Occupied(mut o) = self.depth_targets.entry(dimensions) {
            o.get_mut().refcount += 1;
            return o.get().rbo_id;
        }

        let rbo_id = self.generate_rbo_id();
        let rbo = DepthBuffer::new(
            self.device.as_ref(),
            &mut *self.heaps.lock().unwrap(),
            dimensions.width as _,
            dimensions.height as _,
            self.depth_format,
        );
        if let Some((barrier, pipeline_stages)) = rbo.core.transit(
            (
                hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                hal::image::Layout::DepthStencilAttachmentOptimal,
            ),
            rbo.core.subresource_range.clone(),
        ) {
            unsafe {
                self.command_buffer.pipeline_barrier(
                    pipeline_stages,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
        self.rbos.insert(rbo_id, rbo);
        let target = SharedDepthTarget {
            rbo_id,
            refcount: 1,
        };
        record_gpu_alloc(depth_target_size_in_bytes(&dimensions));
        self.depth_targets.insert(dimensions, target);
        rbo_id
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
            let old_rbo = self.rbos.remove(&t.rbo_id).unwrap();
            old_rbo.deinit(self.device.as_ref(), &mut *self.heaps.lock().unwrap());
            record_gpu_free(depth_target_size_in_bytes(&dimensions));
        }
    }

    fn blit_with_shader(&mut self, src_rect: FramebufferIntRect, dest_rect: FramebufferIntRect) {
        assert!(self.inside_render_pass);
        let (src_layer, view_kind, texture_id, width, height) =
            if self.bound_read_fbo != DEFAULT_READ_FBO {
                let fbo = &self.fbos[&self.bound_read_fbo];
                let img = &self.images[&fbo.texture_id];
                let hal::image::Extent { width, height, .. } = img.kind.level_extent(0);
                let layer = fbo.layer_index;
                (layer, img.view_kind, fbo.texture_id, width, height)
            } else {
                error!("We should not use the main target as blit source");
                return;
            };

        let descriptor_group = (ShaderKind::Service).into();
        let per_draw_bindings = PerDrawBindings(
            [texture_id, texture_id, texture_id],
            TextureFilter::Nearest,
        );

        self.bind_per_draw_textures_impl(descriptor_group, per_draw_bindings);

        let src_bounds = hal::image::Offset {
            x: src_rect.origin.x as i32,
            y: src_rect.origin.y as i32,
            z: 0,
        }..hal::image::Offset {
            x: src_rect.origin.x as i32 + src_rect.size.width as i32,
            y: src_rect.origin.y as i32 + src_rect.size.height as i32,
            z: 1,
        };

        let dst_bounds = hal::image::Offset {
            x: dest_rect.origin.x as i32,
            y: dest_rect.origin.y as i32,
            z: 0,
        }..hal::image::Offset {
            x: dest_rect.origin.x as i32 + dest_rect.size.width as i32,
            y: dest_rect.origin.y as i32 + dest_rect.size.height as i32,
            z: 1,
        };

        let data = {
            // Image extents, layers are treated as depth
            let (sx, dx) = if dst_bounds.start.x > dst_bounds.end.x {
                (src_bounds.end.x, src_bounds.start.x - src_bounds.end.x)
            } else {
                (src_bounds.start.x, src_bounds.end.x - src_bounds.start.x)
            };
            let (sy, dy) = if dst_bounds.start.y > dst_bounds.end.y {
                (src_bounds.end.y, src_bounds.start.y - src_bounds.end.y)
            } else {
                (src_bounds.start.y, src_bounds.end.y - src_bounds.start.y)
            };

            vertex_types::BlitInstance {
                aOffset: [sx as f32 / width as f32, sy as f32 / height as f32],
                aExtent: [dx as f32 / width as f32, dy as f32 / height as f32],
                aZ: src_layer as f32,
                aLevel: 0.0,
            }
        };

        let viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: dst_bounds.start.x.min(dst_bounds.end.x) as _,
                y: dst_bounds.start.y.min(dst_bounds.end.y) as _,
                w: (dst_bounds.end.x - dst_bounds.start.x).abs() as _,
                h: (dst_bounds.end.y - dst_bounds.start.y).abs() as _,
            },
            depth: 0.0..1.0,
        };
        unsafe { self.command_buffer.set_viewports(0, &[viewport.clone()]) };
        self.update_instances(&[data]);

        self.ensure_blit_program(view_kind);
        let (_, program) = self.blit_programs.get_mut(&view_kind).unwrap();
        unsafe {
            self.command_buffer.bind_graphics_pipeline(
                &program
                    .pipelines
                    .get(&(self.surface_format, None, self.render_pass_depth_state, None))
                    .expect(&format!(
                        "Can't find blit program with render_pass_depth_state {:?} and format {:?}",
                        self.render_pass_depth_state, self.surface_format,
                    )),
            );
        }
        self.bound_program = INVALID_PROGRAM_ID;

        unsafe {
            self.command_buffer.bind_graphics_descriptor_sets(
                self.descriptor_data.pipeline_layout(&descriptor_group),
                DESCRIPTOR_SET_PER_DRAW,
                self.bound_per_draw_descriptor.as_ref().map(|descriptor| descriptor.raw()),
                &[],
            );
        }

        program.submit(
            &mut self.command_buffer,
            self.next_id,
            &self.quad_buffer,
            &self.instance_buffers[self.next_id],
            self.instance_buffer_range.clone(),
        );

        // Reset states after the draw
        unsafe { self.command_buffer.set_viewports(0, &[self.viewport.clone()]) };
        Self::replace_active_descriptor_set(
            &mut self.bound_per_draw_descriptor,
            None,
            &self.bound_per_draw_textures,
            &mut self.per_draw_descriptors,
        );
        self.bound_per_draw_textures = PerDrawBindings::default();
    }

    /// Perform a blit between self.bound_read_fbo and self.bound_draw_fbo.
    pub fn blit_render_target_impl(
        &mut self,
        src_rect: FramebufferIntRect,
        dest_rect: FramebufferIntRect,
        filter: TextureFilter,
    ) {
        debug_assert!(self.inside_frame);
        if self.inside_render_pass {
            return self.blit_with_shader(src_rect, dest_rect);
        }

        let (src_format, src_img, src_layer) = if self.bound_read_fbo != DEFAULT_READ_FBO {
            let fbo = &self.fbos[&self.bound_read_fbo];
            let img = &self.images[&fbo.texture_id];
            (img.format, &img.core, fbo.layer_index)
        } else {
            panic!("We should not blit from the main FBO!");
        };

        let (dest_format, dst_img, dest_layer) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let fbo = &self.fbos[&self.bound_draw_fbo];
            let img = &self.images[&fbo.texture_id];
            let layer = fbo.layer_index;
            (img.format, &img.core, layer)
        } else {
            panic!("Blitting to default draw fbo should be handled by now")
        };

        // let src_range = hal::image::SubresourceRange {
        //     aspects: hal::format::Aspects::COLOR,
        //     levels: 0 .. 1,
        //     layers: src_layer .. src_layer + 1,
        // };
        // let dest_range = hal::image::SubresourceRange {
        //     aspects: hal::format::Aspects::COLOR,
        //     levels: 0 .. 1,
        //     layers: dest_layer .. dest_layer + 1,
        // };

        unsafe {
            let (src_image_prev_state, dst_image_prev_state) =
                (src_img.state.get(), dst_img.state.get());
            let transitions = src_img
                .transit(
                    (
                        hal::image::Access::TRANSFER_READ,
                        hal::image::Layout::TransferSrcOptimal,
                    ),
                    src_img.subresource_range.clone(),
                )
                .into_iter()
                .chain(if self.draw_target_usage != DrawTargetUsage::CopyOnly {
                    dst_img.transit(
                        (
                            hal::image::Access::TRANSFER_WRITE,
                            hal::image::Layout::TransferDstOptimal,
                        ),
                        dst_img.subresource_range.clone(),
                    )
                } else {
                    None
                });
            Self::execute_transitions(&mut self.command_buffer, transitions);

            if src_rect.size != dest_rect.size || src_format != dest_format {
                // TODO:(zakorgy) add check if src and dst texture filters are different
                self.command_buffer.blit_image(
                    &src_img.image,
                    hal::image::Layout::TransferSrcOptimal,
                    &dst_img.image,
                    hal::image::Layout::TransferDstOptimal,
                    filter.into(),
                    &[hal::command::ImageBlit {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: src_layer..src_layer + 1,
                        },
                        src_bounds: hal::image::Offset {
                            x: src_rect.origin.x as i32,
                            y: src_rect.origin.y as i32,
                            z: 0,
                        }..hal::image::Offset {
                            x: src_rect.origin.x as i32 + src_rect.size.width as i32,
                            y: src_rect.origin.y as i32 + src_rect.size.height as i32,
                            z: 1,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: dest_layer..dest_layer + 1,
                        },
                        dst_bounds: hal::image::Offset {
                            x: dest_rect.origin.x as i32,
                            y: dest_rect.origin.y as i32,
                            z: 0,
                        }..hal::image::Offset {
                            x: dest_rect.origin.x as i32 + dest_rect.size.width as i32,
                            y: dest_rect.origin.y as i32 + dest_rect.size.height as i32,
                            z: 1,
                        },
                    }],
                );
            } else {
                self.command_buffer.copy_image(
                    &src_img.image,
                    hal::image::Layout::TransferSrcOptimal,
                    &dst_img.image,
                    hal::image::Layout::TransferDstOptimal,
                    &[hal::command::ImageCopy {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: src_layer..src_layer + 1,
                        },
                        src_offset: hal::image::Offset {
                            x: src_rect.origin.x as i32,
                            y: src_rect.origin.y as i32,
                            z: 0,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: dest_layer as _..(dest_layer + 1) as _,
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

            let transitions = src_img
                .transit(src_image_prev_state, src_img.subresource_range.clone())
                .into_iter()
                .chain(
                    // the blit caller code expects to be able to render to the target
                    if self.draw_target_usage != DrawTargetUsage::CopyOnly {
                        dst_img.transit(dst_image_prev_state, dst_img.subresource_range.clone())
                    } else {
                        None
                    },
                );
            Self::execute_transitions(&mut self.command_buffer, transitions);
        }
    }

    /// Perform a blit between src_target and dest_target.
    /// This will overwrite self.bound_read_fbo and self.bound_draw_fbo.
    pub fn blit_render_target(
        &mut self,
        src_target: ReadTarget,
        src_rect: FramebufferIntRect,
        dest_target: DrawTarget,
        dest_rect: FramebufferIntRect,
        filter: TextureFilter,
    ) {
        debug_assert!(self.inside_frame);
        self.bind_read_target(src_target);
        if !self.inside_render_pass {
            self.bind_draw_target(dest_target, DrawTargetUsage::Draw, None);
        }
        self.blit_render_target_impl(src_rect, dest_rect, filter);
    }

    /// Performs a blit while flipping vertically. Useful for blitting textures
    /// (which use origin-bottom-left) to the main framebuffer (which uses
    /// origin-top-left).
    /// We don't have to perform this kind of flip with gfx backends
    /// so we just execute the default blit_render_target method
    pub fn blit_render_target_invert_y(
        &mut self,
        src_target: ReadTarget,
        src_rect: FramebufferIntRect,
        dest_target: DrawTarget,
        dest_rect: FramebufferIntRect,
    ) {
        debug_assert!(self.inside_frame);
        self.blit_render_target(
            src_target,
            src_rect,
            dest_target,
            dest_rect,
            TextureFilter::Linear,
        );
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

    fn free_texture_unconditionaly(&mut self, mut texture: Texture) {
        if texture.supports_depth() {
            self.release_depth_target(texture.get_dimensions());
        }

        if !texture.fbos_with_depth.is_empty() {
            for old in texture.fbos_with_depth.drain(..) {
                debug_assert!(self.bound_draw_fbo != old || self.bound_read_fbo != old);
                let old_fbo = self.fbos.remove(&old).unwrap();
                old_fbo.deinit(self.device.as_ref());
            }
        }

        if !texture.fbos.is_empty() {
            for old in texture.fbos.drain(..) {
                debug_assert!(self.bound_draw_fbo != old || self.bound_read_fbo != old);
                let old_fbo = self.fbos.remove(&old).unwrap();
                old_fbo.deinit(self.device.as_ref());
            }
        }

        self.per_draw_descriptors.retain(&texture.id);
        self.per_pass_descriptors.retain(&texture.id);
        self.per_group_descriptors.retain(&texture.id);

        if texture.is_buffer {
            if let Some(buffer) = self.gpu_cache_buffers.remove(&texture.id) {
                buffer.deinit(self.device.as_ref(), &mut *self.heaps.lock().unwrap());
            }
        } else {
            let image = self.images.remove(&texture.id).expect("Texture not found.");
            image.deinit(self.device.as_ref(), &mut *self.heaps.lock().unwrap());
        }
        record_gpu_free(texture.size_in_bytes());
        texture.id = 0;
    }

    fn free_texture(&mut self, texture: Texture) {
        if texture.still_in_flight(self.frame_id, self.frame_count) {
            self.retained_textures.push(texture);
            return;
        }
        self.free_texture_unconditionaly(texture);
    }

    fn delete_retained_textures(&mut self) {
        let textures: SmallVec<[Texture; 16]> = self.retained_textures.drain(..).collect();
        for texture in textures {
            self.free_texture(texture);
        }
    }

    pub fn delete_texture(&mut self, mut texture: Texture) {
        if texture.size_in_bytes() == 0 {
            texture.id = 0;
            return;
        }

        self.free_texture(texture);
    }

    pub fn retain_cache_buffer(&mut self, texture: Texture) {
        self.retained_textures.push(texture);
    }

    #[cfg(feature = "replay")]
    pub fn delete_external_texture(&mut self, mut external: ExternalTexture) {
        warn!("delete external texture is missing");
        external.id = 0;
    }

    pub fn switch_mode(&mut self, _mode: i32) {
        debug_assert!(self.inside_frame);
    }

    pub fn create_pbo(&mut self) -> PBO {
        self.create_pbo_with_size(0)
    }

    pub fn create_pbo_with_size(&mut self, size: usize) -> PBO {
        PBO {
            id: 0,
            reserved_size: size,
        }
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
            UploadMethod::Immediate => unimplemented!("Immediate upload not implemented"),
            UploadMethod::PixelBuffer(..) => TextureUploader {
                device: self,
                texture,
            },
        }
    }

    pub fn upload_texture_immediate<T: Texel>(&mut self, texture: &Texture, pixels: &[T]) {
        assert!(!self.inside_render_pass);
        texture.bound_in_frame.set(self.frame_id);
        let len = pixels.len() / texture.layer_count as usize;
        for i in 0..texture.layer_count {
            let start = len * i as usize;

            self.images
                .get_mut(&texture.id)
                .expect("Texture not found.")
                .update(
                    self.device.as_ref(),
                    &mut self.command_buffer,
                    &mut self.staging_buffer_pool[self.next_id],
                    DeviceIntRect::new(DeviceIntPoint::new(0, 0), texture.size),
                    i,
                    texels_to_u8_slice(&pixels[start..(start + len)]),
                    None,
                );
        }
        if texture.filter == TextureFilter::Trilinear {
            self.generate_mipmaps(texture);
        }
    }

    #[cfg(feature = "capture")]
    pub fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8> {
        let mut pixels = vec![0; (img_desc.size.width * img_desc.size.height * 4) as usize];
        self.read_pixels_into(
            FramebufferIntRect::new(
                FramebufferIntPoint::zero(),
                FramebufferIntSize::new(img_desc.size.width, img_desc.size.height),
            ),
            ImageFormat::RGBA8,
            &mut pixels,
        );
        pixels
    }

    /// Read rectangle of pixels into the specified output slice.
    pub fn read_pixels_into(
        &mut self,
        rect: FramebufferIntRect,
        read_format: ImageFormat,
        output: &mut [u8],
    ) {
        self.wait_for_resources();

        let bytes_per_pixel = read_format.bytes_per_pixel();
        let size_in_bytes = (bytes_per_pixel * rect.size.width * rect.size.height) as usize;
        assert_eq!(output.len(), size_in_bytes);
        let capture_read =
            cfg!(feature = "capture") && self.bound_read_texture.0 != INVALID_TEXTURE_ID;

        let (image, image_format, layer) = if capture_read {
            let img = &self.images[&self.bound_read_texture.0];
            (&img.core, img.format, self.bound_read_texture.1 as u16)
        } else if self.bound_read_fbo != DEFAULT_READ_FBO {
            let fbo = &self.fbos[&self.bound_read_fbo];
            let img = &self.images[&fbo.texture_id];
            let layer = fbo.layer_index;
            (&img.core, img.format, layer)
        } else {
            if self.readback_texture.is_none() {
                warn!("Readback mode disabled, can't read the content of the main FBO!");
                return;
            }
            let id = &self.readback_texture.as_ref().unwrap().id;
            (&self.images[&id].core, self.surface_format, 0)
        };

        let (fmt_mismatch, stride) = if bytes_per_pixel < image_format.bytes_per_pixel() {
            // Special case which can occur during png save, because we force to read Rgba8 values from an Rgbaf32 texture.
            (
                true,
                (image_format.bytes_per_pixel() / bytes_per_pixel) as usize,
            )
        } else {
            assert_eq!(bytes_per_pixel, image_format.bytes_per_pixel());
            (false, 1)
        };

        assert!(
            output.len() <= DOWNLOAD_BUFFER_SIZE,
            "output len {:?} buffer size {:?}",
            output.len(),
            DOWNLOAD_BUFFER_SIZE
        );
        if self.download_buffer.is_none() {
            self.download_buffer = Some(Buffer::new(
                self.device.as_ref(),
                &mut *self.heaps.lock().unwrap(),
                MemoryUsageValue::Download,
                hal::buffer::Usage::TRANSFER_DST,
                (self.limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                DOWNLOAD_BUFFER_SIZE / stride,
                stride,
            ));
        }
        let download_buffer = self.download_buffer.as_mut().unwrap();
        let mut command_pool = unsafe {
            self.device.create_command_pool(
                self.queue_group_family,
                hal::pool::CommandPoolCreateFlags::empty(),
            )
        }
        .expect("create_command_pool failed");
        unsafe { command_pool.reset(false) };

        let mut cmd_buffer = unsafe { command_pool.allocate_one(Level::Primary) };
        unsafe {
            Self::begin_cmd_buffer(&mut cmd_buffer);

            // let range = hal::image::SubresourceRange {
            //     aspects: hal::format::Aspects::COLOR,
            //     levels: 0 .. 1,
            //     layers: layer .. layer + 1,
            // };
            let buffer_barrier = download_buffer.transit(hal::buffer::Access::TRANSFER_WRITE);
            let prev_image_state = image.state.get();
            match image.transit(
                (
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                ),
                image.subresource_range.clone(),
            ) {
                Some((barrier, pipeline_stages)) => {
                    cmd_buffer.pipeline_barrier(
                        pipeline_stages,
                        hal::memory::Dependencies::empty(),
                        buffer_barrier.into_iter().chain(Some(barrier)),
                    );
                }
                None => {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::TRANSFER..PipelineStage::TRANSFER,
                        hal::memory::Dependencies::empty(),
                        buffer_barrier.into_iter(),
                    );
                }
            };

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
                        layers: layer..layer + 1,
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
            if let Some((barrier, pipeline_stages)) =
                image.transit(prev_image_state, image.subresource_range.clone())
            {
                cmd_buffer.pipeline_barrier(
                    pipeline_stages,
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
            let submission = hal::queue::Submission {
                command_buffers: Some(&cmd_buffer),
                wait_semaphores: None,
                signal_semaphores: None,
            };
            self.queue_group_queues[0]
                .submit::<_, _, B::Semaphore, _, _>(submission, Some(&mut copy_fence));
            self.device
                .wait_for_fence(&copy_fence, !0)
                .expect("wait_for_fence failed");
            self.device.destroy_fence(copy_fence);
        }

        let mut data = vec![0; download_buffer.buffer_size];
        let range = 0..download_buffer.buffer_size as u64;
        if fmt_mismatch {
            let mut f32_data = vec![0f32; download_buffer.buffer_size];
            unsafe {
                let mut mapped = download_buffer
                    .memory_block
                    .map(self.device.as_ref(), range.clone())
                    .expect("Mapping memory block failed");
                let slice = mapped
                    .read(self.device.as_ref(), range)
                    .expect("Read failed");
                f32_data[0..slice.len()].copy_from_slice(&slice);
            }
            download_buffer.memory_block.unmap(self.device.as_ref());
            for i in 0..f32_data.len() {
                data[i] = round_to_int(f32_data[i].min(0f32).max(1f32));
            }
        } else {
            unsafe {
                let mut mapped = download_buffer
                    .memory_block
                    .map(self.device.as_ref(), range.clone())
                    .expect("Mapping memory block failed");
                let slice = mapped
                    .read(self.device.as_ref(), range)
                    .expect("Read failed");
                data[0..slice.len()].copy_from_slice(&slice);
            }
            download_buffer.memory_block.unmap(self.device.as_ref());
        }
        data.truncate(output.len());
        if !capture_read && self.surface_format == ImageFormat::BGRA8 {
            let width = rect.size.width as usize;
            let height = rect.size.height as usize;
            let row_pitch: usize = bytes_per_pixel as usize * width;
            // Vertical flip the result and convert to RGBA
            for y in 0..height as usize {
                for x in 0..width as usize {
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

        unsafe {
            command_pool.reset(false);
            self.device.destroy_command_pool(command_pool);
        }
    }

    /// Get texels of a texture into the specified output slice.
    pub fn get_tex_image_into(
        &mut self,
        _texture: &Texture,
        _format: ImageFormat,
        _output: &mut [u8],
    ) {
        unimplemented!("get_tex_image_into not implemented");
    }

    /// Attaches the provided texture to the current Read FBO binding.
    #[cfg(feature = "capture")]
    fn attach_read_texture_raw(&mut self, texture_id: u32, _target: TextureTarget, layer_id: i32) {
        self.bind_read_texture(texture_id, layer_id);
    }

    #[cfg(feature = "capture")]
    pub fn attach_read_texture_external(
        &mut self,
        _texture_id: u32,
        _target: TextureTarget,
        _layer_id: i32,
    ) {
        unimplemented!("attach_read_texture_external not implemented");
    }

    #[cfg(feature = "capture")]
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
        if self.bound_program != INVALID_PROGRAM_ID {
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
        #[cfg(debug_assertions)]
        debug_assert!(self.shader_is_ready);
        self.draw();
    }

    pub fn draw_triangles_u32(&mut self, _first_vertex: i32, _index_count: i32) {
        debug_assert!(self.inside_frame);
        #[cfg(debug_assertions)]
        debug_assert!(self.shader_is_ready);
        self.draw();
    }

    pub fn draw_nonindexed_lines(&mut self, _first_vertex: i32, _vertex_count: i32) {
        debug_assert!(self.inside_frame);
        #[cfg(debug_assertions)]
        debug_assert!(self.shader_is_ready);
        self.draw();
    }

    pub fn draw_indexed_triangles_instanced_u16(
        &mut self,
        _index_count: i32,
        _instance_count: i32,
    ) {
        debug_assert!(self.inside_frame);
        #[cfg(debug_assertions)]
        debug_assert!(self.shader_is_ready);
        self.draw();
    }

    pub fn end_frame(&mut self) {
        self.reset_draw_target(None);
        self.reset_read_target();

        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        self.frame_id.0 += 1;
    }

    pub fn begin_render_pass(&mut self, last_main_fbo_pass: bool) {
        assert!(!self.last_rp_in_frame_reached);
        assert!(!self.inside_render_pass);
        assert_eq!(self.draw_target_usage, DrawTargetUsage::Draw);

        let (color_clear, depth_clear) = match self.clear_values.remove(&self.bound_draw_fbo) {
            Some(ClearValues { color, depth }) => (Some(color), depth),
            None => (None, None),
        };

        let (render_pass, frame_buffer, rect) = if self.bound_draw_fbo == DEFAULT_DRAW_FBO {
            let new_layout = if last_main_fbo_pass {
                self.last_rp_in_frame_reached = true;
                hal::image::Layout::Present
            } else {
                hal::image::Layout::ColorAttachmentOptimal
            };

            let render_pass = self.render_passes.main_target_pass(
                self.swapchain_image_layouts[self.next_id],
                new_layout,
                color_clear.is_some(),
            );
            let rect = hal::pso::Rect {
                x: 0,
                y: 0,
                w: self.dimensions.0 as _,
                h: self.dimensions.1 as _,
            };

            let frame_buffer = self.frame.as_mut().unwrap().get_or_create_fbo(
                self.device.as_ref(),
                self.swapchain_image_layouts[self.next_id],
                new_layout,
                color_clear.is_some(),
                &self.frame_depth.core.view,
                rect,
                &render_pass,
            );
            self.swapchain_image_layouts[self.next_id] = new_layout;
            (render_pass, frame_buffer, rect)
        } else {
            let format = self.fbos[&self.bound_draw_fbo].format;
            (
                self.render_passes
                    .render_pass(format, self.depth_available, color_clear.is_some()),
                &self.fbos[&self.bound_draw_fbo].fbo,
                self.viewport.rect,
            )
        };

        unsafe {
            self.command_buffer.begin_render_pass(
                render_pass,
                frame_buffer,
                rect,
                color_clear.into_iter().chain(depth_clear.into_iter()),
                hal::command::SubpassContents::Inline,
            );
            self.command_buffer.set_viewports(0, &[self.viewport.clone()]);
            let scissor_rect = if self.scissor_enabled {
                self.scissor_rect
            } else {
                self.viewport.rect
            };
            self.command_buffer.set_scissors(0, &[scissor_rect]);
        }
        self.inside_render_pass = true;
    }

    pub fn end_render_pass(&mut self) {
        if self.inside_render_pass {
            unsafe { self.command_buffer.end_render_pass() };
            self.inside_render_pass = false;
        }
    }

    pub fn clear_target_rects(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rects: impl IntoIterator<Item = FramebufferIntRect>,
    ) {
        assert!(self.inside_render_pass);

        let rects = rects.into_iter().map(|rect| {
            hal::pso::ClearRect {
                rect: hal::pso::Rect {
                    x: rect.origin.x as i16,
                    y: rect.origin.y as i16,
                    w: rect.size.width as i16,
                    h: rect.size.height as i16,
                },
                layers: 0..1,
            }
        });

        let color_clear = color.map(|c| hal::command::AttachmentClear::Color {
            index: 0,
            value: ClearColor { float32: c },
        });

        let depth_clear = depth.map(|d| hal::command::AttachmentClear::DepthStencil {
            depth: Some(d),
            stencil: None,
        });

        unsafe {
            self.command_buffer
                .clear_attachments(color_clear.into_iter().chain(depth_clear), rects);
        }
    }

    fn clear_target_image(&mut self, color: Option<[f32; 4]>, depth: Option<f32>) {
        assert!(!self.inside_render_pass);
        let (img, layer, dimg) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let fbo = &self.fbos[&self.bound_draw_fbo];
            let img = &self.images[&fbo.texture_id];
            let dimg = if depth.is_some() {
                Some(&self.rbos[&fbo.rbo].core)
            } else {
                None
            };
            (&img.core, fbo.layer_index, dimg)
        } else {
            panic!("This should be handled by attachment load operations");
        };

        assert_eq!(self.draw_target_usage, DrawTargetUsage::Draw);

        //Note: this function is assumed to be called within an active FBO
        // thus, we bring back the targets into renderable state
        unsafe {
            if let Some(color) = color {
                if let Some((barrier, pipeline_stages)) = img.transit(
                    (
                        hal::image::Access::TRANSFER_WRITE,
                        hal::image::Layout::TransferDstOptimal,
                    ),
                    img.subresource_range.clone(),
                ) {
                    self.command_buffer.pipeline_barrier(
                        pipeline_stages,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                self.command_buffer.clear_image(
                    &img.image,
                    hal::image::Layout::TransferDstOptimal,
                    ClearValue {
                        color: ClearColor { float32: color },
                    },
                    Some(hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: 0..1,
                        layers: layer..layer + 1,
                    }),
                );
                if let Some((barrier, pipeline_stages)) = img.transit(
                    (
                        hal::image::Access::COLOR_ATTACHMENT_WRITE,
                        hal::image::Layout::ColorAttachmentOptimal,
                    ),
                    img.subresource_range.clone(),
                ) {
                    self.command_buffer.pipeline_barrier(
                        pipeline_stages,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }

            if let (Some(depth), Some(dimg)) = (depth, dimg) {
                assert_ne!(self.current_depth_test, None);
                let prev_dimg_state = dimg.state.get();
                if let Some((barrier, pipeline_stages)) = dimg.transit(
                    (
                        hal::image::Access::TRANSFER_WRITE,
                        hal::image::Layout::TransferDstOptimal,
                    ),
                    dimg.subresource_range.clone(),
                ) {
                    self.command_buffer.pipeline_barrier(
                        pipeline_stages,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                self.command_buffer.clear_image(
                    &dimg.image,
                    hal::image::Layout::TransferDstOptimal,
                    ClearValue {
                        depth_stencil: ClearDepthStencil { depth, stencil: 0 },
                    },
                    Some(dimg.subresource_range.clone()),
                );
                if let Some((barrier, pipeline_stages)) =
                    dimg.transit(prev_dimg_state, dimg.subresource_range.clone())
                {
                    self.command_buffer.pipeline_barrier(
                        pipeline_stages,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }
        }
    }

    pub fn clear_target(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<FramebufferIntRect>,
        can_use_load_op: bool,
    ) {
        if color.is_none() && depth.is_none() {
            return;
        }
        if rect.is_some() {
            self.clear_target_rects(color, depth, rect);
        } else if can_use_load_op {
            let color = color.map(|c| ClearValue {
                color: ClearColor { float32: c },
            });
            let depth = depth.map(|d| ClearValue {
                depth_stencil: ClearDepthStencil {
                    depth: d,
                    stencil: 0,
                },
            });
            if depth.is_some() {
                assert!(color.is_some());
            }
            match self.clear_values.entry(self.bound_draw_fbo) {
                Entry::Occupied(mut o) => {
                    let ClearValues {
                        color: old_color,
                        depth: old_depth,
                    } = o.get_mut();
                    if let Some(c) = color {
                        *old_color = c;
                    }
                    if !depth.is_none() {
                        *old_depth = depth;
                    }
                }
                Entry::Vacant(v) => {
                    v.insert(ClearValues {
                        color: color.unwrap(),
                        depth,
                    });
                }
            }
        } else {
            self.clear_target_image(color, depth);
        }
    }

    pub fn clear_rt_if_needed(&mut self) {
        assert!(!self.inside_render_pass);
        if let Some(ClearValues { color, depth }) = self.clear_values.remove(&self.bound_draw_fbo) {
            unsafe { self.clear_target_image(Some(color.color.float32), depth.map(|d| d.depth_stencil.depth)) };
        };
    }

    pub fn enable_depth(&mut self) {
        assert!(
            self.depth_available,
            "Enabling depth test without depth target"
        );
        self.current_depth_test = Some(LESS_EQUAL_TEST);
    }

    pub fn disable_depth(&mut self) {
        self.current_depth_test = None;
    }

    pub fn set_depth_func(&mut self, _depth_func: DepthFunction) {
        // TODO add Less depth function
        //self.current_depth_test = depth_func;
    }

    pub fn enable_depth_write(&mut self) {
        assert!(
            self.depth_available,
            "Enabling depth test without depth target"
        );
        self.current_depth_test = Some(LESS_EQUAL_WRITE);
    }

    pub fn disable_depth_write(&mut self) {
        if self.current_depth_test == Some(LESS_EQUAL_WRITE) {
            self.current_depth_test = Some(LESS_EQUAL_TEST);
        }
    }

    pub fn disable_stencil(&self) {
        warn!("disable stencil is missing")
    }

    pub fn set_scissor_rect(&mut self, rect: FramebufferIntRect) {
        self.scissor_rect = hal::pso::Rect {
            x: rect.origin.x as _,
            y: rect.origin.y as _,
            w: rect.size.width as _,
            h: rect.size.height as _,
        };
    }

    pub fn enable_scissor(&mut self) {
        self.scissor_enabled = true;
    }

    pub fn disable_scissor(&mut self) {
        self.scissor_enabled = false;
    }

    pub fn set_blend(&mut self, enable: bool) {
        if !enable {
            self.current_blend_state = None;
        }
        #[cfg(debug_assertions)]
        {
            self.shader_is_ready = false;
        }
    }

    fn set_blend_state(&mut self, blend_state: Option<BlendState>) {
        self.current_blend_state = blend_state;
        #[cfg(debug_assertions)]
        {
            self.shader_is_ready = false;
        }
    }

    pub fn set_blend_mode_alpha(&mut self) {
        self.set_blend_state(Some(ALPHA));
    }

    pub fn set_blend_mode_premultiplied_alpha(&mut self) {
        self.set_blend_state(Some(BlendState::PREMULTIPLIED_ALPHA));
    }

    pub fn set_blend_mode_premultiplied_dest_out(&mut self) {
        self.set_blend_state(Some(PREMULTIPLIED_DEST_OUT));
    }

    pub fn set_blend_mode_multiply(&mut self) {
        self.set_blend_state(Some(BlendState::MULTIPLY));
    }

    pub fn set_blend_mode_max(&mut self) {
        self.set_blend_state(Some(MAX));
    }

    pub fn set_blend_mode_min(&mut self) {
        self.set_blend_state(Some(MIN));
    }

    pub fn set_blend_mode_subpixel_pass0(&mut self) {
        self.set_blend_state(Some(SUBPIXEL_PASS0));
    }

    pub fn set_blend_mode_subpixel_pass1(&mut self) {
        self.set_blend_state(Some(SUBPIXEL_PASS1));
    }

    pub fn set_blend_mode_subpixel_with_bg_color_pass0(&mut self) {
        self.set_blend_state(Some(SUBPIXEL_WITH_BG_COLOR_PASS0));
    }

    pub fn set_blend_mode_subpixel_with_bg_color_pass1(&mut self) {
        self.set_blend_state(Some(SUBPIXEL_WITH_BG_COLOR_PASS1));
    }

    pub fn set_blend_mode_subpixel_with_bg_color_pass2(&mut self) {
        self.set_blend_state(Some(SUBPIXEL_WITH_BG_COLOR_PASS2));
    }

    pub fn set_blend_mode_subpixel_constant_text_color(&mut self, color: ColorF) {
        self.set_blend_state(Some(SUBPIXEL_CONSTANT_TEXT_COLOR));
        // color is an unpremultiplied color.
        self.blend_color = ColorF::new(color.r, color.g, color.b, 1.0);
    }

    pub fn set_blend_mode_subpixel_dual_source(&mut self) {
        self.set_blend_state(Some(SUBPIXEL_DUAL_SOURCE));
    }

    pub fn set_blend_mode_show_overdraw(&mut self) {
        self.set_blend_state(Some(OVERDRAW));
    }

    pub fn set_blend_mode_advanced(&mut self, _mode: MixBlendMode) {
        unimplemented!("set_blend_mode_advanced is unimplemented");
    }

    pub fn supports_features(&self, features: hal::Features) -> bool {
        self.features.contains(features)
    }

    pub fn echo_driver_messages(&self) {
        warn!("echo_driver_messages is unimplemeneted");
    }

    pub fn set_next_frame_id(&mut self) {
        self.last_rp_in_frame_reached = false;
        if let Some((barrier, pipeline_stages)) = self.frame_depth.core.transit(
            (
                hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                hal::image::Layout::DepthStencilAttachmentOptimal,
            ),
            self.frame_depth.core.subresource_range.clone(),
        ) {
            unsafe {
                self.command_buffer.pipeline_barrier(
                    pipeline_stages,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
        unsafe {
            if let Some(ref mut surface) = self.surface.as_mut() {
                if self.frame.is_some() {
                    return;
                }
                match surface.acquire_image(!0) {
                    Ok((swapchain_image, suboptimal)) => {
                        if suboptimal.is_some() {
                            warn!("The swapchain no longer matches the surface, but we still can use it.");
                        }
                        self.frame = Some(Frame::new(swapchain_image));
                    }
                    Err(acquire_error) => {
                        error!("Acquire error {:?}, recrating swapchian.", acquire_error);
                        self.recreate_swapchain(None);
                    }
                }
            }
        }
    }

    pub fn submit_to_gpu(&mut self) {
        unsafe {
            self.command_buffer.finish();
            match self.surface.as_mut() {
                Some(surface) => {
                    let submission = hal::queue::Submission {
                        command_buffers: &[&self.command_buffer],
                        wait_semaphores: None,
                        signal_semaphores: Some(&self.render_finished_semaphores[self.next_id]),
                    };
                    self.queue_group_queues[0]
                        .submit(submission, Some(&mut self.frame_fence[self.next_id].inner));
                    self.frame_fence[self.next_id].is_submitted = true;

                    let frame = self.frame.take().unwrap();

                    // present frame
                    match self.queue_group_queues[0].present_surface(
                        surface,
                        frame.swapchain_image,
                        Some(&self.render_finished_semaphores[self.next_id]),
                    ) {
                        Ok(suboptimal) => {
                            if suboptimal.is_some() {
                                warn!("The swapchain no longer matches the surface, but we still can use it.");
                            }
                        }
                        Err(presenterr) => {
                            error!("Present error {:?}, recrating swapchian.", presenterr);
                            self.recreate_swapchain(None);
                        }
                    }
                    for (_, fb) in frame.framebuffers {
                        self.device.destroy_framebuffer(fb);
                    }
                }
                None => {
                    let submission = hal::queue::Submission {
                        command_buffers: &[&self.command_buffer],
                        wait_semaphores: None,
                        signal_semaphores: None,
                    };
                    self.queue_group_queues[0].submit::<_, _, B::Semaphore, _, _>(
                        submission,
                        Some(&mut self.frame_fence[self.next_id].inner),
                    );
                    self.frame_fence[self.next_id].is_submitted = true;
                }
            }
        };
        self.reset_next_frame_resources();
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
        for command_pool in &mut self.command_pools {
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

    pub fn deinit(mut self) {
        self.device.wait_idle().unwrap();
        for texture in self.retained_textures.drain(..).collect::<Vec<_>>() {
            self.free_texture_unconditionaly(texture);
        }
        if let Some(texture) = self.readback_texture.take() {
            self.free_texture_unconditionaly(texture);
        }
        unsafe {
            if self.save_cache && self.cache_path.is_some() {
                let pipeline_cache = self
                    .device
                    .create_pipeline_cache(None)
                    .expect("Failed to create pipeline cache");
                let data;
                if let Some(cache) = self.pipeline_cache {
                    self.device
                        .merge_pipeline_caches(&cache, Some(&pipeline_cache))
                        .expect("merge_pipeline_caches failed");
                    data = self
                        .device
                        .get_pipeline_cache_data(&cache)
                        .expect("get_pipeline_cache_data failed");
                    self.device.destroy_pipeline_cache(cache);
                } else {
                    data = self
                        .device
                        .get_pipeline_cache_data(&pipeline_cache)
                        .expect("get_pipeline_cache_data failed");
                };
                self.device.destroy_pipeline_cache(pipeline_cache);

                if data.len() > 0 {
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .open(&self.cache_path.as_ref().unwrap())
                        .expect("File open/creation failed");

                    use std::io::Write;
                    file.write(&data).expect("File write failed");
                }
            } else {
                if let Some(cache) = self.pipeline_cache {
                    self.device.destroy_pipeline_cache(cache);
                }
            }

            Self::replace_active_descriptor_set(
                &mut self.bound_per_draw_descriptor,
                None,
                &self.bound_per_draw_textures,
                &mut self.per_draw_descriptors,
            );

            Self::replace_active_descriptor_set(
                &mut self.bound_per_pass_descriptor,
                None,
                &self.bound_per_pass_textures,
                &mut self.per_pass_descriptors,
            );

            for descriptor_group in [DescriptorGroup::Default, DescriptorGroup::Clip, DescriptorGroup::Primitive].iter() {
                Self::replace_active_descriptor_set(
                    &mut self.bound_per_group_descriptors[*descriptor_group as usize],
                    None,
                    &(*descriptor_group, self.bound_per_group_textures),
                    &mut self.per_group_descriptors,
                );
            }

            let mut heaps = Arc::try_unwrap(self.heaps).unwrap().into_inner().unwrap();

            self.command_pools[self.next_id].return_cmd_buffer(self.command_buffer);
            for command_pool in self.command_pools {
                command_pool.destroy(self.device.as_ref());
            }
            for staging_buffer_pool in self.staging_buffer_pool {
                staging_buffer_pool.deinit(self.device.as_ref(), &mut heaps);
            }
            self.quad_buffer.deinit(self.device.as_ref(), &mut heaps);
            for instance_buffer in self.instance_buffers {
                instance_buffer.deinit(self.device.as_ref(), &mut heaps);
            }
            for buffer in self.free_instance_buffers {
                buffer.deinit(self.device.as_ref(), &mut heaps);
            }
            if let Some(buffer) = self.download_buffer {
                buffer.deinit(self.device.as_ref(), &mut heaps);
            }
            if let Some(frame) = self.frame {
                frame.deinit(self.device.as_ref())
            }
            self.frame_depth.deinit(self.device.as_ref(), &mut heaps);
            if let Some(mut surface) = self.surface {
                surface.unconfigure_swapchain(&self.device);
            }
            for (_, image) in self.images {
                image.deinit(self.device.as_ref(), &mut heaps);
            }
            for (_, rbo) in self.fbos {
                rbo.deinit(self.device.as_ref());
            }
            for (_, rbo) in self.rbos {
                rbo.deinit(self.device.as_ref(), &mut heaps);
            }

            if let Some(buffer) = self.gpu_cache_buffer {
                self.device
                    .as_ref()
                    .destroy_buffer(Arc::try_unwrap(buffer.buffer).unwrap());
            }

            for (_, buffer) in self.gpu_cache_buffers.into_iter() {
                buffer.deinit(self.device.as_ref(), &mut heaps);
            }
            self.device.destroy_sampler(self.sampler_linear);
            self.device.destroy_sampler(self.sampler_nearest);

            self.per_draw_descriptors.free(&mut self.desc_allocator);
            self.per_group_descriptors.free(&mut self.desc_allocator);
            self.per_pass_descriptors.free(&mut self.desc_allocator);

            self.uniform_buffer_handler.deinit(self.device.as_ref(), &mut heaps, &mut self.desc_allocator);
            self.desc_allocator.dispose(self.device.as_ref());
            for (_, program) in self.programs {
                program.deinit(self.device.as_ref(), &mut heaps)
            }
            for (_, (_, program)) in self.blit_programs {
                program.deinit(self.device.as_ref(), &mut heaps)
            }
            heaps.dispose(self.device.as_ref());
            for (_, (vs_module, fs_module)) in self.shader_modules {
                self.device.destroy_shader_module(vs_module);
                self.device.destroy_shader_module(fs_module);
            }
            self.descriptor_data.deinit(self.device.as_ref());
            self.render_passes.deinit(self.device.as_ref());
            for fence in self.frame_fence {
                self.device.destroy_fence(fence.inner);
            }
            for semaphore in self.render_finished_semaphores {
                self.device.destroy_semaphore(semaphore);
            }
        }
        // We must ensure these are dropped before `self._instance` or we segfault with Vulkan
        mem::drop(self.device);
        mem::drop(self.queue_group_queues);
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
        format_override: Option<ImageFormat>,
        data: &[T],
    ) -> usize {
        assert!(!self.device.inside_render_pass);
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
                for j in 0..height {
                    for i in 0..width {
                        let offset = i * data_stride + j * data_stride * width;
                        let src = &data[j * row_length + i * data_stride..];
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
                for j in 0..height {
                    for i in 0..width {
                        let offset = i * data_stride + j * data_stride * width;
                        let src = &data[j * row_length + i * data_stride..];
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
                self.device.device.as_ref(),
                &mut self.device.command_buffer,
                &mut self.device.staging_buffer_pool[self.device.next_id],
                rect,
                layer_index,
                data,
                format_override,
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
