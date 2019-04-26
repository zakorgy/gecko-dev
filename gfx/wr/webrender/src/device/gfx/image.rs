/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{DeviceIntRect, ImageFormat};
use hal::{self, Device as BackendDevice};
use rendy_memory::{Block, Heaps, MemoryBlock, MemoryUsageValue};

use std::cell::Cell;
use super::buffer::BufferPool;
use super::command::CommandPool;
use super::render_pass::RenderPass;
use super::TextureId;
use super::super::{RBOId, Texture};

const DEPTH_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::DEPTH,
    levels: 0 .. 1,
    layers: 0 .. 1,
};

pub(super) struct ImageCore<B: hal::Backend> {
    pub(super) image: B::Image,
    pub(super) memory_block: Option<MemoryBlock<B>>,
    pub(super) view: B::ImageView,
    pub(super) subresource_range: hal::image::SubresourceRange,
    pub(super) state: Cell<hal::image::State>,
}

impl<B: hal::Backend> ImageCore<B> {
    pub(super) fn from_image(
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
            memory_block: None,
            view,
            subresource_range,
            state: Cell::new((hal::image::Access::empty(), hal::image::Layout::Undefined)),
        }
    }

    pub(super) fn create(
        device: &B::Device,
        heaps: &mut Heaps<B>,
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

        let memory_block = heaps
            .allocate(
                device,
                requirements.type_mask as u32,
                MemoryUsageValue::Data,
                requirements.size,
                requirements.alignment,
            )
            .expect("Allocate memory failed");

        unsafe {
            device
                .bind_image_memory(
                    &memory_block.memory(),
                    memory_block.range().start,
                    &mut image,
                )
                .expect("Bind image memory failed")
        };

        ImageCore {
            memory_block: Some(memory_block),
            ..Self::from_image(device, image, view_kind, format, subresource_range)
        }
    }

    fn _reset(&self) {
        self.state
            .set((hal::image::Access::empty(), hal::image::Layout::Undefined));
    }

    pub(super) fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        unsafe { device.destroy_image_view(self.view) };
        if let Some(memory_block) = self.memory_block {
            unsafe {
                device.destroy_image(self.image);
                heaps.free(device, memory_block);
            }
        }
    }

    pub(super) fn transit(
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

pub(super) struct Image<B: hal::Backend> {
    pub(super) core: ImageCore<B>,
    pub(super) kind: hal::image::Kind,
    pub(super) format: ImageFormat,
}

impl<B: hal::Backend> Image<B> {
    pub(super) fn new(
        device: &B::Device,
        heaps: &mut Heaps<B>,
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
            ImageFormat::RGBAF32 => hal::format::Format::Rgba32Sfloat,
            ImageFormat::RGBAI32 => hal::format::Format::Rgba32Sint,
        };
        let kind = hal::image::Kind::D2(image_width as _, image_height as _, image_depth as _, 1);

        let core = ImageCore::create(
            device,
            heaps,
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

    pub(super) fn update(
        &self,
        device: &B::Device,
        cmd_pool: &mut CommandPool<B>,
        staging_buffer_pool: &mut BufferPool<B>,
        rect: DeviceIntRect,
        layer_index: i32,
        image_data: &[u8],
    ) {
        use hal::pso::PipelineStage;
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

    pub fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        self.core.deinit(device, heaps);
    }
}

pub(super) struct Framebuffer<B: hal::Backend> {
    pub(super) texture_id: TextureId,
    pub(super) layer_index: u16,
    pub(super) format: ImageFormat,
    image_view: B::ImageView,
    pub(super) fbo: B::Framebuffer,
    pub(super) rbo: RBOId,
}

impl<B: hal::Backend> Framebuffer<B> {
    pub(super) fn new(
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
                    Some(&image_view).into_iter().chain(depth.into_iter()),
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
            texture_id: texture.id,
            layer_index,
            format: texture.format,
            image_view,
            fbo,
            rbo,
        }
    }

    pub(super) fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_framebuffer(self.fbo);
            device.destroy_image_view(self.image_view);
        }
    }
}

pub(super) struct DepthBuffer<B: hal::Backend> {
    pub(super) core: ImageCore<B>,
}

impl<B: hal::Backend> DepthBuffer<B> {
    pub(super) fn new(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        pixel_width: u32,
        pixel_height: u32,
        depth_format: hal::format::Format,
    ) -> Self {
        let core = ImageCore::create(
            device,
            heaps,
            hal::image::Kind::D2(pixel_width, pixel_height, 1, 1),
            hal::image::ViewKind::D2,
            1,
            depth_format,
            hal::image::Usage::TRANSFER_DST | hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
            DEPTH_RANGE,
        );
        DepthBuffer { core }
    }

    pub(super) fn deinit(self, device: &B::Device, heaps: &mut Heaps<B>) {
        self.core.deinit(device, heaps);
    }
}
