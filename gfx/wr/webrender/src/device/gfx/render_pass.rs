/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ImageFormat;
use hal::Device;

pub(super) struct HalRenderPasses<B: hal::Backend> {
    pub(super) r8: B::RenderPass,
    pub(super) r8_depth: B::RenderPass,
    pub(super) bgra8: B::RenderPass,
    pub(super) bgra8_depth: B::RenderPass,
    pub(super) rgba8: B::RenderPass,
    pub(super) rgba8_depth: B::RenderPass,
    pub(super) rgbaf32: B::RenderPass,
    pub(super) rgbaf32_depth: B::RenderPass,
}

impl<B: hal::Backend> HalRenderPasses<B> {
    pub(super) fn get_render_pass(
        &self,
        format: ImageFormat,
        depth_enabled: bool,
    ) -> &B::RenderPass {
        match format {
            ImageFormat::R8 if depth_enabled => &self.r8_depth,
            ImageFormat::R8 => &self.r8,
            ImageFormat::BGRA8 if depth_enabled => &self.bgra8_depth,
            ImageFormat::BGRA8 => &self.bgra8,
            ImageFormat::RGBA8 if depth_enabled => &self.rgba8_depth,
            ImageFormat::RGBA8 => &self.rgba8,
            ImageFormat::RGBAF32 if depth_enabled => &self.rgbaf32,
            ImageFormat::RGBAF32 => &self.rgbaf32_depth,
            f => unimplemented!("No render pass for image format {:?}", f),
        }
    }

    pub(super) fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_render_pass(self.r8);
            device.destroy_render_pass(self.r8_depth);
            device.destroy_render_pass(self.bgra8);
            device.destroy_render_pass(self.bgra8_depth);
            device.destroy_render_pass(self.rgba8);
            device.destroy_render_pass(self.rgba8_depth);
            device.destroy_render_pass(self.rgbaf32);
            device.destroy_render_pass(self.rgbaf32_depth);
        }
    }

    pub fn create_render_passes(
        device: &B::Device,
        surface_format: hal::format::Format,
        depth_format: hal::format::Format,
    ) -> HalRenderPasses<B> {
        let attachment_r8 = hal::pass::Attachment {
            format: Some(hal::format::Format::R8Unorm),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::ColorAttachmentOptimal
                .. hal::image::Layout::ColorAttachmentOptimal,
        };

        let attachment_bgra8 = hal::pass::Attachment {
            format: Some(surface_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::ColorAttachmentOptimal
                .. hal::image::Layout::ColorAttachmentOptimal,
        };

        let attachment_rgba8 = hal::pass::Attachment {
            format: Some(hal::format::Format::Rgba8Unorm),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::ColorAttachmentOptimal
                .. hal::image::Layout::ColorAttachmentOptimal,
        };

        let attachment_rgbaf32 = hal::pass::Attachment {
            format: Some(hal::format::Format::Rgba32Sfloat),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
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
                hal::pass::AttachmentLoadOp::Load,
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

        let subpass_rgba8 = hal::pass::SubpassDesc {
            colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let subpass_depth_rgba8 = hal::pass::SubpassDesc {
            colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let subpass_rgbaf32 = hal::pass::SubpassDesc {
            colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let subpass_depth_rgbaf32 = hal::pass::SubpassDesc {
            colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        use std::iter;
        HalRenderPasses {
            r8: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_r8),
                    &[subpass_r8],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            r8_depth: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_r8).chain(iter::once(&attachment_depth)),
                    &[subpass_depth_r8],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            rgbaf32: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_rgbaf32),
                    &[subpass_rgbaf32],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            rgbaf32_depth: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_rgbaf32).chain(iter::once(&attachment_depth)),
                    &[subpass_depth_rgbaf32],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            rgba8: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_rgba8),
                    &[subpass_rgba8],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            rgba8_depth: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_rgba8).chain(iter::once(&attachment_depth)),
                    &[subpass_depth_rgba8],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            bgra8: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_bgra8),
                    &[subpass_bgra8],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            bgra8_depth: unsafe {
                device.create_render_pass(
                    &[attachment_bgra8, attachment_depth],
                    &[subpass_depth_bgra8],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
        }
    }
}
