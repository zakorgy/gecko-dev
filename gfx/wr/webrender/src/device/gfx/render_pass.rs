/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ImageFormat;
use hal::device::Device;
use hal::image::Layout;

pub(super) struct HalRenderPasses<B: hal::Backend> {
    // passes of inermediate targets
    pub(super) r8: B::RenderPass,
    pub(super) r8_clear: B::RenderPass,
    pub(super) r8_depth: B::RenderPass,
    pub(super) bgra8: B::RenderPass,
    pub(super) bgra8_clear: B::RenderPass,
    pub(super) bgra8_depth_clear: B::RenderPass,
    pub(super) rgbaf32: B::RenderPass,
    pub(super) rgbaf32_depth: B::RenderPass,

    // this is used by intermediate and main targets as well
    pub(super) bgra8_depth: B::RenderPass,

    // main target passes
    pub(super) cao_to_present: B::RenderPass,
    pub(super) undef_to_cao: B::RenderPass,
    pub(super) undef_to_present: B::RenderPass,
}

impl<B: hal::Backend> HalRenderPasses<B> {
    pub(super) fn render_pass(
        &self,
        format: ImageFormat,
        depth_enabled: bool,
        clear: bool,
    ) -> &B::RenderPass {
        match format {
            ImageFormat::R8 => match (depth_enabled, clear) {
                (true, true) => panic!("We should not use this case"),
                (true, false) => &self.r8_depth,
                (false, true) => &self.r8_clear,
                (false, false) => &self.r8,
            },
            ImageFormat::BGRA8 => match (depth_enabled, clear) {
                (true, true) => &self.bgra8_depth_clear,
                (true, false) => &self.bgra8_depth,
                (false, true) => &self.bgra8_clear,
                (false, false) => &self.bgra8,
            },
            ImageFormat::RGBAF32 if depth_enabled => &self.rgbaf32_depth,
            ImageFormat::RGBAF32 => &self.rgbaf32,
            f => unimplemented!("No render pass for image format {:?}", f),
        }
    }

    pub(super) fn main_target_pass(
        &self,
        old_layout: Layout,
        new_layout: Layout,
        clear: bool
    ) -> &B::RenderPass {
        match (old_layout, new_layout, clear) {
            (Layout::ColorAttachmentOptimal, Layout::ColorAttachmentOptimal, false) => &self.bgra8_depth,
            (Layout::ColorAttachmentOptimal, Layout::Present, false) => &self.cao_to_present,
            (Layout::Present, Layout::ColorAttachmentOptimal, true) => &self.undef_to_cao,
            (Layout::Present, Layout::Present, true) => &self.undef_to_present,
            (Layout::Undefined, Layout::ColorAttachmentOptimal, true) => &self.undef_to_cao,
            (Layout::Undefined, Layout::Present, true) => &self.undef_to_present,
            conf => unimplemented!("No render pass for configuration {:?}", conf),
        }
    }

    pub(super) fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_render_pass(self.r8);
            device.destroy_render_pass(self.r8_depth);
            device.destroy_render_pass(self.r8_clear);
            device.destroy_render_pass(self.bgra8);
            device.destroy_render_pass(self.bgra8_depth);
            device.destroy_render_pass(self.bgra8_clear);
            device.destroy_render_pass(self.bgra8_depth_clear);
            device.destroy_render_pass(self.rgbaf32);
            device.destroy_render_pass(self.rgbaf32_depth);

            device.destroy_render_pass(self.cao_to_present);
            device.destroy_render_pass(self.undef_to_cao);
            device.destroy_render_pass(self.undef_to_present);
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
            layouts: Layout::ColorAttachmentOptimal
                .. Layout::ColorAttachmentOptimal,
        };

        let attachment_r8_clear = hal::pass::Attachment {
            format: Some(hal::format::Format::R8Unorm),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::ColorAttachmentOptimal
                .. Layout::ColorAttachmentOptimal,
        };

        let attachment_bgra8 = hal::pass::Attachment {
            format: Some(surface_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::ColorAttachmentOptimal
                .. Layout::ColorAttachmentOptimal,
        };

        let attachment_bgra8_clear = hal::pass::Attachment {
            format: Some(surface_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::ColorAttachmentOptimal
                .. Layout::ColorAttachmentOptimal,
        };

        let attachment_rgbaf32 = hal::pass::Attachment {
            format: Some(hal::format::Format::Rgba32Sfloat),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::ColorAttachmentOptimal
                .. Layout::ColorAttachmentOptimal,
        };

        let attachment_depth = hal::pass::Attachment {
            format: Some(depth_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::DepthStencilAttachmentOptimal
                .. Layout::DepthStencilAttachmentOptimal,
        };

        let attachment_depth_clear = hal::pass::Attachment {
            format: Some(depth_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::DepthStencilAttachmentOptimal
                .. Layout::DepthStencilAttachmentOptimal,
        };

        let attachment_cao_to_present = hal::pass::Attachment {
            format: Some(surface_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::ColorAttachmentOptimal
                .. Layout::Present,
        };

        let attachment_undef_to_cao = hal::pass::Attachment {
            format: Some(surface_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined
                .. Layout::ColorAttachmentOptimal,
        };

        let attachment_undef_to_present = hal::pass::Attachment {
            format: Some(surface_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined
                .. Layout::Present,
        };

        let subpass_r8 = hal::pass::SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let subpass_depth_r8 = hal::pass::SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let subpass_bgra8 = hal::pass::SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let subpass_depth_bgra8 = hal::pass::SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let subpass_rgbaf32 = hal::pass::SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let subpass_depth_rgbaf32 = hal::pass::SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        use std::iter;
        HalRenderPasses {
            r8: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_r8),
                    &[subpass_r8.clone()],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            r8_clear: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_r8_clear),
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
            bgra8: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_bgra8),
                    &[subpass_bgra8.clone()],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            bgra8_clear: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_bgra8_clear),
                    &[subpass_bgra8.clone()],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            bgra8_depth_clear: unsafe {
                device.create_render_pass(
                    &[attachment_bgra8_clear, attachment_depth_clear.clone()],
                    &[subpass_depth_bgra8.clone()],
                    &[],
                )
            }
            .expect("create_render_pass failed"),

            bgra8_depth: unsafe {
                device.create_render_pass(
                    &[attachment_bgra8, attachment_depth.clone()],
                    &[subpass_depth_bgra8.clone()],
                    &[],
                )
            }
            .expect("create_render_pass failed"),

            // main target passes
            cao_to_present: unsafe {
                device.create_render_pass(
                    &[attachment_cao_to_present, attachment_depth],
                    &[subpass_depth_bgra8.clone()],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            undef_to_cao: unsafe {
                device.create_render_pass(
                    &[attachment_undef_to_cao, attachment_depth_clear.clone()],
                    &[subpass_depth_bgra8.clone()],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
            undef_to_present: unsafe {
                device.create_render_pass(
                    &[attachment_undef_to_present, attachment_depth_clear],
                    &[subpass_depth_bgra8],
                    &[],
                )
            }
            .expect("create_render_pass failed"),
        }
    }
}
