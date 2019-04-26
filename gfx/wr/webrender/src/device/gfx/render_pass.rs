/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ImageFormat;
use hal::Device;

pub(super) struct RenderPass<B: hal::Backend> {
    pub(super) r8: B::RenderPass,
    pub(super) r8_depth: B::RenderPass,
    pub(super) bgra8: B::RenderPass,
    pub(super) bgra8_depth: B::RenderPass,
}

impl<B: hal::Backend> RenderPass<B> {
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
            f => unimplemented!("No render pass for image format {:?}", f),
        }
    }

    pub(super) fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_render_pass(self.r8);
            device.destroy_render_pass(self.r8_depth);
            device.destroy_render_pass(self.bgra8);
            device.destroy_render_pass(self.bgra8_depth);
        }
    }
}
