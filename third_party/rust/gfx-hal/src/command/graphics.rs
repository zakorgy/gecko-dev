//! `CommandBuffer` methods for graphics operations.
use std::borrow::Borrow;
use std::ops::Range;

use super::{
    ClearColorRaw,
    ClearDepthStencilRaw,
    ClearValueRaw,
    CommandBuffer,
    DescriptorSetOffset,
    Level,
    Primary,
    RawCommandBuffer,
    RenderPassInlineEncoder,
    RenderPassSecondaryEncoder,
    Shot,
};
use crate::queue::capability::{Graphics, GraphicsOrCompute, Supports};
use crate::Backend;
use crate::{buffer, image, pso, query};

/// A universal clear color supporting integer formats
/// as well as the standard floating-point.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClearColor {
    /// Standard floating-point `vec4` color
    Sfloat(pso::ColorValue),
    /// Integer vector to clear `ivec4` targets.
    Sint([i32; 4]),
    /// Unsigned int vector to clear `uvec4` targets.
    Uint([u32; 4]),
}

macro_rules! impl_clear {
    { $( $ty:ty = $sub:ident[$a:expr, $b:expr, $c:expr, $d:expr], )* } => {
        $(
            impl From<$ty> for ClearColor {
                fn from(v: $ty) -> Self {
                    ClearColor::$sub([v[$a], v[$b], v[$c], v[$d]])
                }
            }
        )*
    }
}

impl_clear! {
    [f32; 4] = Sfloat[0, 1, 2, 3],
    [f32; 3] = Sfloat[0, 1, 2, 0],
    [f32; 2] = Sfloat[0, 1, 0, 0],
    [i32; 4] = Sint  [0, 1, 2, 3],
    [i32; 3] = Sint  [0, 1, 2, 0],
    [i32; 2] = Sint  [0, 1, 0, 0],
    [u32; 4] = Uint [0, 1, 2, 3],
    [u32; 3] = Uint [0, 1, 2, 0],
    [u32; 2] = Uint [0, 1, 0, 0],
}

impl From<f32> for ClearColor {
    fn from(v: f32) -> Self {
        ClearColor::Sfloat([v, 0.0, 0.0, 0.0])
    }
}
impl From<i32> for ClearColor {
    fn from(v: i32) -> Self {
        ClearColor::Sint([v, 0, 0, 0])
    }
}
impl From<u32> for ClearColor {
    fn from(v: u32) -> Self {
        ClearColor::Uint([v, 0, 0, 0])
    }
}

impl From<ClearColor> for ClearColorRaw {
    fn from(cv: ClearColor) -> Self {
        match cv {
            ClearColor::Sfloat(cv) => ClearColorRaw { float32: cv },
            ClearColor::Sint(cv) => ClearColorRaw { int32: cv },
            ClearColor::Uint(cv) => ClearColorRaw { uint32: cv },
        }
    }
}

/// Depth-stencil target clear values.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClearDepthStencil(pub pso::DepthValue, pub pso::StencilValue);

impl From<ClearDepthStencil> for ClearDepthStencilRaw {
    fn from(value: ClearDepthStencil) -> Self {
        ClearDepthStencilRaw {
            depth: value.0,
            stencil: value.1,
        }
    }
}

/// General clear values for attachments (color or depth-stencil).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClearValue {
    ///
    Color(ClearColor),
    ///
    DepthStencil(ClearDepthStencil),
}

impl From<ClearValue> for ClearValueRaw {
    fn from(value: ClearValue) -> Self {
        match value {
            ClearValue::Color(color) => ClearValueRaw {
                color: color.into(),
            },
            ClearValue::DepthStencil(ds) => ClearValueRaw {
                depth_stencil: ds.into(),
            },
        }
    }
}

/// Attachment clear description for the current subpass.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AttachmentClear {
    /// Clear color attachment.
    Color {
        /// Index inside the `SubpassDesc::colors` array.
        index: usize,
        /// Value to clear with.
        value: ClearColor,
    },
    /// Clear depth-stencil attachment.
    DepthStencil {
        /// Depth value to clear with.
        depth: Option<pso::DepthValue>,
        /// Stencil value to clear with.
        stencil: Option<pso::StencilValue>,
    },
}

/// Parameters for an image resolve operation,
/// where a multi-sampled image is copied into a single-sampled
/// image.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageResolve {
    /// Source image and layers.
    pub src_subresource: image::SubresourceLayers,
    /// Source image offset.
    pub src_offset: image::Offset,
    /// Destination image and layers.
    pub dst_subresource: image::SubresourceLayers,
    /// Destination image offset.
    pub dst_offset: image::Offset,
    /// Image extent.
    pub extent: image::Extent,
}

/// Parameters for an image blit operation, where a portion of one image
/// is copied into another, possibly with scaling and filtering.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageBlit {
    /// Source image and layers.
    pub src_subresource: image::SubresourceLayers,
    /// Source image bounds.
    pub src_bounds: Range<image::Offset>,
    /// Destination image and layers.
    pub dst_subresource: image::SubresourceLayers,
    /// Destination image bounds.
    pub dst_bounds: Range<image::Offset>,
}

impl<B: Backend, C: Supports<Graphics>, S: Shot, L: Level> CommandBuffer<B, C, S, L> {
    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn clear_image<T>(
        &mut self,
        image: &B::Image,
        layout: image::Layout,
        color: ClearColor,
        depth_stencil: ClearDepthStencil,
        subresource_ranges: T,
    ) where
        T: IntoIterator,
        T::Item: Borrow<image::SubresourceRange>,
    {
        self.raw.clear_image(
            image,
            layout,
            color.into(),
            depth_stencil.into(),
            subresource_ranges,
        )
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn bind_index_buffer(&mut self, ibv: buffer::IndexBufferView<B>) {
        self.raw.bind_index_buffer(ibv)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn bind_vertex_buffers<I, T>(&mut self, first_binding: pso::BufferIndex, buffers: I)
    where
        I: IntoIterator<Item = (T, buffer::Offset)>,
        T: Borrow<B::Buffer>,
    {
        self.raw.bind_vertex_buffers(first_binding, buffers)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn bind_graphics_pipeline(&mut self, pipeline: &B::GraphicsPipeline) {
        self.raw.bind_graphics_pipeline(pipeline)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn bind_graphics_descriptor_sets<I, J>(
        &mut self,
        layout: &B::PipelineLayout,
        first_set: usize,
        sets: I,
        offsets: J,
    ) where
        I: IntoIterator,
        I::Item: Borrow<B::DescriptorSet>,
        J: IntoIterator,
        J::Item: Borrow<DescriptorSetOffset>,
    {
        self.raw
            .bind_graphics_descriptor_sets(layout, first_set, sets, offsets)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_viewports<T>(&mut self, first_viewport: u32, viewports: T)
    where
        T: IntoIterator,
        T::Item: Borrow<pso::Viewport>,
    {
        self.raw.set_viewports(first_viewport, viewports)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_scissors<T>(&mut self, first_scissor: u32, scissors: T)
    where
        T: IntoIterator,
        T::Item: Borrow<pso::Rect>,
    {
        self.raw.set_scissors(first_scissor, scissors)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_stencil_reference(&mut self, faces: pso::Face, value: pso::StencilValue) {
        self.raw.set_stencil_reference(faces, value);
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_stencil_read_mask(&mut self, faces: pso::Face, value: pso::StencilValue) {
        self.raw.set_stencil_read_mask(faces, value);
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_stencil_write_mask(&mut self, faces: pso::Face, value: pso::StencilValue) {
        self.raw.set_stencil_write_mask(faces, value);
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_blend_constants(&mut self, cv: pso::ColorValue) {
        self.raw.set_blend_constants(cv)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_depth_bounds(&mut self, bounds: Range<f32>) {
        self.raw.set_depth_bounds(bounds)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_line_width(&mut self, width: f32) {
        self.raw.set_line_width(width);
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn set_depth_bias(&mut self, depth_bias: pso::DepthBias) {
        self.raw.set_depth_bias(depth_bias);
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn push_graphics_constants(
        &mut self,
        layout: &B::PipelineLayout,
        stages: pso::ShaderStageFlags,
        offset: u32,
        constants: &[u32],
    ) {
        self.raw
            .push_graphics_constants(layout, stages, offset, constants)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn resolve_image<T>(
        &mut self,
        src: &B::Image,
        src_layout: image::Layout,
        dst: &B::Image,
        dst_layout: image::Layout,
        regions: T,
    ) where
        T: IntoIterator,
        T::Item: Borrow<ImageResolve>,
    {
        self.raw
            .resolve_image(src, src_layout, dst, dst_layout, regions)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn blit_image<T>(
        &mut self,
        src: &B::Image,
        src_layout: image::Layout,
        dst: &B::Image,
        dst_layout: image::Layout,
        filter: image::Filter,
        regions: T,
    ) where
        T: IntoIterator,
        T::Item: Borrow<ImageBlit>,
    {
        self.raw
            .blit_image(src, src_layout, dst, dst_layout, filter, regions)
    }
}

impl<B: Backend, C: Supports<Graphics>, S: Shot> CommandBuffer<B, C, S, Primary> {
    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn begin_render_pass_inline<T>(
        &mut self,
        render_pass: &B::RenderPass,
        frame_buffer: &B::Framebuffer,
        render_area: pso::Rect,
        clear_values: T,
    ) -> RenderPassInlineEncoder<B>
    where
        T: IntoIterator,
        T::Item: Borrow<ClearValue>,
    {
        RenderPassInlineEncoder::new(self, render_pass, frame_buffer, render_area, clear_values)
    }

    /// Creates a new secondary render pass.
    pub unsafe fn begin_render_pass_secondary<T>(
        &mut self,
        render_pass: &B::RenderPass,
        frame_buffer: &B::Framebuffer,
        render_area: pso::Rect,
        clear_values: T,
    ) -> RenderPassSecondaryEncoder<B>
    where
        T: IntoIterator,
        T::Item: Borrow<ClearValue>,
    {
        RenderPassSecondaryEncoder::new(self, render_pass, frame_buffer, render_area, clear_values)
    }
}

impl<B: Backend, C: Supports<GraphicsOrCompute>, S: Shot, L: Level> CommandBuffer<B, C, S, L> {
    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn begin_query(&mut self, query: query::Query<B>, flags: query::ControlFlags) {
        self.raw.begin_query(query, flags)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn end_query(&mut self, query: query::Query<B>) {
        self.raw.end_query(query)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn reset_query_pool(&mut self, pool: &B::QueryPool, queries: Range<query::Id>) {
        self.raw.reset_query_pool(pool, queries)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn copy_query_pool_results(
        &mut self,
        pool: &B::QueryPool,
        queries: Range<query::Id>,
        buffer: &B::Buffer,
        offset: buffer::Offset,
        stride: buffer::Offset,
        flags: query::ResultFlags,
    ) {
        self.raw
            .copy_query_pool_results(pool, queries, buffer, offset, stride, flags)
    }

    /// Identical to the `RawCommandBuffer` method of the same name.
    pub unsafe fn write_timestamp(&mut self, stage: pso::PipelineStage, query: query::Query<B>) {
        self.raw.write_timestamp(stage, query)
    }
}
