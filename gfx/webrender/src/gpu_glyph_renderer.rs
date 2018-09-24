/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! GPU glyph rasterization using Pathfinder.

use api::{DeviceIntPoint, DeviceIntRect, DeviceUintSize, FontRenderMode};
use api::{ImageFormat, TextureTarget};
use debug_colors;
use device::{self, Device, PrimitiveType, ShaderKind, Texture};
use device::{TextureFilter, TextureSampler, VAO, VertexArrayKind};
use euclid::{Point2D, Size2D, Transform3D, TypedVector2D, Vector2D};
use hal;
use internal_types::RenderTargetInfo;
use pathfinder_gfx_utils::ShelfBinPacker;
use profiler::GpuProfileTag;
use renderer::{self, ImageBufferKind, Renderer, RendererError, RendererStats};
use shade::LazilyCompiledShader;
use tiling::GlyphJob;
#[cfg(not(feature = "gleam"))]
use vertex_types::{VectorStencilInstance, VectorCoverInstance};

// The area lookup table in uncompressed grayscale TGA format (TGA image format 3).
static AREA_LUT_TGA_BYTES: &'static [u8] = include_bytes!("../res/area-lut.tga");

const HORIZONTAL_BIN_PADDING: i32 = 3;

const GPU_TAG_GLYPH_STENCIL: GpuProfileTag = GpuProfileTag {
    label: "Glyph Stencil",
    color: debug_colors::STEELBLUE,
};
const GPU_TAG_GLYPH_COVER: GpuProfileTag = GpuProfileTag {
    label: "Glyph Cover",
    color: debug_colors::LIGHTSTEELBLUE,
};

pub struct GpuGlyphRenderer<B: hal::Backend> {
    pub area_lut_texture: Texture,
    pub vector_stencil_vao: VAO,
    pub vector_cover_vao: VAO,

    // These are Pathfinder shaders, used for rendering vector graphics.
    vector_stencil: LazilyCompiledShader<B>,
    vector_cover: LazilyCompiledShader<B>,
}

impl<B: hal::Backend> GpuGlyphRenderer<B> {
    #[cfg(not(feature = "gleam"))]
    pub fn new(_device: &mut Device<B>, _prim_vao: &VAO, _precache_shaders: bool)
               -> Result<GpuGlyphRenderer<B>, RendererError> {
        unimplemented!();
    }

    #[cfg(feature = "gleam")]
    pub fn new(device: &mut Device<B>, prim_vao: &VAO, precache_shaders: bool)
               -> Result<GpuGlyphRenderer<B>, RendererError> {
        // Make sure the area LUT is uncompressed grayscale TGA, 8bpp.
        debug_assert!(AREA_LUT_TGA_BYTES[2] == 3);
        debug_assert!(AREA_LUT_TGA_BYTES[16] == 8);
        let area_lut_width = (AREA_LUT_TGA_BYTES[12] as u32) |
            ((AREA_LUT_TGA_BYTES[13] as u32) << 8);
        let area_lut_height = (AREA_LUT_TGA_BYTES[14] as u32) |
            ((AREA_LUT_TGA_BYTES[15] as u32) << 8);
        let area_lut_pixels =
            &AREA_LUT_TGA_BYTES[18..(18 + area_lut_width * area_lut_height) as usize];

        let mut area_lut_texture = device.create_texture(TextureTarget::Default, ImageFormat::R8);
        device.init_texture(&mut area_lut_texture,
                            area_lut_width,
                            area_lut_height,
                            TextureFilter::Linear,
                            None,
                            1,
                            Some(area_lut_pixels));

        let vector_stencil_vao =
            device.create_vao_with_new_instances(&device::desc::VECTOR_STENCIL, prim_vao);
        let vector_cover_vao = device.create_vao_with_new_instances(&device::desc::VECTOR_COVER,
                                                                    prim_vao);

        // Load Pathfinder vector graphics shaders.
        let vector_stencil = try!{
            LazilyCompiledShader::new(ShaderKind::VectorStencil,
                                      "pf_vector_stencil",
                                      &[ImageBufferKind::Texture2D.get_feature_string()],
                                      device,
                                      precache_shaders)
        };
        let vector_cover = try!{
            LazilyCompiledShader::new(ShaderKind::VectorCover,
                                      "pf_vector_cover",
                                      &[ImageBufferKind::Texture2D.get_feature_string()],
                                      device,
                                      precache_shaders)
        };

        Ok(GpuGlyphRenderer {
            area_lut_texture,
            vector_stencil_vao,
            vector_cover_vao,
            vector_stencil,
            vector_cover,
        })
    }
}

impl<B: hal::Backend> Renderer<B> {
    /// Renders glyphs using the vector graphics shaders (Pathfinder).
    pub fn stencil_glyphs(&mut self,
                          glyphs: &[GlyphJob],
                          projection: &Transform3D<f32>,
                          target_size: &DeviceUintSize,
                          stats: &mut RendererStats)
                          -> Option<StenciledGlyphPage> {
        if glyphs.is_empty() {
            return None
        }

        let _timer = self.gpu_profile.start_timer(GPU_TAG_GLYPH_STENCIL);

        // Initialize temporary framebuffer.
        // FIXME(pcwalton): Cache this!
        // FIXME(pcwalton): Use RF32, not RGBAF32!
        let mut current_page = StenciledGlyphPage {
            texture: self.device.create_texture(TextureTarget::Default, ImageFormat::RGBAF32),
            glyphs: vec![],
        };
        self.device.init_texture::<f32>(&mut current_page.texture,
                                        target_size.width,
                                        target_size.height,
                                        TextureFilter::Nearest,
                                        Some(RenderTargetInfo {
                                            has_depth: false,
                                        }),
                                        1,
                                        None);

        // Allocate all target rects.
        let mut packer = ShelfBinPacker::new(&target_size.to_i32().to_untyped(),
                                             &Vector2D::new(HORIZONTAL_BIN_PADDING, 0));
        let mut glyph_indices: Vec<_> = (0..(glyphs.len())).collect();
        glyph_indices.sort_by(|&a, &b| {
            glyphs[b].target_rect.size.height.cmp(&glyphs[a].target_rect.size.height)
        });
        for &glyph_index in &glyph_indices {
            let glyph = &glyphs[glyph_index];
            let x_scale = x_scale_for_render_mode(glyph.render_mode);
            let stencil_size = Size2D::new(glyph.target_rect.size.width * x_scale,
                                           glyph.target_rect.size.height);
            match packer.add(&stencil_size) {
                Err(_) => return None,
                Ok(origin) => {
                    current_page.glyphs.push(VectorCoverInstanceAttrs {
                        target_rect: glyph.target_rect,
                        stencil_origin: DeviceIntPoint::from_untyped(&origin),
                        subpixel: (glyph.render_mode == FontRenderMode::Subpixel) as u16,
                    })
                }
            }
        }

        // Initialize path info.
        // TODO(pcwalton): Cache this texture!
        let mut path_info_texture = self.device.create_texture(TextureTarget::Default,
                                                               ImageFormat::RGBAF32);

        let mut path_info_texels = Vec::with_capacity(glyphs.len() * 12);
        for (stenciled_glyph_index, &glyph_index) in glyph_indices.iter().enumerate() {
            let glyph = &glyphs[glyph_index];
            let stenciled_glyph = &current_page.glyphs[stenciled_glyph_index];
            let x_scale = x_scale_for_render_mode(glyph.render_mode) as f32;
            let glyph_origin = TypedVector2D::new(-glyph.origin.x as f32 * x_scale,
                                                  -glyph.origin.y as f32);
            let subpixel_offset = TypedVector2D::new(glyph.subpixel_offset.x * x_scale,
                                                     glyph.subpixel_offset.y);
            let rect = stenciled_glyph.stencil_rect()
                                      .to_f32()
                                      .translate(&glyph_origin)
                                      .translate(&subpixel_offset);
            path_info_texels.extend_from_slice(&[
                x_scale, 0.0, 0.0, -1.0,
                rect.origin.x, rect.max_y(), 0.0, 0.0,
                rect.size.width, rect.size.height,
                glyph.embolden_amount.x,
                glyph.embolden_amount.y,
            ]);
        }

        self.device.init_texture(&mut path_info_texture,
                                 3,
                                 glyphs.len() as u32,
                                 TextureFilter::Nearest,
                                 None,
                                 1,
                                 Some(&path_info_texels));

        self.gpu_glyph_renderer.vector_stencil.bind(&mut self.device,
                                                    projection,
                                                    &mut self.renderer_errors);

        self.device.bind_draw_target(Some((&current_page.texture, 0)), Some(*target_size));
        self.device.clear_target(Some([0.0, 0.0, 0.0, 0.0]), None, None);

        self.device.set_blend(true);
        self.device.set_blend_mode_subpixel_pass1();

        let mut instance_data = vec![];
        for (path_id, &glyph_id) in glyph_indices.iter().enumerate() {
            let glyph = &glyphs[glyph_id];
            instance_data.extend(glyph.mesh
                                      .stencil_segments
                                      .iter()
                                      .zip(glyph.mesh.stencil_normals.iter())
                                      .map(|(segment, normals)| {
                VectorStencilInstanceAttrs {
                    from_position: segment.from,
                    ctrl_position: segment.ctrl,
                    to_position: segment.to,
                    from_normal: normals.from,
                    ctrl_normal: normals.ctrl,
                    to_normal: normals.to,
                    path_id: path_id as u16,
                }
            }));
        }

        self.device.bind_texture(TextureSampler::color(0),
                                 &self.gpu_glyph_renderer.area_lut_texture);
        self.device.bind_texture(TextureSampler::color(1), &path_info_texture);
        self.draw_instanced_batch_with_previously_bound_textures(&instance_data,
                                                                 VertexArrayKind::VectorStencil,
                                                                 stats);

        self.device.delete_texture(path_info_texture);

        Some(current_page)
    }

    /// Blits glyphs from the stencil texture to the texture cache.
    ///
    /// Deletes the stencil texture at the end.
    /// FIXME(pcwalton): This is bad. Cache it somehow.
    pub fn cover_glyphs(&mut self,
                        stencil_page: StenciledGlyphPage,
                        projection: &Transform3D<f32>,
                        stats: &mut RendererStats) {
        debug_assert!(!stencil_page.glyphs.is_empty());

        let _timer = self.gpu_profile.start_timer(GPU_TAG_GLYPH_COVER);

        self.gpu_glyph_renderer.vector_cover.bind(&mut self.device,
                                                  projection,
                                                  &mut self.renderer_errors);

        self.device.bind_texture(TextureSampler::color(0), &stencil_page.texture);
        self.draw_instanced_batch_with_previously_bound_textures(&stencil_page.glyphs,
                                                                 VertexArrayKind::VectorCover,
                                                                 stats);

        self.device.delete_texture(stencil_page.texture);
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct VectorStencilInstanceAttrs {
    from_position: Point2D<f32>,
    ctrl_position: Point2D<f32>,
    to_position: Point2D<f32>,
    from_normal: Vector2D<f32>,
    ctrl_normal: Vector2D<f32>,
    to_normal: Vector2D<f32>,
    path_id: u16,
}

#[cfg(not(feature = "gleam"))]
impl PrimitiveType for VectorStencilInstanceAttrs {
    type Primitive = VectorStencilInstance;
    fn to_primitive_type(&self) -> VectorStencilInstance {
        VectorStencilInstance {
            aFromPosition: [self.from_position.x, self.from_position.y],
            aCtrlPosition: [self.ctrl_position.x, self.ctrl_position.y],
            aToPosition: [self.to_position.x, self.to_position.y],
            aFromNormal: [self.from_normal.x, self.from_normal.y],
            aCtrlNormal: [self.ctrl_normal.x, self.ctrl_normal.y],
            aToNormal: [self.to_normal.x, self.to_normal.y],
            aPathID: self.path_id as _,
            aPad: 0,
        }
    }
}

#[cfg(feature = "gleam")]
impl PrimitiveType for VectorStencilInstanceAttrs { }

pub struct StenciledGlyphPage {
    texture: Texture,
    glyphs: Vec<VectorCoverInstanceAttrs>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct VectorCoverInstanceAttrs {
    target_rect: DeviceIntRect,
    stencil_origin: DeviceIntPoint,
    subpixel: u16,
}

#[cfg(not(feature = "gleam"))]
impl PrimitiveType for VectorCoverInstanceAttrs {
    type Primitive = VectorCoverInstance;
    fn to_primitive_type(&self) -> VectorCoverInstance {
        VectorCoverInstance {
            aTargetRect: [
                self.target_rect.origin.x,
                self.target_rect.origin.y,
                self.target_rect.size.width,
                self.target_rect.size.height,
            ],
            aStencilOrigin: [self.stencil_origin.x, self.stencil_origin.y],
            aSubpixel: self.subpixel as _,
            aPad: 0,
        }
    }
}

#[cfg(feature = "gleam")]
impl PrimitiveType for VectorCoverInstanceAttrs { }

impl VectorCoverInstanceAttrs {
    fn stencil_rect(&self) -> DeviceIntRect {
        DeviceIntRect::new(self.stencil_origin, self.target_rect.size)
    }
}

fn x_scale_for_render_mode(render_mode: FontRenderMode) -> i32 {
    match render_mode {
        FontRenderMode::Subpixel => 3,
        FontRenderMode::Mono | FontRenderMode::Alpha => 1,
    }
}
