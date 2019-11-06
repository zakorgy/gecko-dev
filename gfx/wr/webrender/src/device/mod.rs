/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::shader_source::SHADERS;
use api::{ImageFormat, TextureTarget};
use api::VoidPtrToSizeFn;
use api::units::{DeviceIntSize, DeviceIntPoint, DeviceIntRect};
use api::units::{FramebufferIntSize, FramebufferIntPoint, FramebufferIntRect};
use euclid::Transform3D;
#[cfg(feature = "gl")]
use gleam::gl as gleam_gl;
use crate::internal_types::{FastHashMap, LayerIndex, Swizzle, ORTHO_FAR_PLANE, ORTHO_NEAR_PLANE};
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::ops::Add;
use std::os::raw::c_void;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use webrender_build::shader::{parse_shader_source, shader_source_from_file};
use webrender_build::shader::ProgramSourceDigest;

cfg_if! {
    if #[cfg(feature = "gl")] {
        mod gl;
        pub mod query_gl;

        pub use self::gl::*;
        pub use self::query_gl as query;
    } else {
        mod gfx;
        pub mod query_gfx;

        pub use self::gfx::*;
        pub use self::gfx::PrimitiveType;
        pub use self::query_gfx as query;
    }
}

/// Sequence number for frames, as tracked by the device layer.
#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GpuFrameId(usize);

/// Tracks the total number of GPU bytes allocated across all WebRender instances.
///
/// Assuming all WebRender instances run on the same thread, this doesn't need
/// to be atomic per se, but we make it atomic to satisfy the thread safety
/// invariants in the type system. We could also put the value in TLS, but that
/// would be more expensive to access.
static GPU_BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

/// Returns the number of GPU bytes currently allocated.
pub fn total_gpu_bytes_allocated() -> usize {
    GPU_BYTES_ALLOCATED.load(Ordering::Relaxed)
}

/// Records an allocation in VRAM.
fn record_gpu_alloc(num_bytes: usize) {
    GPU_BYTES_ALLOCATED.fetch_add(num_bytes, Ordering::Relaxed);
}

/// Records an deallocation in VRAM.
fn record_gpu_free(num_bytes: usize) {
    let old = GPU_BYTES_ALLOCATED.fetch_sub(num_bytes, Ordering::Relaxed);
    assert!(old >= num_bytes, "Freeing {} bytes but only {} allocated", num_bytes, old);
}

impl GpuFrameId {
    pub fn new(value: usize) -> Self {
        GpuFrameId(value)
    }
}

impl Add<usize> for GpuFrameId {
    type Output = GpuFrameId;

    fn add(self, other: usize) -> GpuFrameId {
        GpuFrameId(self.0 + other)
    }
}

const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";

#[cfg(not(feature = "gl"))]
pub type IdType = u32;

#[cfg(feature = "gl")]
pub type IdType = gleam_gl::GLuint;

pub struct TextureSlot(pub usize);

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureFilter {
    Nearest,
    Linear,
    Trilinear,
}

#[cfg(not(feature = "gl"))]
impl From<TextureFilter> for crate::hal::image::Filter {
    fn from(filter: TextureFilter) -> crate::hal::image::Filter {
        match filter {
            TextureFilter::Nearest => crate::hal::image::Filter::Nearest,
            _ => crate::hal::image::Filter::Linear,
        }
    }
}

#[cfg(not(feature = "gl"))]
impl Default for TextureFilter {
    fn default() -> Self {
        TextureFilter::Nearest
    }
}

#[derive(Debug)]
pub enum VertexAttributeKind {
    F32,
    U8Norm,
    U16Norm,
    I32,
    U16,
}

#[derive(Debug)]
pub struct VertexAttribute {
    pub name: &'static str,
    pub count: u32,
    pub kind: VertexAttributeKind,
}

#[derive(Debug)]
pub struct VertexDescriptor {
    pub vertex_attributes: &'static [VertexAttribute],
    pub instance_attributes: &'static [VertexAttribute],
}

/// Method of uploading texel data from CPU to GPU.
#[derive(Debug, Clone)]
pub enum UploadMethod {
    /// Just call `glTexSubImage` directly with the CPU data pointer
    Immediate,
    /// Accumulate the changes in PBO first before transferring to a texture.
    PixelBuffer(VertexUsageHint),
}

/// Plain old data that can be used to initialize a texture.
pub unsafe trait Texel: Copy {}
unsafe impl Texel for u8 {}
unsafe impl Texel for f32 {}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct RBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(IdType);

pub struct PBO {
    id: IdType,
    reserved_size: usize,
}

impl PBO {
    pub fn get_reserved_size(&self) -> usize {
        self.reserved_size
    }
}

impl Drop for PBO {
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking() || self.id == 0,
            "renderer::deinit not called"
        );
    }
}

pub struct BoundPBO<'a, B: hal::Backend> {
    #[cfg(feature = "gl")]
    device: &'a mut Device<B>,
    pub data: &'a [u8],
    #[cfg(not(feature = "gl"))]
    phantom_data: std::marker::PhantomData<B>,
}

/// Returns the size in bytes of a depth target with the given dimensions.
fn depth_target_size_in_bytes(dimensions: &DeviceIntSize) -> usize {
    // DEPTH24 textures generally reserve 3 bytes for depth and 1 byte
    // for stencil, so we measure them as 32 bits.
    let pixels = dimensions.width * dimensions.height;
    (pixels as usize) * 4
}

// Get a shader string by name, from the built in resources or
// an override path, if supplied.
fn get_shader_source(shader_name: &str, base_path: Option<&PathBuf>) -> Cow<'static, str> {
    if let Some(ref base) = base_path {
        let shader_path = base.join(&format!("{}.glsl", shader_name));
        Cow::Owned(shader_source_from_file(&shader_path))
    } else {
        Cow::Borrowed(
            SHADERS
            .get(shader_name)
            .expect("Shader not found")
            .source
        )
    }
}

/// Creates heap-allocated strings for both vertex and fragment shaders. Public
/// to be accessible to tests.
pub fn build_shader_strings(
     gl_version_string: &str,
     features: &str,
     base_filename: &str,
     override_path: Option<&PathBuf>,
) -> (String, String) {
    let mut vs_source = String::new();
    do_build_shader_string(
        gl_version_string,
        features,
        SHADER_KIND_VERTEX,
        base_filename,
        override_path,
        |s| vs_source.push_str(s),
    );

    let mut fs_source = String::new();
    do_build_shader_string(
        gl_version_string,
        features,
        SHADER_KIND_FRAGMENT,
        base_filename,
        override_path,
        |s| fs_source.push_str(s),
    );

    (vs_source, fs_source)
}

/// Walks the given shader string and applies the output to the provided
/// callback. Assuming an override path is not used, does no heap allocation
/// and no I/O.
fn do_build_shader_string<F: FnMut(&str)>(
    gl_version_string: &str,
    features: &str,
    kind: &str,
    base_filename: &str,
    override_path: Option<&PathBuf>,
    mut output: F,
) {
    build_shader_prefix_string(gl_version_string, features, kind, base_filename, &mut output);
    build_shader_main_string(base_filename, override_path, &mut output);
}

/// Walks the prefix section of the shader string, which manages the various
/// defines for features etc.
fn build_shader_prefix_string<F: FnMut(&str)>(
    gl_version_string: &str,
    features: &str,
    kind: &str,
    base_filename: &str,
    output: &mut F,
) {
    // GLSL requires that the version number comes first.
    output(gl_version_string);

    // Insert the shader name to make debugging easier.
    let name_string = format!("// {}\n", base_filename);
    output(&name_string);

    // Define a constant depending on whether we are compiling VS or FS.
    output(kind);

    // Add any defines that were passed by the caller.
    output(features);
}

/// Walks the main .glsl file, including any imports.
fn build_shader_main_string<F: FnMut(&str)>(
    base_filename: &str,
    override_path: Option<&PathBuf>,
    output: &mut F,
) {
    let shared_source = get_shader_source(base_filename, override_path);
    parse_shader_source(shared_source, &|f| get_shader_source(f, override_path), output);
}

pub trait FileWatcherHandler: Send {
    fn file_changed(&self, path: PathBuf);
}

#[cfg_attr(feature = "replay", derive(Clone))]
pub struct ExternalTexture {
    id: IdType,
    #[cfg_attr(not(feature = "gl"), allow(dead_code))]
    target: IdType,
    _swizzle: Swizzle,
}


impl ExternalTexture {
    pub fn new(id: u32, target: TextureTarget, _swizzle: Swizzle) -> Self {
        ExternalTexture {
            id,
            #[cfg(feature = "gl")]
            target: get_gl_target(target),
            #[cfg(not(feature = "gl"))]
            target: target as _,
            _swizzle,
        }
    }

    #[cfg(feature = "replay")]
    pub fn internal_id(&self) -> IdType {
        self.id
    }
}

bitflags! {
    #[derive(Default)]
    pub struct TextureFlags: u32 {
        /// This texture corresponds to one of the shared texture caches.
        const IS_SHARED_TEXTURE_CACHE = 1 << 0;
    }
}

/// WebRender interface to an OpenGL texture.
///
/// Because freeing a texture requires various device handles that are not
/// reachable from this struct, manual destruction via `Device` is required.
/// Our `Drop` implementation asserts that this has happened.
#[cfg_attr(not(feature = "gl"), derive(Clone))]
pub struct Texture {
    id: IdType,
    target: IdType,
    layer_count: i32,
    format: ImageFormat,
    size: DeviceIntSize,
    filter: TextureFilter,
    flags: TextureFlags,
    /// An internally mutable swizzling state that may change between batches.
    active_swizzle: Cell<Swizzle>,
    /// Framebuffer Objects, one for each layer of the texture, allowing this
    /// texture to be rendered to. Empty if this texture is not used as a render
    /// target.
    fbos: Vec<FBOId>,
    /// Same as the above, but with a depth buffer attached.
    ///
    /// FBOs are cheap to create but expensive to reconfigure (since doing so
    /// invalidates framebuffer completeness caching). Moreover, rendering with
    /// a depth buffer attached but the depth write+test disabled relies on the
    /// driver to optimize it out of the rendering pass, which most drivers
    /// probably do but, according to jgilbert, is best not to rely on.
    ///
    /// So we lazily generate a second list of FBOs with depth. This list is
    /// empty if this texture is not used as a render target _or_ if it is, but
    /// the depth buffer has never been requested.
    ///
    /// Note that we always fill fbos, and then lazily create fbos_with_depth
    /// when needed. We could make both lazy (i.e. render targets would have one
    /// or the other, but not both, unless they were actually used in both
    /// configurations). But that would complicate a lot of logic in this module,
    /// and FBOs are cheap enough to create.
    fbos_with_depth: Vec<FBOId>,
    /// If we are unable to blit directly to a texture array then we need
    /// an intermediate renderbuffer.
    blit_workaround_buffer: Option<(RBOId, FBOId)>,
    last_frame_used: GpuFrameId,

    #[cfg(not(feature = "gl"))]
    pub bound_in_frame: Cell<GpuFrameId>,
    #[cfg(not(feature = "gl"))]
    is_buffer: bool,
}

impl Texture {
    pub fn id(&self) -> IdType {
        self.id
    }

    pub fn get_dimensions(&self) -> DeviceIntSize {
        self.size
    }

    pub fn get_layer_count(&self) -> i32 {
        self.layer_count
    }

    pub fn get_format(&self) -> ImageFormat {
        self.format
    }

    pub fn get_filter(&self) -> TextureFilter {
        self.filter
    }

    pub fn supports_depth(&self) -> bool {
        !self.fbos_with_depth.is_empty()
    }

    pub fn used_in_frame(&self, frame_id: GpuFrameId) -> bool {
        self.last_frame_used == frame_id
    }

    /// Returns true if this texture was used within `threshold` frames of
    /// the current frame.
    pub fn used_recently(&self, current_frame_id: GpuFrameId, threshold: usize) -> bool {
        self.last_frame_used + threshold >= current_frame_id
    }

    /// Returns the flags for this texture.
    pub fn flags(&self) -> &TextureFlags {
        &self.flags
    }

    /// Returns a mutable borrow of the flags for this texture.
    pub fn flags_mut(&mut self) -> &mut TextureFlags {
        &mut self.flags
    }

    /// Returns the number of bytes (generally in GPU memory) that each layer of
    /// this texture consumes.
    pub fn layer_size_in_bytes(&self) -> usize {
        assert!(self.layer_count > 0 || self.size.width + self.size.height == 0);
        let bpp = self.format.bytes_per_pixel() as usize;
        let w = self.size.width as usize;
        let h = self.size.height as usize;
        bpp * w * h
    }

    /// Returns the number of bytes (generally in GPU memory) that this texture
    /// consumes.
    pub fn size_in_bytes(&self) -> usize {
        self.layer_size_in_bytes() * (self.layer_count as usize)
    }

    #[cfg(feature = "replay")]
    pub fn into_external(mut self) -> ExternalTexture {
        let ext = ExternalTexture {
            id: self.id,
            target: self.target,
            _swizzle: Swizzle::default(),
        };
        self.id = 0; // don't complain, moved out
        ext
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        debug_assert!(thread::panicking() || self.id == 0);
    }
}

/// The interfaces that an application can implement to handle ProgramCache update
pub trait ProgramCacheObserver {
    fn update_disk_cache(&self, entries: Vec<Arc<ProgramBinary>>);
    fn try_load_shader_from_disk(&self, digest: &ProgramSourceDigest, program_cache: &Rc<ProgramCache>);
    fn notify_program_binary_failed(&self, program_binary: &Arc<ProgramBinary>);
}

struct ProgramCacheEntry {
    /// The binary.
    binary: Arc<ProgramBinary>,
    /// True if the binary has been linked, i.e. used for rendering.
    #[cfg(feature = "gl")]
    linked: bool,
}

pub struct ProgramCache {
    entries: RefCell<FastHashMap<ProgramSourceDigest, ProgramCacheEntry>>,

    /// True if we've already updated the disk cache with the shaders used during startup.
    #[cfg(feature = "gl")]
    updated_disk_cache: Cell<bool>,

    /// Optional trait object that allows the client
    /// application to handle ProgramCache updating
    #[cfg(feature = "gl")]
    program_cache_handler: Option<Box<dyn ProgramCacheObserver>>,
}

impl ProgramCache {
    pub fn new(_program_cache_observer: Option<Box<dyn ProgramCacheObserver>>) -> Rc<Self> {
        Rc::new(
            ProgramCache {
                entries: RefCell::new(FastHashMap::default()),
                #[cfg(feature = "gl")]
                updated_disk_cache: Cell::new(false),
                #[cfg(feature = "gl")]
                program_cache_handler: _program_cache_observer,
            }
        )
    }

    /// Notify that we've rendered the first few frames, and that the shaders
    /// we've loaded correspond to the shaders needed during startup, and thus
    /// should be the ones cached to disk.
    #[cfg(feature = "gl")]
    fn startup_complete(&self) {
        if self.updated_disk_cache.get() {
            return;
        }

        if let Some(ref handler) = self.program_cache_handler {
            let active_shaders = self.entries.borrow().values()
                .filter(|e| e.linked).map(|e| e.binary.clone())
                .collect::<Vec<_>>();
            handler.update_disk_cache(active_shaders);
            self.updated_disk_cache.set(true);
        }
    }

    /// Load ProgramBinary to ProgramCache.
    /// The function is typically used to load ProgramBinary from disk.
    #[cfg(all(feature = "serialize_program", feature = "gl"))]
    pub fn load_program_binary(&self, program_binary: Arc<ProgramBinary>) {
        let digest = program_binary.source_digest.clone();
        let entry = ProgramCacheEntry {
            binary: program_binary,
            linked: false,
        };
        self.entries.borrow_mut().insert(digest, entry);
    }

    /// Returns the number of bytes allocated for shaders in the cache.
    pub fn report_memory(&self, op: VoidPtrToSizeFn) -> usize {
        self.entries.borrow().values()
            .map(|e| unsafe { op(e.binary.bytes.as_ptr() as *const c_void ) })
            .sum()
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ProgramSourceInfo {
    base_filename: &'static str,
    features: String,
    digest: ProgramSourceDigest,
}

#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramBinary {
    bytes: Vec<u8>,
    #[cfg(feature = "gl")]
    format: gleam_gl::GLenum,
    #[cfg(feature = "gl")]
    source_digest: ProgramSourceDigest,
}

#[derive(Debug, Copy, Clone)]
pub enum VertexUsageHint {
    Static,
    Dynamic,
    Stream,
}

#[derive(Debug)]
pub struct Capabilities {
    /// Whether multisampled render targets are supported.
    pub supports_multisampling: bool,
    /// Whether the function `glCopyImageSubData` is available.
    pub supports_copy_image_sub_data: bool,
    /// Whether we are able to use `glBlitFramebuffers` with the draw fbo
    /// bound to a non-0th layer of a texture array. This is buggy on
    /// Adreno devices.
    pub supports_blit_to_texture_array: bool,
    /// Whether we can use the pixel local storage functionality that
    /// is available on some mobile GPUs. This allows fast access to
    /// the per-pixel tile memory.
    pub supports_pixel_local_storage: bool,
    /// Whether advanced blend equations are supported.
    pub supports_advanced_blend_equation: bool,
    /// Whether KHR_debug is supported for getting debug messages from
    /// the driver.
    pub supports_khr_debug: bool,
    /// Whether we can configure texture units to do swizzling on sampling.
    pub supports_texture_swizzle: bool,
}

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error message
    Link(String, String),        // name, error message
}

#[derive(Eq, PartialEq, Hash, Debug, Copy, Clone)]
pub enum ShaderKind {
    Primitive,
    Cache(VertexArrayKind),
    ClipCache,
    Brush,
    Text,
    #[allow(dead_code)]
    VectorStencil,
    #[allow(dead_code)]
    VectorCover,
    Resolve,
    Composite,
    #[cfg(all(not(feature = "gl")))]
    DebugColor,
    #[cfg(all(not(feature = "gl")))]
    DebugFont,
}

#[derive(Eq, PartialEq, Hash, Debug, Copy, Clone)]
pub enum VertexArrayKind {
    Primitive,
    Blur,
    Clip,
    #[cfg_attr(not(feature = "gl"), allow(dead_code))]
    VectorStencil,
    #[cfg_attr(not(feature = "gl"), allow(dead_code))]
    VectorCover,
    Border,
    Scale,
    LineDecoration,
    Gradient,
    Resolve,
    SvgFilter,
    Composite,
}

/// A refcounted depth target, which may be shared by multiple textures across
/// the device.
struct SharedDepthTarget {
    /// The Render Buffer Object representing the depth target.
    rbo_id: RBOId,
    /// Reference count. When this drops to zero, the RBO is deleted.
    refcount: usize,
}

#[cfg(debug_assertions)]
impl Drop for SharedDepthTarget {
    fn drop(&mut self) {
        debug_assert!(thread::panicking() || self.refcount == 0);
    }
}
/// Contains the parameters necessary to bind a draw target.
#[derive(Clone, Copy, Debug)]
pub enum DrawTarget {
    /// Use the device's default draw target, with the provided dimensions,
    /// which are used to set the viewport.
    Default {
        /// Target rectangle to draw.
        rect: FramebufferIntRect,
        /// Total size of the target.
        total_size: FramebufferIntSize,
    },
    /// Use the provided texture.
    Texture {
        /// Size of the texture in pixels
        dimensions: DeviceIntSize,
        /// The slice within the texture array to draw to
        layer: LayerIndex,
        /// Whether to draw with the texture's associated depth target
        with_depth: bool,
        /// Workaround buffers for devices with broken texture array copy implementation
        blit_workaround_buffer: Option<(RBOId, FBOId)>,
        /// FBO that corresponds to the selected layer / depth mode
        fbo_id: FBOId,
        /// Native GL texture ID
        id: IdType,
        /// Native GL texture target
        target: IdType,
    },
    /// Use an FBO attached to an external texture.
    External {
        fbo: FBOId,
        size: FramebufferIntSize,
    },
}

impl DrawTarget {
    pub fn new_default(size: DeviceIntSize) -> Self {
        let total_size = FramebufferIntSize::from_untyped(size.to_untyped());
        DrawTarget::Default {
            rect: total_size.into(),
            total_size,
        }
    }

    /// Returns true if this draw target corresponds to the default framebuffer.
    pub fn is_default(&self) -> bool {
        match *self {
            DrawTarget::Default {..} => true,
            _ => false,
        }
    }

    #[cfg(feature = "gl")]
    pub fn from_texture(
        texture: &Texture,
        layer: usize,
        with_depth: bool,
    ) -> Self {
        let fbo_id = if with_depth {
            texture.fbos_with_depth[layer]
        } else {
            texture.fbos[layer]
        };

        DrawTarget::Texture {
            dimensions: texture.get_dimensions(),
            fbo_id,
            with_depth,
            layer,
            blit_workaround_buffer: texture.blit_workaround_buffer,
            id: texture.id,
            target: texture.target,
        }
    }

    #[cfg(not(feature = "gl"))]
    pub fn from_texture(
        texture: &Texture,
        layer: usize,
        with_depth: bool,
        frame_id: GpuFrameId,
    ) -> Self {
        let fbo_id = if with_depth {
            texture.fbos_with_depth[layer]
        } else {
            texture.fbos[layer]
        };
        texture.bound_in_frame.set(frame_id);

        DrawTarget::Texture {
            dimensions: texture.get_dimensions(),
            fbo_id,
            with_depth,
            layer,
            blit_workaround_buffer: texture.blit_workaround_buffer,
            id: texture.id,
            target: texture.target,
        }
    }

    /// Returns the dimensions of this draw-target.
    pub fn dimensions(&self) -> DeviceIntSize {
        match *self {
            DrawTarget::Default { total_size, .. } => DeviceIntSize::from_untyped(total_size.to_untyped()),
            DrawTarget::Texture { dimensions, .. } => dimensions,
            DrawTarget::External { size, .. } => DeviceIntSize::from_untyped(size.to_untyped()),
        }
    }

    pub fn to_framebuffer_rect(&self, device_rect: DeviceIntRect) -> FramebufferIntRect {
        let mut fb_rect = FramebufferIntRect::from_untyped(&device_rect.to_untyped());
        match *self {
            DrawTarget::Default { ref rect, .. } => {
                if cfg!(feature = "gl") {
                    // perform a Y-flip here
                    fb_rect.origin.y = rect.origin.y + rect.size.height - fb_rect.origin.y - fb_rect.size.height;
                }
                fb_rect.origin.x += rect.origin.x;
            }
            DrawTarget::Texture { .. } | DrawTarget::External { .. } => (),
        }
        fb_rect
    }

    /// Given a scissor rect, convert it to the right coordinate space
    /// depending on the draw target kind. If no scissor rect was supplied,
    /// returns a scissor rect that encloses the entire render target.
    pub fn build_scissor_rect(
        &self,
        scissor_rect: Option<DeviceIntRect>,
        content_origin: DeviceIntPoint,
    ) -> FramebufferIntRect {
        let dimensions = self.dimensions();

        match scissor_rect {
            Some(scissor_rect) => match *self {
                DrawTarget::Default { ref rect, .. } => {
                    self.to_framebuffer_rect(scissor_rect.translate(-content_origin.to_vector()))
                        .intersection(rect)
                        .unwrap_or_else(FramebufferIntRect::zero)
                }
                DrawTarget::Texture { .. } | DrawTarget::External { .. } => {
                    FramebufferIntRect::from_untyped(&scissor_rect.to_untyped())
                }
            }
            None => {
                FramebufferIntRect::new(
                    FramebufferIntPoint::zero(),
                    FramebufferIntSize::from_untyped(dimensions.to_untyped()),
                )
            }
        }
    }
}

/// Contains the parameters necessary to bind a texture-backed read target.
#[derive(Clone, Copy, Debug)]
pub enum ReadTarget {
    /// Use the device's default draw target.
    Default,
    /// Use the provided texture,
    Texture {
        /// ID of the FBO to read from.
        fbo_id: FBOId,
    },
    /// Use an FBO attached to an external texture.
    External {
        fbo: FBOId,
    },
}

impl ReadTarget {
    pub fn from_texture(
        texture: &Texture,
        layer: usize,
    ) -> Self {
        ReadTarget::Texture {
            fbo_id: texture.fbos[layer],
        }
    }
}

impl From<DrawTarget> for ReadTarget {
    fn from(t: DrawTarget) -> Self {
        match t {
            DrawTarget::Default { .. } => ReadTarget::Default,
            DrawTarget::Texture { fbo_id, .. } =>
                ReadTarget::Texture { fbo_id },
            DrawTarget::External { fbo, .. } =>
                ReadTarget::External { fbo },
        }
    }
}

bitflags! {
    /// Flags that control how shaders are pre-cached, if at all.
    #[derive(Default)]
    pub struct ShaderPrecacheFlags: u32 {
        /// Needed for const initialization
        const EMPTY                 = 0;

        /// Only start async compile
        const ASYNC_COMPILE         = 1 << 2;

        /// Do a full compile/link during startup
        const FULL_COMPILE          = 1 << 3;
    }
}

/// Enumeration of the texture samplers used across the various WebRender shaders.
///
/// Each variant corresponds to a uniform declared in shader source. We only bind
/// the variants we need for a given shader, so not every variant is bound for every
/// batch.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TextureSampler {
    Color0,
    Color1,
    Color2,
    PrevPassAlpha,
    PrevPassColor,
    GpuCache,
    TransformPalette,
    RenderTasks,
    Dither,
    PrimitiveHeadersF,
    PrimitiveHeadersI,
}

impl TextureSampler {
    pub(crate) fn color(n: usize) -> TextureSampler {
        match n {
            0 => TextureSampler::Color0,
            1 => TextureSampler::Color1,
            2 => TextureSampler::Color2,
            _ => {
                panic!("There are only 3 color samplers.");
            }
        }
    }
}

impl Into<TextureSlot> for TextureSampler {
    fn into(self) -> TextureSlot {
        match self {
            TextureSampler::Color0 => TextureSlot(0),
            TextureSampler::Color1 => TextureSlot(1),
            TextureSampler::Color2 => TextureSlot(2),
            TextureSampler::PrevPassAlpha => TextureSlot(3),
            TextureSampler::PrevPassColor => TextureSlot(4),
            TextureSampler::GpuCache => TextureSlot(5),
            TextureSampler::TransformPalette => TextureSlot(6),
            TextureSampler::RenderTasks => TextureSlot(7),
            TextureSampler::Dither => TextureSlot(8),
            TextureSampler::PrimitiveHeadersF => TextureSlot(9),
            TextureSampler::PrimitiveHeadersI => TextureSlot(10),
        }
    }
}

/// A structure defining a particular workflow of texture transfers.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct TextureFormatPair<T> {
    /// Format the GPU natively stores texels in.
    pub internal: T,
    /// Format we expect the users to provide the texels in.
    pub external: T,
}

impl<T: Copy> From<T> for TextureFormatPair<T> {
    fn from(value: T) -> Self {
        TextureFormatPair {
            internal: value,
            external: value,
        }
    }
}

#[cfg(not(feature = "gl"))]
pub struct FormatDesc;

pub(crate) fn create_projection(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    main_frame_buffer: bool
) -> Transform3D<f32, euclid::UnknownUnit, euclid::UnknownUnit> {
    let projection = Transform3D::ortho(
        left,
        right,
        bottom,
        top,
        ORTHO_NEAR_PLANE,
        ORTHO_FAR_PLANE,
    );
    if main_frame_buffer && cfg!(not(feature = "gl")) {
        return projection.post_scale(1.0, -1.0, 0.5);
    }
    projection
}
