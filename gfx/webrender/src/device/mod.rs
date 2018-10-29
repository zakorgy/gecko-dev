/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::shader_source;
use api::{DeviceUintSize, ImageFormat, TextureTarget};
use euclid::Transform3D;
#[cfg(feature = "gleam")]
use gleam::gl::GLuint;
use internal_types::{FastHashMap, LayerIndex, ORTHO_FAR_PLANE, ORTHO_NEAR_PLANE};
#[cfg(not(feature = "gleam"))]
use std::cell::Cell;
use std::cell::RefCell;
use std::io::Read;
use std::fs::File;
use std::ops::Add;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::thread;

cfg_if! {
    if #[cfg(feature = "gleam")] {
        mod gl;
        pub mod query_gl;

        pub use self::gl::*;
        pub use self::query_gl as query;
    } else {
        mod gfx;
        pub mod query_gfx;

        pub use self::gfx::*;
        pub use self::query_gfx as query;
    }
}

const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";
const SHADER_IMPORT: &str = "#include ";

#[cfg(not(feature = "gleam"))]
pub type IdType = u32;

#[cfg(feature = "gleam")]
pub type IdType = GLuint;

/// Plain old data that can be used to initialize a texture.
pub unsafe trait Texel: Copy {}
unsafe impl Texel for u8 {}
unsafe impl Texel for f32 {}

#[cfg(feature = "debug_renderer")]
pub struct Capabilities {
    pub supports_multisampling: bool,
}

#[derive(Debug, Copy, Clone)]
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
    #[cfg(all(not(feature = "gleam"), feature = "debug_renderer"))]
    DebugColor,
    #[cfg(all(not(feature = "gleam"), feature = "debug_renderer"))]
    DebugFont,
}

#[derive(Debug, Copy, Clone)]
pub enum VertexArrayKind {
    Primitive,
    Blur,
    Clip,
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    VectorStencil,
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    VectorCover,
    Border,
    Scale,
    LineDecoration,
}

/// The interfaces that an application can implement to handle ProgramCache update
pub trait ProgramCacheObserver {
    fn notify_binary_added(&self, program_binary: &Arc<ProgramBinary>);
    fn notify_program_binary_failed(&self, program_binary: &Arc<ProgramBinary>);
}

pub struct ProgramCache {
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    binaries: RefCell<FastHashMap<ProgramSources, Arc<ProgramBinary>>>,

    /// Optional trait object that allows the client
    /// application to handle ProgramCache updating
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    program_cache_handler: Option<Box<ProgramCacheObserver>>,
}

impl ProgramCache {
    pub fn new(program_cache_observer: Option<Box<ProgramCacheObserver>>) -> Rc<Self> {
        Rc::new(
            ProgramCache {
                binaries: RefCell::new(FastHashMap::default()),
                program_cache_handler: program_cache_observer,
            }
        )
    }
    /// Load ProgramBinary to ProgramCache.
    /// The function is typically used to load ProgramBinary from disk.
    #[cfg(feature = "serialize_program")]
    pub fn load_program_binary(&self, program_binary: Arc<ProgramBinary>) {
        let sources = program_binary.sources.clone();
        self.binaries.borrow_mut().insert(sources, program_binary);
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct RBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(IdType);

#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FrameId(usize);

impl FrameId {
    pub fn new(value: usize) -> Self {
        FrameId(value)
    }
}

impl Add<usize> for FrameId {
    type Output = FrameId;

    fn add(self, other: usize) -> FrameId {
        FrameId(self.0 + other)
    }
}

pub struct TextureSlot(pub usize);

#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureFilter {
    Nearest,
    Linear,
    Trilinear,
}

#[derive(Debug)]
pub enum VertexAttributeKind {
    F32,
    #[cfg(feature = "debug_renderer")]
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReadPixelsFormat {
    Standard(ImageFormat),
    Rgba8,
}

pub trait FileWatcherHandler: Send {
    fn file_changed(&self, path: PathBuf);
}

#[cfg_attr(feature = "replay", derive(Clone))]
pub struct ExternalTexture {
    id: IdType,
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    target: IdType,
}


impl ExternalTexture {
    pub fn new(id: u32, target: TextureTarget) -> Self {
        ExternalTexture {
            id,
            #[cfg(feature = "gleam")]
            target: get_gl_target(target),
            #[cfg(not(feature = "gleam"))]
            target: target as _,
        }
    }

    #[cfg(feature = "replay")]
    pub fn internal_id(&self) -> IdType {
        self.id
    }
}

/// A refcounted depth target, which may be shared by multiple textures across
/// the device.
struct SharedDepthTarget {
    /// The Render Buffer Object representing the depth target.
    rbo_id: RBOId,
    /// Reference count. When this drops to zero, the RBO is deleted.
    refcount: usize,
}

#[cfg(feature = "debug")]
impl Drop for SharedDepthTarget {
    fn drop(&mut self) {
        debug_assert!(thread::panicking() || self.refcount == 0);
    }
}

/// WebRender interface to an OpenGL texture.
///
/// Because freeing a texture requires various device handles that are not
/// reachable from this struct, manual destruction via `Device` is required.
/// Our `Drop` implementation asserts that this has happened.
pub struct Texture {
    pub id: IdType,
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    target: IdType,
    layer_count: i32,
    format: ImageFormat,
    width: u32,
    height: u32,
    filter: TextureFilter,
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
    last_frame_used: FrameId,
    #[cfg(not(feature = "gleam"))]
    bound_in_frame: Cell<FrameId>,
}

impl Texture {
    pub fn get_dimensions(&self) -> DeviceUintSize {
        DeviceUintSize::new(self.width, self.height)
    }

    pub fn get_layer_count(&self) -> i32 {
        self.layer_count
    }

    pub fn get_format(&self) -> ImageFormat {
        self.format
    }

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    pub fn get_filter(&self) -> TextureFilter {
        self.filter
    }

    pub fn supports_depth(&self) -> bool {
        !self.fbos_with_depth.is_empty()
    }

    pub fn used_in_frame(&self, frame_id: FrameId) -> bool {
        self.last_frame_used == frame_id
    }

    /// Returns true if this texture was used within `threshold` frames of
    /// the current frame.
    pub fn used_recently(&self, current_frame_id: FrameId, threshold: usize) -> bool {
        self.last_frame_used + threshold >= current_frame_id
    }

    /// Returns the number of bytes (generally in GPU memory) that this texture
    /// consumes.
    pub fn size_in_bytes(&self) -> usize {
        assert!(self.layer_count > 0 || self.width + self.height == 0);
        let bpp = self.format.bytes_per_pixel() as usize;
        let w = self.width as usize;
        let h = self.height as usize;
        let count = self.layer_count as usize;
        bpp * w * h * count
    }

    #[cfg(feature = "replay")]
    pub fn into_external(mut self) -> ExternalTexture {
        let ext = ExternalTexture {
            id: self.id,
            target: self.target,
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

/// Contains the parameters necessary to bind a texture-backed draw target.
#[derive(Clone, Copy)]
pub struct TextureDrawTarget<'a> {
    /// The target texture.
    pub texture: &'a Texture,
    /// The slice within the texture array to draw to.
    pub layer: LayerIndex,
    /// Whether to draw with the texture's associated depth target.
    pub with_depth: bool,
}

/// Contains the parameters necessary to bind a texture-backed read target.
#[derive(Clone, Copy)]
pub struct TextureReadTarget<'a> {
    /// The source texture.
    pub texture: &'a Texture,
    /// The slice within the texture array to read from.
    pub layer: LayerIndex,
}

impl<'a> From<TextureDrawTarget<'a>> for TextureReadTarget<'a> {
    fn from(t: TextureDrawTarget<'a>) -> Self {
        TextureReadTarget {
            texture: t.texture,
            layer: t.layer,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum VertexUsageHint {
    Static,
    Dynamic,
    Stream,
}

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error message
    Link(String, String),        // name, error message
}

/// Flags that control how shaders are pre-cached, if at all.
bitflags! {
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
pub(crate) enum TextureSampler {
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

// Get a shader string by name, from the built in resources or
// an override path, if supplied.
fn get_shader_source(shader_name: &str, base_path: &Option<PathBuf>) -> Option<String> {
    if let Some(ref base) = *base_path {
        let shader_path = base.join(&format!("{}.glsl", shader_name));
        if shader_path.exists() {
            let mut source = String::new();
            File::open(&shader_path)
                .unwrap()
                .read_to_string(&mut source)
                .unwrap();
            return Some(source);
        }
    }

    shader_source::SHADERS
        .get(shader_name)
        .map(|s| s.to_string())
}

// Parse a shader string for imports. Imports are recursively processed, and
// prepended to the list of outputs.
fn parse_shader_source(source: String, base_path: &Option<PathBuf>, output: &mut String) {
    for line in source.lines() {
        if line.starts_with(SHADER_IMPORT) {
            let imports = line[SHADER_IMPORT.len() ..].split(',');

            // For each import, get the source, and recurse.
            for import in imports {
                if let Some(include) = get_shader_source(import, base_path) {
                    parse_shader_source(include, base_path, output);
                }
            }
        } else {
            output.push_str(line);
            output.push_str("\n");
        }
    }
}

pub fn build_shader_strings(
    gl_version_string: &str,
    features: &str,
    base_filename: &str,
    override_path: &Option<PathBuf>,
) -> (String, String) {
    // Construct a list of strings to be passed to the shader compiler.
    let mut vs_source = String::new();
    let mut fs_source = String::new();

    // GLSL requires that the version number comes first.
    vs_source.push_str(gl_version_string);
    fs_source.push_str(gl_version_string);

    // Insert the shader name to make debugging easier.
    let name_string = format!("// {}\n", base_filename);
    vs_source.push_str(&name_string);
    fs_source.push_str(&name_string);

    // Define a constant depending on whether we are compiling VS or FS.
    vs_source.push_str(SHADER_KIND_VERTEX);
    fs_source.push_str(SHADER_KIND_FRAGMENT);

    // Add any defines that were passed by the caller.
    vs_source.push_str(features);
    fs_source.push_str(features);

    // Parse the main .glsl file, including any imports
    // and append them to the list of sources.
    let mut shared_result = String::new();
    if let Some(shared_source) = get_shader_source(base_filename, override_path) {
        parse_shader_source(shared_source, override_path, &mut shared_result);
    }

    vs_source.push_str(&shared_result);
    fs_source.push_str(&shared_result);

    (vs_source, fs_source)
}

pub(crate) fn create_projection(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    main_frame_buffer: bool
) -> Transform3D<f32> {
    let projection = Transform3D::ortho(
        left,
        right,
        bottom,
        top,
        ORTHO_NEAR_PLANE,
        ORTHO_FAR_PLANE,
    );
    if main_frame_buffer && cfg!(not(feature = "gleam")) {
        return projection.post_scale(1.0, -1.0, 1.0);
    }
    projection
}

pub(crate) mod desc {
    #![cfg_attr(not(feature = "gleam"), allow(dead_code))]
    use device::{VertexAttribute, VertexAttributeKind, VertexDescriptor};

    pub const PRIM_INSTANCES: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aData",
                count: 4,
                kind: VertexAttributeKind::I32,
            },
        ],
    };

    pub const BLUR: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aBlurRenderTaskAddress",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aBlurSourceTaskAddress",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aBlurDirection",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
        ],
    };

    pub const LINE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aTaskRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aLocalSize",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aWavyLineThickness",
                count: 1,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aStyle",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aOrientation",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
        ],
    };

    pub const BORDER: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aTaskOrigin",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor0",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor1",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aFlags",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aWidths",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aRadii",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipParams1",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipParams2",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const SCALE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aScaleRenderTaskAddress",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aScaleSourceTaskAddress",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
        ],
    };

    pub const CLIP: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aClipRenderTaskAddress",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aClipTransformId",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aPrimTransformId",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aClipSegment",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aClipDataResourceAddress",
                count: 4,
                kind: VertexAttributeKind::U16,
            },
        ],
    };

    pub const GPU_CACHE_UPDATE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::U16Norm,
            },
            VertexAttribute {
                name: "aValue",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[],
    };

    pub const VECTOR_STENCIL: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aFromPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aCtrlPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aToPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aFromNormal",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aCtrlNormal",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aToNormal",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aPathID",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aPad",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
        ],
    };

    pub const VECTOR_COVER: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aTargetRect",
                count: 4,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aStencilOrigin",
                count: 2,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aSubpixel",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aPad",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
        ],
    };
}
