/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![cfg_attr(
not(any(feature = "gfx-hal", feature = "gl")),
allow(dead_code, unused_imports)
)]

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
#[cfg(feature = "gl")]
extern crate gfx_backend_empty as back;

#[cfg(feature = "gl")]
use gleam::gl;
#[cfg(feature = "gl")]
use glutin;
use std::env;
use std::path::PathBuf;
use webrender;
use winit;
use webrender::{DebugFlags, ShaderPrecacheFlags};
use webrender::api::*;
use webrender::api::units::*;
#[cfg(feature = "gfx-hal")]
use webrender::hal::Instance;

struct Notifier {
    events_proxy: winit::EventsLoopProxy,
}

impl Notifier {
    fn new(events_proxy: winit::EventsLoopProxy) -> Notifier {
        Notifier { events_proxy }
    }
}

impl RenderNotifier for Notifier {
    fn clone(&self) -> Box<dyn RenderNotifier> {
        Box::new(Notifier {
            events_proxy: self.events_proxy.clone(),
        })
    }

    fn wake_up(&self) {
        #[cfg(not(target_os = "android"))]
        let _ = self.events_proxy.wakeup();
    }

    fn new_frame_ready(&self,
                       _: DocumentId,
                       _scrolled: bool,
                       _composite_needed: bool,
                       _render_time: Option<u64>) {
        self.wake_up();
    }
}

pub trait HandyDandyRectBuilder {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect;
    fn by(&self, w: i32, h: i32) -> LayoutRect;
}
// Allows doing `(x, y).to(x2, y2)` or `(x, y).by(width, height)` with i32
// values to build a f32 LayoutRect
impl HandyDandyRectBuilder for (i32, i32) {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect {
        LayoutRect::new(
            LayoutPoint::new(self.0 as f32, self.1 as f32),
            LayoutSize::new((x2 - self.0) as f32, (y2 - self.1) as f32),
        )
    }

    fn by(&self, w: i32, h: i32) -> LayoutRect {
        LayoutRect::new(
            LayoutPoint::new(self.0 as f32, self.1 as f32),
            LayoutSize::new(w as f32, h as f32),
        )
    }
}

pub trait Example {
    const TITLE: &'static str = "WebRender Sample App";
    const PRECACHE_SHADER_FLAGS: ShaderPrecacheFlags = ShaderPrecacheFlags::EMPTY;
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;

    fn render(
        &mut self,
        api: &RenderApi,
        builder: &mut DisplayListBuilder,
        txn: &mut Transaction,
        device_size: DeviceIntSize,
        pipeline_id: PipelineId,
        document_id: DocumentId,
    );
    fn on_event(
        &mut self,
        _: winit::WindowEvent,
        _: &RenderApi,
        _: DocumentId,
    ) -> bool {
        false
    }
    #[cfg(feature = "gl")]
    fn get_image_handlers(
        &mut self,
        _gl: &dyn gl::Gl,
    ) -> (Option<Box<dyn webrender::ExternalImageHandler>>,
          Option<Box<dyn webrender::OutputImageHandler>>) {
        (None, None)
    }
    #[cfg(not(feature = "gl"))]
    fn get_image_handlers(
        &mut self,
    ) -> (Option<Box<dyn webrender::ExternalImageHandler>>,
          Option<Box<dyn webrender::OutputImageHandler>>) {
        (None, None)
    }
    #[cfg(feature = "gl")]
    fn draw_custom(&mut self, _gl: &dyn gl::Gl) {
    }
}

#[cfg(any(feature = "gfx-hal", feature = "gl"))]
pub fn main_wrapper<E: Example>(
    example: &mut E,
    options: Option<webrender::RendererOptions>,
) {
    env_logger::init();

    #[cfg(target_os = "macos")]
    {
        use core_foundation::{self as cf, base::TCFType};
        let i = cf::bundle::CFBundle::main_bundle().info_dictionary();
        let mut i = unsafe { i.to_mutable() };
        i.set(
            cf::string::CFString::new("NSSupportsAutomaticGraphicsSwitching"),
            cf::boolean::CFBoolean::true_value().into_CFType(),
        );
    }

    let args: Vec<String> = env::args().collect();
    let res_path = if args.len() > 1 {
        Some(PathBuf::from(&args[1]))
    } else {
        None
    };

    let mut events_loop = winit::EventsLoop::new();
    let window_builder = winit::WindowBuilder::new()
        .with_title(E::TITLE)
        .with_multitouch()
        .with_dimensions(winit::dpi::LogicalSize::new(E::WIDTH as f64, E::HEIGHT as f64));

    #[cfg(feature = "gl")]
    let (gl, init, windowed_context) = {
        let windowed_context = glutin::ContextBuilder::new()
            .with_gl(glutin::GlRequest::GlThenGles {
                opengl_version: (3, 2),
                opengles_version: (3, 0),
            })
            .build_windowed(window_builder, &events_loop)
            .unwrap();

        let windowed_context = unsafe { windowed_context.make_current().unwrap() };

        let gl = match windowed_context.get_api() {
            glutin::Api::OpenGl => unsafe {
                gl::GlFns::load_with(
                    |symbol| windowed_context.get_proc_address(symbol) as *const _
                )
            },
            glutin::Api::OpenGlEs => unsafe {
                gl::GlesFns::load_with(
                    |symbol| windowed_context.get_proc_address(symbol) as *const _
                )
            },
            glutin::Api::WebGl => unimplemented!(),
        };

        println!("OpenGL version {}", gl.get_string(gl::VERSION));
        let init = gl.clone().into();
        (gl, init, windowed_context)
    };

    #[cfg(feature = "gfx-hal")]
    let (init, window) = {
        let window = window_builder.build(&events_loop).unwrap();
        let instance = back::Instance::create("gfx-rs instance", 1).expect("Instance creation failed");
        let mut adapters = instance.enumerate_adapters();
        let adapter = adapters.remove(0);
        let surface = Some(
            unsafe { instance.create_surface(&window) }.expect("Surface creation failed")
        );
        let winit::dpi::LogicalSize { width, height } = window.get_inner_size().unwrap();
        let init = {
            let cache_dir = dirs::cache_dir().expect("User's cache directory not found");
            let cache_path = Some(PathBuf::from(&cache_dir).join("pipeline_cache.bin"));

            #[cfg(feature = "vulkan")]
            let backend_api = webrender::BackendApiType::Vulkan;
            #[cfg(feature = "metal")]
            let backend_api = webrender::BackendApiType::Metal;
            #[cfg(feature = "dx12")]
            let backend_api = webrender::BackendApiType::Dx12;

            webrender::DeviceInit {
                instance,
                adapter,
                surface,
                window_size: (width as i32, height as i32),
                descriptor_count: None,
                cache_path,
                save_cache: true,
                backend_api,
            }
        };
        (init, window)
    };

    println!("Shader resource path: {:?}", res_path);

    #[cfg(feature = "gl")]
    let window = windowed_context.window();
    let device_pixel_ratio = window.get_hidpi_factor() as f32;
    println!("Device pixel ratio: {}", device_pixel_ratio);

    println!("Loading shaders...");
    let mut debug_flags = DebugFlags::ECHO_DRIVER_MESSAGES | DebugFlags::TEXTURE_CACHE_DBG;

    #[cfg(feature = "gfx-hal")]
    let heaps_config = {
        let config_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("webrender/res/mem_config.ron");
        let source = std::fs::read_to_string(&config_path)
            .expect(&format!("Unable to open memory config file from {:?}", config_path));
        ron::de::from_str(&source).expect("Unable to parse HeapsConfig")
    };

    let opts = webrender::RendererOptions {
        resource_override_path: res_path,
        precache_flags: E::PRECACHE_SHADER_FLAGS,
        device_pixel_ratio,
        clear_color: Some(ColorF::new(0.3, 0.0, 0.0, 1.0)),
        //scatter_gpu_cache_updates: false,
        debug_flags,
        //allow_texture_swizzling: false,
        #[cfg(feature = "gfx-hal")]
        heaps_config,
        ..options.unwrap_or(webrender::RendererOptions::default())
    };

    #[cfg(feature = "gl")]
    let window = windowed_context.window();

    let mut device_size = {
        let size = window
            .get_inner_size()
            .unwrap()
            .to_physical(device_pixel_ratio as f64);
        DeviceIntSize::new(size.width as i32, size.height as i32)
    };
    let notifier = Box::new(Notifier::new(events_loop.create_proxy()));

    let (mut renderer, sender): (webrender::Renderer<back::Backend>, _) = webrender::Renderer::new(
        init,
        notifier,
        opts,
        None,
        device_size,
    ).unwrap();
    let api = sender.create_api();
    let document_id = api.add_document(device_size, 0);

    #[cfg(feature = "gl")]
    let (external, output) = example.get_image_handlers(&*gl);

    #[cfg(feature = "gfx-hal")]
    let (external, output) = example.get_image_handlers();

    if let Some(output_image_handler) = output {
        renderer.set_output_image_handler(output_image_handler);
    }

    if let Some(external_image_handler) = external {
        renderer.set_external_image_handler(external_image_handler);
    }

    let epoch = Epoch(0);
    let pipeline_id = PipelineId(0, 0);
    let mut layout_size = device_size.to_f32() / euclid::Scale::new(device_pixel_ratio);
    let mut builder = DisplayListBuilder::new(pipeline_id, layout_size);
    let mut txn = Transaction::new();

    example.render(
        &api,
        &mut builder,
        &mut txn,
        device_size,
        pipeline_id,
        document_id,
    );
    txn.set_display_list(
        epoch,
        Some(ColorF::new(0.3, 0.0, 0.0, 1.0)),
        layout_size,
        builder.finalize(),
        true,
    );
    txn.set_root_pipeline(pipeline_id);
    txn.generate_frame();
    api.send_transaction(document_id, txn);

    println!("Entering event loop");
    events_loop.run_forever(|global_event| {
        let mut txn = Transaction::new();
        let mut custom_event = true;

        let old_flags = debug_flags;
        let win_event = match global_event {
            winit::Event::WindowEvent { event, .. } => event,
            _ => return winit::ControlFlow::Continue,
        };
        match win_event {
            winit::WindowEvent::CloseRequested => return winit::ControlFlow::Break,
            // skip high-frequency events
            winit::WindowEvent::AxisMotion { .. } |
            winit::WindowEvent::CursorMoved { .. } => return winit::ControlFlow::Continue,
            winit::WindowEvent::KeyboardInput {
                input: winit::KeyboardInput {
                    state: winit::ElementState::Pressed,
                    virtual_keycode: Some(key),
                    ..
                },
                ..
            } => match key {
                winit::VirtualKeyCode::Escape => return winit::ControlFlow::Break,
                winit::VirtualKeyCode::P => debug_flags.toggle(DebugFlags::PROFILER_DBG),
                winit::VirtualKeyCode::O => debug_flags.toggle(DebugFlags::RENDER_TARGET_DBG),
                winit::VirtualKeyCode::I => debug_flags.toggle(DebugFlags::TEXTURE_CACHE_DBG),
                winit::VirtualKeyCode::S => debug_flags.toggle(DebugFlags::COMPACT_PROFILER),
                winit::VirtualKeyCode::T => debug_flags.toggle(DebugFlags::PICTURE_CACHING_DBG),
                winit::VirtualKeyCode::Q => debug_flags.toggle(
                    DebugFlags::GPU_TIME_QUERIES | DebugFlags::GPU_SAMPLE_QUERIES
                ),
                winit::VirtualKeyCode::F => debug_flags.toggle(
                    DebugFlags::NEW_FRAME_INDICATOR | DebugFlags::NEW_SCENE_INDICATOR
                ),
                winit::VirtualKeyCode::G => debug_flags.toggle(DebugFlags::GPU_CACHE_DBG),
                winit::VirtualKeyCode::Key1 => txn.set_document_view(
                    device_size.into(),
                    1.0
                ),
                winit::VirtualKeyCode::Key2 => txn.set_document_view(
                    device_size.into(),
                    2.0
                ),
                winit::VirtualKeyCode::M => api.notify_memory_pressure(),
                winit::VirtualKeyCode::C => {
                    let path: PathBuf = "../captures/example".into();
                    //TODO: switch between SCENE/FRAME capture types
                    // based on "shift" modifier, when `glutin` is updated.
                    let bits = CaptureBits::all();
                    api.save_capture(path, bits);
                },
                _ => {
                    custom_event = example.on_event(
                        win_event,
                        &api,
                        document_id,
                    )
                },
            },
            winit::WindowEvent::Resized(dims) => {
                let new_size = ((dims.width as f32 * device_pixel_ratio) as i32, (dims.height as f32 * device_pixel_ratio) as i32);
                #[cfg(not(feature = "gl"))]
                { device_size = renderer.resize(Some(new_size)) };
                #[cfg(feature = "gl")]
                { device_size = DeviceIntSize::new(new_size.0,new_size.1) };

                layout_size = device_size.to_f32() / euclid::Scale::new(device_pixel_ratio);
                api.set_document_view(
                    document_id,
                    DeviceIntRect::new(DeviceIntPoint::zero(), device_size),
                    device_pixel_ratio,
                );
                return winit::ControlFlow::Continue;
            }
            other => custom_event = example.on_event(
                other,
                &api,
                document_id,
            ),
        };

        if debug_flags != old_flags {
            api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
        }

        if custom_event {
            let mut builder = DisplayListBuilder::new(pipeline_id, layout_size);

            example.render(
                &api,
                &mut builder,
                &mut txn,
                device_size,
                pipeline_id,
                document_id,
            );
            txn.set_display_list(
                epoch,
                Some(ColorF::new(0.3, 0.0, 0.0, 1.0)),
                layout_size,
                builder.finalize(),
                true,
            );
            txn.generate_frame();
        }
        api.send_transaction(document_id, txn);

        renderer.update();
        renderer.render(device_size).unwrap();
        let _ = renderer.flush_pipeline_info();
        #[cfg(feature = "gl")]
        example.draw_custom(&*gl);
        #[cfg(feature = "gl")]
        windowed_context.swap_buffers().ok();

        winit::ControlFlow::Continue
    });
    api.shut_down(true);

    renderer.deinit();
}

#[cfg(not(any(feature = "gfx-hal", feature = "gl")))]
pub fn main_wrapper<E: Example>(
    _example: &mut E,
    _options: Option<webrender::RendererOptions>,
) {
    println!("You need to enable one of the native API features (dx12/gl/metal/vulkan) in order to run this example.");
}
