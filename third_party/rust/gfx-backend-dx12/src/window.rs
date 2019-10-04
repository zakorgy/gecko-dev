use std::collections::VecDeque;
use std::mem;

#[cfg(feature = "winit")]
use winit;

use winapi::shared::dxgi1_4;
use winapi::shared::windef::{HWND, RECT};
use winapi::um::winuser::GetClientRect;

use hal::{self, format as f, image as i, CompositeAlpha};
use {native, resource as r, Backend, Instance, PhysicalDevice, QueueFamily};

use std::os::raw::c_void;

impl Instance {
    pub fn create_surface_from_hwnd(&self, hwnd: *mut c_void) -> Surface {
        Surface {
            factory: self.factory,
            wnd_handle: hwnd as *mut _,
        }
    }

    #[cfg(feature = "winit")]
    pub fn create_surface(&self, window: &winit::Window) -> Surface {
        use winit::os::windows::WindowExt;
        self.create_surface_from_hwnd(window.get_hwnd() as *mut _)
    }

    pub fn create_surface_from_raw(
        &self,
        has_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Surface, hal::window::InitError> {
        match has_handle.raw_window_handle() {
            raw_window_handle::RawWindowHandle::Windows(handle) => {
                Ok(self.create_surface_from_hwnd(handle.hwnd))
            }
            _ => Err(hal::window::InitError::UnsupportedWindowHandle),
        }
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Surface {
    #[derivative(Debug = "ignore")]
    pub(crate) factory: native::WeakPtr<dxgi1_4::IDXGIFactory4>,
    pub(crate) wnd_handle: HWND,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    fn get_extent(&self) -> (u32, u32) {
        unsafe {
            let mut rect: RECT = mem::zeroed();
            if GetClientRect(self.wnd_handle as *mut _, &mut rect as *mut RECT) == 0 {
                panic!("GetClientRect failed");
            }
            (
                (rect.right - rect.left) as u32,
                (rect.bottom - rect.top) as u32,
            )
        }
    }
}

impl hal::Surface<Backend> for Surface {
    fn supports_queue_family(&self, queue_family: &QueueFamily) -> bool {
        match queue_family {
            &QueueFamily::Present => true,
            _ => false,
        }
    }

    fn compatibility(
        &self,
        _: &PhysicalDevice,
    ) -> (
        hal::SurfaceCapabilities,
        Option<Vec<f::Format>>,
        Vec<hal::PresentMode>,
    ) {
        let (width, height) = self.get_extent();
        let extent = hal::window::Extent2D { width, height };

        let capabilities = hal::SurfaceCapabilities {
            image_count: 2 ..= 16, // we currently use a flip effect which supports 2..=16 buffers
            current_extent: Some(extent),
            extents: extent ..= extent,
            max_image_layers: 1,
            usage: i::Usage::COLOR_ATTACHMENT | i::Usage::TRANSFER_SRC | i::Usage::TRANSFER_DST,
            composite_alpha: CompositeAlpha::OPAQUE, //TODO
        };

        // Sticking to FLIP swap effects for the moment.
        // We also expose sRGB buffers but they are handled internally as UNORM.
        // Roughly ordered by popularity..
        let formats = vec![
            f::Format::Bgra8Srgb,
            f::Format::Bgra8Unorm,
            f::Format::Rgba8Srgb,
            f::Format::Rgba8Unorm,
            f::Format::A2b10g10r10Unorm,
            f::Format::Rgba16Sfloat,
        ];

        let present_modes = vec![
            hal::PresentMode::Fifo, //TODO
        ];

        (capabilities, Some(formats), present_modes)
    }
}

#[derive(Debug)]
pub struct Swapchain {
    pub(crate) inner: native::WeakPtr<dxgi1_4::IDXGISwapChain3>,
    pub(crate) next_frame: usize,
    pub(crate) frame_queue: VecDeque<usize>,
    #[allow(dead_code)]
    pub(crate) rtv_heap: r::DescriptorHeap,
    // need to associate raw image pointers with the swapchain so they can be properly released
    // when the swapchain is destroyed
    pub(crate) resources: Vec<native::Resource>,
}

impl hal::Swapchain<Backend> for Swapchain {
    unsafe fn acquire_image(
        &mut self,
        _timout_ns: u64,
        _semaphore: Option<&r::Semaphore>,
        _fence: Option<&r::Fence>,
    ) -> Result<(hal::SwapImageIndex, Option<hal::window::Suboptimal>), hal::AcquireError> {
        // TODO: sync

        if false {
            // TODO: we need to block this at some point? (running out of backbuffers)
            //let num_images = self.images.len();
            let num_images = 1;
            let index = self.next_frame;
            self.frame_queue.push_back(index);
            self.next_frame = (self.next_frame + 1) % num_images;
        }

        // TODO:
        Ok((self.inner.GetCurrentBackBufferIndex(), None))
    }
}

unsafe impl Send for Swapchain {}
unsafe impl Sync for Swapchain {}
