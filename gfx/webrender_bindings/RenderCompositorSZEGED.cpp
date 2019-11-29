/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "RenderCompositorSZEGED.h"

#include "GLContext.h"
#include "GLContextProvider.h"
#include "mozilla/gfx/Logging.h"
#include "mozilla/webrender/RenderThread.h"
#include "mozilla/widget/CompositorWidget.h"

#if defined(XP_WIN)
#pragma comment(lib, "d3d12.lib")       // located in DirectX SDK
#pragma comment(lib, "D3DCompiler.lib")
#include <d3d12.h>
#endif

namespace mozilla {
namespace wr {

/* static */ UniquePtr<RenderCompositor>
RenderCompositorSZEGED::Create(RefPtr<widget::CompositorWidget>&& aWidget)
{
  UniquePtr<RenderCompositorSZEGED> compositor = MakeUnique<RenderCompositorSZEGED>(std::move(aWidget));
  if (!compositor->Initialize()) {
    gfxCriticalNote << "Failed RenderCompositorSZEGED creation";
    return nullptr;
  }
  // gfxCriticalNote << "RenderCompositorSZEGED creation completed!";
  return compositor;
}

RenderCompositorSZEGED::RenderCompositorSZEGED(RefPtr<widget::CompositorWidget>&& aWidget)
  : RenderCompositor(std::move(aWidget))
{
}

RenderCompositorSZEGED::~RenderCompositorSZEGED()
{
  gfxCriticalNote << "in ~RenderCompositorSZEGED";
}

bool
RenderCompositorSZEGED::Initialize()
{
  const auto flags = gl::CreateContextFlags::NONE;

  nsCString discardFailureId;

  #if defined(XP_WIN)
    mGL = gl::GLContextProviderWGL::CreateHeadless(flags, &discardFailureId);
  #endif

  #if defined(XP_MACOSX)
    mGL = gl::GLContextProviderCGL::CreateHeadless(flags, &discardFailureId);
  #endif

  #if !(defined(XP_MACOSX) || defined(XP_WIN))
    mGL = gl::GLContextProviderGLX::CreateHeadless(flags, &discardFailureId);
  #endif

  if (!mGL) {
    gfxCriticalNote << "Failed GL context creation for WebRender: " << gfx::hexa(mGL.get());
    return false;
  }

  if (!mGL->MakeCurrent()) {
    gfxCriticalNote << "Failed GL context creation for WebRender: " << gfx::hexa(mGL.get());
    return false;
  }

  return true;
}

bool
RenderCompositorSZEGED::BeginFrame(layers::NativeLayer* aNativeLayer)
{
    if (!mGL->MakeCurrent()) {
      gfxCriticalNote << "Failed to make render context current, can't draw.";
      return false;
    }
    return true;
}

void
RenderCompositorSZEGED::EndFrame()
{
  mGL->SwapBuffers();
}

bool
RenderCompositorSZEGED::WaitForGPU() {
  return true;
}

void
RenderCompositorSZEGED::Pause()
{
}

bool
RenderCompositorSZEGED::Resume()
{
  return true;
}

LayoutDeviceIntSize
RenderCompositorSZEGED::GetBufferSize()
{
  return mWidget->GetClientSize();
}

} // namespace wr
} // namespace mozilla
