/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WEBGL_FRAMEBUFFER_H_
#define WEBGL_FRAMEBUFFER_H_

#include <vector>

#include "mozilla/LinkedList.h"
#include "mozilla/WeakPtr.h"
#include "nsWrapperCache.h"

#include "WebGLObjectModel.h"
#include "WebGLRenderbuffer.h"
#include "WebGLStrongTypes.h"
#include "WebGLTexture.h"
#include "WebGLTypes.h"

namespace mozilla {

class WebGLFramebuffer;
class WebGLRenderbuffer;
class WebGLTexture;

template<typename T>
class PlacementArray;

namespace gl {
    class GLContext;
} // namespace gl

class WebGLFBAttachPoint final
{
    friend class WebGLFramebuffer;
public:
    WebGLFramebuffer* const mFB;
    const GLenum mAttachmentPoint;

private:
    WebGLRefPtr<WebGLTexture> mTexturePtr;
    WebGLRefPtr<WebGLRenderbuffer> mRenderbufferPtr;
    TexImageTarget mTexImageTarget;
    GLint mTexImageLayer;
    uint32_t mTexImageLevel;

    ////

    WebGLFBAttachPoint();
    WebGLFBAttachPoint(WebGLFramebuffer* fb, GLenum attachmentPoint);
    explicit WebGLFBAttachPoint(WebGLFBAttachPoint&) = default; // Make this private.

public:
    ~WebGLFBAttachPoint();

    ////

    void Unlink();

    bool IsDefined() const;
    bool IsDeleteRequested() const;

    const webgl::FormatUsageInfo* Format() const;
    uint32_t Samples() const;

    bool HasAlpha() const;
    bool IsReadableFloat() const;

    void Clear();

    void SetTexImage(WebGLTexture* tex, TexImageTarget target,
                     GLint level, GLint layer = 0);
    void SetRenderbuffer(WebGLRenderbuffer* rb);

    WebGLTexture* Texture() const { return mTexturePtr; }
    WebGLRenderbuffer* Renderbuffer() const { return mRenderbufferPtr; }

    TexImageTarget ImageTarget() const {
        return mTexImageTarget;
    }
    GLint Layer() const {
        return mTexImageLayer;
    }
    uint32_t MipLevel() const {
        return mTexImageLevel;
    }
    void AttachmentName(nsCString* out) const;

    bool HasUninitializedImageData() const;
    void SetImageDataStatus(WebGLImageDataStatus x) const;

    void Size(uint32_t* const out_width, uint32_t* const out_height) const;

    bool HasImage() const;
    bool IsComplete(WebGLContext* webgl, nsCString* const out_info) const;

    void Resolve(gl::GLContext* gl) const;

    JS::Value GetParameter(WebGLContext* webgl, JSContext* cx,
                           GLenum target, GLenum attachment, GLenum pname,
                           ErrorResult* const out_error) const;

    void OnBackingStoreRespecified() const;

    bool IsEquivalentForFeedback(const WebGLFBAttachPoint& other) const {
        if (!IsDefined() || !other.IsDefined())
            return false;

#define _(X) X == other.X
        return ( _(mRenderbufferPtr) &&
                 _(mTexturePtr) &&
                 _(mTexImageTarget.get()) &&
                 _(mTexImageLevel) &&
                 _(mTexImageLayer) );
#undef _
    }

    ////

    struct Ordered {
        const WebGLFBAttachPoint& mRef;

        explicit Ordered(const WebGLFBAttachPoint& ref)
            : mRef(ref)
        { }

        bool operator<(const Ordered& other) const {
            MOZ_ASSERT(mRef.IsDefined() && other.mRef.IsDefined());

#define ORDER_BY(X) if (X != other.X) return X < other.X;

            ORDER_BY(mRef.mRenderbufferPtr)
            ORDER_BY(mRef.mTexturePtr)
            ORDER_BY(mRef.mTexImageTarget.get())
            ORDER_BY(mRef.mTexImageLevel)
            ORDER_BY(mRef.mTexImageLayer)

#undef ORDER_BY
            return false;
        }
    };
};

class WebGLFramebuffer final
    : public nsWrapperCache
    , public WebGLRefCountedObject<WebGLFramebuffer>
    , public LinkedListElement<WebGLFramebuffer>
    , public SupportsWeakPtr<WebGLFramebuffer>
{
    friend class WebGLContext;

public:
    MOZ_DECLARE_WEAKREFERENCE_TYPENAME(WebGLFramebuffer)

    const GLuint mGLName;

private:
    uint64_t mNumFBStatusInvals;

protected:
#ifdef ANDROID
    // Bug 1140459: Some drivers (including our test slaves!) don't
    // give reasonable answers for IsRenderbuffer, maybe others.
    // This shows up on Android 2.3 emulator.
    //
    // So we track the `is a Framebuffer` state ourselves.
    bool mIsFB;
#endif

    ////

    WebGLFBAttachPoint mDepthAttachment;
    WebGLFBAttachPoint mStencilAttachment;
    WebGLFBAttachPoint mDepthStencilAttachment;

    // In theory, this number can be unbounded based on the driver. However, no driver
    // appears to expose more than 8. We might as well stop there too, for now.
    // (http://opengl.gpuinfo.org/gl_stats_caps_single.php?listreportsbycap=GL_MAX_COLOR_ATTACHMENTS)
    static const size_t kMaxColorAttachments = 8; // jgilbert's MacBook Pro exposes 8.
    WebGLFBAttachPoint mColorAttachments[kMaxColorAttachments];

    ////

    std::vector<const WebGLFBAttachPoint*> mColorDrawBuffers; // Non-null
    const WebGLFBAttachPoint* mColorReadBuffer; // Null if NONE

    ////

    struct ResolvedData {
        // IsFeedback
        std::vector<const WebGLFBAttachPoint*> texDrawBuffers; // Non-null
        std::set<WebGLFBAttachPoint::Ordered> drawSet;
        std::set<WebGLFBAttachPoint::Ordered> readSet;

        explicit ResolvedData(const WebGLFramebuffer& parent);
    };

    mutable UniquePtr<const ResolvedData> mResolvedCompleteData;

    ////

public:
    NS_INLINE_DECL_CYCLE_COLLECTING_NATIVE_REFCOUNTING(WebGLFramebuffer)
    NS_DECL_CYCLE_COLLECTION_SCRIPT_HOLDER_NATIVE_CLASS(WebGLFramebuffer)

    WebGLFramebuffer(WebGLContext* webgl, GLuint fbo);

    WebGLContext* GetParentObject() const { return mContext; }
    virtual JSObject* WrapObject(JSContext* cx, JS::Handle<JSObject*> givenProto) override;

private:
    ~WebGLFramebuffer() {
        DeleteOnce();
    }

public:
    void Delete();

    ////

    bool HasDuplicateAttachments() const;
    bool HasDefinedAttachments() const;
    bool HasIncompleteAttachments(nsCString* const out_info) const;
    bool AllImageRectsMatch() const;
    bool AllImageSamplesMatch() const;
    FBStatus PrecheckFramebufferStatus(nsCString* const out_info) const;

protected:
    Maybe<WebGLFBAttachPoint*> GetAttachPoint(GLenum attachment); // Fallible
    Maybe<WebGLFBAttachPoint*> GetColorAttachPoint(GLenum attachment); // Fallible
    void ResolveAttachments() const;
    void RefreshDrawBuffers() const;
    void RefreshReadBuffer() const;
    bool ResolveAttachmentData() const;

public:
    void DetachTexture(const WebGLTexture* tex);
    void DetachRenderbuffer(const WebGLRenderbuffer* rb);
    bool ValidateAndInitAttachments() const;
    bool ValidateClearBufferType(GLenum buffer, uint32_t drawBuffer,
                                 GLenum funcType) const;

    bool ValidateForColorRead(const webgl::FormatUsageInfo** out_format,
                              uint32_t* out_width, uint32_t* out_height) const;

    ////////////////
    // Getters

#define GETTER(X) const decltype(m##X)& X() const { return m##X; }

    GETTER(DepthAttachment)
    GETTER(StencilAttachment)
    GETTER(DepthStencilAttachment)
    GETTER(ColorDrawBuffers)
    GETTER(ColorReadBuffer)
    GETTER(ResolvedCompleteData)

#undef GETTER

    const auto& ColorAttachment0() const {
        return mColorAttachments[0];
    }

    const auto& AnyDepthAttachment() const {
        if (mDepthStencilAttachment.IsDefined())
            return mDepthStencilAttachment;
        return mDepthAttachment;
    }

    const auto& AnyStencilAttachment() const {
        if (mDepthStencilAttachment.IsDefined())
            return mDepthStencilAttachment;
        return mStencilAttachment;
    }

    ////////////////
    // Invalidation

    bool IsResolvedComplete() const { return bool(mResolvedCompleteData); }
    void InvalidateFramebufferStatus();
    void RefreshResolvedData();

    ////////////////
    // WebGL funcs

    bool IsCheckFramebufferStatusComplete() const {
        return CheckFramebufferStatus() == LOCAL_GL_FRAMEBUFFER_COMPLETE;
    }

    FBStatus CheckFramebufferStatus() const;
    void FramebufferRenderbuffer(GLenum attachment, GLenum rbtarget,
                                 WebGLRenderbuffer* rb);
    void FramebufferTexture2D(GLenum attachment,
                              GLenum texImageTarget, WebGLTexture* tex, GLint level);
    void FramebufferTextureLayer(GLenum attachment,
                                 WebGLTexture* tex, GLint level, GLint layer);
    void DrawBuffers(const dom::Sequence<GLenum>& buffers);
    void ReadBuffer(GLenum attachPoint);

    JS::Value GetAttachmentParameter(JSContext* cx, GLenum target,
                                     GLenum attachment, GLenum pname,
                                     ErrorResult* const out_error);

    static void BlitFramebuffer(WebGLContext* webgl,
                                GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                                GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
                                GLbitfield mask, GLenum filter);
};

} // namespace mozilla

#endif // WEBGL_FRAMEBUFFER_H_
