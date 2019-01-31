/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define VECS_PER_SPECIFIC_BRUSH 1

#include shared,prim_shared,brush

// TODO(gw): Consider whether we should even have separate shader compilations
//           for the various YUV modes. To save on the number of shaders we
//           need to compile, it might be worth just doing this as an
//           uber-shader instead.
// TODO(gw): Regardless of the above, we should remove the separate shader
//           compilations for the different color space matrix below. That
//           can be provided by a branch in the VS and pushed through the
//           interpolators, or even as a uniform that breaks batches, rather
//           that needing to compile to / select a different shader when
//           there is a different color space.

varying vec2 vLocalPos;

varying vec3 vUv_Y;
flat varying vec4 vUvBounds_Y;
 varying vec3 vUv_U;
flat varying vec4 vUvBounds_U;
 varying vec3 vUv_V;
flat varying vec4 vUvBounds_V;
 varying vec3 vUv_UV;
flat varying vec4 vUvBounds_UV;
 varying vec3 vUv_YUV;
flat varying vec4 vUvBounds_YUV;

#define YUV_PLANAR        0U
#define YUV_NV12          1U
#define YUV_INTERLEAVED   2U

#define YUV_REC709        1U

#ifdef WR_FEATURE_TEXTURE_RECT
    #define TEX_SIZE(sampler) vec2(1.0)
#else
    #define TEX_SIZE(sampler) vec2(textureSize(sampler, 0).xy)
#endif

flat varying float vCoefficient;

#ifdef WR_VERTEX_SHADER
void write_uv_rect(
    int resource_id,
    vec2 f,
    vec2 texture_size,
    out vec3 uv,
    out vec4 uv_bounds
) {
    ImageResource res = fetch_image_resource(resource_id);
    vec2 uv0 = res.uv_rect.p0;
    vec2 uv1 = res.uv_rect.p1;

    uv.xy = mix(uv0, uv1, f);
    uv.z = res.layer;

    uv_bounds = vec4(uv0 + vec2(0.5), uv1 - vec2(0.5));

    #ifndef WR_FEATURE_TEXTURE_RECT
        uv.xy /= texture_size;
        uv_bounds /= texture_size.xyxy;
    #endif
}

struct YuvPrimitive {
    float coefficient;
};

YuvPrimitive fetch_yuv_primitive(int address) {
    vec4 data = fetch_from_gpu_cache_1(address);
    return YuvPrimitive(data.x);
}

void brush_vs(
    VertexInfo vi,
    int prim_address,
    RectWithSize local_rect,
    RectWithSize segment_rect,
    ivec4 user_data,
    mat4 transform,
    PictureTask pic_task,
    int brush_flags,
    vec4 unused
) {
    vec2 f = (vi.local_pos - local_rect.p0) / local_rect.size;

    YuvPrimitive prim = fetch_yuv_primitive(prim_address);
    vCoefficient = prim.coefficient;

    if (alpha_pass) {
        vLocalPos = vi.local_pos;
    } else {
        vLocalPos = vec2(0.0);
    }

    if (yuv_format == YUV_PLANAR) {
        write_uv_rect(user_data.x, f, TEX_SIZE(sColor0), vUv_Y, vUvBounds_Y);
        write_uv_rect(user_data.y, f, TEX_SIZE(sColor1), vUv_U, vUvBounds_U);
        write_uv_rect(user_data.z, f, TEX_SIZE(sColor2), vUv_V, vUvBounds_V);

        vUv_UV = vec3(0.0);
        vUvBounds_UV = vec4(0.0);
        vUv_YUV = vec3(0.0);
        vUvBounds_YUV = vec4(0.0);
    } else if (yuv_format == YUV_NV12) {
        write_uv_rect(user_data.x, f, TEX_SIZE(sColor0), vUv_Y, vUvBounds_Y);
        write_uv_rect(user_data.y, f, TEX_SIZE(sColor1), vUv_UV, vUvBounds_UV);
        vUv_U = vec3(0.0);
        vUvBounds_U = vec4(0.0);
        vUv_V = vec3(0.0);
        vUvBounds_V = vec4(0.0);
        vUv_YUV = vec3(0.0);
        vUvBounds_YUV = vec4(0.0);
    } else if (yuv_format == YUV_INTERLEAVED) {
        write_uv_rect(user_data.x, f, TEX_SIZE(sColor0), vUv_YUV, vUvBounds_YUV);
        vUv_U = vec3(0.0);
        vUvBounds_U = vec4(0.0);
        vUv_V = vec3(0.0);
        vUvBounds_V = vec4(0.0);
        vUv_Y = vec3(0.0);
        vUvBounds_Y = vec4(0.0);
        vUv_UV = vec3(0.0);
        vUvBounds_UV = vec4(0.0);
    }

}
#endif

#ifdef WR_FRAGMENT_SHADER

// The constants added to the Y, U and V components are applied in the fragment shader.
// From Rec601:
// [R]   [1.1643835616438356,  0.0,                 1.5960267857142858   ]   [Y -  16]
// [G] = [1.1643835616438358, -0.3917622900949137, -0.8129676472377708   ] x [U - 128]
// [B]   [1.1643835616438356,  2.017232142857143,   8.862867620416422e-17]   [V - 128]
//
// For the range [0,1] instead of [0,255].
//
// The matrix is stored in column-major.
const mat3 YuvColorMatrix601 = mat3(
    1.16438,  1.16438, 1.16438,
    0.0,     -0.39176, 2.01723,
    1.59603, -0.81297, 0.0
);

// From Rec709:
// [R]   [1.1643835616438356,  4.2781193979771426e-17, 1.7927410714285714]   [Y -  16]
// [G] = [1.1643835616438358, -0.21324861427372963,   -0.532909328559444 ] x [U - 128]
// [B]   [1.1643835616438356,  2.1124017857142854,     0.0               ]   [V - 128]
//
// For the range [0,1] instead of [0,255]:
//
// The matrix is stored in column-major.
const mat3 YuvColorMatrix709 = mat3(
    1.16438,  1.16438,  1.16438,
    0.0    , -0.21325,  2.11240,
    1.79274, -0.53291,  0.0
);


Fragment brush_fs() {
    vec3 yuv_value;

    if (yuv_format == YUV_PLANAR) {
        // The yuv_planar format should have this third texture coordinate.
        vec2 uv_y = clamp(vUv_Y.xy, vUvBounds_Y.xy, vUvBounds_Y.zw);
        vec2 uv_u = clamp(vUv_U.xy, vUvBounds_U.xy, vUvBounds_U.zw);
        vec2 uv_v = clamp(vUv_V.xy, vUvBounds_V.xy, vUvBounds_V.zw);
        yuv_value.x = TEX_SAMPLE(sColor0, vec3(uv_y, vUv_Y.z)).r;
        yuv_value.y = TEX_SAMPLE(sColor1, vec3(uv_u, vUv_U.z)).r;
        yuv_value.z = TEX_SAMPLE(sColor2, vec3(uv_v, vUv_V.z)).r;
    } else if (yuv_format == YUV_NV12) {
        vec2 uv_y = clamp(vUv_Y.xy, vUvBounds_Y.xy, vUvBounds_Y.zw);
        vec2 uv_uv = clamp(vUv_UV.xy, vUvBounds_UV.xy, vUvBounds_UV.zw);
        yuv_value.x = TEX_SAMPLE(sColor0, vec3(uv_y, vUv_Y.z)).r;
        yuv_value.yz = TEX_SAMPLE(sColor1, vec3(uv_uv, vUv_UV.z)).rg;
    } else if (yuv_format == YUV_INTERLEAVED) {
        // "The Y, Cb and Cr color channels within the 422 data are mapped into
        // the existing green, blue and red color channels."
        // https://www.khronos.org/registry/OpenGL/extensions/APPLE/APPLE_rgb_422.txt
        vec2 uv_y = clamp(vUv_YUV.xy, vUvBounds_YUV.xy, vUvBounds_YUV.zw);
        yuv_value = TEX_SAMPLE(sColor0, vec3(uv_y, vUv_YUV.z)).gbr;
    } else {
        yuv_value = vec3(0.0);
    }

    // See the YuvColorMatrix definition for an explanation of where the constants come from.
    vec3 rgb = yuv_value - vec3(0.06275, 0.50196, 0.50196);
    if (yuv_color_space == YUV_REC709) {
        rgb = YuvColorMatrix709 * rgb;
    } else {
        rgb = YuvColorMatrix601 * rgb;
    }
    vec4 color = vec4(rgb, 1.0);

    if (alpha_pass) {
        color *= init_transform_fs(vLocalPos);
    }

    return Fragment(color);
}
#endif
