/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

varying vec3 vUv;
flat varying vec4 vUvRect;

#ifdef WR_VERTEX_SHADER

in int aScaleRenderTaskAddress;
in int aScaleSourceTaskAddress;

struct ScaleTask {
    RenderTaskCommonData common_data;
};

ScaleTask fetch_scale_task(int address) {
    RenderTaskData task_data = fetch_render_task_data(address);

    ScaleTask task = ScaleTask(task_data.common_data);

    return task;
}

void main(void) {
    ScaleTask scale_task = fetch_scale_task(aScaleRenderTaskAddress);
    RenderTaskCommonData src_task = fetch_render_task_common_data(aScaleSourceTaskAddress);

    RectWithSize src_rect = src_task.task_rect;
    RectWithSize target_rect = scale_task.common_data.task_rect;

    vec2 texture_size;
    if (color_target) {
        texture_size = vec2(textureSize(sPrevPassColor, 0).xy);
    } else {
        texture_size = vec2(textureSize(sPrevPassAlpha, 0).xy);
    }

    vUv.z = src_task.texture_layer_index;

    vUvRect = vec4(src_rect.p0 + vec2(0.5),
                   src_rect.p0 + src_rect.size - vec2(0.5)) / texture_size.xyxy;

    vec2 pos = target_rect.p0 + target_rect.size * aPosition.xy;
    vUv.xy = (src_rect.p0 + src_rect.size * aPosition.xy) / texture_size;

    gl_Position = uTransform * vec4(pos, 0.0, 1.0);
}

#endif

#ifdef WR_FRAGMENT_SHADER

void main(void) {
    vec2 st = clamp(vUv.xy, vUvRect.xy, vUvRect.zw);
    if (color_target) {
        oFragColor = vec4(texture(sPrevPassColor, vec3(st, vUv.z)));
    } else {
        oFragColor = vec4(texture(sPrevPassAlpha, vec3(st, vUv.z)).r);
    }

}

#endif
