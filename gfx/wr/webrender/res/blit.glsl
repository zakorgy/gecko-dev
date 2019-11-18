/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#include shared,shared_other

varying vec4 vUv;

#ifdef WR_VERTEX_SHADER
in vec2 aOffset;
in vec2 aExtent;
in float aZ;
in float aLevel;

void main(void) {
    vec2 coord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

    gl_Position = vec4(vec2(-1.0, -1.0) + coord * vec2(2.0, 2.0), 0.0, 1.0);
    vUv = vec4(aOffset + coord * aExtent, aZ, aLevel);
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
#if defined(WR_FEATURE_TEXTURE_EXTERNAL) || defined(WR_FEATURE_TEXTURE_RECT) || defined(WR_FEATURE_TEXTURE_2D)
    oFragColor = texture(sColor0, vUv.xy);
#else
    oFragColor = textureLod(sColor0, vUv.xyz, vUv.w);
#endif
}
#endif
