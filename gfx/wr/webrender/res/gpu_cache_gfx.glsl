/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

layout(set = 1, binding = 5, std430) buffer sGpuCache
{
    vec4 gpu_cache[];
};

#define VECS_PER_IMAGE_RESOURCE     2



int get_gpu_cache_address(ivec2 address) {
    return int(address.y * WR_MAX_VERTEX_TEXTURE_WIDTH + address.x);
}

// TODO(zakorgy): Until we need to adopt to the GL shaders we need these helper `fetch_from_xxx` functions.
// Later we can leave these methods and acces `gpu_cache` directly using the address as index.

vec4[2] fetch_from_gpu_cache_2_direct(ivec2 addr) {
    int address = get_gpu_cache_address(addr);
    return vec4[2](
        gpu_cache[address],
        gpu_cache[address + 1]
    );
}

vec4[2] fetch_from_gpu_cache_2(int address) {
    return vec4[2](
        gpu_cache[address],
        gpu_cache[address + 1]
    );
}

#ifdef WR_VERTEX_SHADER

vec4[8] fetch_from_gpu_cache_8(int address) {
    return vec4[8](
        gpu_cache[address],
        gpu_cache[address + 1],
        gpu_cache[address + 2],
        gpu_cache[address + 3],
        gpu_cache[address + 4],
        gpu_cache[address + 5],
        gpu_cache[address + 6],
        gpu_cache[address + 7]
    );
}

vec4[3] fetch_from_gpu_cache_3(int address) {
    return vec4[3](
        gpu_cache[address],
        gpu_cache[address + 1],
        gpu_cache[address + 2]
    );
}

vec4[3] fetch_from_gpu_cache_3_direct(ivec2 addr) {
    int address = get_gpu_cache_address(addr);
    return vec4[3](
        gpu_cache[address],
        gpu_cache[address + 1],
        gpu_cache[address + 2]
    );
}

vec4[4] fetch_from_gpu_cache_4_direct(ivec2 addr) {
    int address = get_gpu_cache_address(addr);
    return vec4[4](
        gpu_cache[address],
        gpu_cache[address + 1],
        gpu_cache[address + 2],
        gpu_cache[address + 3]
    );
}

vec4[4] fetch_from_gpu_cache_4(int address) {
    return vec4[4](
        gpu_cache[address],
        gpu_cache[address + 1],
        gpu_cache[address + 2],
        gpu_cache[address + 3]
    );
}

vec4 fetch_from_gpu_cache_1_direct(ivec2 addr) {
    int address = get_gpu_cache_address(addr);
    return gpu_cache[address];
}

vec4 fetch_from_gpu_cache_1(int address) {
    return gpu_cache[address];
}

//TODO: image resource is too specific for this module

struct ImageResource {
    RectWithEndpoint uv_rect;
    float layer;
    vec3 user_data;
};

ImageResource fetch_image_resource(int address) {
    //Note: number of blocks has to match `renderer::BLOCKS_PER_UV_RECT`
    vec4 data[2] = fetch_from_gpu_cache_2(address);
    RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
    return ImageResource(uv_rect, data[1].x, data[1].yzw);
}

ImageResource fetch_image_resource_direct(ivec2 address) {
    vec4 data[2] = fetch_from_gpu_cache_2_direct(address);
    RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
    return ImageResource(uv_rect, data[1].x, data[1].yzw);
}

// Fetch optional extra data for a texture cache resource. This can contain
// a polygon defining a UV rect within the texture cache resource.
struct ImageResourceExtra {
    vec2 st_tl;
    vec2 st_tr;
    vec2 st_bl;
    vec2 st_br;
};

ImageResourceExtra fetch_image_resource_extra(int address) {
    vec4 data[2] = fetch_from_gpu_cache_2(address + VECS_PER_IMAGE_RESOURCE);
    return ImageResourceExtra(
        data[0].xy,
        data[0].zw,
        data[1].xy,
        data[1].zw
    );
}

#endif //WR_VERTEX_SHADER
