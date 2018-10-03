/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use gfx_hal::pso::{AttributeDesc, DescriptorRangeDesc, DescriptorSetLayoutBinding};
use gfx_hal::pso::{DescriptorType, Element, ShaderStageFlags, VertexBufferDesc};
use gfx_hal::format::Format;
use ron::de::from_str;
use ron::ser::{to_string_pretty, PrettyConfig};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::mem;
use std::path::Path;
use std::process::{self, Command, Stdio};
use vertex_types::*;

const SHADER_IMPORT: &str = "#include ";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";
const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_PREFIX: &str = "#define WR_MAX_VERTEX_TEXTURE_WIDTH 1024U\n";
const SHADER_VERSION_VK: &'static str = "#version 450\n";
const VK_EXTENSIONS: &'static str = "#extension GL_ARB_shading_language_420pack : enable\n\
                                     #extension GL_ARB_explicit_attrib_location : enable\n\
                                     #extension GL_ARB_separate_shader_objects : enable\n";

// https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#features-limits
const MAX_INPUT_ATTRIBUTES: u32 = 16;

#[derive(Deserialize)]
struct Shader {
    name: String,
    source_name: String,
    features: Vec<String>,
}

#[derive(Serialize)]
struct PipelineRequirements {
    attribute_descriptors: Vec<AttributeDesc>,
    bindings_map: HashMap<String, usize>,
    descriptor_range_descriptors: Vec<DescriptorRangeDesc>,
    descriptor_set_layouts: Vec<DescriptorSetLayoutBinding>,
    vertex_buffer_descriptors: Vec<VertexBufferDesc>,
}

fn create_shaders(out_dir: &str, shaders: &HashMap<String, String>) -> Vec<String> {
    fn get_shader_source(shader_name: &str, shaders: &HashMap<String, String>) -> Option<String> {
        if let Some(shader_file) = shaders.get(shader_name) {
            let shader_file_path = Path::new(shader_file);
            if let Ok(mut shader_source_file) = File::open(shader_file_path) {
                let mut source = String::new();
                shader_source_file.read_to_string(&mut source).unwrap();
                Some(source)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn parse_shader_source(source: &str, shaders: &HashMap<String, String>, output: &mut String) {
        for line in source.lines() {
            if line.starts_with(SHADER_IMPORT) {
                let imports = line[SHADER_IMPORT.len() ..].split(",");
                // For each import, get the source, and recurse.
                for import in imports {
                    if let Some(include) = get_shader_source(import, shaders) {
                        parse_shader_source(&include, shaders, output);
                    }
                }
            } else {
                output.push_str(line);
                output.push_str("\n");
            }
        }
    }

    fn build_shader_strings(
        base_filename: &str,
        features: &str,
        shaders: &HashMap<String, String>,
    ) -> (String, String) {
        // Construct a list of strings to be passed to the shader compiler.
        let mut vs_source = String::new();
        let mut fs_source = String::new();

        vs_source.push_str(SHADER_VERSION_VK);
        fs_source.push_str(SHADER_VERSION_VK);

        // Define a constant depending on whether we are compiling VS or FS.
        vs_source.push_str(SHADER_KIND_VERTEX);
        fs_source.push_str(SHADER_KIND_FRAGMENT);

        // Add any defines that were passed by the caller.
        vs_source.push_str(features);
        fs_source.push_str(features);

        // Parse the main .glsl file, including any imports
        // and append them to the list of sources.
        let mut shared_result = String::new();
        if let Some(shared_source) = get_shader_source(base_filename, shaders) {
            parse_shader_source(&shared_source, shaders, &mut shared_result);
        }

        //vs_source.push_str(SHADER_LINE_MARKER);
        vs_source.push_str(&shared_result);
        //fs_source.push_str(SHADER_LINE_MARKER);
        fs_source.push_str(&shared_result);

        (vs_source, fs_source)
    }

    let mut file = File::open("shaders.ron").expect("Unable to open shaders.ron");
    let mut source = String::new();
    file.read_to_string(&mut source).unwrap();
    let shader_configs: Vec<Shader> = from_str(&source).expect("Unable to parse shaders.ron");

    let mut file_names = Vec::new();
    for shader in &shader_configs {
        for config in &shader.features {
            let mut features = String::new();

            features.push_str(SHADER_PREFIX);
            features.push_str(format!("//Source: {}.glsl\n", shader.source_name).as_str());

            let mut file_name_postfix = String::new();
            for feature in config.split(",") {
                if !feature.is_empty() {
                    features.push_str(&format!("#define WR_FEATURE_{}\n", feature));
                    if shader.name == shader.source_name {
                        file_name_postfix
                            .push_str(&format!("_{}", feature.to_lowercase().as_str()));
                    }
                }
            }

            features.push_str(VK_EXTENSIONS);

            let (mut vs_source, mut fs_source) =
                build_shader_strings(&shader.source_name, &features, shaders);

            let mut filename = shader.name.clone();
            filename.push_str(file_name_postfix.as_str());
            let (mut vs_name, mut fs_name) = (filename.clone(), filename);
            vs_name.push_str(".vert");
            fs_name.push_str(".frag");
            println!("vs_name = {}, shader.name = {}", vs_name, shader.name);
            let (vs_file_path, fs_file_path) = (
                Path::new(out_dir).join(vs_name.clone()),
                Path::new(out_dir).join(fs_name.clone()),
            );
            let (mut vs_file, mut fs_file) = (
                File::create(vs_file_path).unwrap(),
                File::create(fs_file_path).unwrap(),
            );
            write!(vs_file, "{}", vs_source).unwrap();
            write!(fs_file, "{}", fs_source).unwrap();
            file_names.push(vs_name);
            file_names.push(fs_name);
        }
    }
    file_names
}

fn process_glsl_for_spirv(file_path: &Path, file_name: &str) -> Option<PipelineRequirements> {
    let mut new_data = String::new();
    let mut binding = 1; // 0 is reserved for Locals
    let mut in_location = 0;
    let mut out_location = 0;
    let mut varying_location = 0;
    let mut attribute_descriptors: Vec<AttributeDesc> = Vec::new();
    let mut bindings_map: HashMap<String, usize> = HashMap::new();
    let mut descriptor_set_layouts: Vec<DescriptorSetLayoutBinding> = Vec::new();
    let mut vertex_offset = 0;
    let mut instance_offset = 0;
    // Since the .vert and .frag files for the same shader use the same layout qualifiers
    // we extract layout datas from .vert files only.
    let write_ron = file_name.ends_with(".vert");

    // Mapping from glsl sampler variable name to a tuple,
    // in which the first item is the corresponding expression used in vulkan glsl files,
    // the second is the layout binding index.
    // e.g.: sColor0 -> ("sampler2DArray(tColor0, sColor0)", 0)
    let mut sampler_mapping: HashMap<String, (String, i8)> = HashMap::new();

    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let mut l = line.unwrap();
        let trimmed = l.trim_left();

        // Replace uniforms in shader:
        //      Sampler uniforms are splitted to texture + sampler.
        //      Other types occur in a group. These are replaced with a static string for now.
        if trimmed.starts_with("uniform") {
            if trimmed.contains("sampler") {
                let code = split_code(trimmed);
                replace_sampler_definition_with_texture_and_sampler(
                    &mut binding,
                    code,
                    &mut descriptor_set_layouts,
                    &mut bindings_map,
                    &mut new_data,
                    &mut sampler_mapping,
                    write_ron,
                );

                // Replace non-sampler uniforms with a structure.
                // We just place a predifened structure to the position of the last non-uniform
                // variable (uDevicePixelRatio), since all shader uses the same variables.
            } else if trimmed.starts_with("uniform float uDevicePixelRatio") {
                replace_non_sampler_uniforms(&mut new_data);
                if write_ron {
                    add_locals_to_descriptor_set_layout(&mut descriptor_set_layouts, &mut bindings_map);
                }
            }

            // Adding location info for non-uniform variables.
        } else if trimmed.contains(';') && // If the line contains a semicolon we assume it is a variable declaration.
            (trimmed.starts_with("varying ") || trimmed.starts_with("flat varying ")
                || trimmed.starts_with("in ") || trimmed.starts_with("out "))
            {
                extend_non_uniform_variables_with_location_info(
                    &mut attribute_descriptors,
                    &mut in_location,
                    &mut out_location,
                    &mut varying_location,
                    &mut instance_offset,
                    trimmed,
                    &mut new_data,
                    &mut vertex_offset,
                    write_ron,
                );
                // Replacing sampler variables with the corresponding expression from sampler_mapping.
            } else if l.contains("TEX_SAMPLE(") || l.contains("TEXEL_FETCH(") || l.contains("TEX_SIZE(")
            || l.contains("texelFetch(") || l.contains("texture(")
            || l.contains("textureLod(") || l.contains("textureSize(")
            {
                let mut line = l.clone();
                for (k, v) in sampler_mapping.iter() {
                    if line.contains(k) {
                        line = line.replace(k, &v.0);
                    }
                }
                // Change the sampling coordinates of the Dither texture. Basically it's a vertical flip.
                if line.contains("sDither") {
                    line = line.replace("pos", "ivec2(pos.x, matrix_mask - pos.y)");
                }
                new_data.push_str(&line);
                new_data.push('\n');
            } else {
            new_data.push_str(&l);
            new_data.push('\n');
        }
    }
    let mut file = File::create(file_path).unwrap();
    file.write(new_data.as_bytes()).unwrap();
    if write_ron {
        let descriptor_range_descriptors = create_desciptor_range_descriptors(sampler_mapping.len());
        let vertex_buffer_descriptors = create_vertex_buffer_descriptors(file_name);
        let pipeline_requirmenets = PipelineRequirements {
            attribute_descriptors,
            bindings_map,
            descriptor_set_layouts,
            descriptor_range_descriptors,
            vertex_buffer_descriptors,
        };
        return Some(pipeline_requirmenets);
    }
    None
}

fn split_code(line: &str) -> Vec<&str> {
    line.split(';')
        .collect::<Vec<&str>>()[0]
        .split(' ')
        .collect::<Vec<&str>>()
}

fn replace_sampler_definition_with_texture_and_sampler(
    binding: &mut usize,
    code: Vec<&str>,
    descriptor_set_layouts: &mut Vec<DescriptorSetLayoutBinding>,
    bindings_map: &mut HashMap<String, usize>,
    new_data: &mut String,
    sampler_mapping: &mut HashMap<String, (String, i8)>,
    write_ron: bool,
) {
    // Get the name of the sampler.
    let (sampler_name, code) = code.split_last().unwrap();

    // Get the exact type of the sampler.
    let (sampler_type, code) = code.split_last().unwrap();
    let mut code_str = String::new();
    for i in 0 .. code.len() {
        code_str.push_str(code[i]);
        code_str.push(' ');
    }

    let texture_type = sampler_type.replace("sampler", "texture");
    let texture_name = sampler_name.replacen('s', "t", 1);

    // If the sampler is redefined we use the same binding index, but the sampler type gets updated.
    // Note: This should be handled by parsing the defines and use them to determine which definition is correct.
    //       Since only sColor samplers are involved in this case, the last definition for these samplers
    //       which uses sampler/texture2DArray is good for us to go with.
    if let Some(&(_, binding)) = sampler_mapping.get(*sampler_name) {
        let mut layout_str = format!(
            "layout(set = 0, binding = {}) {}{} {};\n",
            binding, code_str, texture_type, texture_name
        );
        new_data.push_str(&layout_str);
        sampler_mapping.insert(
            String::from(*sampler_name),
            (
                format!("{}({}, {})", sampler_type, texture_name, sampler_name),
                binding,
            ),
        );

        layout_str = format!(
            "layout(set = 0, binding = {}) {}sampler {};\n",
            binding + 1,
            code_str,
            sampler_name
        );
        new_data.push_str(&layout_str);

        // Replace sampler definition with a texture and a sampler.
    } else {
        let mut layout_str = format!(
            "layout(set = 0, binding = {}) {}{} {};\n",
            binding, code_str, texture_type, texture_name
        );
        if write_ron {
            descriptor_set_layouts.push(
                DescriptorSetLayoutBinding {
                    binding: *binding as u32,
                    ty: DescriptorType::SampledImage,
                    count: 1,
                    stage_flags: ShaderStageFlags::ALL,
                    immutable_samplers: false,
                });
            bindings_map.insert(texture_name.clone(), *binding);
        }
        new_data.push_str(&layout_str);
        sampler_mapping.insert(
            String::from(*sampler_name),
            (
                format!("{}({}, {})", sampler_type, texture_name, sampler_name),
                *binding as i8,
            ),
        );
        *binding += 1;

        layout_str = format!(
            "layout(set = 0, binding = {}) {}sampler {};\n",
            binding, code_str, sampler_name
        );
        if write_ron {
            descriptor_set_layouts.push(
                DescriptorSetLayoutBinding {
                    binding: *binding as u32,
                    ty: DescriptorType::Sampler,
                    count: 1,
                    stage_flags: ShaderStageFlags::ALL,
                    immutable_samplers: false,
                });
            bindings_map.insert(String::from(*sampler_name), *binding);
        }
        new_data.push_str(&layout_str);
        *binding += 1;
    }
}

fn replace_non_sampler_uniforms(new_data: &mut String) {
    new_data.push_str(
        "\tlayout(set = 0, binding = 0) uniform Locals {\n\
         \t\tuniform mat4 uTransform;       // Orthographic projection\n\
         \t\tuniform float uDevicePixelRatio;\n\
         \t\t// A generic uniform that shaders can optionally use to configure\n\
         \t\t// an operation mode for this batch.\n\
         \t\tuniform int uMode;\n\
         \t};\n",
    );
}

fn add_locals_to_descriptor_set_layout(
    descriptor_set_layouts: &mut Vec<DescriptorSetLayoutBinding>,
    bindings_map: &mut HashMap<String, usize>,
) {
    descriptor_set_layouts.push(
        DescriptorSetLayoutBinding {
            binding: 0,
            ty: DescriptorType::UniformBuffer,
            count: 1,
            stage_flags: ShaderStageFlags::VERTEX,
            immutable_samplers: false,
        }
    );
    bindings_map.insert("Locals".to_owned(), 0);
}

fn extend_non_uniform_variables_with_location_info(
    attribute_descriptors: &mut Vec<AttributeDesc>,
    in_location: &mut u32,
    out_location: &mut u32,
    varying_location: &mut u32,
    instance_offset: &mut u32,
    line: &str,
    new_data: &mut String,
    vertex_offset: &mut u32,
    write_ron: bool,
) {
    let layout_str;
    let location_size = calculate_location_size(line);
    if line.starts_with("in") {
        layout_str = format!("layout(location = {}) {}\n", in_location, line);
        if write_ron {
            add_attribute_descriptors(
                attribute_descriptors,
                in_location,
                instance_offset,
                line,
                vertex_offset,
            );
        }
        *in_location += location_size;
        assert!(*in_location < MAX_INPUT_ATTRIBUTES);
    } else if line.starts_with("out") {
        layout_str = format!("layout(location = {}) {}\n", out_location, line);
        *out_location += location_size;
    } else {
        layout_str = format!("layout(location = {}) {}\n", varying_location, line);
        *varying_location += location_size;
    }
    new_data.push_str(&layout_str)
}

fn calculate_location_size(line: &str) -> u32 {
    let mut multiplier = 1;
    if line.ends_with("];") {
        multiplier = line.split(|c| c == '[' || c == ']').nth(1).unwrap().parse::<u32>().unwrap();
    }
    let res = match line.split_whitespace().rev().nth(1).unwrap() {
        "mat4" => 4,
        "mat3" => 3,
        _ => 1,
    };
    res * multiplier
}

fn add_attribute_descriptors(
    attribute_descriptors: &mut Vec<AttributeDesc>,
    in_location: &mut u32,
    instance_offset: &mut u32,
    line: &str,
    vertex_offset: &mut u32,
) {
    let def = split_code(line);
    let var_name = def[2].trim_right_matches(';');
    let (format, offset) = match def[1] {
        "int" => (Format::R32Int, 4),
        "ivec4" => (Format::Rgba32Int, 16),
        "vec2" => (Format::Rg32Float, 8),
        "vec3" => (Format::Rgb32Float, 12),
        "vec4" => (Format::Rgba32Float, 16),
        _ => unimplemented!(),
    };
    match var_name {
        "aColor" | "aColorTexCoord" | "aPosition" => {
            attribute_descriptors.push(
                AttributeDesc {
                    location: *in_location,
                    binding: 0,
                    element: Element {
                        format,
                        offset: *vertex_offset,
                    }
                }
            );
            *vertex_offset += offset;
        }
        _ => {
            attribute_descriptors.push(
                AttributeDesc {
                    location: *in_location,
                    binding: 1,
                    element: Element {
                        format,
                        offset: *instance_offset,
                    }
                }
            );
            *instance_offset += offset;
        }
    };
}

fn create_desciptor_range_descriptors(count: usize) -> Vec<DescriptorRangeDesc> {
    vec![
        DescriptorRangeDesc {
            ty: DescriptorType::SampledImage,
            count,
        },
        DescriptorRangeDesc {
            ty: DescriptorType::Sampler,
            count,
        },
        DescriptorRangeDesc {
            ty: DescriptorType::UniformBuffer,
            count: 1,
        },
    ]
}

fn create_vertex_buffer_descriptors(file_name: &str) -> Vec<VertexBufferDesc> {
    let mut descriptors = vec![
        VertexBufferDesc {
            binding: 0,
            stride: mem::size_of::<Vertex>() as _,
            rate: 0,
        }
    ];
    if file_name.starts_with("cs_blur") {
        descriptors.push(
            VertexBufferDesc {
                binding: 1,
                stride: mem::size_of::<BlurInstance>() as _,
                rate: 1,
            }
        );
    } else if file_name.starts_with("cs_border") {
        descriptors.push(
            VertexBufferDesc {
                binding: 1,
                stride: mem::size_of::<BorderInstance>() as _,
                rate: 1,
            }
        );
    } else if file_name.starts_with("cs_clip") {
        descriptors.push(
            VertexBufferDesc {
                binding: 1,
                stride: mem::size_of::<ClipMaskInstance>() as _,
                rate: 1,
            }
        );
    } else if file_name.starts_with("cs_scale") {
        descriptors.push(
            VertexBufferDesc {
                binding: 1,
                stride: mem::size_of::<ScalingInstance>() as _,
                rate: 1,
            }
        );
    } else if file_name.starts_with("debug_color") {
        descriptors = vec![
            VertexBufferDesc {
                binding: 0,
                stride: mem::size_of::<DebugColorVertex>() as _,
                rate: 0,
            },
        ];
    } else if file_name.starts_with("debug_font") {
        descriptors = vec![
            VertexBufferDesc {
                binding: 0,
                stride: mem::size_of::<DebugFontVertex>() as _,
                rate: 0,
            },
        ];
        // Primitive and brush shaders
    } else {
        descriptors.push(
            VertexBufferDesc {
                binding: 1,
                stride: mem::size_of::<PrimitiveInstanceData>() as _,
                rate: 1,
            }
        );
    }
    descriptors
}

fn compile_glsl_to_spirv(file_name_vector: Vec<String>, out_dir: &str) ->  HashMap<String, PipelineRequirements> {
    let mut requirements = HashMap::new();
    for mut file_name in file_name_vector {
        let file_path = Path::new(&out_dir).join(&file_name);
        if let Some(req) = process_glsl_for_spirv(&file_path, &file_name) {
            requirements.insert(file_name.trim_right_matches(".vert").to_owned(), req);
        }
        file_name.push_str(".spv");
        let spirv_file_path = Path::new(&out_dir).join(&file_name);
        #[cfg(target_os="linux")]
        let mut glslang_cmd = Command::new(Path::new("./tools/glslang-validator-linux"));
        #[cfg(target_os="macos")]
        let mut glslang_cmd = Command::new(Path::new("./tools/glslang-validator-mac"));
        #[cfg(target_os = "windows")]
        let mut glslang_cmd = Command::new(Path::new("./tools/glslang-validator-win.exe"));
        glslang_cmd
            .arg("-V")
            .arg("-o")
            .arg(&spirv_file_path)
            .arg(&file_path);
        if glslang_cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .unwrap()
            .code()
            .unwrap() != 0
            {
                println!("Error while compiling spirv: {:?}", file_name);
                process::exit(1)
            };
        #[cfg(target_os="linux")]
        let mut spirv_val_cmd = Command::new(Path::new("./tools/spirv-val-linux"));
        #[cfg(target_os="macos")]
        let mut spirv_val_cmd = Command::new(Path::new("./tools/spirv-val-mac"));
        #[cfg(target_os = "windows")]
        let mut spirv_val_cmd = Command::new(Path::new("./tools/spirv-val-win.exe"));
        spirv_val_cmd.arg(&spirv_file_path);
        if spirv_val_cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .unwrap()
            .code()
            .unwrap() != 0
            {
                println!("Error while validating spirv shader: {:?}", file_name);
                process::exit(1)
            }
    }
    requirements
}

fn write_ron_to_file(requriements: HashMap<String, PipelineRequirements>, out_dir: &str) {
    let ron_file_path = Path::new(&out_dir).join("shader_bindings.ron");
    let mut ron_file = File::create(&ron_file_path).unwrap();
    let pretty = PrettyConfig {
        enumerate_arrays: true,
        ..PrettyConfig::default()
    };
    ron_file
        .write(
            to_string_pretty(&requriements, pretty)
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
}

pub fn gfx_main(out_dir: &str, shader_map: HashMap<String, String>) {
    println!("cargo:rerun-if-changed=shaders.ron");
    let shaders = create_shaders(&out_dir, &shader_map);
    let requirements = compile_glsl_to_spirv(shaders, &out_dir);
    write_ron_to_file(requirements, &out_dir);
}
