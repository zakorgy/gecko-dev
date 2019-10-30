# U-Szeged version

This version of WebRender is using next-gen backends (vulkan/dx12/metal) provided by the [gfx-hal](https://crates.io/crates/gfx-hal) API.

The current state is **experimental**.
At the moment the **vulkan, dx12,** and **metal** backends are usable, but in performance they are behind the original (this year's goal is to change this).

By default this WebRender builds with gfx-hal, but we kept the original OpenGL backend and you can enable it with the **gleam** feature in WebRender and with the **gl** feature in Wrench.

To run **Wrench** reftests e.g. with vulkan backend, use the following comand:
```
cd wrench
cargo run --features=vulkan reftest
```

It was tested on Linux (Ubuntu 18.04), Windows 10 and macOS (Mojave).

# WebRender

[![Version](https://img.shields.io/crates/v/webrender.svg)](https://crates.io/crates/webrender)

GPU renderer for the Web content, used by Servo.

Note that the canonical home for this code is in gfx/wr folder of the
mozilla-central repository at https://hg.mozilla.org/mozilla-central. The
Github repository at https://github.com/servo/webrender should be considered
a downstream mirror, although it contains additional metadata (such as Github
wiki pages) that do not exist in mozilla-central. Pull requests against the
Github repository are still being accepted, although once reviewed, they will
be landed on mozilla-central first and then mirrored back. If you are familiar
with the mozilla-central contribution workflow, filing bugs in
[Bugzilla](https://bugzilla.mozilla.org/enter_bug.cgi?product=Core&component=Graphics%3A%20WebRender)
and submitting patches there would be preferred.

## Update as a Dependency
After updating shaders in WebRender, go to servo and:

  * Go to the servo directory and do ./mach update-cargo -p webrender
  * Create a pull request to servo


## Use WebRender with Servo
To use a local copy of WebRender with servo, go to your servo build directory and:

  * Edit Cargo.toml
  * Add at the end of the file:

```
[patch."https://github.com/servo/webrender"]
"webrender" = { path = "<path>/webrender" }
"webrender_api" = { path = "<path>/webrender_api" }
```

where `<path>` is the path to your local copy of WebRender.

  * Build as normal

## Documentation

The Wiki has a [few pages](https://github.com/servo/webrender/wiki/) describing the internals and conventions of WebRender.

## Testing

Tests run using OSMesa to get consistent rendering across platforms.

Still there may be differences depending on font libraries on your system, for
example.

See [this gist](https://gist.github.com/finalfantasia/129cae811e02bf4551ac) for
how to make the text tests useful in Fedora, for example.
