# Simple JPEG XL encoder implementation

[![Build/Test](https://github.com/libjxl/libjxl-tiny/actions/workflows/build_test.yml/badge.svg)](
https://github.com/libjxl/libjxl-tiny/actions/workflows/build_test.yml)

<img src="doc/jxl.svg" width="100" align="right" alt="JXL logo">

This repository contains a simpler encoder implementation of JPEG XL, aimed at
photographic images without an alpha channel. The goal is to guide hardware
implementations of the encoder where support for the full set of encoding tools
is not feasible. The color management is outside the scope of this library, the
encoder input is given as a portable float map (PFM) in the linear sRGB
colorspace, where individual sample values can be outside the [0.0, 1.0] range
for out-of-gammut colors. For more details, see [the overview of the coding tools](doc/coding_tools.md).

JPEG XL is in the final stages of standardization and its codestream and file format
are frozen.

The library API, command line options, and tools in this repository are subject
to change, however files encoded with `cjxl_tiny` conform to the JPEG XL format
specification and can be decoded with current and future `djxl` decoders or
`libjxl` decoding library.

## Quick start guide

For more details and other workflows see the "Advanced guide" below.

### Checking out the code

```bash
git clone https://github.com/libjxl/libjxl-tiny.git --recursive --shallow-submodules
```

This repository uses git submodules to handle some third party dependencies
under `third_party`, that's why is important to pass `--recursive`. If you
didn't check out with `--recursive`, or any submodule has changed, run:

```bash
git submodule update --init --recursive --depth 1 --recommend-shallow
```

The `--shallow-submodules` and `--depth 1 --recommend-shallow` options create
shallow clones which only downloads the commits requested, and is all that is
needed to build `libjxl-tiny`. Should full clones be necessary, you could always run:

```bash
git submodule foreach git fetch --unshallow
git submodule update --init --recursive
```

which pulls the rest of the commits in the submodules.

Important: If you downloaded a zip file or tarball from the web interface you
won't get the needed submodules and the code will not compile. You can download
these external dependencies from source running `./deps.sh`. The git workflow
described above is recommended instead.

### Installing dependencies

Required dependencies for compiling the code, in a Debian/Ubuntu based
distribution run:

```bash
sudo apt install cmake pkg-config
```

We recommend using a recent Clang compiler (version 7 or newer), for that
install clang and set `CC` and `CXX` variables.

```bash
sudo apt install clang
export CC=clang CXX=clang++
```

### Building

```bash
cd libjxl-tiny
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF ..
cmake --build . -- -j$(nproc)
```

The encoder tool `cjxl_tiny` will be available in the `build/encoder` directory.

### <a name="installing"></a> Installing

```bash
sudo cmake --install .
```

### Basic encoder

To encode a source image to JPEG XL with default settings:

```bash
build/encoder/cjxl_tiny input.pfm output.jxl
```

For more settings run `build/encoder/cjxl_tiny --help`

## Advanced guide

### Building with Docker

We build a common environment based on Debian/Ubuntu using Docker. Other
systems may have different combinations of versions and dependencies that
have not been tested and may not work. For those cases we recommend using the
Docker container as explained in the
[step by step guide](doc/developing_in_docker.md).

### Building JPEG XL for developers

For experienced developers, we provide build instructions for several other environments:

*   [Building on Debian](doc/developing_in_debian.md)
*   Building on Windows with [vcpkg](doc/developing_in_windows_vcpkg.md) (Visual Studio 2019)
*   Building on Windows with [MSYS2](doc/developing_in_windows_msys.md)
*   [Cross Compiling for Windows with Crossroad](doc/developing_with_crossroad.md)

If you encounter any difficulties, please use Docker instead.

## License

This software is available under a 3-clause BSD license which can be found in
the [LICENSE](LICENSE) file, with an "Additional IP Rights Grant" as outlined in
the [PATENTS](PATENTS) file.

Please note that the PATENTS file only mentions Google since Google is the legal
entity receiving the Contributor License Agreements (CLA) from all contributors
to the JPEG XL Project, including the initial main contributors to the JPEG XL
format: Cloudinary and Google.

## Additional documentation

### Codec description

*   [JPEG XL Format Overview](doc/format_overview.md)
*   [Introductory paper](https://www.spiedigitallibrary.org/proceedings/Download?fullDOI=10.1117%2F12.2529237) (open-access)
*   [XL Overview](doc/xl_overview.md) - a brief introduction to the source code modules
*   [JPEG XL white paper](https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf)
*   [JPEG XL official website](https://jpeg.org/jpegxl)
*   [JPEG XL community website](https://jpegxl.info)

### Development process

*   [More information on testing/build options](doc/building_and_testing.md)
*   [Git guide for JPEG XL](doc/developing_in_github.md) - for developers

### Contact

If you encounter a bug or other issue with the software, please open an Issue here.

There is a [subreddit about JPEG XL](https://www.reddit.com/r/jpegxl/), and
informal chatting with developers and early adopters of `libjxl` can be done on the
[JPEG XL Discord server](https://discord.gg/DqkQgDRTFu).
