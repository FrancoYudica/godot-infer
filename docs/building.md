# Building godot-infer

## Prerequisites

- [CMake](https://cmake.org/) 3.24+
- Visual Studio 2019+ Build Tools (MSVC, C++20-capable) -- `cl.exe`/`link.exe` is the compiler on Windows
- [Ninja](https://ninja-build.org/) (all configure presets use it)
- [LLVM/Clang](https://releases.llvm.org/) 18+ (`clang-tidy`, `clang-format`, `clangd`) -- for editor tooling and lint/format only, not for building
- [vcpkg](https://vcpkg.io/) installed, with `VCPKG_ROOT` pointing to it (see step 2)

> **Platform support**: this build currently only covers **Windows**. Linux is supported by the CMake build itself but isn't walked through here yet. macOS/iOS/Android haven't been ported.

These steps use PowerShell.

## 1. Clone

The repo uses git submodules for `godot-cpp` and the ONNX proto definitions, so clone recursively:

```powershell
git clone --recursive git@github.com:FrancoYudica/godot-infer.git
cd godot-infer
```

## 2. Install vcpkg and set VCPKG_ROOT

If you don't already have vcpkg, clone and bootstrap it anywhere on your machine (it doesn't need to live inside this repo):

```powershell
git clone https://github.com/microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

Then point `VCPKG_ROOT` at that folder, **persistently** (not just for the current shell session), so every future terminal and the CMake commands below all pick it up automatically:

```powershell
[System.Environment]::SetEnvironmentVariable("VCPKG_ROOT", "C:\path\to\vcpkg", "User")
```

(GUI alternative: Start menu -> search "Environment Variables" -> *Edit the system environment variables* -> *Environment Variables...* -> under *User variables*, *New...* -> Name `VCPKG_ROOT`, Value the full path to your vcpkg folder.)

Open a **new** PowerShell window afterward, then sanity-check it's set:

```powershell
echo $env:VCPKG_ROOT
```

## 3. Install dependencies via vcpkg

The project ships a `vcpkg.json` manifest. Running `vcpkg install` from the project root reads it and installs protobuf and Abseil locally into `vcpkg_installed/`. Nothing goes into your global vcpkg directory.

Static linking is required. The GDExtension must be a single self-contained binary, with no external runtime DLLs to ship alongside it. Windows needs the explicit `-static` triplet since vcpkg's default `x64-windows` triplet links dynamically.

```powershell
vcpkg install --triplet x64-windows-static
```

This step takes a while the first time since both libraries are compiled from source. Subsequent runs are instant.

## 4. Build

All configure/build invocations are codified in `src/CMakePresets.json` -- run them from **`src/`**, not the repo root. `cmake --preset` resolves `CMakePresets.json` relative to the current directory.

Two independent choices affect what you get:

- **`GODOTCPP_TARGET`** (`template_debug` or `template_release`): baked into each preset. Picks which Godot API variant the extension is built against, and determines the output filename (`lib_godot_infer.windows.template_debug.*` vs `...template_release.*`). Godot's exporter looks for this specifically -- a "Release" export in the Godot editor loads the `template_release`-named binary.
- **Debug vs Release preset**: how *your own* C++ code is compiled (optimizations, debug symbols, runtime). Each preset bakes in both `CMAKE_BUILD_TYPE` and `GODOTCPP_TARGET` together.

All commands below need a **"Developer PowerShell for VS"** shell (Start menu -> search for it), so `cl.exe`/the Windows SDK `INCLUDE`/`LIB` environment are on `PATH`. Inside VSCode, the CMake Tools extension's **Kit** picker detects installed Visual Studio instances and activates this environment automatically -- no manual shell needed there.

### Development build (editor / testing)

```powershell
cd src
cmake --preset windows-msvc-debug
cmake --build --preset windows-msvc-debug
```

### Shipping build (Release export)

```powershell
cd src
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
```

The extension is written to `demo/addons/godot_infer/bin/`.

### Running the C++ unit test suite

Presets default to `BUILD_TESTS=OFF`. Append an override the same way you would any other `-D` flag:

```powershell
cmake --preset windows-msvc-debug -DBUILD_TESTS=ON
cmake --build --preset windows-msvc-debug
ctest --preset windows-msvc-debug
```

Notes:
- There's no default triplet baked into `CMakeLists.txt` itself -- each preset sets `VCPKG_TARGET_TRIPLET` explicitly.
- Each preset has its own binary directory under `build/` (e.g. `build/windows-msvc-debug/`).
- `CMakeLists.txt` pins the MSVC runtime to match vcpkg's static-triplet libs for whichever CMake config you pick (`/MT` for Release, `/MTd` for Debug), working around a godot-cpp CMake quirk where `linux.cmake`/`windows.cmake` disagree on the default.

## 5. Editor support (clangd)

`.clangd` points `CompilationDatabase` at `build/windows-msvc-debug`, which the `windows-msvc-debug` preset keeps up to date (`CMAKE_EXPORT_COMPILE_COMMANDS: ON` is set in the shared `base` preset). Reconfigure that preset whenever you add or remove a `.cpp` file; clangd picks up the change automatically.

If you want clangd pointed at a specific LLVM install (e.g. a standalone one rather than the VSCode extension's bundled version), set it in your **own user settings** -- not the shared workspace `.vscode/settings.json`, since that file is committed and the path is machine-specific:

```json
"clangd.path": "C:\\path\\to\\your\\llvm\\bin\\clangd.exe"
```

Restart the language server afterward.
