# Test Generation Suite

Generates ONNX models together with their input/output data for a set of operators. The engine loads the resulting `tests.json` to run each model and validate its output against the expected values.

## Setup

```bash
cd tests
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

By default the artifacts are written to `tests/generated_tests/`. Pass `--base_path` to change the output location:

```bash
python main.py --base_path ../demo/addons/ml/tests
```

## Output

`generated_tests/` contains one `.onnx` file per test case and a single `tests.json` that lists every case with its name, input data, input shape, and expected output.

## C++ unit-test fixtures

`python generate_cpp_fixtures.py` reuses the same operator builders to write single-operator `.onnx` models (no `tests.json`, no numeric data) into `fixtures/parser/`, consumed by `cpp_tests/stages/compile/parser/parser_test.cpp`. Run manually whenever the fixture set changes; the output is checked into git, not regenerated at build time.

Build with `-DBUILD_TESTS=ON` and run the suite with ctest:

```bash
cmake --build build --target parser_test
ctest --test-dir build --output-on-failure
```

With the default Visual Studio generator (multi-config), pass the config explicitly: `ctest --test-dir build --output-on-failure -C Debug`. Single-config generators (e.g. the Ninja build CI uses) don't need it.