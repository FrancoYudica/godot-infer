import onnx
from onnx import parser as onnx_parser

IR_VERSION = 13
OPSET = 13


def _build_model(text):
    """Parses ONNX's compact text DSL into a ModelProto (see onnx.parser)."""
    return onnx_parser.parse_model(text)


def _gemm_test1(fixtures_dir):
    filepath = f"{fixtures_dir}/GemmTest1.onnx"
    model = _build_model(f"""
    <
        ir_version: {IR_VERSION},
        opset_import: [ "" : {OPSET} ]
    >
    GemmModel (float[N, 3] input) => (float[N, 3] output)
    <
        float[3,3] W = {{1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0}},
        float[3] B = {{0.0,0.0,0.0}}
    >
    {{
        output = Gemm<alpha=1.0, beta=1.0, transB=1>(input, W, B)
    }}
    """)
    onnx.save(model, filepath)
    return filepath


def _relu_test1(fixtures_dir):
    filepath = f"{fixtures_dir}/ReluTest1.onnx"
    model = _build_model(f"""
    <
        ir_version: {IR_VERSION},
        opset_import: [ "" : {OPSET} ]
    >
    ReluModel (float[N, 3] input) => (float[N, 3] output)
    {{
        output = Relu(input)
    }}
    """)
    onnx.save(model, filepath)
    return filepath


def _sigmoid_test1(fixtures_dir):
    filepath = f"{fixtures_dir}/SigmoidTest1.onnx"
    model = _build_model(f"""
    <
        ir_version: {IR_VERSION},
        opset_import: [ "" : {OPSET} ]
    >
    SigmoidModel (float[N, 3] input) => (float[N, 3] output)
    {{
        output = Sigmoid(input)
    }}
    """)
    onnx.save(model, filepath)
    return filepath


def _conv_test1(fixtures_dir):
    filepath = f"{fixtures_dir}/ConvTest1.onnx"
    w_vals = ", ".join(["1.0"] * (8 * 3 * 3 * 3))
    b_vals = ", ".join(["0.0"] * 8)
    model = _build_model(f"""
    <
        ir_version: {IR_VERSION},
        opset_import: [ "" : {OPSET} ]
    >
    Conv2DModel (float[batch, 3, height, width] input) => (float[batch, 8, height, width] output)
    <
        float[8,3,3,3] W = {{{w_vals}}},
        float[8] B = {{{b_vals}}}
    >
    {{
        output = Conv<kernel_shape=[3,3], strides=[2,2], pads=[1,1,1,1], dilations=[1,1]>(input, W, B)
    }}
    """)
    onnx.save(model, filepath)
    return filepath


def _conv_transpose_test1(fixtures_dir):
    filepath = f"{fixtures_dir}/ConvTransposeTest1.onnx"
    w_vals = ", ".join(["1.0"] * (1 * 1 * 3 * 3))
    model = _build_model(f"""
    <
        ir_version: {IR_VERSION},
        opset_import: [ "" : {OPSET} ]
    >
    ConvTransposeModel (float[batch, 1, height, width] input) => (float[batch, 1, height, width] output)
    <
        float[1,1,3,3] W = {{{w_vals}}},
        float[1] B = {{1.0}}
    >
    {{
        output = ConvTranspose<kernel_shape=[3,3], strides=[1,1], pads=[1,1,1,1], dilations=[1,1]>(input, W, B)
    }}
    """)
    onnx.save(model, filepath)
    return filepath


def _max_pool_test1(fixtures_dir):
    filepath = f"{fixtures_dir}/MaxPoolTest1.onnx"
    model = _build_model(f"""
    <
        ir_version: {IR_VERSION},
        opset_import: [ "" : {OPSET} ]
    >
    MaxPoolModel (float[batch, 3, 4, 4] input) => (float[batch, 3, h, w] output)
    {{
        output = MaxPool<kernel_shape=[2,2], strides=[1,1], pads=[0,0,0,0], dilations=[1,1]>(input)
    }}
    """)
    onnx.save(model, filepath)
    return filepath


def _im2col_test1(fixtures_dir):
    filepath = f"{fixtures_dir}/Im2ColTest1.onnx"
    # Im2Col is not a standard ONNX operator -- it's specific to this
    # engine -- so this is deliberately not schema-checked, same as before.
    model = _build_model(f"""
    <
        ir_version: {IR_VERSION},
        opset_import: [ "" : {OPSET} ]
    >
    Im2ColModel (float[1, 1, 3, 3] input) => (float[1, 1, 3, 3] output)
    {{
        output = Im2Col<kernel_shape=[3,3], strides=[1,1], pads=[0,0,0,0], dilations=[1,1]>(input)
    }}
    """)
    onnx.save(model, filepath)
    return filepath


def _reshape_test1(fixtures_dir):
    filepath = f"{fixtures_dir}/ReshapeTest1.onnx"
    model = _build_model(f"""
    <
        ir_version: {IR_VERSION},
        opset_import: [ "" : {OPSET} ]
    >
    ReshapeModel (float[2, 3] input) => (float[3, 2] output)
    <
        int64[2] shape = {{3, 2}}
    >
    {{
        output = Reshape(input, shape)
    }}
    """)
    onnx.save(model, filepath)
    return filepath


_BUILDERS = [
    _gemm_test1,
    _relu_test1,
    _sigmoid_test1,
    _conv_test1,
    _conv_transpose_test1,
    _max_pool_test1,
    _im2col_test1,
    _reshape_test1,
]


def generate(fixtures_root):
    fixtures_dir = fixtures_root / "parser"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    for build in _BUILDERS:
        filepath = build(str(fixtures_dir))
        print(f"[*] Wrote {filepath}")
