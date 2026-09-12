import onnx
from onnx import helper, TensorProto
from core import TestBuilder, TestData

class ReshapeBasic(TestBuilder):

    def __init__(self, base_path):
        super().__init__(base_path, "reshape_basic")

    def create_model(self):
        input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 2])
        shape_init = helper.make_tensor('shape', TensorProto.INT64, [2], [3, 2])
        node = helper.make_node('Reshape', ['input', 'shape'], ['output'])
        graph = helper.make_graph([node], 'ReshapeModel', [input_info], [output_info], [shape_init])
        onnx.save(helper.make_model(graph), self.model_filepath)

    def create_test_data(self):
        return TestData(
            name=self.name,
            input_data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            input_shape=[2.0, 3.0],
            output_data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
