#ifndef ML_INFERENCE_ENGINE_H
#define ML_INFERENCE_ENGINE_H
#include "core/core.hpp"
#include "inference_descriptor.hpp"
#include "inference_task.hpp"
#include "stages/execute/deletion_stack.hpp"
#include "stages/execute/io/io.hpp"
#include "io/onnx_resource.hpp"
#include "stages/execute/operators/operators.hpp"
#include "stages/compile/lowering/lowering.hpp"
#include "stages/compile/parser/parser.hpp"
#include "stages/execute/tensors/tensors.hpp"

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <unordered_set>
namespace godot {

struct GraphContext {
    gdinfer::compile::lowering::Physical::Graph graph;
    gdinfer::compile::lowering::SourceOpTable source_ops;
    Ref<gdinfer::execute::tensors::TensorResourceManager> initializers_tm;
};

class MLInferenceEngine : public RefCounted {
    GDCLASS(MLInferenceEngine, RefCounted)

  public:
    void init();
    void destroy();
    uint32_t register_model(Ref<ONNXResource> resource);
    void unload_model(uint32_t model_rid);
    Ref<InferenceTask> queue_request(
        uint32_t model_rid,
        Ref<InferenceDescriptor> request);
    void print_model(uint32_t model_rid);

    void destroy_task(Ref<InferenceTask> task);
    godot::Variant get_task_output(
        Ref<InferenceTask> task,
        const String& output_name);
    void set_capture_timestamps(bool enabled) { _capture_timestamps = enabled; }
    bool get_capture_timestamps() const { return _capture_timestamps; }

  protected:
    static void _bind_methods();

  private:
    void _process_pending_tasks();
    void _process_task(Ref<InferenceTask> task);
    void _run_node(
        const gdinfer::compile::lowering::Physical::Graph& graph,
        const gdinfer::compile::lowering::SourceOpTable& source_ops,
        const gdinfer::execute::shape_inference::DispatchData& dispatch_data,
        const gdinfer::compile::lowering::Physical::Node& node,
        int64_t compute_list,
        Ref<gdinfer::execute::tensors::TensorResourceManager> initializers_tm,
        Ref<gdinfer::execute::tensors::TensorResourceManager> activations_tm,
        const gdinfer::execute::shape_inference::ShapeTable& shape_table);
    void _allocate_activations(
        const gdinfer::compile::lowering::Physical::Graph& graph,
        const gdinfer::execute::shape_inference::ShapeTable& shape_table,
        Ref<gdinfer::execute::tensors::TensorResourceManager> activations_tm);

    void _free_all_resources();
    bool _has_graph(uint32_t graph_rid);
    bool _validate_inputs(
        const gdinfer::compile::lowering::Physical::Graph& graph,
        gdinfer::execute::shape_inference::ShapeTable& shape_table);

    void _capture_timestamp(const String& label);
    void _collect_task_timestamps();
    bool _destroy_graph_if_unused(uint32_t model_rid);
    void _destroy_pending_graphs_if_unused();

  private:
    RenderingDevice* _rd;
    gdinfer::execute::tensors::StorageBufferPool _sb_pool;
    gdinfer::execute::operators::OperatorRegistry _operator_registry;
    gdinfer::execute::io::InputHandlerRegistry _input_registry;
    gdinfer::execute::io::OutputHandlerRegistry _output_registry;
    std::unordered_map<uint32_t, GraphContext> _graphs;
    std::vector<Ref<InferenceTask>> _pending_tasks;
    std::vector<Ref<InferenceTask>> _executing_tasks;
    std::unordered_map<uint32_t, Ref<InferenceTask>> _tasks;
    bool _initialized = false;
    bool _destroying = false;
    bool _capture_timestamps = false;

    gdinfer::execute::DeletionStack _frame_deletion_stack;

    uint32_t _next_graph_id = 1;
    uint32_t _next_task_id = 0;

    std::unordered_set<uint32_t> _pending_graph_deletions;
};
} // namespace godot

#endif
