#include "operator.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer::execute::operators {
class ReshapeOperator : public IOperator {
  public:
    virtual bool init(godot::RenderingDevice* rd, ShaderCache* shader_cache) override;
    virtual std::optional<dispatch::DispatchDescriptor> build(
        const Physical::Node& node,
        const OperatorContext& ctx) override;
};

} // namespace gdinfer::execute::operators
