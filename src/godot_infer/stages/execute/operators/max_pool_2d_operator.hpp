#include "operator.hpp"

namespace gdinfer::execute::operators {
class MaxPool2DOperator : public IOperator {
  public:
    virtual bool init(godot::RenderingDevice* rd, ShaderCache* shader_cache) override;
    virtual std::optional<dispatch::DispatchDescriptor> build(
        const Physical::Node& node,
        const OperatorContext& ctx) override;

  private:
    godot::RID _shader;
    godot::RID _pipeline;
};

} // namespace gdinfer::execute::operators
