#include "operator.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer::execute::operators {
class ReshapeOperator : public IOperator {
  public:
    virtual bool init(godot::RenderingDevice* rd) override;
    virtual void dispatch(
        const Physical::Node& node,
        const OperatorContext& ctx) override;
    void destroy(godot::RenderingDevice* rd) override;
};

} // namespace gdinfer::execute::operators
