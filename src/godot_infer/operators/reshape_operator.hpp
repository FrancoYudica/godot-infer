#include "operator.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer {
class ReshapeOperator : public IOperator {
  public:
    virtual bool init(godot::RenderingDevice* rd) override;
    virtual void dispatch(
        const gdinfer::Physical::Node& node,
        const OperatorContext& ctx) override;
    void destroy(godot::RenderingDevice* rd) override;

  private:
    void _dispatch_transpose(
        const gdinfer::Physical::Node& node,
        const OperatorContext& ctx,
        const gdinfer::Physical::ReshapeAttrs& attrs);

    RID _shader;
    RID _pipeline;

    struct PushConstants {
        uint32_t rows;
        uint32_t cols;
        uint32_t _padding[2];
    };
};

} // namespace gdinfer