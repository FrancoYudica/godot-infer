#include "operator.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace gdinfer::execute::operators {
class MatrixTransposeOperator : public IOperator {
  public:
    virtual bool init(godot::RenderingDevice* rd) override;
    virtual void dispatch(
        const Physical::Node& node,
        const OperatorContext& ctx) override;
    void destroy(godot::RenderingDevice* rd) override;

  private:
    RID _shader;
    RID _pipeline;

    struct PushConstants {
        uint32_t rows;
        uint32_t cols;
        uint32_t _padding[2];
    };
};

} // namespace gdinfer::execute::operators
