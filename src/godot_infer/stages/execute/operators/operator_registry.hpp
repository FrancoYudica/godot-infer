#pragma once
#include "core/core.hpp"
#include "operator.hpp"

namespace gdinfer::execute::operators {

class OperatorRegistry {
  public:
    bool init(godot::RenderingDevice* rd, ShaderCache* shader_cache);

    IOperator* get(Physical::Operator op) const;

    bool supports(Physical::Operator op) const;

    void destroy();

  private:
    template <typename T>
    bool _register(Physical::Operator op, godot::RenderingDevice* rd, ShaderCache* shader_cache) {
        auto impl = std::make_unique<T>();
        if (impl->init(rd, shader_cache)) {
            _operators[op] = std::move(impl);
            return true;
        }
        return false;
    }
    std::unordered_map<Physical::Operator, std::unique_ptr<IOperator>> _operators;
};

} // namespace gdinfer::execute::operators