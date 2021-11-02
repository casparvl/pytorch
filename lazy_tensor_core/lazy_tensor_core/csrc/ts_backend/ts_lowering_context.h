#pragma once

#include <torch/jit.h>
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "torch/csrc/jit/runtime/graph_executor.h"

namespace torch_lazy_tensors {
namespace compiler {

using TSOpVector = std::vector<torch::jit::Value*>;

class NodeLowering;

namespace ts_backend {

class GenericComputationTS : public GenericComputation {
 public:
  GenericComputationTS(std::shared_ptr<torch::jit::Graph> graph)
      : graph_executor_(std::move(graph), "") {
        for (torch::jit::Value* input : graph_executor_.graph()->inputs()) {
          parameter_names_.push_back(input->debugName());
        }
      }

  int parameters_size() const override { return parameter_names_.size(); }

  const std::vector<lazy_tensors::Shape>& parameter_shapes() const override {
    throw std::runtime_error("TODO(whc) implement TS computation shapes or change interface");
    return parameter_shapes_;
  }

  const std::vector<std::string>& parameter_names() const override { return parameter_names_; }

  const lazy_tensors::Shape& result_shape() const override {
    throw std::runtime_error("TODO(whc) implement TS computation shapes or change interface");
    return result_shape_;
  }

  std::shared_ptr<torch::jit::Graph> graph() const { return graph_executor_.graph(); }

  torch::jit::GraphExecutor& graph_executor() { return graph_executor_; }

 private:
  torch::jit::GraphExecutor graph_executor_;
  std::vector<std::string> parameter_names_;
  std::vector<lazy_tensors::Shape> parameter_shapes_;
  lazy_tensors::Shape result_shape_;
};

class TSLoweringContext : public ir::LoweringContext {
 public:
  TSLoweringContext(const std::string& name, Device device);

  TSLoweringContext(const std::string& name, Device device,
                    c10::ArrayRef<torch::lazy::Node*> post_order,
                    ir::Util::EmissionMap emit_status);

  lazy_tensors::Shape GetResultShape(size_t index) const override;

  size_t AddResult(const torch::lazy::Output& output) override;

  lazy_tensors::StatusOr<std::shared_ptr<GenericComputation>>
  Build() override;

  // Retrieves the lowered operation for a output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  torch::jit::Value* GetOutputOp(const torch::lazy::Output& output);

  // Assigns the given TS operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const torch::lazy::Output& output, torch::jit::Value* op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  torch::jit::Value* GetParameter(
      const std::shared_ptr<Data>& data);

  std::shared_ptr<torch::jit::Graph> graph() const { return graph_; }

 private:
  struct Parameter {
    torch::jit::Value* param;
    size_t index = 0;
  };

  size_t AddResult(torch::jit::Value* op);

  std::shared_ptr<torch::jit::Graph> graph_;
  std::unordered_map<Data::OpaqueHandle, Parameter>
      parameters_map_;
  std::vector<torch::jit::Value*> root_tuple_;
  torch::lazy::OutputMap<torch::jit::Value*> emitted_outputs_;
  std::unique_ptr<NodeLowering> lowering_;
};

}  // namespace ts_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors