#pragma once

#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/shape.h"
#include "torch/csrc/lazy/core/ir.h"
#include "ts_node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
using namespace torch_lazy_tensors::compiler;
using NodePtr = torch::lazy::NodePtr;
using Node = torch::lazy::Node;
using OpKind = torch::lazy::OpKind;
using OpList = torch::lazy::OpList;

namespace ops {
  using NodePtr = torch::lazy::NodePtr;

}

// Helper that makes it easy to access the TsNode::shape() method
// from an torch::lazy::Output* that holds a Node* that points to a TsNode
// TODO(whc) remove these once migrating to codegen and cleaning up Shape use
lazy_tensors::Shape GetShapeFromTsOutput(const torch::lazy::Output& output);
lazy_tensors::Shape GetShapeFromTsValue(const torch::lazy::Value& value);
lazy_tensors::Shape GetShapeFromTsNode(const torch::lazy::Node& value);
void TsNodeSetShapeDeferred(
    NodePtr node, const std::function<lazy_tensors::Shape()>& shape_fn);

class TsNode : public torch::lazy::Node {
 public:
  TsNode(OpKind op, OpList operands, lazy_tensors::Shape shape,
         size_t num_outputs = 1, torch::lazy::hash_t hash_seed = torch::lazy::kHashSeed);

  // Same as the constructor above, but the shape is generated by a function,
  // only if needed (shape cache miss).
  TsNode(OpKind op, OpList operands,
         const std::function<lazy_tensors::Shape()>& shape_fn,
         size_t num_outputs = 1, torch::lazy::hash_t hash_seed = torch::lazy::kHashSeed);

  // The shape is set later.
  TsNode(OpKind op, OpList operands, size_t num_outputs = 1,
         torch::lazy::hash_t hash_seed = torch::lazy::kHashSeed);

  void SetShapeDeferred(const std::function<lazy_tensors::Shape()>& shape_fn);

  // Contructor used to create leaf nodes.
  TsNode(OpKind op, lazy_tensors::Shape shape, size_t num_outputs = 1,
         torch::lazy::hash_t hash_seed = torch::lazy::kHashSeed);

  virtual ~TsNode() = default;

  lazy_tensors::Shape GetOpShape(
      const std::function<lazy_tensors::Shape()>& shape_fn) const;

  // Retrieves the full shape of the IR Node. Note that if this is a
  // multi-output node, the returned shape will be a tuple.
  const lazy_tensors::Shape& shape() const;

  // Retrieves the shape of the output at a given index. If the node is not a
  // multi-output node, output_index must be zero.
  const lazy_tensors::Shape& shape(size_t output_index) const;

  virtual std::string ToString() const override;

  static torch::lazy::hash_t GetOpHash(OpKind op,
                                       const lazy_tensors::Shape& shape,
                                       torch::lazy::hash_t hash_seed);

  virtual const std::vector<torch::lazy::Output>& operands() const override {
    return operands_as_outputs_;
  }
  virtual const torch::lazy::Output& operand(size_t i) const override {
    return operands_as_outputs_.at(i);
  }

  // TODO(whc) We'll delete Clone since it's not used.  But it needs to be
  // removed from all the legacy ops, so I'm moving it from Node to TsNode
  // for now, and we'll delete it later once we've moved more ops to codegen
  virtual NodePtr Clone(OpList operands) const {
    LOG(ERROR) << "Cloning not implemented for TsNode";
  }

  // Lower is a backend-specific method since it returns a backend specific
  // type. hence, it is convenient to define it differently per-backend rather
  // than at Node API
  virtual TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                           ts_backend::TSLoweringContext* loctx) const;

 private:
  // Adds node's index output number as operand.
  void AddOperand(NodePtr node, size_t index = 0);

  lazy_tensors::Shape shape_;
  // A node holds a real reference to its operands.
  std::vector<NodePtr> operands_;
  // Outputs do not hold references on the nodes, and neither do the uses, since
  // otherwise we get into circular reference counting.
  std::vector<torch::lazy::Output> operands_as_outputs_;
};

}  // namespace ir
}  // namespace torch_lazy_tensors
