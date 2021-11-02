#pragma once

#include <atomic>
#include <ostream>
#include <vector>

#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensors/str_cat.h"
#include "lazy_tensors/str_join.h"

namespace lazy_tensors {

class Shape {
 public:
  Shape() : at_element_type_(c10::ScalarType::Undefined) {}

  Shape(at::ScalarType element_type, c10::ArrayRef<int64_t> dimensions);

  Shape(c10::ArrayRef<Shape> element_shapes)
      : is_tuple_(true),
        at_element_type_(c10::ScalarType::Undefined),
        element_shapes_(element_shapes.begin(), element_shapes.end()) {
    CHECK(element_shapes.size() > 0);
    // TODO(whc) it's not really clear what the definition of element shape
    // should be for a tuple shape.  However, for tuple shapes, we appear
    // to be accessing the element_type field in some places.  Fix this.
    at_element_type_ = element_shapes[0].at_element_type();
  }

  std::string ToString(bool print_layout = false) const {
    return lazy_tensors::StrCat(toString(at_element_type_), "[",
                                c10::Join(",", dimensions_), "]");
  }

  c10::ScalarType at_element_type() const { return at_element_type_; }
  void set_element_type(at::ScalarType value);

  int64_t rank() const { return dimensions_.size(); }

  int64_t dimensions(int index) const {
    CHECK_LT(index, dimensions_.size());
    return dimensions_[index];
  }

  c10::ArrayRef<int64_t> dimensions() const { return dimensions_; }

  void set_dimensions(int index, int64_t value) {
    CHECK_LT(index, dimensions_.size());
    dimensions_[index] = value;
  }

  // TODO(whc) remove tuple support? or keep it (But make dimensions() methods
  // work consistently with it somehow?)
  bool IsTuple() const { return is_tuple_; }
  int tuple_shapes_size() const { return element_shapes_.size(); }

  const Shape& tuple_shapes(int index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, element_shapes_.size());
    return element_shapes_[index];
  }
  const std::vector<Shape>& tuple_shapes() const { return element_shapes_; }

  bool operator==(const Shape& other) const {
    return at_element_type_ == other.at_element_type_ &&
           dimensions_ == other.dimensions_;
  }

 private:
  bool is_tuple_ = false;
  c10::ScalarType at_element_type_;
  std::vector<int64_t> dimensions_;
  std::vector<Shape> element_shapes_;
};

inline std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.ToString();
}

}  // namespace lazy_tensors
