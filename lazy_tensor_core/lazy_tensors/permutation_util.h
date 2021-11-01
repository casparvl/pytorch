#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Logging.h>

#include <vector>

namespace lazy_tensors {

std::vector<int64_t> InversePermutation(
    c10::ArrayRef<int64_t> input_permutation);

bool IsPermutation(c10::ArrayRef<int64_t> permutation);

}  // namespace lazy_tensors
