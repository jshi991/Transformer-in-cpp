#ifndef MYTORCH_CHECKPOINT_H
#define MYTORCH_CHECKPOINT_H

#include "mytorch/jensor.h"

#include <string>
#include <vector>

namespace mytorch {

void save_checkpoint(const std::string& path, const std::vector<Jensor<float>*>& params);
void load_checkpoint(const std::string& path, const std::vector<Jensor<float>*>& params);

}  // namespace mytorch
#endif  // MYTORCH_CHECKPOINT_H
