#include <iostream>

#include "teuk/modes.hpp"

int main() {
  const teuk::ModeRegistry registry({-4, -2, 0, 2, 4}, {-4, 0, 4});
  std::cout << "Ordered daughter-mode pairs from explicit signed content:\n";
  for (const auto& pair : registry.ordered_pairs()) {
    if ((pair.m1 == -2 || pair.m1 == 2) &&
        (pair.m2 == -2 || pair.m2 == 2)) {
      std::cout << '(' << pair.m1 << ',' << pair.m2 << ")->" << pair.target
                << '\n';
    }
  }
  return 0;
}
