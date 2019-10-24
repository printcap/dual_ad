#include "dual.hpp"

auto main() -> int {
  using dfloat = dual<double>;

  dfloat w1{2.0};
  dfloat w2{1.0};
  std::array<dfloat, 2> w{w1, w2};

  auto expr = [](std::array<dfloat, 2>& w) {
    return 1.0 / (1.0 + exp(-(2.0 * w[0] + 3.0 * w[1])));
  };

  // compute value and gradient of lambda expression
  auto [val, grad] = gradient(expr, w);

  std::cout << "value: " << val << std::endl;
  std::cout << "grad:  (" << grad[0] << ", " << grad[1] << ")\n";
}
