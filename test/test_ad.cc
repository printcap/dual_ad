#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/dual.hpp"

using dfloat = dual<double>;

TEST_CASE("Differentiates addition", "[ops]") {
  dfloat x{1, 1};
  dfloat y{2, 0};
  auto z = x + y;
  REQUIRE(z.value() == Approx(3.0));
  REQUIRE(z.deriv() == Approx(1.0));

  dfloat u{2, 0};
  dfloat v{3, 1};
  auto w = u + v;
  REQUIRE(w.value() == Approx(5.0));
  REQUIRE(w.deriv() == Approx(1.0));
}

TEST_CASE("Differentiates subtraction", "[ops]") {
  dfloat x{1, 1};
  dfloat y{2, 0};
  auto z = x - y;
  REQUIRE(z.value() == Approx(-1.0));
  REQUIRE(z.deriv() == Approx(1.0));

  dfloat u{3, 0};
  dfloat v{2, 1};
  auto w = u - v;
  REQUIRE(w.value() == Approx(1.0));
  REQUIRE(w.deriv() == Approx(-1.0));
}

TEST_CASE("Differentiates multiplication", "[ops]") {
  dfloat x{2, 1};
  dfloat y{3, 0};
  auto z = x * y;
  REQUIRE(z.value() == Approx(6.0));
  REQUIRE(z.deriv() == Approx(3.0));

  dfloat u{4, 0};
  dfloat v{5, 1};
  auto w = u * v;
  REQUIRE(w.value() == Approx(20.0));
  REQUIRE(w.deriv() == Approx(4.0));
}

TEST_CASE("Differentiates division", "[ops]") {
  dfloat x{4, 1};
  dfloat y{2, 0};
  auto z = x / y;
  REQUIRE(z.value() == Approx(2.0));
  REQUIRE(z.deriv() == Approx(0.5));

  dfloat u{3, 0};
  dfloat v{2, 1};
  auto w = u / v;
  REQUIRE(w.value() == Approx(1.5));
  REQUIRE(w.deriv() == Approx(-0.75));
}

TEST_CASE("Computes gradient sin(x)*cos(y)", "[grad]") {
  dfloat x{M_PI / 4};
  dfloat y{M_PI / 4};
  std::array<dfloat, 2> w{x, y};
  auto f = [](std::array<dfloat, 2>& w) { return sin(w[0]) * cos(w[1]); };
  auto [val, grad] = gradient(f, w);
  REQUIRE(val == Approx(0.5));
  REQUIRE(grad[0] == Approx(0.5));
  REQUIRE(grad[1] == Approx(-0.5));
}

TEST_CASE("Computes gradient of activation", "[grad]") {
  dfloat w1{2.0};
  dfloat w2{3.0};
  std::array<dfloat, 2> w{w1, w2};
  auto f = [](std::array<dfloat, 2>& w) {
    return 1.0 / (1.0 + exp(-(3.0 * w[0] - 2.0 * w[1])));
  };
  auto [val, grad] = gradient(f, w);
  REQUIRE(val == Approx(0.5));
  REQUIRE(grad[0] == Approx(0.75));
  REQUIRE(grad[1] == Approx(-0.5));
}
