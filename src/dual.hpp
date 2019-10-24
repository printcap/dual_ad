#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

template <typename T>
class dual {
  const T value_;
  T deriv_;

 public:
  dual() : value_(0), deriv_(0) {}
  dual(T value, T deriv = 0) : value_(value), deriv_(deriv) {}

  auto value() const noexcept -> T { return value_; }
  auto deriv() const noexcept -> T { return deriv_; }

  friend auto operator+(const dual& lhs, const dual& rhs) -> dual {
    return dual(lhs.value_ + rhs.value_, lhs.deriv_ + rhs.deriv_);
  }

  friend auto operator-(const dual& lhs, const dual& rhs) -> dual {
    return dual(lhs.value_ - rhs.value_, lhs.deriv_ - rhs.deriv_);
  }

  friend auto operator-(const dual& d) -> dual {
    return dual(-d.value_, -d.deriv_);
  }

  friend auto operator*(const dual& lhs, const dual& rhs) -> dual {
    return dual(lhs.value_ * rhs.value_,
                lhs.value_ * rhs.deriv_ + rhs.value_ * lhs.deriv_);
  }

  friend auto operator/(const dual& lhs, const dual& rhs) -> dual {
    return dual(lhs.value_ / rhs.value_,
                (lhs.deriv_ * rhs.value_ - lhs.value_ * rhs.deriv_) /
                    (rhs.value_ * rhs.value_));
  }

  friend auto sin(dual x) -> dual {
    return dual(sin(x.value_), x.deriv_ * cos(x.value_));
  }

  friend auto cos(dual x) -> dual {
    return dual(cos(x.value_), -x.deriv_ * sin(x.value_));
  }

  friend auto exp(dual x) -> dual {
    return dual(exp(x.value_), x.deriv_ * exp(x.value_));
  }

  friend auto operator<<(std::ostream& os, const dual& t) -> std::ostream& {
    std::cout << t.value_;
    return os;
  }

  template <size_t N, typename F>
  friend auto gradient(F f, std::array<dual, N>& x0)
      -> std::tuple<T, std::array<T, N>> {
    for (auto& x : x0) {
      x.deriv_ = 0;
    }

    T value{};
    std::array<T, N> grad;
    for (int i = 0; i < N; ++i) {
      x0[i].deriv_ = 1;
      auto dual_value = f(x0);
      if (i == 0) {
        value = dual_value.value_;
      }
      grad[i] = dual_value.deriv_;
      x0[i].deriv_ = 0;
    }

    return std::make_tuple(value, grad);
  }
};
