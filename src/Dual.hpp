#pragma once

template <typename T = float>
struct Dual
{
    T real_ = T{0.0};
    T dual_ = T{1.0};
};

template <typename T>
[[nodiscard]] Dual<T> operator+(Dual<T>&& a, Dual<T>&& b) noexcept
{
    return {a.real_ + b.real_, a.dual_ + b.dual_};
}

template <typename T>
[[nodiscard]] Dual<T> operator-(Dual<T>&& a, Dual<T>&& b) noexcept
{
    return {a.real_ - b.real_, a.dual_ - b.dual_};
}

template <typename T>
[[nodiscard]] constexpr Dual<T> operator*(Dual<T>&& a, Dual<T>&& b) noexcept
{
    return {
        a.real_ * b.real_,
        a.real_ * b.dual_ + b.real_ * a.dual_,
    };
}
