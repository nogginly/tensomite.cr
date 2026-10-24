module Tensomite
  # `Float32` transcendental functions for host kernels, in pure Crystal.
  #
  # They replace the C library's because a call into it cannot be vectorised: its
  # `tanhf` costs about 12 ns per element where a loop of arithmetic costs well
  # under 1. Each function is range reduction and a short polynomial, written
  # with selects rather than branches so that loops over it vectorise, and each
  # is within 2 ulp of the correctly rounded result over its range (see
  # `spec/maths_spec.cr`).
  #
  # The same functions run on every host device and operating system, so host
  # results no longer depend on the platform's maths library.
  #
  # Values whose result would be subnormal (`exp` below about -87.3) give zero.
  module Maths
    private LOG2E    =  1.44269504088896341_f32
    private LN2_HI   =          0.693359375_f32
    private LN2_LO   =       -2.12194440e-4_f32
    private EXP_HI   =          88.72283905_f32
    private EXP_LO   =          -87.3365447_f32
    private SQRT_1_2 = 0.707106781186547524_f32
    private TANH_BIG =                 9.01_f32
    # 1.5 · 2²³: adding and subtracting it rounds a Float32 of magnitude below
    # 2²² to the nearest integer, ties to even, in plain arithmetic that vectorises.
    private ROUNDER = 12582912_f32

    # Returns e raised to *x*.
    #
    # 1. Writes `x = n·ln 2 + r`, with `n` the nearest integer to `x / ln 2` and
    #    `|r| ≤ ln 2 / 2`, subtracting `n·ln 2` in two parts to keep `r` exact.
    # 2. Approximates `e^r` by a degree-6 polynomial.
    # 3. Multiplies by `2^n`, built from exponent bits as `2^⌊n/2⌋ · 2^(n - ⌊n/2⌋)`,
    #    since `n` reaches 128 near the top of the range, where `2^n` alone
    #    would overflow.
    @[AlwaysInline]
    def self.exp(x : Float32) : Float32
      clamped = x > EXP_HI || x < EXP_LO ? 0_f32 : x
      n = (clamped * LOG2E + ROUNDER) - ROUNDER
      r = clamped - n * LN2_HI - n * LN2_LO
      z = r * r
      p = 1.9875691500e-4_f32
      p = p * r + 1.3981999507e-3_f32
      p = p * r + 8.3334519073e-3_f32
      p = p * r + 4.1665795894e-2_f32
      p = p * r + 1.6666665459e-1_f32
      p = p * r + 5.0000001201e-1_f32
      p = p * z + r + 1_f32
      whole = n.to_i32!
      half = whole >> 1
      result = p * ((half &+ 127) << 23).unsafe_as(Float32) * ((whole &- half &+ 127) << 23).unsafe_as(Float32)
      result = x > EXP_HI ? Float32::INFINITY : result
      result = x < EXP_LO ? 0_f32 : result
      x.nan? ? x : result
    end

    # Returns the natural logarithm of *x*: `-∞` at 0, NaN below 0.
    #
    # 1. Splits `x` into `m · 2^e` from its bits, with `m` in `[√½, √2)`, scaling
    #    subnormal inputs by `2^23` first.
    # 2. Approximates `log(m)` by `f - f²/2 + f³·P(f)`, with `f = m - 1` and `P`
    #    of degree 8.
    # 3. Adds `e · ln 2`, in two parts.
    @[AlwaysInline]
    def self.log(x : Float32) : Float32
      subnormal = x < Float32::MIN_POSITIVE
      scaled = subnormal ? x * 8388608_f32 : x
      bits = scaled.unsafe_as(Int32)
      e = ((bits >> 23) & 0xff) - 126 - (subnormal ? 23 : 0)
      m = ((bits & 0x007fffff) | 0x3f000000).unsafe_as(Float32)
      small = m < SQRT_1_2
      e = small ? e - 1 : e
      f = small ? m + m - 1_f32 : m - 1_f32
      z = f * f
      y = 7.0376836292e-2_f32
      y = y * f - 1.1514610310e-1_f32
      y = y * f + 1.1676998740e-1_f32
      y = y * f - 1.2420140846e-1_f32
      y = y * f + 1.4249322787e-1_f32
      y = y * f - 1.6668057665e-1_f32
      y = y * f + 2.0000714765e-1_f32
      y = y * f - 2.4999993993e-1_f32
      y = y * f + 3.3333331174e-1_f32
      y = y * f * z
      ef = e.to_f32
      y = y + ef * LN2_LO
      y = y - 0.5_f32 * z
      result = f + y + ef * LN2_HI
      result = x == Float32::INFINITY ? x : result
      result = x == 0_f32 ? -Float32::INFINITY : result
      result = x < 0_f32 ? Float32::NAN : result
      x.nan? ? x : result
    end

    # Returns the hyperbolic tangent of *x*.
    #
    # Below `|x| = 0.625` it is an odd polynomial in `x`; above, `1 - 2 / (e^{2|x|} + 1)`
    # with the sign of `x`; beyond `|x| = 9.01` it is ±1 to `Float32` precision.
    @[AlwaysInline]
    def self.tanh(x : Float32) : Float32
      a = x.abs
      z = x * x
      p = -5.70498872745e-3_f32
      p = p * z + 2.06390887954e-2_f32
      p = p * z - 5.37397155531e-2_f32
      p = p * z + 1.33314422036e-1_f32
      p = p * z - 3.33332819422e-1_f32
      near = p * z * x + x
      far = 1_f32 - 2_f32 / (exp(a + a) + 1_f32)
      far = a > TANH_BIG ? 1_f32 : far
      far = x < 0_f32 ? -far : far
      result = a < 0.625_f32 ? near : far
      result = x == 0_f32 ? x : result
      x.nan? ? x : result
    end
  end
end
