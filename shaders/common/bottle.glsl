// Match JS bottleHalfFrac exactly
float bottleHalfFrac(float t) {
  if (t < 0.03) return 0.0;
  if (t < 0.06) {
    float u = (t - 0.03) / 0.03;
    return 0.247 * (u * u * (3.0 - 2.0 * u));
  }
  if (t < 0.82) {
    float u = (t - 0.06) / 0.76;
    float e = u * u * (3.0 - 2.0 * u);
    return mix(0.247, 0.50, e);
  }
  if (t < 0.93) {
    float u = (t - 0.82) / 0.11;
    float e = u * u * (3.0 - 2.0 * u);
    return mix(0.50, 0.40, e);
  }
  if (t < 0.96) {
    float u = (t - 0.93) / 0.03;
    float e = u * u * (3.0 - 2.0 * u);
    return mix(0.40, 0.32, e);
  }
  if (t < 0.99) {
    float u = (t - 0.96) / 0.03;
    return 0.32 * (1.0 - u);
  }
  return 0.0;
}
