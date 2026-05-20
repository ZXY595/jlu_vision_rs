use std::f64::consts::PI;

#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

pub fn logistic(x: f64, min: f64, max: f64) -> f64 {
    min + (max - min) / (1.0 + (-x).exp())
}

pub fn logistic_inverse(y: f64, min: f64, max: f64) -> f64 {
    -((max - y) / (y - min)).ln()
}

pub fn logistic_derivative(y: f64, min: f64, max: f64) -> f64 {
    (y - min) * (max - y) / (max - min)
}

pub fn normalize_angle(angle: f64) -> f64 {
    let a = angle % (2.0 * PI);
    if a > PI {
        a - 2.0 * PI
    } else if a < -PI {
        a + 2.0 * PI
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_logistic_roundtrip() {
        let x = 0.5;
        let y = logistic(x, 0.1, 0.5);
        let x2 = logistic_inverse(y, 0.1, 0.5);
        assert_relative_eq!(x, x2, epsilon = 1e-10);
    }

    #[test]
    fn test_logistic_bounds() {
        assert_relative_eq!(logistic(100.0, 0.1, 0.5), 0.5, epsilon = 1e-10);
        assert_relative_eq!(logistic(-100.0, 0.1, 0.5), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_angle() {
        assert_relative_eq!(normalize_angle(3.5), 3.5 - 2.0 * PI, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(-3.5), -3.5 + 2.0 * PI, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(0.5), 0.5, epsilon = 1e-10);
    }
}
