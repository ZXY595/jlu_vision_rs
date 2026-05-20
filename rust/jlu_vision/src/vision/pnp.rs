use crate::core::camera;
use crate::core::math;
use crate::core::types::{Armor, ArmorType, CameraInfo};
use nalgebra::Rotation3;
use nalgebra::{Matrix3, Matrix3x4, Point2, Point3, UnitQuaternion, Vector3};

#[derive(Debug, Clone)]
pub struct PnPConfig {
    pub project_error_ratio_thres: f64,
    pub roll_thres_degree: f64,
    pub outpost_is_small_armor: bool,
}
impl Default for PnPConfig {
    fn default() -> Self {
        Self {
            project_error_ratio_thres: 3.,
            roll_thres_degree: 30.,
            outpost_is_small_armor: false,
        }
    }
}

pub struct PnPResult {
    pub rvec: Vector3<f64>,
    pub tvec: Vector3<f64>,
    pub rvec_alt: Vector3<f64>,
    pub tvec_alt: Vector3<f64>,
    pub error0: f64,
    pub error1: f64,
}

pub struct PnPSolver {
    camera_matrix: Matrix3<f64>,
    distortion_coefficients: [f64; 5],
    config: PnPConfig,
}

impl PnPSolver {
    pub fn new(camera_info: &CameraInfo, config: PnPConfig) -> Self {
        Self {
            camera_matrix: camera_info.matrix,
            distortion_coefficients: camera_info.distortion_coefficients,
            config,
        }
    }

    pub fn distance_to_center(&self, point: &Point2<f64>) -> f64 {
        let cx = self.camera_matrix[(0, 2)];
        let cy = self.camera_matrix[(1, 2)];
        ((point.x - cx).powi(2) + (point.y - cy).powi(2)).sqrt()
    }

    pub fn solve(&self, armor: &mut Armor) -> Option<PnPResult> {
        let obj = self.get_object_points(armor.armor_type);
        let img = [
            Point2::new(armor.left_light.bottom.x, armor.left_light.bottom.y),
            Point2::new(armor.left_light.top.x, armor.left_light.top.y),
            Point2::new(armor.right_light.top.x, armor.right_light.top.y),
            Point2::new(armor.right_light.bottom.x, armor.right_light.bottom.y),
        ];
        let result = self.ippe(&obj, &img)?;
        let sorted = self.sort_result(armor, result)?;
        armor.position = Point3::new(sorted.tvec.x, sorted.tvec.y, sorted.tvec.z);
        let rot = nalgebra::Rotation3::from_scaled_axis(sorted.rvec);
        armor.orientation = UnitQuaternion::from_rotation_matrix(&rot);
        Some(sorted)
    }

    fn get_object_points(&self, at: ArmorType) -> [Point3<f64>; 4] {
        let t = if at == ArmorType::Outpost && self.config.outpost_is_small_armor {
            ArmorType::Three
        } else {
            at
        };
        crate::core::types::get_armor_points(t)
    }

    fn ippe(&self, obj: &[Point3<f64>; 4], img: &[Point2<f64>; 4]) -> Option<PnPResult> {
        let h = Self::compute_homography(obj, img, &self.camera_matrix)?;
        let (r1, r2, t1, t2) = decompose_homography(&h, &self.camera_matrix)?;
        let e0 = self.reproj_err(obj, img, &r1, &t1);
        let e1 = self.reproj_err(obj, img, &r2, &t2);
        Some(PnPResult {
            rvec: rot_to_rvec(&r1),
            tvec: t1,
            rvec_alt: rot_to_rvec(&r2),
            tvec_alt: t2,
            error0: e0,
            error1: e1,
        })
    }

    fn compute_homography(
        obj: &[Point3<f64>; 4],
        img: &[Point2<f64>; 4],
        k: &Matrix3<f64>,
    ) -> Option<Matrix3<f64>> {
        let ki = k.try_inverse()?;
        let mut a = Matrix3x4::zeros();
        for i in 0..4 {
            let p = ki * img[i].to_homogeneous();
            let u = p.x / p.z;
            let v = p.y / p.z;
            let x = obj[i].x;
            let y = obj[i].y;
            let r = 2 * i;
            a[(r, 0)] = -x;
            a[(r, 1)] = -y;
            a[(r, 2)] = -1.;
            a[(r, 6)] = u * x;
            a[(r, 7)] = u * y;
            a[(r, 8)] = u;
            a[(r + 1, 3)] = -x;
            a[(r + 1, 4)] = -y;
            a[(r + 1, 5)] = -1.;
            a[(r + 1, 6)] = v * x;
            a[(r + 1, 7)] = v * y;
            a[(r + 1, 8)] = v;
        }
        let svd = a.svd(true, true);
        let vt = svd.v_t?;
        let hv = vt.row(8);
        let mut h = Matrix3::zeros();
        for r in 0..3 {
            for c in 0..3 {
                h[(r, c)] = hv[r * 3 + c];
            }
        }
        if h[(2, 2)].abs() > 1e-10 {
            h /= h[(2, 2)];
        }
        Some(k * h)
    }

    fn reproj_err(
        &self,
        obj: &[Point3<f64>; 4],
        img: &[Point2<f64>; 4],
        r: &Matrix3<f64>,
        t: &Vector3<f64>,
    ) -> f64 {
        let mut e = 0.;
        for i in 0..4 {
            let pt = r * obj[i].coords + t;
            if pt.z <= 0. {
                return f64::MAX;
            }
            let proj = camera::project(
                &Point3::from(pt),
                &CameraInfo {
                    matrix: self.camera_matrix,
                    distortion_coefficients: self.distortion_coefficients,
                },
            );
            e += (proj - img[i]).norm();
        }
        e / 4.
    }

    fn sort_result(&self, armor: &Armor, mut r: PnPResult) -> Option<PnPResult> {
        if r.error1 / r.error0 > self.config.project_error_ratio_thres {
            return Some(r);
        }
        let rc2r = Matrix3::new(0., 0., 1., -1., 0., 0., 0., -1., 0.);
        let r0 = rvec_to_rot(&r.rvec);
        let r1 = rvec_to_rot(&r.rvec_alt);
        let rpy0 = Rotation3::from_matrix(&(rc2r * r0)).euler_angles();
        let rpy1 = Rotation3::from_matrix(&(rc2r * r1)).euler_angles();
        let rd = math::rad_to_deg(
            math::normalize_angle(rpy0.0)
                .clamp(-std::f64::consts::PI / 2., std::f64::consts::PI / 2.),
        );
        if rd.abs() > self.config.roll_thres_degree {
            return Some(r);
        }
        let la = (armor.left_light.top.y)
            .atan2(armor.left_light.top.x)
            .to_degrees();
        let ra = (armor.right_light.top.y)
            .atan2(armor.right_light.top.x)
            .to_degrees();
        let mut aa = (la + ra) / 2. + 90.;
        if armor.armor_type == ArmorType::Outpost {
            aa = -aa;
        }
        if (aa > 0. && rpy0.2 > 0. && rpy1.2 < 0.) || (aa < 0. && rpy0.2 < 0. && rpy1.2 > 0.) {
            std::mem::swap(&mut r.rvec, &mut r.rvec_alt);
            std::mem::swap(&mut r.tvec, &mut r.tvec_alt);
        }
        Some(r)
    }
}

fn decompose_homography(
    h: &Matrix3<f64>,
    k: &Matrix3<f64>,
) -> Option<(Matrix3<f64>, Matrix3<f64>, Vector3<f64>, Vector3<f64>)> {
    let ki = k.try_inverse()?;
    let mut hn = ki * h;
    let s = (hn.column(0).norm() + hn.column(1).norm()) / 2.;
    hn /= s;
    let r1 = hn.column(0).into_owned();
    let r2 = hn.column(1).into_owned();
    let t = hn.column(2).into_owned();
    let r12 = Matrix3::from_columns(&[r1, r2, r1.cross(&r2)]);
    let svd = r12.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let mut r = u * vt;
    if r.determinant() < 0. {
        r = -r;
    }
    let pos = t.dot(&r.column(2)) > 0.;
    let mut rc = r;
    rc[(0, 2)] = -rc[(0, 2)];
    rc[(1, 2)] = -rc[(1, 2)];
    rc[(2, 2)] = -rc[(2, 2)];
    if pos {
        Some((r, rc, t, t))
    } else {
        Some((r, rc, -t, -t))
    }
}
fn rvec_to_rot(rvec: &Vector3<f64>) -> Matrix3<f64> {
    let t = rvec.norm();
    if t < 1e-10 {
        return Matrix3::identity();
    }
    let k = rvec / t;
    let kx = Matrix3::new(0., -k.z, k.y, k.z, 0., -k.x, -k.y, k.x, 0.);
    Matrix3::identity() + t.sin() * kx + (1. - t.cos()) * kx * kx
}
fn rot_to_rvec(r: &Matrix3<f64>) -> Vector3<f64> {
    let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let ct = ((tr - 1.) / 2.).clamp(-1., 1.);
    let t = ct.acos();
    if t.abs() < 1e-10 {
        return Vector3::zeros();
    }
    let ts = 2. * t.sin();
    if ts.abs() < 1e-10 {
        return Vector3::zeros();
    }
    Vector3::new(
        (r[(2, 1)] - r[(1, 2)]) / ts * t,
        (r[(0, 2)] - r[(2, 0)]) / ts * t,
        (r[(1, 0)] - r[(0, 1)]) / ts * t,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_rvec_roundtrip() {
        let rv = Vector3::new(0.5, -0.3, 1.2);
        let r = rvec_to_rot(&rv);
        let rv2 = rot_to_rvec(&r);
        assert!((rv - rv2).norm() < 1e-6);
    }
    #[test]
    fn test_homography_decomp() {
        let k = Matrix3::new(1000., 0., 640., 0., 1000., 360., 0., 0., 1.);
        let rot = nalgebra::Rotation3::from_euler_angles(0.1, -0.2, 0.3);
        let r = rot.matrix();
        let t = Vector3::new(0., 0., 2.);
        let mut h =
            k * Matrix3::from_columns(&[r.column(0).into_owned(), r.column(1).into_owned(), t]);
        h /= h[(2, 2)];
        assert!(decompose_homography(&h, &k).is_some());
    }
}
