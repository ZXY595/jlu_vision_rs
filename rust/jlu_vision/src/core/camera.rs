use crate::core::types::CameraInfo;

pub fn project(
    point_3d: &nalgebra::Point3<f64>,
    camera_info: &CameraInfo,
) -> nalgebra::Point2<f64> {
    let fx = camera_info.camera_matrix[0][0];
    let fy = camera_info.camera_matrix[1][1];
    let cx = camera_info.camera_matrix[0][2];
    let cy = camera_info.camera_matrix[1][2];
    let k1 = camera_info.distortion_coefficients[0];
    let k2 = camera_info.distortion_coefficients[1];
    let p1 = camera_info.distortion_coefficients[2];
    let p2 = camera_info.distortion_coefficients[3];
    let k3 = camera_info.distortion_coefficients[4];
    let xn = point_3d.x / point_3d.z;
    let yn = point_3d.y / point_3d.z;
    let r2 = xn * xn + yn * yn;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    let x_dist = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
    let y_dist = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
    nalgebra::Point2::new(fx * x_dist + cx, fy * y_dist + cy)
}

pub fn back_project(
    pixel: &nalgebra::Point2<f64>,
    depth: f64,
    camera_info: &CameraInfo,
) -> nalgebra::Point3<f64> {
    let fx = camera_info.camera_matrix[0][0];
    let fy = camera_info.camera_matrix[1][1];
    let cx = camera_info.camera_matrix[0][2];
    let cy = camera_info.camera_matrix[1][2];
    nalgebra::Point3::new(
        (pixel.x - cx) * depth / fx,
        (pixel.y - cy) * depth / fy,
        depth,
    )
}
