#![allow(non_upper_case_globals)]

use factrs::{
    assign_symbols,
    containers::Graph,
    linalg::{Const, ForwardProp, Numeric, VectorX, vectorx},
    mark,
    noise::GaussianNoise,
    residuals::{Residual1, Residual2, Residual3},
    variables::{MatrixLieGroup as _, SE3, SO2, Variable, VectorVar1, VectorVar3},
};

use crate::core::types::{self, ArmorType, CameraInfo};

assign_symbols! {
Position: VectorVar3; Velocity: VectorVar3; Rotation: SO2; Vyaw: VectorVar1; RadiusA: VectorVar1; RadiusB: VectorVar1; Dz: VectorVar1; ArmorPose: SE3
}
#[derive(Clone, Debug)]
pub struct TranslationResidual {
    pub dt: f64,
}

#[mark]
impl Residual3 for TranslationResidual {
    type Differ = ForwardProp<Const<3>>;
    type V1 = VectorVar3;
    type V2 = VectorVar3;
    type V3 = VectorVar3;
    type DimIn = Const<3>;
    type DimOut = Const<3>;
    fn residual3<T: Numeric>(
        &self,
        xp: VectorVar3<T>,
        vp: VectorVar3<T>,
        xc: VectorVar3<T>,
    ) -> VectorX<T> {
        VectorVar3::from(xp.0 + vp.0 * T::from(self.dt)).ominus(&xc)
    }
}

#[derive(Clone, Debug)]
pub struct YawResidual {
    pub dt: f64,
}

#[mark]
impl Residual3 for YawResidual {
    type Differ = ForwardProp<Const<1>>;
    type V1 = SO2;
    type V2 = VectorVar1;
    type V3 = SO2;
    type DimIn = Const<1>;
    type DimOut = Const<1>;
    fn residual3<T: Numeric>(&self, rp: SO2<T>, wp: VectorVar1<T>, rc: SO2<T>) -> VectorX<T> {
        let pred = rp.compose(&SO2::from_theta(wp.0[0] * T::from(self.dt)));
        pred.ominus(&rc)
    }
}

#[derive(Clone, Debug)]
pub struct VelocityResidual;

#[mark]
impl Residual2 for VelocityResidual {
    type Differ = ForwardProp<Const<3>>;
    type V1 = VectorVar3;
    type V2 = VectorVar3;
    type DimIn = Const<3>;
    type DimOut = Const<3>;
    fn residual2<T: Numeric>(&self, vp: VectorVar3<T>, vc: VectorVar3<T>) -> VectorX<T> {
        vp.ominus(&vc)
    }
}

#[derive(Clone, Debug)]
pub struct VyawResidual;

#[mark]
impl Residual2 for VyawResidual {
    type Differ = ForwardProp<Const<1>>;
    type V1 = VectorVar1;
    type V2 = VectorVar1;
    type DimIn = Const<1>;
    type DimOut = Const<1>;
    fn residual2<T: Numeric>(&self, wp: VectorVar1<T>, wc: VectorVar1<T>) -> VectorX<T> {
        wp.ominus(&wc)
    }
}

#[derive(Clone, Debug)]
pub struct ReprojectionResidual {
    armor_point: nalgebra::Point3<f64>,
    pixel_observed: nalgebra::Vector2<f64>,
    camera_info: CameraInfo,
}

impl ReprojectionResidual {
    pub fn new(
        armor_type: ArmorType,
        point_index: usize,
        pixel: nalgebra::Vector2<f64>,
        camera_info: &CameraInfo,
    ) -> Self {
        let points = types::get_armor_points(armor_type);
        Self {
            armor_point: points[point_index],
            pixel_observed: pixel,
            camera_info: camera_info.clone(),
        }
    }
}

#[mark]
impl Residual1 for ReprojectionResidual {
    type Differ = ForwardProp<Const<6>>;
    type V1 = SE3;
    type DimIn = Const<6>;
    type DimOut = Const<2>;
    fn residual1<T: Numeric>(&self, pose: SE3<T>) -> VectorX<T> {
        let pc = pose.apply(self.armor_point.cast().coords.as_view());
        let fx = T::from(self.camera_info.camera_matrix[0][0]);
        let fy = T::from(self.camera_info.camera_matrix[1][1]);
        let cx = T::from(self.camera_info.camera_matrix[0][2]);
        let cy = T::from(self.camera_info.camera_matrix[1][2]);
        let k1 = T::from(self.camera_info.distortion_coefficients[0]);
        let k2 = T::from(self.camera_info.distortion_coefficients[1]);
        let p1 = T::from(self.camera_info.distortion_coefficients[2]);
        let p2 = T::from(self.camera_info.distortion_coefficients[3]);
        let xn = pc.x / pc.z;
        let yn = pc.y / pc.z;
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let radial = T::from(1.0) + k1 * r2 + k2 * r4;
        let xd = xn * radial + T::from(2.0) * p1 * xn * yn + p2 * (r2 + T::from(2.0) * xn * xn);
        let yd = yn * radial + p1 * (r2 + T::from(2.0) * yn * yn) + T::from(2.0) * p2 * xn * yn;
        vectorx![
            fx * xd + cx - T::from(self.pixel_observed.x),
            fy * yd + cy - T::from(self.pixel_observed.y)
        ]
    }
}

pub struct FactorGraph {
    graph: Graph,
}
impl FactorGraph {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
    }
    pub fn inner(&self) -> &Graph {
        &self.graph
    }
    pub fn add_translation(&mut self, xp: Position, vp: Velocity, xc: Position, dt: f64) {
        self.graph.add_factor(
            factrs::containers::FactorBuilder::new3(TranslationResidual { dt }, xp, vp, xc)
                .noise(GaussianNoise::<3>::from_diag_sigmas(0.1, 0.1, 0.1))
                .build(),
        );
    }
    pub fn add_yaw(&mut self, rp: Rotation, wp: Vyaw, rc: Rotation, dt: f64) {
        self.graph.add_factor(
            factrs::containers::FactorBuilder::new3(YawResidual { dt }, rp, wp, rc)
                .noise(GaussianNoise::<1>::from_diag_sigmas(0.01))
                .build(),
        );
    }
    pub fn add_velocity(&mut self, vp: Velocity, vc: Velocity) {
        self.graph.add_factor(
            factrs::containers::FactorBuilder::new2(VelocityResidual, vp, vc)
                .noise(GaussianNoise::<3>::from_diag_sigmas(0.5, 0.5, 0.5))
                .build(),
        );
    }
    pub fn add_vyaw(&mut self, wp: Vyaw, wc: Vyaw) {
        self.graph.add_factor(
            factrs::containers::FactorBuilder::new2(VyawResidual, wp, wc)
                .noise(GaussianNoise::<1>::from_diag_sigmas(0.1))
                .build(),
        );
    }
    pub fn add_reprojection(
        &mut self,
        ak: ArmorPose,
        at: ArmorType,
        pi: usize,
        px: nalgebra::Vector2<f64>,
        camera: &CameraInfo,
    ) {
        self.graph.add_factor(
            factrs::containers::FactorBuilder::new1(ReprojectionResidual::new(at, pi, px, camera), ak)
                .noise(GaussianNoise::<2>::from_diag_sigmas(1.0, 1.0))
                .build(),
        );
    }
}

impl Default for FactorGraph {
    fn default() -> Self {
        Self::new()
    }
}
