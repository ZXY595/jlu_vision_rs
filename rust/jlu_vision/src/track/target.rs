use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use factrs::{
    containers::Values,
    optimizers::{GaussNewton, Optimizer},
    variables::{SE3, SO2, SO3, VectorVar1, VectorVar3},
};
use nalgebra::{Isometry3, Point3, Rotation2, Vector2};

use crate::core::math;
use crate::core::types::*;
use crate::track::factors::*;

fn armor_key(frame_k: u64, idx: ArmorIndex) -> ArmorPose {
    ArmorPose((frame_k * 4) as u32 + idx as u32)
}

fn armor_between_yaw(i: ArmorIndex, n: usize) -> f64 {
    i as usize as f64 * (2.0 * std::f64::consts::PI / n as f64)
}

fn armor_from_center(
    center: &Point3<f64>,
    center_yaw: f64,
    radius: f64,
    dz: f64,
    i: ArmorIndex,
    n: usize,
) -> ArmorPositionWithYaw {
    let y = center_yaw + armor_between_yaw(i, n);
    ArmorPositionWithYaw {
        position: Point3::new(
            center.x + radius * y.cos(),
            center.y + radius * y.sin(),
            center.z + dz,
        ),
        yaw: Rotation2::new(y),
    }
}

fn match_armors(
    pred: &[ArmorPositionWithYaw],
    obs: &[ArmorDetection],
    max_d: f64,
    max_y: f64,
) -> Vec<(usize, usize)> {
    let mut m = vec![];
    let mut u = [false; 4];
    for (oi, o) in obs.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (pi, p) in pred.iter().enumerate().take(4) {
            let d = (o.position - p.position).norm();
            let mut yd = (o.yaw.angle() - p.yaw.angle()).abs();
            yd = yd.min(2.0 * std::f64::consts::PI - yd);
            if d < max_d && yd < max_y && !u[pi] && best.is_none_or(|(_, s)| yd < s) {
                best = Some((pi, yd));
            }
        }
        if let Some((idx, _)) = best {
            u[idx] = true;
            m.push((idx, oi));
        }
    }
    m
}

pub struct RobotTarget {
    pub armor_type: ArmorType,
    pub camera_info: CameraInfo,
    pub state: Mutex<RobotTargetState>,
    pub track_state: Mutex<TrackState>,
    pub max_match_distance: f64,
    pub max_yaw_diff: f64,
    pub lost_threshold: Duration,
}

impl RobotTarget {
    pub fn new(
        armor_type: ArmorType,
        camera_info: CameraInfo,
        default_radius: f64,
        default_dz: f64,
    ) -> Self {
        Self {
            armor_type,
            camera_info,
            state: Mutex::new(RobotTargetState {
                state: TargetState {
                    armor_type,
                    center_position: Point3::origin(),
                    center_velocity: nalgebra::Vector3::zeros(),
                    center_yaw: 0.,
                    center_vyaw: 0.,
                    armors: vec![],
                },
                radius_a: default_radius,
                radius_b: default_radius,
                dz: default_dz,
            }),
            track_state: Mutex::new(TrackState {
                state: TrackStateKind::Lost,
                stamp_last_update: SystemTime::UNIX_EPOCH,
                stamp_last_tracking: SystemTime::UNIX_EPOCH,
                k: 0,
            }),
            max_match_distance: 0.5,
            max_yaw_diff: 0.5,
            lost_threshold: Duration::from_secs_f64(1.0),
        }
    }

    pub fn get_state(&self) -> (TargetState, TrackStateKind) {
        let s = self.state.lock().unwrap();
        let t = self.track_state.lock().unwrap();
        let mut b = s.state.clone();
        b.armors = Self::get_armors(&s.state, s.radius_a, s.radius_b, s.dz, 4);
        (b, t.state)
    }

    fn get_armors(
        state: &TargetState,
        radius_a: f64,
        radius_b: f64,
        dz: f64,
        count: usize,
    ) -> Vec<ArmorPositionWithYaw> {
        (0..count)
            .map(|i| {
                let idx = unsafe { std::mem::transmute::<u8, ArmorIndex>(i as _) };
                let radius = if i == 0 || i == 2 { radius_a } else { radius_b };
                let z = if i == 0 || i == 2 { 0.0 } else { dz };
                armor_from_center(
                    &state.center_position,
                    state.center_yaw,
                    radius,
                    z,
                    idx,
                    count,
                )
            })
            .collect()
    }

    pub fn update(
        &self,
        armors: &[ArmorDetection],
        stamp: SystemTime,
        t_co: &Isometry3<f64>,
    ) -> TrackStateKind {
        let mut track = self.track_state.lock().unwrap();
        let dt = stamp
            .duration_since(track.stamp_last_update)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64();
        let dts = track
            .stamp_last_update
            .duration_since(track.stamp_last_tracking)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64();
        if track.state != TrackStateKind::Lost && dt + dts > self.lost_threshold.as_secs_f64() {
            track.state = TrackStateKind::Lost;
            track.k = 0;
            return TrackStateKind::Lost;
        }
        let armor_obs: Vec<ArmorDetection> = armors
            .iter()
            .map(|a| {
                let mut a = a.clone();
                a.position = t_co * a.position;
                a
            })
            .collect();
        let mut target_state = { self.state.lock().unwrap().clone() };
        target_state.state = target_state.state.predict(dt);
        if track.state == TrackStateKind::Lost {
            if armor_obs.is_empty() {
                return TrackStateKind::Lost;
            }
            target_state = Self::init_from_armor(&armor_obs[0], self.armor_type);
        }
        let pred = Self::get_armors(
            &target_state.state,
            target_state.radius_a,
            target_state.radius_b,
            target_state.dz,
            4,
        );
        let matched = match_armors(
            &pred,
            &armor_obs,
            self.max_match_distance,
            self.max_yaw_diff,
        );

        let mut values = Values::new();
        let mut graph = FactorGraph::new();

        values.insert(
            Position(track.k as _),
            VectorVar3::from(target_state.state.center_position.coords),
        );
        values.insert(
            Velocity(track.k as _),
            VectorVar3::from(target_state.state.center_velocity),
        );
        values.insert(
            Rotation(track.k as _),
            SO2::from_theta(target_state.state.center_yaw),
        );
        values.insert(
            Vyaw(track.k as _),
            VectorVar1::new(target_state.state.center_vyaw),
        );

        if track.k == 0 {
            values.insert(
                RadiusA(0),
                VectorVar1::new(math::logistic_inverse(target_state.radius_a, 0.05, 0.3)),
            );
            values.insert(
                RadiusB(0),
                VectorVar1::new(math::logistic_inverse(target_state.radius_b, 0.05, 0.3)),
            );
            values.insert(Dz(0), VectorVar1::new(target_state.dz));
        }

        if track.k > 0 {
            graph.add_translation(
                Position((track.k - 1) as _),
                Velocity((track.k - 1) as _),
                Position(track.k as _),
                dt,
            );
            graph.add_yaw(
                Rotation((track.k - 1) as _),
                Vyaw((track.k - 1) as _),
                Rotation(track.k as _),
                dt,
            );
            graph.add_velocity(Velocity((track.k - 1) as _), Velocity(track.k as _));
            graph.add_vyaw(Vyaw((track.k - 1) as _), Vyaw(track.k as _));
        }

        for &(ai, oi) in &matched {
            let obs = &armors[oi];
            let idx = unsafe { std::mem::transmute::<u8, ArmorIndex>(ai as _) };
            values.insert(
                armor_key(track.k, idx),
                SE3::from_rot_trans(
                    SO3::from_vec(
                        nalgebra::UnitQuaternion::from_euler_angles(
                            obs.roll,
                            obs.pitch,
                            obs.yaw.angle(),
                        )
                        .coords,
                    ),
                    obs.position.coords,
                ),
            );

            for pi in 0..4 {
                graph.add_reprojection(
                    armor_key(track.k, idx),
                    self.armor_type,
                    pi,
                    Vector2::new(obs.points[pi].x, obs.points[pi].y),
                    &self.camera_info,
                );
            }
        }

        // Optimize
        let params = factrs::optimizers::BaseOptParams::default();
        let mut opt = GaussNewton::new(params, graph.inner().clone());
        if let Ok(opt_values) = opt.optimize(values) {
            if let Some(xc) = opt_values.get(Position(track.k as _)) {
                target_state.state.center_position = Point3::new(xc.0[0], xc.0[1], xc.0[2]);
            }
            if let Some(vc) = opt_values.get(Velocity(track.k as _)) {
                target_state.state.center_velocity =
                    nalgebra::Vector3::new(vc.0[0], vc.0[1], vc.0[2]);
            }
            if let Some(rc) = opt_values.get(Rotation(track.k as _)) {
                target_state.state.center_yaw = math::normalize_angle(rc.to_theta());
            }
            if let Some(wc) = opt_values.get(Vyaw(track.k as _)) {
                target_state.state.center_vyaw = wc.0[0];
            }
            if let Some(a_val) = opt_values.get(RadiusA(0)) {
                target_state.radius_a = math::logistic(a_val.0[0], 0.05, 0.3);
            }
            if let Some(b_val) = opt_values.get(RadiusB(0)) {
                target_state.radius_b = math::logistic(b_val.0[0], 0.05, 0.3);
            }
            if let Some(z_val) = opt_values.get(Dz(0)) {
                target_state.dz = z_val.0[0];
            }
        }

        *self.state.lock().unwrap() = target_state;
        let ns = if matched.is_empty() {
            TrackStateKind::TempLost
        } else {
            TrackStateKind::Tracking
        };
        track.state = ns;
        track.stamp_last_update = stamp;
        if ns == TrackStateKind::Tracking {
            track.stamp_last_tracking = stamp;
        }
        track.k += 1;
        ns
    }

    fn init_from_armor(armor: &ArmorDetection, armor_type: ArmorType) -> RobotTargetState {
        let dr = 0.2;
        let cx = armor.position.x + dr * armor.yaw.angle().cos();
        let cy = armor.position.y + dr * armor.yaw.angle().sin();
        RobotTargetState {
            state: TargetState {
                armor_type,
                center_position: Point3::new(cx, cy, armor.position.z),
                center_velocity: nalgebra::Vector3::zeros(),
                center_yaw: armor.yaw.angle(),
                center_vyaw: 0.,
                armors: vec![],
            },
            radius_a: dr,
            radius_b: dr,
            dz: 0.,
        }
    }
}
