use crate::core::math;
use crate::core::types::*;
use nalgebra::{Isometry3, Point2, Point3, Vector3};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub struct BuffAimResult {
    pub yaw: f64,
    pub pitch: f64,
    pub fly_time: f64,
    pub blade_index: BuffBladeIndex,
}

pub struct BuffTracker {
    camera_info: CameraInfo,
    state: Mutex<BuffState>,
    track_state: Mutex<TrackState>,
    roll_history: Mutex<VecDeque<(f64, f64)>>,
    big_params: Mutex<[f64; 4]>,
    bullet_speed: f64,
    lost_threshold: Duration,
    pub is_big_buff: bool,
}

impl BuffTracker {
    pub fn new(camera_info: CameraInfo, bullet_speed: f64, is_big_buff: bool) -> Self {
        Self {
            camera_info,
            state: Mutex::new(BuffState {
                center: Point3::origin(),
                roll: 0.,
                vroll: 0.,
                radius: BUFF_RADIUS,
                blades: vec![],
            }),
            track_state: Mutex::new(TrackState {
                state: TrackStateKind::Lost,
                stamp_last_update: SystemTime::UNIX_EPOCH,
                stamp_last_tracking: SystemTime::UNIX_EPOCH,
                k: 0,
            }),
            roll_history: Mutex::new(VecDeque::with_capacity(100)),
            big_params: Mutex::new([0., 1.884, 0., 0.]),
            bullet_speed,
            lost_threshold: Duration::from_secs_f64(1.0),
            is_big_buff,
        }
    }

    pub fn update(
        &self,
        blades: &[RuneObject],
        stamp: SystemTime,
        t_co: &Isometry3<f64>,
    ) -> TrackStateKind {
        let mut track = self.track_state.lock().unwrap();
        let dt = stamp
            .duration_since(track.stamp_last_update)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64();
        if track.state != TrackStateKind::Lost && dt > self.lost_threshold.as_secs_f64() {
            track.state = TrackStateKind::Lost;
            track.k = 0;
            return TrackStateKind::Lost;
        }
        if blades.is_empty() {
            if track.state == TrackStateKind::Tracking {
                track.state = TrackStateKind::TempLost;
            }
            return track.state;
        }
        let bps: Vec<BladePositionRollPoints> = blades
            .iter()
            .filter_map(|b| self.solve_pnp(b, t_co))
            .collect();
        if bps.is_empty() {
            track.state = TrackStateKind::Lost;
            return TrackStateKind::Lost;
        }
        let mut state = self.state.lock().unwrap();
        let ac: Vector3<f64> =
            bps.iter().map(|b| b.position.coords).sum::<Vector3<f64>>() / bps.len() as f64;
        let roll = bps[0].position.y.atan2(bps[0].position.x);
        let nv = if track.k > 0 && dt > 1e-6 {
            math::normalize_angle(roll - state.roll) / dt
        } else {
            state.vroll
        };
        state.center = Point3::from(ac);
        state.vroll = nv;
        state.roll = roll;
        state.blades = bps
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let idx = unsafe { std::mem::transmute::<u8, BuffBladeIndex>(i as u8) };
                (idx, b.position)
            })
            .collect();
        if self.is_big_buff {
            let mut h = self.roll_history.lock().unwrap();
            let ts = h.front().map(|(t, _)| *t).unwrap_or(0.);
            h.push_back((ts + dt, roll));
            if h.len() > 100 {
                h.pop_front();
            }
            if h.len() >= 10
                && let Some(p) = self.fit_curve(&h)
            {
                *self.big_params.lock().unwrap() = p;
            }
        }
        track.state = TrackStateKind::Tracking;
        track.stamp_last_tracking = stamp;
        track.stamp_last_update = stamp;
        track.k += 1;
        TrackStateKind::Tracking
    }

    pub fn aim(&self) -> AimCommand {
        let state = self.state.lock().unwrap();
        if state.blades.is_empty() {
            return AimCommand {
                control: false,
                ..Default::default()
            };
        }
        let target = self.select_blade(&state);
        let (yaw, pitch) = self.compute_aim(&state, target);
        AimCommand {
            control: true,
            fire_thres_yaw: 0.,
            fire_thres_pitch: 0.,
            target_yaw: yaw,
            target_pitch: pitch,
            yaw,
            yaw_vel: 0.,
            yaw_acc: 0.,
            pitch,
            pitch_vel: 0.,
            pitch_acc: 0.,
            bullet_id: 0,
        }
    }

    fn solve_pnp(
        &self,
        blade: &RuneObject,
        t_co: &Isometry3<f64>,
    ) -> Option<BladePositionRollPoints> {
        let fx = self.camera_info.camera_matrix[0][0];
        let fy = self.camera_info.camera_matrix[1][1];
        let cx = self.camera_info.camera_matrix[0][2];
        let cy = self.camera_info.camera_matrix[1][2];
        let hp = ((blade.points.top_left.y - blade.points.bottom_left.y).abs()
            + (blade.points.top_right.y - blade.points.bottom_right.y).abs())
            / 2.;
        let bh = 0.317;
        let depth = fy * bh / hp.max(1.0);
        let x = (blade.points.center.x - cx) * depth / fx;
        let y = (blade.points.center.y - cy) * depth / fy;
        let pc = Point3::new(x, y, depth);
        let po = t_co * pc;
        Some(BladePositionRollPoints {
            position: po,
            roll: 0.,
            pitch: 0.,
            yaw: 0.,
            points: [
                Point2::new(blade.points.center.x, blade.points.center.y),
                Point2::new(blade.points.bottom_right.x, blade.points.bottom_right.y),
                Point2::new(blade.points.top_right.x, blade.points.top_right.y),
                Point2::new(blade.points.top_left.x, blade.points.top_left.y),
                Point2::new(blade.points.bottom_left.x, blade.points.bottom_left.y),
            ],
        })
    }

    fn select_blade(&self, state: &BuffState) -> BuffBladeIndex {
        state
            .blades
            .iter()
            .min_by(|(_, a), (_, b)| {
                let da = a.y.atan2(a.x).abs();
                let db = b.y.atan2(b.x).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| *i)
            .unwrap_or(BuffBladeIndex::_0)
    }

    fn compute_aim(&self, state: &BuffState, bi: BuffBladeIndex) -> (f64, f64) {
        let ft = state.center.coords.norm() / self.bullet_speed;
        let rp = if self.is_big_buff {
            let p = self.big_params.lock().unwrap();
            self.predict_big(state.roll, state.vroll, ft, &p)
        } else {
            state.roll + state.vroll * ft
        };
        let ba = rp + bi as usize as f64 * 2. * std::f64::consts::PI / 5.;
        let bp = Point3::new(
            state.center.x + BUFF_RADIUS * ba.cos(),
            state.center.y + BUFF_RADIUS * ba.sin(),
            state.center.z,
        );
        (
            bp.y.atan2(bp.x),
            (-bp.z).atan2((bp.x.powi(2) + bp.y.powi(2)).sqrt()),
        )
    }

    fn predict_big(&self, roll: f64, _vroll: f64, dt: f64, p: &[f64; 4]) -> f64 {
        roll + p[0] * (p[1] * dt + p[2]).sin() + p[3]
    }

    fn fit_curve(&self, h: &VecDeque<(f64, f64)>) -> Option<[f64; 4]> {
        let n = h.len();
        if n < 5 {
            return None;
        }
        let omega = 1.884;
        let mut ata = nalgebra::Matrix3::zeros();
        let mut atb = nalgebra::Vector3::zeros();
        for &(t, roll) in h.iter() {
            let c = (omega * t).cos();
            let s = (omega * t).sin();
            ata[(0, 0)] += c * c;
            ata[(0, 1)] += c * s;
            ata[(0, 2)] += c;
            ata[(1, 0)] += s * c;
            ata[(1, 1)] += s * s;
            ata[(1, 2)] += s;
            ata[(2, 0)] += c;
            ata[(2, 1)] += s;
            ata[(2, 2)] += 1.;
            atb[0] += roll * c;
            atb[1] += roll * s;
            atb[2] += roll;
        }
        let svd: nalgebra::SVD<f64, nalgebra::Const<3>, nalgebra::Const<3>> = ata.svd(true, true);
        let x = svd.solve(&atb, 1e-6).ok()?;
        let a = (x[0].powi(2) + x[1].powi(2)).sqrt();
        let c = x[1].atan2(x[0]);
        let d = x[2];
        Some([a, omega, c, d])
    }
}
