use crate::core::types::{AimCommand, ArmorPositionWithYaw, TargetState};

#[derive(Debug, Clone)]
pub struct Planner {
    pub bullet_speed: f64,
    pub yaw_offset: f64,
    pub pitch_offset: f64,
    pub fire_thres_yaw: f64,
    pub fire_thres_pitch: f64,
}
impl Default for Planner {
    fn default() -> Self {
        Self {
            bullet_speed: 15.,
            yaw_offset: 0.,
            pitch_offset: 0.,
            fire_thres_yaw: 0.01,
            fire_thres_pitch: 0.01,
        }
    }
}
impl Planner {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn plan(&self, target: &TargetState) -> AimCommand {
        let armor = target.armors.first().cloned().unwrap_or(ArmorPositionWithYaw {
            position: target.center_position,
            yaw: nalgebra::Rotation2::new(target.center_yaw),
        });
        let pos = armor.position;
        let dist = (pos.x.powi(2) + pos.y.powi(2) + pos.z.powi(2)).sqrt();
        let bt = dist / self.bullet_speed;
        let pp = pos + target.center_velocity * bt;
        let ty = pp.y.atan2(pp.x);
        let tp = (-pp.z).atan2((pp.x.powi(2) + pp.y.powi(2)).sqrt());
        let ye = (ty - self.yaw_offset).abs();
        let pe = (tp - self.pitch_offset).abs();
        AimCommand {
            control: true,
            fire_thres_yaw: ye,
            fire_thres_pitch: pe,
            target_yaw: ty,
            target_pitch: tp,
            yaw: ty,
            yaw_vel: 0.,
            yaw_acc: 0.,
            pitch: tp,
            pitch_vel: 0.,
            pitch_acc: 0.,
            bullet_id: 0,
        }
    }
}
