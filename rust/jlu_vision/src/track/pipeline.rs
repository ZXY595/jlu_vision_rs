use crate::core::tf::TfBuffer;
use crate::core::types::*;
use crate::track::planner::Planner;
use crate::track::target::RobotTarget;
use async_channel::{Receiver, Sender};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    pub camera_info: CameraInfo,
    pub odom_frame_id: String,
    pub camera_frame_id: String,
    pub default_radius: f64,
    pub default_dz: f64,
    pub aiming_change_count: u32,
}
impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            camera_info: CameraInfo {
                camera_matrix: [[1000., 0., 720.], [0., 1000., 540.], [0., 0., 1.]],
                distortion_coefficients: [0.; 5],
            },
            odom_frame_id: "odom".into(),
            camera_frame_id: "camera".into(),
            default_radius: 0.2,
            default_dz: 0.,
            aiming_change_count: 5,
        }
    }
}

pub struct TrackerPipeline {
    config: TrackerConfig,
}

impl TrackerPipeline {
    pub fn new(config: TrackerConfig) -> Self {
        Self { config }
    }

    pub async fn run(self, armor_rx: Receiver<Vec<Armor>>, cmd_tx: Sender<AimCommand>) {
        log::info!("Tracker pipeline started");
        let tf = TfBuffer::new();
        let planner = Planner::new();
        let mut targets: HashMap<ArmorType, RobotTarget> = HashMap::new();
        let mut aiming = ArmorType::Negative;
        let mut cc = 0u32;
        for armor_ty in [
            ArmorType::One,
            ArmorType::Two,
            ArmorType::Three,
            ArmorType::Four,
            ArmorType::Sentry,
        ] {
            targets.insert(
                armor_ty,
                RobotTarget::new(
                    armor_ty,
                    self.config.camera_info.clone(),
                    self.config.default_radius,
                    self.config.default_dz,
                ),
            );
        }
        while let Ok(armors) = armor_rx.recv().await {
            if armors.is_empty() {
                let _ = cmd_tx
                    .send(AimCommand {
                        control: false,
                        ..Default::default()
                    })
                    .await;
                continue;
            }
            let stamp = armors[0].stamp;
            let t_co = tf
                .lookup(
                    &self.config.odom_frame_id,
                    &self.config.camera_frame_id,
                    stamp,
                    std::time::Duration::from_millis(50),
                )
                .unwrap_or(nalgebra::Isometry3::identity());
            let dets: Vec<ArmorDetection> = armors
                .iter()
                .filter(|a| !a.heart_beat)
                .map(ArmorDetection::from)
                .collect();
            let mut all_lost = true;
            let mut best_at = ArmorType::Negative;
            let mut best_d = f64::MAX;
            for (&at, t) in &targets {
                let ma: Vec<ArmorDetection> = dets
                    .iter()
                    .filter(|a| a.armor_type == at)
                    .cloned()
                    .collect();
                if t.update(&ma, stamp, &t_co) == TrackStateKind::Tracking {
                    all_lost = false;
                }
                if let Some(f) = ma.first() {
                    let d = f.position.coords.norm();
                    if d < best_d {
                        best_d = d;
                        best_at = at;
                    }
                }
            }
            if all_lost {
                aiming = ArmorType::Negative;
                cc = 0;
                let _ = cmd_tx
                    .send(AimCommand {
                        control: false,
                        ..Default::default()
                    })
                    .await;
                continue;
            }
            if best_at != ArmorType::Negative && best_at != aiming {
                cc += 1;
                if cc >= self.config.aiming_change_count {
                    aiming = best_at;
                    cc = 0;
                }
            } else if best_at == aiming {
                cc = 0;
            }
            if let Some(t) = targets.get(&aiming) {
                let (ts, _) = t.get_state();
                let cmd = planner.plan(&ts);
                log::debug!(
                    "Tracker: aiming {:?}, yaw={:.3} pitch={:.3}",
                    aiming,
                    cmd.yaw,
                    cmd.pitch
                );
                let _ = cmd_tx.send(cmd).await;
            }
        }
        log::info!("Tracker pipeline stopped");
    }
}
