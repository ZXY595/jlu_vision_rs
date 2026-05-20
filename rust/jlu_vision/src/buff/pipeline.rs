use crate::buff::detector::{BuffDetector, BuffDetectorConfig};
use crate::buff::tracker::BuffTracker;
use crate::core::types::*;
use async_channel::{Receiver, Sender};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct BuffPipelineConfig {
    pub camera_info: CameraInfo,
    pub bullet_speed: f64,
    pub is_big_buff: bool,
    pub detector_config: BuffDetectorConfig,
}
impl Default for BuffPipelineConfig {
    fn default() -> Self {
        Self {
            camera_info: CameraInfo {
                camera_matrix: [[2000., 0., 720.], [0., 2000., 540.], [0., 0., 1.]],
                distortion_coefficients: [0.; 5],
            },
            bullet_speed: 15.,
            is_big_buff: false,
            detector_config: BuffDetectorConfig::default(),
        }
    }
}

pub struct BuffPipeline {
    detector: Arc<BuffDetector>,
    tracker: Arc<BuffTracker>,
}

impl BuffPipeline {
    pub fn new(config: BuffPipelineConfig) -> Self {
        let mut d = BuffDetector::new(config.detector_config);
        d.set_image_size(1440, 1080);
        let t = BuffTracker::new(config.camera_info, config.bullet_speed, config.is_big_buff);
        Self {
            detector: Arc::new(d),
            tracker: Arc::new(t),
        }
    }

    pub fn process(&self, frame: &ImageFrame, output: &[f32]) -> Option<AimCommand> {
        let runes = self.detector.postprocess(output);
        if runes.is_empty() {
            return None;
        }
        let ti = nalgebra::Isometry3::identity();
        if self.tracker.update(&runes, frame.stamp, &ti) != TrackStateKind::Tracking {
            return None;
        }
        Some(self.tracker.aim())
    }

    pub async fn run(self, image_rx: Receiver<(ImageFrame, Vec<f32>)>, cmd_tx: Sender<AimCommand>) {
        log::info!(
            "Buff pipeline started ({} buff)",
            if self.tracker.is_big_buff {
                "big"
            } else {
                "small"
            }
        );
        while let Ok((frame, output)) = image_rx.recv().await {
            if let Some(cmd) = self.process(&frame, &output) {
                let _ = cmd_tx.send(cmd).await;
            }
        }
        log::info!("Buff pipeline stopped");
    }
}
