use crate::core::types::{Armor, CameraInfo, ImageFrame};
use crate::vision::detector::{MockBackend, YoloDetector};
use crate::vision::pca::{LightCornerCorrector, PcaConfig};
use crate::vision::pnp::{PnPConfig, PnPSolver};
use async_channel::Receiver;
use futures_lite::stream::Stream;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub model_path: PathBuf,
    pub camera_info: CameraInfo,
    pub score_thresh: f32,
    pub nms_iou_thresh: f32,
    pub accept_thresh: f32,
    pub use_pca: bool,
    pub pca_config: PcaConfig,
    pub pnp_config: PnPConfig,
}
impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("assets/0526.onnx"),
            camera_info: CameraInfo {
                camera_matrix: [[2000., 0., 720.], [0., 2000., 540.], [0., 0., 1.]],
                distortion_coefficients: [0.; 5],
            },
            score_thresh: 0.3,
            nms_iou_thresh: 0.45,
            accept_thresh: 0.5,
            use_pca: true,
            pca_config: PcaConfig::default(),
            pnp_config: PnPConfig::default(),
        }
    }
}

pub struct DetectorPipeline {
    detector: Arc<YoloDetector>,
    corrector: Option<LightCornerCorrector>,
    solver: PnPSolver,
}

impl DetectorPipeline {
    pub fn new(config: DetectorConfig) -> anyhow::Result<Self> {
        let detector = Arc::new(YoloDetector::new(
            Box::new(MockBackend),
            config.score_thresh,
            config.nms_iou_thresh,
            config.accept_thresh,
        ));
        let corrector = config
            .use_pca
            .then(|| LightCornerCorrector::new(config.pca_config));
        let solver = PnPSolver::new(&config.camera_info, config.pnp_config);
        log::info!(
            "Detector pipeline initialized: model={}, PCA={}",
            config.model_path.display(),
            config.use_pca
        );
        Ok(Self {
            detector,
            corrector,
            solver,
        })
    }

    pub fn process(&self, frame: &ImageFrame) -> Vec<Armor> {
        let mut armors = self.detector.detect(&frame.image).unwrap_or_default();
        for armor in &mut armors {
            armor.frame_id = frame.frame_id.clone();
            armor.stamp = frame.stamp;
            if let Some(ref corrector) = self.corrector {
                let gray = image::DynamicImage::ImageRgb8(frame.image.clone()).into_luma8();
                armor.key_frame = corrector.correct(armor, &gray);
            }
            armor.distance_to_image_center = self.solver.distance_to_center(&armor.center);
            let _ = self.solver.solve(armor);
        }
        armors
    }

    pub fn into_stream(self, image_rx: Receiver<ImageFrame>) -> impl Stream<Item = Vec<Armor>> {
        let me = Arc::new(self);
        futures_lite::stream::unfold((image_rx, me), |(rx, slf)| async move {
            let frame = rx.recv().await.ok()?;
            let armors = slf.process(&frame);
            Some((armors, (rx, slf)))
        })
    }
}
