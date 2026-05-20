use crate::core::math;
use crate::core::types::{Armor, ArmorType, EnemyColor, LightBar};
use nalgebra::Point2;
use std::collections::HashMap;

pub trait InferBackend: Send + Sync {
    fn preprocess(&self, image: &image::RgbImage) -> anyhow::Result<YoloOutput>;
}

pub struct YoloOutput {
    pub data: Vec<f32>,
    pub num_detections: usize,
}

pub struct MockBackend;
impl InferBackend for MockBackend {
    fn preprocess(&self, _: &image::RgbImage) -> anyhow::Result<YoloOutput> {
        Ok(YoloOutput {
            data: vec![],
            num_detections: 0,
        })
    }
}

const INPUT_SIZE: u32 = 640;

pub struct YoloDetector {
    backend: Box<dyn InferBackend>,
    score_thresh: f32,
    nms_iou_thresh: f32,
    accept_thresh: f32,
    type_map: HashMap<i64, ArmorType>,
    color_map: HashMap<i64, EnemyColor>,
}

impl YoloDetector {
    pub fn new(
        backend: Box<dyn InferBackend>,
        score_thresh: f32,
        nms_iou_thresh: f32,
        accept_thresh: f32,
    ) -> Self {
        Self {
            backend,
            score_thresh,
            nms_iou_thresh,
            accept_thresh,
            type_map: [
                (0, ArmorType::Sentry),
                (1, ArmorType::One),
                (2, ArmorType::Two),
                (3, ArmorType::Three),
                (4, ArmorType::Four),
                (5, ArmorType::Negative),
                (6, ArmorType::Outpost),
                (7, ArmorType::Base),
                (8, ArmorType::Negative),
            ]
            .into(),
            color_map: [
                (0, EnemyColor::Blue),
                (1, EnemyColor::Red),
                (2, EnemyColor::Extinguished),
            ]
            .into(),
        }
    }

    pub fn detect(&self, image: &image::RgbImage) -> anyhow::Result<Vec<Armor>> {
        let output = self.backend.preprocess(image)?;
        Ok(self.postprocess(&output, image.width(), image.height()))
    }

    fn postprocess(&self, output: &YoloOutput, img_w: u32, img_h: u32) -> Vec<Armor> {
        if output.num_detections == 0 {
            return vec![];
        }
        let h = img_h as f32;
        let w = img_w as f32;
        let scale = INPUT_SIZE as f32 / h.max(w);
        let num_det = output.num_detections;

        let mut raw = vec![];

        for r in 0..num_det {
            let row = &output.data[r * 22..];
            if row.len() < 22 {
                continue;
            }
            let score = math::logistic(row[8] as f64, 0.0, 1.0) as f32;
            if score < self.score_thresh {
                continue;
            }
            let (color_id, _) = (9..13)
                .map(|i| (i - 9, row[i]))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            let (class_id, _) = (13..22)
                .map(|i| (i - 13, row[i]))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            let kp: [[f32; 2]; 4] = [
                [row[0] / scale, row[1] / scale],
                [row[2] / scale, row[3] / scale],
                [row[4] / scale, row[5] / scale],
                [row[6] / scale, row[7] / scale],
            ];
            let min_x = kp.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
            let max_x = kp.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
            let min_y = kp.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
            let max_y = kp.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
            raw.push(RawDet {
                bbox: [min_x, min_y, max_x, max_y],
                keypoints: kp,
                confidence: score,
                color_id: color_id as i64,
                class_id: class_id as i64,
            });
        }
        let indices = nms(&raw, self.nms_iou_thresh);
        indices
            .iter()
            .filter_map(|&i| {
                let d = &raw[i];
                if d.confidence < self.accept_thresh {
                    return None;
                }
                let at = self
                    .type_map
                    .get(&d.class_id)
                    .copied()
                    .unwrap_or(ArmorType::Negative);
                if at == ArmorType::Negative {
                    return None;
                }
                let color = self
                    .color_map
                    .get(&d.color_id)
                    .copied()
                    .unwrap_or(EnemyColor::Blue);
                let ll = LightBar {
                    top: Point2::new(d.keypoints[1][0] as f64, d.keypoints[1][1] as f64),
                    bottom: Point2::new(d.keypoints[0][0] as f64, d.keypoints[0][1] as f64),
                };
                let rl = LightBar {
                    top: Point2::new(d.keypoints[3][0] as f64, d.keypoints[3][1] as f64),
                    bottom: Point2::new(d.keypoints[2][0] as f64, d.keypoints[2][1] as f64),
                };
                Some(Armor {
                    armor_type: at,
                    armor_color: color,
                    distance_to_image_center: 0.,
                    position: nalgebra::Point3::new(0., 0., 0.),
                    orientation: nalgebra::UnitQuaternion::identity(),
                    left_light: ll,
                    right_light: rl,
                    center: Point2::new(
                        (ll.top.x + rl.bottom.x) / 2.,
                        (ll.top.y + rl.bottom.y) / 2.,
                    ),
                    confidence: d.confidence,
                    key_frame: true,
                    heart_beat: false,
                    frame_id: String::new(),
                    stamp: std::time::SystemTime::now(),
                })
            })
            .collect()
    }
}

fn nms(dets: &[RawDet], iou_thresh: f32) -> Vec<usize> {
    let n = dets.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        dets[b]
            .confidence
            .partial_cmp(&dets[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = vec![];
    let mut sup = vec![false; n];
    for &i in &idx {
        if sup[i] {
            continue;
        }
        keep.push(i);
        for &j in &idx {
            if sup[j] || i == j {
                continue;
            }
            if iou(&dets[i].bbox, &dets[j].bbox) > iou_thresh {
                sup[j] = true;
            }
        }
    }
    keep
}
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.) * (y2 - y1).max(0.);
    let aa = (a[2] - a[0]) * (a[3] - a[1]);
    let ab = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (aa + ab - inter + 1e-6)
}
struct RawDet {
    bbox: [f32; 4],
    keypoints: [[f32; 2]; 4],
    confidence: f32,
    color_id: i64,
    class_id: i64,
}
