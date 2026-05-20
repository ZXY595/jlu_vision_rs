use crate::core::types::*;
use nalgebra::Point2;

const INPUT_SIZE: u32 = 480;
const STRIDES: [i32; 3] = [8, 16, 32];
const NP: usize = 5;
const NC: usize = 2;
const NCL: usize = 2;
const CPA: usize = NP * 2 + 1 + NC + NCL;

#[derive(Debug, Clone, Copy)]
struct GS {
    g0: i32,
    g1: i32,
    stride: i32,
}
#[derive(Debug, Clone)]
struct TM {
    scale: f32,
    hw: f32,
    #[expect(dead_code)]
    hh: f32,
}

#[derive(Debug, Clone)]
pub struct BuffDetectorConfig {
    pub conf_th: f32,
    pub nms_th: f32,
    pub top_k: usize,
}
impl Default for BuffDetectorConfig {
    fn default() -> Self {
        Self {
            conf_th: 0.3,
            nms_th: 0.5,
            top_k: 100,
        }
    }
}

pub struct BuffDetector {
    config: BuffDetectorConfig,
    gs: Vec<GS>,
    tm: Option<TM>,
}

impl BuffDetector {
    pub fn new(config: BuffDetectorConfig) -> Self {
        let gs = STRIDES
            .iter()
            .flat_map(|&s| {
                let n = INPUT_SIZE as i32 / s;
                (0..n).flat_map(move |g1| (0..n).map(move |g0| GS { g0, g1, stride: s }))
            })
            .collect();
        Self {
            config,
            gs,
            tm: None,
        }
    }
    pub fn set_image_size(&mut self, iw: u32, ih: u32) {
        let s = INPUT_SIZE as f32 / ih.max(iw) as f32;
        let rh = (ih as f32 * s).round() as i32;
        let rw = (iw as f32 * s).round() as i32;
        let ph = INPUT_SIZE as i32 - rh;
        let pw = INPUT_SIZE as i32 - rw;
        self.tm = Some(TM {
            scale: s,
            hw: pw as f32 / 2.,
            hh: ph as f32 / 2.,
        });
    }

    pub fn postprocess(&self, output: &[f32]) -> Vec<RuneObject> {
        let Some(t) = &self.tm else { return vec![] };
        let na = output.len() / CPA;
        let mut objs = vec![];
        for ai in 0..na.min(self.gs.len()) {
            let b = ai * CPA;
            let conf = output[b + NP * 2];
            if conf < self.config.conf_th {
                continue;
            }
            let g = &self.gs[ai];
            let s = g.stride as f32;
            let mut kp = [[0.0f32; 2]; NP];
            for p in 0..NP {
                kp[p][0] = (output[b + p * 2] + g.g0 as f32) * s;
                kp[p][1] = (output[b + p * 2 + 1] + g.g1 as f32) * s;
            }
            kp.iter_mut()
                .take(NP)
                .flat_map(|p| p.iter_mut())
                .for_each(|p| *p = (*p - t.hw) / t.scale);
            let cs = NP * 2 + 1;
            let ci = (0..NC)
                .max_by(|&a, &b| output[b + cs + a].partial_cmp(&output[b + cs + b]).unwrap())
                .unwrap_or(0);
            let cls = cs + NC;
            let cli = (0..NCL)
                .max_by(|&a, &b| {
                    output[b + cls + a]
                        .partial_cmp(&output[b + cls + b])
                        .unwrap()
                })
                .unwrap_or(0);
            let min_x = kp.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
            let max_x = kp.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
            let min_y = kp.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
            let max_y = kp.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
            objs.push(RuneObject {
                color: if ci == 0 {
                    EnemyColor::Red
                } else {
                    EnemyColor::Blue
                },
                blade_type: if cli == 0 {
                    BuffBladeType::Inactivated
                } else {
                    BuffBladeType::Activated
                },
                points: BuffRunePoints {
                    center: Point2::new(kp[0][0] as f64, kp[0][1] as f64),
                    bottom_right: Point2::new(kp[1][0] as f64, kp[1][1] as f64),
                    top_right: Point2::new(kp[2][0] as f64, kp[2][1] as f64),
                    top_left: Point2::new(kp[3][0] as f64, kp[3][1] as f64),
                    bottom_left: Point2::new(kp[4][0] as f64, kp[4][1] as f64),
                },
                prob: conf,
                bbox: [min_x, min_y, max_x, max_y],
                area: (max_x - min_x) * (max_y - min_y),
                frame_id: String::new(),
                stamp: std::time::SystemTime::now(),
            });
        }
        objs.sort_by(|a, b| {
            b.prob
                .partial_cmp(&a.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if objs.len() > self.config.top_k {
            objs.truncate(self.config.top_k);
        }
        nms(&objs, self.config.nms_th)
    }
}

fn nms(objects: &[RuneObject], th: f32) -> Vec<RuneObject> {
    let n = objects.len();
    if n == 0 {
        return vec![];
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        objects[b]
            .prob
            .partial_cmp(&objects[a].prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = vec![];
    let mut sup = vec![false; n];
    for &i in &idx {
        if sup[i] {
            continue;
        }
        keep.push(i);
        let ab: [f32; 4] = objects[i].bbox;
        for &j in &idx {
            if sup[j] || i == j {
                continue;
            }
            let bb: [f32; 4] = objects[j].bbox;
            let x1 = ab[0].max(bb[0]);
            let y1 = ab[1].max(bb[1]);
            let x2 = ab[2].min(bb[2]);
            let y2 = ab[3].min(bb[3]);
            let inter = (x2 - x1).max(0.) * (y2 - y1).max(0.);
            let u = (ab[2] - ab[0]) * (ab[3] - ab[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter;
            if inter / u > th {
                sup[j] = true;
            }
        }
    }
    keep.iter().map(|&i| objects[i].clone()).collect()
}
