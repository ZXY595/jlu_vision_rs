use crate::core::types::{Armor, LightBar};
use nalgebra::{Matrix2, Point2, Vector2};

#[derive(Debug, Clone)]
pub struct PcaConfig {
    pub max_search_dist: f64,
    pub search_step: f64,
    pub brightness_thresh: u8,
}
impl Default for PcaConfig {
    fn default() -> Self {
        Self {
            max_search_dist: 20.,
            search_step: 1.,
            brightness_thresh: 60,
        }
    }
}

pub struct LightCornerCorrector {
    config: PcaConfig,
}

impl LightCornerCorrector {
    pub fn new(config: PcaConfig) -> Self {
        Self { config }
    }

    pub fn correct(&self, armor: &mut Armor, gray: &image::GrayImage) -> bool {
        let la = self.find_axis(gray, &armor.left_light);
        let ra = self.find_axis(gray, &armor.right_light);
        if let (Some(la), Some(ra)) = (la, ra)
            && let (Some(lt), Some(lb), Some(rt), Some(rb)) = (
                self.find_corner(gray, &armor.left_light, &la, true),
                self.find_corner(gray, &armor.left_light, &la, false),
                self.find_corner(gray, &armor.right_light, &ra, true),
                self.find_corner(gray, &armor.right_light, &ra, false),
            )
        {
            armor.left_light.top = lt;
            armor.left_light.bottom = lb;
            armor.right_light.top = rt;
            armor.right_light.bottom = rb;
            return true;
        }
        false
    }

    fn find_axis(&self, gray: &image::GrayImage, light: &LightBar) -> Option<SymmetryAxis> {
        let (w, h) = (gray.width(), gray.height());
        let wh = (w as f64) - 1.0;
        let hh = (h as f64) - 1.0;
        let min_x = (light.top.x.min(light.bottom.x) - 10.0).max(0.0) as u32;
        let max_x = (light.top.x.max(light.bottom.x) + 10.0).min(wh) as u32;
        let min_y = (light.top.y.min(light.bottom.y) - 5.0).max(0.0) as u32;
        let max_y = (light.top.y.max(light.bottom.y) + 5.0).min(hh) as u32;
        let (mut sx, mut sy, mut n) = (0., 0., 0u32);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if gray.get_pixel(x, y)[0] > self.config.brightness_thresh {
                    sx += x as f64;
                    sy += y as f64;
                    n += 1;
                }
            }
        }
        if n < 5 {
            return None;
        }
        let (cx, cy) = (sx / n as f64, sy / n as f64);
        let (mut cxx, mut cyy, mut cxy) = (0., 0., 0.);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if gray.get_pixel(x, y)[0] > self.config.brightness_thresh {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    cxx += dx * dx;
                    cyy += dy * dy;
                    cxy += dx * dy;
                }
            }
        }
        let cov = Matrix2::new(cxx, cxy, cxy, cyy);
        let eig = cov.symmetric_eigen();
        let dir = if eig.eigenvalues.x > eig.eigenvalues.y {
            eig.eigenvectors.column(0).into_owned()
        } else {
            eig.eigenvectors.column(1).into_owned()
        };
        Some(SymmetryAxis {
            centroid: Point2::new(cx, cy),
            direction: dir,
        })
    }

    fn find_corner(
        &self,
        gray: &image::GrayImage,
        light: &LightBar,
        axis: &SymmetryAxis,
        is_top: bool,
    ) -> Option<Point2<f64>> {
        let sign: f64 = if is_top { -1. } else { 1. };
        let (w, h) = (gray.width(), gray.height());
        let start = if is_top { light.top } else { light.bottom };
        let (mut best, mut bb, mut found) = (start, 0u8, false);
        for s in 0..((self.config.max_search_dist / self.config.search_step) as i32) {
            let t = s as f64 * self.config.search_step * sign;
            let x = (start.x + axis.direction.x * t).round();
            let y = (start.y + axis.direction.y * t).round();
            if x < 0. || y < 0. || x >= w as f64 || y >= h as f64 {
                break;
            }
            let p = gray.get_pixel(x as u32, y as u32);
            if p[0] > bb && p[0] > self.config.brightness_thresh {
                bb = p[0];
                best = Point2::new(x, y);
                found = true;
            }
        }
        if found { Some(best) } else { Some(start) }
    }
}

struct SymmetryAxis {
    #[expect(dead_code)]
    centroid: Point2<f64>,
    direction: Vector2<f64>,
}
