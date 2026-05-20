use nalgebra::Isometry3;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
struct StampedTransform {
    transform: Isometry3<f64>,
    stamp: SystemTime,
    child_frame_id: String,
}

#[derive(Debug, Default)]
pub struct TfBuffer {
    transforms: HashMap<String, Vec<StampedTransform>>,
}

impl TfBuffer {
    pub fn new() -> Self {
        Self {
            transforms: HashMap::new(),
        }
    }

    pub fn set_transform(
        &mut self,
        parent_frame_id: &str,
        child_frame_id: &str,
        transform: Isometry3<f64>,
        stamp: SystemTime,
    ) {
        self.transforms
            .entry(parent_frame_id.to_string())
            .or_default()
            .push(StampedTransform {
                transform,
                stamp,
                child_frame_id: child_frame_id.to_string(),
            });
    }

    pub fn lookup(
        &self,
        target_frame: &str,
        source_frame: &str,
        stamp: SystemTime,
        tolerance: Duration,
    ) -> Option<Isometry3<f64>> {
        if target_frame == source_frame {
            return Some(Isometry3::identity());
        }
        self.lookup_chain(target_frame, source_frame, stamp, tolerance, 0)
    }

    fn lookup_chain(
        &self,
        target: &str,
        source: &str,
        stamp: SystemTime,
        tolerance: Duration,
        depth: usize,
    ) -> Option<Isometry3<f64>> {
        if depth > 100 {
            return None;
        }
        for (parent, children) in &self.transforms {
            for child in children {
                if stamp.duration_since(child.stamp).unwrap_or(Duration::MAX) > tolerance
                    && child.stamp.duration_since(stamp).unwrap_or(Duration::MAX) > tolerance
                {
                    continue;
                }
                if child.child_frame_id == source {
                    if parent == target {
                        return Some(child.transform);
                    }
                    let rest = self.lookup_chain(target, parent, stamp, tolerance, depth + 1)?;
                    return Some(rest * child.transform);
                }
            }
        }
        None
    }

    pub fn prune(&mut self, max_age: Duration, now: SystemTime) {
        for transforms in self.transforms.values_mut() {
            transforms.retain(|t| now.duration_since(t.stamp).unwrap_or(Duration::MAX) < max_age);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_identity() {
        let b = TfBuffer::new();
        assert!(
            b.lookup("a", "a", SystemTime::now(), Duration::from_secs(1))
                .is_some()
        );
    }
    #[test]
    fn test_direct() {
        let mut b = TfBuffer::new();
        let now = SystemTime::now();
        b.set_transform("odom", "cam", Isometry3::translation(1., 0., 0.), now);
        let r = b
            .lookup("odom", "cam", now, Duration::from_secs(1))
            .unwrap();
        assert!((r.translation.vector.x - 1.).abs() < 1e-10);
    }
}
