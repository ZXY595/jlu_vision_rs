use nalgebra::{Matrix3, Point2, Point3, Rotation2, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use strum::{Display, EnumIter, EnumString};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, EnumString, Display, EnumIter,
)]
pub enum ArmorType {
    #[strum(serialize = "One")]
    One,
    #[strum(serialize = "Two")]
    Two,
    #[strum(serialize = "Three")]
    Three,
    #[strum(serialize = "Four")]
    Four,
    #[strum(serialize = "Sentry")]
    Sentry,
    #[strum(serialize = "Outpost")]
    Outpost,
    #[strum(serialize = "Base")]
    Base,
    #[strum(serialize = "Negative")]
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, EnumString, Display)]
pub enum EnemyColor {
    #[strum(serialize = "Red")]
    Red,
    #[strum(serialize = "Blue")]
    Blue,
    #[strum(serialize = "Extinguished")]
    Extinguished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, EnumString, Display)]
pub enum TaskMode {
    #[strum(serialize = "Idle")]
    Idle,
    #[strum(serialize = "Armor")]
    Armor,
    #[strum(serialize = "Buff")]
    Buff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArmorIndex {
    _0 = 0,
    _1 = 1,
    _2 = 2,
    _3 = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArmorPointPosition {
    LeftBottom = 0,
    LeftTop = 1,
    RightTop = 2,
    RightBottom = 3,
}

#[derive(Debug, Clone, Copy)]
pub struct LightBar {
    pub top: Point2<f64>,
    pub bottom: Point2<f64>,
}

#[derive(Debug, Clone)]
pub struct Armor {
    pub armor_type: ArmorType,
    pub armor_color: EnemyColor,
    pub distance_to_image_center: f64,
    pub position: Point3<f64>,
    pub orientation: UnitQuaternion<f64>,
    pub left_light: LightBar,
    pub right_light: LightBar,
    pub center: Point2<f64>,
    pub confidence: f32,
    pub key_frame: bool,
    pub heart_beat: bool,
    pub frame_id: String,
    pub stamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct ImageFrame {
    pub image: image::RgbImage,
    pub frame_id: String,
    pub stamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct ArmorDetection {
    pub position: Point3<f64>,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: Rotation2<f64>,
    pub points: [Point2<f64>; 4],
    pub armor_type: ArmorType,
    pub armor_color: EnemyColor,
    pub confidence: f32,
    pub frame_id: String,
    pub stamp: SystemTime,
}

impl From<&Armor> for ArmorDetection {
    fn from(armor: &Armor) -> Self {
        let (roll, pitch, yaw) = armor.orientation.euler_angles();
        Self {
            position: armor.position,
            roll,
            pitch,
            yaw: Rotation2::new(yaw),
            points: [
                armor.left_light.bottom,
                armor.left_light.top,
                armor.right_light.top,
                armor.right_light.bottom,
            ],
            armor_type: armor.armor_type,
            armor_color: armor.armor_color,
            confidence: armor.confidence,
            frame_id: armor.frame_id.clone(),
            stamp: armor.stamp,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArmorPositionWithYaw {
    pub position: Point3<f64>,
    pub yaw: Rotation2<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct AimCommand {
    pub control: bool,
    pub fire_thres_yaw: f64,
    pub fire_thres_pitch: f64,
    pub target_yaw: f64,
    pub target_pitch: f64,
    pub yaw: f64,
    pub yaw_vel: f64,
    pub yaw_acc: f64,
    pub pitch: f64,
    pub pitch_vel: f64,
    pub pitch_acc: f64,
    pub bullet_id: u32,
}

#[derive(Debug, Clone)]
pub struct CameraInfo {
    pub matrix: Matrix3<f64>,
    pub distortion_coefficients: [f64; 5],
}

pub fn get_armor_points(armor_type: ArmorType) -> [Point3<f64>; 4] {
    let (half_w, half_h) = match armor_type {
        ArmorType::Outpost => (0.115, 0.065),
        _ => (0.065, 0.0275),
    };
    [
        Point3::new(-half_w, -half_h, 0.0),
        Point3::new(-half_w, half_h, 0.0),
        Point3::new(half_w, half_h, 0.0),
        Point3::new(half_w, -half_h, 0.0),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackStateKind {
    Lost,
    TempLost,
    Tracking,
}

#[derive(Debug, Clone)]
pub struct TrackState {
    pub state: TrackStateKind,
    pub stamp_last_update: SystemTime,
    pub stamp_last_tracking: SystemTime,
    pub k: u64,
}

#[derive(Debug, Clone)]
pub struct TargetState {
    pub armor_type: ArmorType,
    pub center_position: Point3<f64>,
    pub center_velocity: Vector3<f64>,
    pub center_yaw: f64,
    pub center_vyaw: f64,
    pub armors: Vec<ArmorPositionWithYaw>,
}

impl TargetState {
    pub fn predict(&self, dt: f64) -> Self {
        let mut next = self.clone();
        next.center_position += self.center_velocity * dt;
        next.center_yaw =
            crate::core::math::normalize_angle(next.center_yaw + self.center_vyaw * dt);
        let yaw = Rotation2::new(next.center_yaw);
        next.armors = self
            .armors
            .iter()
            .map(|a| ArmorPositionWithYaw {
                position: a.position + self.center_velocity * dt,
                yaw,
            })
            .collect();
        next
    }
}

#[derive(Debug, Clone)]
pub struct RobotTargetState {
    pub state: TargetState,
    pub radius_a: f64,
    pub radius_b: f64,
    pub dz: f64,
}
#[derive(Debug, Clone)]
pub struct OutpostTargetState {
    pub base: TargetState,
    pub radius: f64,
    pub dz_0: f64,
    pub dz_1: f64,
    pub dz_2: f64,
}

// ============================================================================
// Buff types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuffBladeType {
    Inactivated,
    Activated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuffBladeIndex {
    _0 = 0,
    _1 = 1,
    _2 = 2,
    _3 = 3,
    _4 = 4,
}

#[derive(Debug, Clone)]
pub struct BuffRunePoints {
    pub center: Point2<f64>,
    pub bottom_right: Point2<f64>,
    pub top_right: Point2<f64>,
    pub top_left: Point2<f64>,
    pub bottom_left: Point2<f64>,
}

#[derive(Debug, Clone)]
pub struct RuneObject {
    pub color: EnemyColor,
    pub blade_type: BuffBladeType,
    pub points: BuffRunePoints,
    pub prob: f32,
    pub bbox: [f32; 4],
    pub area: f32,
    pub frame_id: String,
    pub stamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct BladePositionRollPoints {
    pub position: Point3<f64>,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub points: [Point2<f64>; 5],
}

#[derive(Debug, Clone)]
pub struct BuffState {
    pub center: Point3<f64>,
    pub roll: f64,
    pub vroll: f64,
    pub radius: f64,
    pub blades: Vec<(BuffBladeIndex, Point3<f64>)>,
}

pub const BUFF_BLADE_OBJ_POINTS: [Point3<f64>; 5] = [
    Point3::new(0.0, 0.000, 0.000),
    Point3::new(0.0, -0.186, 0.5415),
    Point3::new(0.0, -0.160, 0.8585),
    Point3::new(0.0, 0.160, 0.8585),
    Point3::new(0.0, 0.186, 0.5415),
];
pub const BUFF_RADIUS: f64 = 0.7;
