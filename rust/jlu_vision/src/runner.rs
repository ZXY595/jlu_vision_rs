use crate::buff::pipeline::{BuffPipeline, BuffPipelineConfig};
use crate::core::types::*;
use crate::track::pipeline::{TrackerConfig, TrackerPipeline};
use crate::vision::pipeline::{DetectorConfig, DetectorPipeline};
use async_channel::{Receiver, Sender};
use futures_lite::stream::StreamExt as _;
use std::path::PathBuf;
use std::pin::pin;
use std::time::Duration;

pub struct AppArgs {
    pub config_dir: PathBuf,
    pub debug: bool,
}

pub async fn run(args: AppArgs) -> anyhow::Result<()> {
    let (image_tx, image_rx): (Sender<ImageFrame>, Receiver<ImageFrame>) =
        async_channel::bounded(8);
    let (armor_tx, armor_rx): (Sender<Vec<Armor>>, Receiver<Vec<Armor>>) =
        async_channel::bounded(8);
    let (cmd_tx, _cmd_rx): (Sender<AimCommand>, Receiver<AimCommand>) = async_channel::bounded(8);
    let cmd_tx2 = cmd_tx.clone();

    smol::spawn(camera_mock(image_tx, args.config_dir.clone())).detach();
    let detector = DetectorPipeline::new(DetectorConfig::default())?;
    smol::spawn(detector_pipeline(detector, image_rx, armor_tx)).detach();
    let tracker = TrackerPipeline::new(TrackerConfig::default());
    smol::spawn(tracker.run(armor_rx, cmd_tx)).detach();
    let buff = BuffPipeline::new(BuffPipelineConfig::default());
    smol::spawn(buff_mock(buff, cmd_tx2)).detach();

    smol::future::pending::<()>().await;
    Ok(())
}

async fn camera_mock(tx: Sender<ImageFrame>, _cd: PathBuf) {
    log::info!("Camera mock started");
    let mut c: u64 = 0;
    loop {
        let frame = ImageFrame {
            image: image::RgbImage::new(1440, 1080),
            frame_id: format!("camera_{c}"),
            stamp: std::time::SystemTime::now(),
        };
        if tx.send(frame).await.is_err() {
            log::error!("Camera mock: rx dropped");
            break;
        }
        c += 1;
        smol::Timer::after(Duration::from_millis(33)).await;
    }
}

async fn detector_pipeline(
    detector: DetectorPipeline,
    image_rx: Receiver<ImageFrame>,
    armor_tx: Sender<Vec<Armor>>,
) {
    log::info!("Detector pipeline started");
    let mut s = pin!(detector.into_stream(image_rx));
    while let Some(armors) = s.next().await {
        if armor_tx.send(armors).await.is_err() {
            log::error!("Detector: armor rx dropped");
            break;
        }
    }
    log::info!("Detector pipeline stopped");
}

async fn buff_mock(buff: BuffPipeline, cmd_tx: Sender<AimCommand>) {
    log::info!("Buff pipeline started (mock)");
    loop {
        let frame = ImageFrame {
            image: image::RgbImage::new(1440, 1080),
            frame_id: String::new(),
            stamp: std::time::SystemTime::now(),
        };
        if let Some(cmd) = buff.process(&frame, &[]) {
            let _ = cmd_tx.send(cmd).await;
        }
        smol::Timer::after(Duration::from_millis(33)).await;
    }
}
