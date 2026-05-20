use clap::Parser;
use jlu_vision::runner;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "jlu_vision", version = "0.1.0")]
struct Args {
    #[arg(short, long, default_value = "configs")]
    config_dir: PathBuf,
    #[arg(short, long)]
    debug: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    if !args.config_dir.exists() {
        anyhow::bail!("Config directory not found: {:?}", args.config_dir);
    }
    log::info!(
        "Starting JLU Vision pipeline... config={:?} debug={}",
        args.config_dir,
        args.debug
    );
    let app_args = runner::AppArgs {
        config_dir: args.config_dir,
        debug: args.debug,
    };
    smol::block_on(runner::run(app_args))
}
