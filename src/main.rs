//! The `dataset-converter` command line.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};

/// Convert object-detection datasets between the Ultralytics YOLO layout and Hugging Face Parquet.
#[derive(Debug, Parser)]
#[command(version)]
struct Cli {
    /// The format of INPUT.
    #[arg(long, value_enum)]
    from: Format,
    /// The format to write to OUTPUT.
    #[arg(long, value_enum)]
    to: Format,
    /// The dataset to convert.
    input: PathBuf,
    /// A missing or empty folder for the converted dataset.
    output: PathBuf,
}

/// A dataset format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// The Ultralytics YOLO layout: a folder with `data.yaml`, `images/` and `labels/`
    Yolo,
    /// Hugging Face Parquet: a folder whose `data/` holds `<split>-NNNNN-of-NNNNN.parquet` files
    Parquet,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let summaries = match (cli.from, cli.to) {
        (Format::Yolo, Format::Parquet) => {
            dataset_converter::yolo_to_parquet(&cli.input, &cli.output)?
        }
        (Format::Parquet, Format::Yolo) => {
            dataset_converter::parquet_to_yolo(&cli.input, &cli.output)?
        }
        (Format::Yolo, Format::Yolo) | (Format::Parquet, Format::Parquet) => {
            bail!("--from and --to name the same format, so there is nothing to convert")
        }
    };
    for summary in summaries {
        println!("{summary}");
    }
    Ok(())
}
