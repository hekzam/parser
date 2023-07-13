use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// The image to process
    pub image: String,
    /// The position file to process
    pub positions: String,
}
