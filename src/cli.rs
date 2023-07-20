use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// The images to process, must all be from the same exam.
    pub images: Vec<String>,
    /// The position file to process
    #[arg(short, long)]
    pub positions: String,
    /// The position file to process
    #[arg(short, long)]
    pub model: String,
}
