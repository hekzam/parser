use std::{fs, path::PathBuf};

use crate::utils::Content_;

use opencv::{core, Result};
mod convert;
///Structs meant for data conversion
mod structs;
pub use structs::Kind;

/// Reads the json position file, and converts it into usable data
pub fn read(file: PathBuf) -> Result<Content_<f64>> {
    let cnts = fs::read_to_string(file)
        .map_err(|e| opencv::Error::new(core::BadOrigin, e.to_string()))?;
    let c: structs::Content = serde_json::from_str(&cnts)
        .map_err(|e| opencv::Error::new(core::StsParseError, e.to_string()))?;
    Ok(c.into())
}
