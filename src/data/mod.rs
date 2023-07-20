use std::{fs, path::PathBuf};

use crate::utils::Content_;

use opencv::{core, Result};
mod convert;
///Structs meant for data conversion
pub mod structs;
pub use structs::content::Kind;

/// Reads the json position file, and converts it into usable data
pub fn read_content(file: PathBuf) -> Result<Content_<f64>> {
    let cnts = fs::read_to_string(file)
        .map_err(|e| opencv::Error::new(core::BadOrigin, e.to_string()))?;
    let c: structs::content::Content = serde_json::from_str(&cnts)
        .map_err(|e| opencv::Error::new(core::StsParseError, e.to_string()))?;
    Ok(c.into())
}

pub fn read_model(file: PathBuf) -> Result<structs::model::Model> {
    let file = fs::read_to_string(file)
        .map_err(|e| opencv::Error::new(core::BadOrigin, e.to_string()))?;
    serde_json::from_str(&file)
        .map_err(|e| opencv::Error::new(core::StsParseError, e.to_string()))
}