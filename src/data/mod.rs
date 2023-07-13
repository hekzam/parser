use std::{path::PathBuf, fs};

use crate::utils::Content_;

use opencv::{Result, core};
///Structs meant for data conversion
mod structs;
mod convert;


pub fn read(file: PathBuf) -> Result<Content_<f64>> {
    let cnts = fs::read_to_string(file).unwrap();
    let c: structs::Content = serde_json::from_str(&cnts).map_err(|e| opencv::Error::new(core::StsParseError, e.to_string()))?;
    Ok(c.into())
}
