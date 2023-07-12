use std::{path::PathBuf, fs};

///Structs meant for data conversion
mod structs;


fn read(file: PathBuf) {
    let cnts = fs::read_to_string(file).unwrap();
    //let c = serde_json::from_str(&cnts);
}