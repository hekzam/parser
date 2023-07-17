use std::collections::HashMap;

use serde::Deserialize;

type Float = f64;
type Int = u8;
type Text = String;

/// Position and general information about a document.
#[derive(Deserialize, Debug)]
pub enum Kind {
    Binary,
    Numeric,
    Text,
}

/// Position and general information about a document.
#[derive(Deserialize, Debug)]
pub struct Content {
    pub q: HashMap<Text, Question>,
    pub mk: Markers,
    pub qr: Rectangle,
    pub md: Meta,
}

/// Document metadata
#[derive(Deserialize, Debug)]
pub struct Meta {
    pub id: Int,
    pub hash: Vec<Int>,
    pub w: Float,
    pub h: Float,
    pub n: Int,
}
/// Struct describing a rectangle
#[derive(Deserialize, Debug)]
pub struct Rectangle {
    pub x: Float,
    pub y: Float,
    pub dx: Float,
    pub dy: Float,
}
/// The markers on a document
#[derive(Deserialize, Debug)]
pub struct Markers {
    pub d: Float,
    pub m: Position, 
    pub l: Position,
    pub s: Position,
}
/// A position
#[derive(Deserialize, Debug)]
pub struct Position {
    pub x: Float,
    pub y: Float,
}
/// A question box
#[derive(Deserialize, Debug)]
pub struct Question {
    pub t: Kind,
    pub p: Int,
    pub at: Rectangle,
}