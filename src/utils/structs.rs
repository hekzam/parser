use std::{collections::HashMap, fmt::Debug};
use super::angle::Deg;
pub use opencv::core::{Point_, Rect_, Size_, VecN};
use crate::data::Kind;

pub type QRDelimitor = Vec<Point_<f32>>;
#[derive(Debug, Clone)]
/// Hash, exam ID, and page number
pub struct QRData {
    pub hash: Vec<u8>, 
    pub id: u8,
    pub page: u8
}

#[derive(Default, Copy, Clone, Debug)]
pub struct OriRect2D<T> {
    /// In [0;360]
    pub angle: Deg<T>,
    pub rect: Rect_<T>,
}

#[derive(Clone, Debug)]
pub struct Question_<T> {
    pub kind: Kind,
    pub page: u8,
    pub rect: Rect_<T>
}

#[derive(Clone, Debug)]
pub enum Answer {
    Binary(bool),
}

#[derive(Default, Clone, Debug)]
pub struct Metadata_<T> {
    pub id: u8,
    pub hash: Vec<u8>,
    pub size: Size_<T>,
    /// The page count
    pub pages: u8,
}

#[derive(Clone, Debug)]
pub struct Content_<T> {
    pub questions: HashMap<String, Question_<T>>,
    pub pointers: Pointers<T>,
    pub qr_code: Rect_<T>,
    pub metadata: Metadata_<T>,
}

/// Pointers go by groups of 3, they only can express linear transformations.
/// The actual positions should be the square's centers.
#[derive(Debug, Clone, Default)]
pub struct Pointers<T> {
    /// The marker's width, optional
    pub diameter: T,
    /// The master is the pointer between the two subordinates. It is the pointer closest to the other two, and is supposed to be at a right angle.
    pub master: Point_<T>,
    /// The short is the pointer closest to the master. It represents the paper's short side (NOTE: Paysage vs portrait?)
    pub short: Point_<T>,
    /// The long is the pointer furthest to the master. It represents the paper's long side
    pub long: Point_<T>,
}