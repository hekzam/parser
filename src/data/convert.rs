use std::collections::HashMap;

use crate::utils::*;
use super::structs::*;


impl From<Rectangle> for Rect_<f64> {
    fn from(value: Rectangle) -> Self {
        Rect_ {
            x: value.x,
            y: value.y,
            width: value.dx,
            height: value.dy
        }
    }
}
impl From<Position> for Point_<f64> {
    fn from(value: Position) -> Self {
        Point_ {
            x: value.x,
            y: value.y
        }
    }
}
impl From<Markers> for Pointers<f64> {
    fn from(value: Markers) -> Self {
        Pointers {
            diameter: value.d,
            master: value.m.into(),
            short: value.s.into(),
            long: value.l.into(),
        }
    }
}
impl From<Question> for Question_<f64> {
    fn from(value: Question) -> Self {
        Question_ {
            id: value.id,
            page: value.p,
            rect: value.at.into()
        }
    }
}
impl From<Meta> for Metadata_<f64> {
    fn from(value: Meta) -> Self {
        Metadata_ {
            id: value.id,
            hash: value.hash,
            size: Size_ { width: value.w, height: value.h },
            pages: value.n
        }
    }
}
impl From<Content> for Content_<f64> {
    fn from(value: Content) -> Self {
        Content_{
            questions: HashMap::from_iter(value.q.into_iter().map(|(s,q)| (s,q.into()))),
            pointers: value.mk.into(),
            qr_code: value.qr.into(),
            metadata: value.md.into()
        }
    }
}
