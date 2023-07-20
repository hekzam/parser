use std::collections::HashMap;

use crate::utils::*;
use super::structs::content;


impl From<content::Rectangle> for Rect_<f64> {
    fn from(value: content::Rectangle) -> Self {
        Rect_ {
            x: value.x,
            y: value.y,
            width: value.dx,
            height: value.dy
        }
    }
}
impl From<content::Position> for Point_<f64> {
    fn from(value: content::Position) -> Self {
        Point_ {
            x: value.x,
            y: value.y
        }
    }
}
impl From<content::Markers> for Pointers<f64> {
    fn from(value: content::Markers) -> Self {
        Pointers {
            diameter: value.d,
            master: value.m.into(),
            short: value.s.into(),
            long: value.l.into(),
        }
    }
}
impl From<content::Question> for Question_<f64> {
    fn from(value: content::Question) -> Self {
        Question_ {
            kind: value.t,
            page: value.p,
            rect: value.at.into()
        }
    }
}
impl From<content::Meta> for Metadata_<f64> {
    fn from(value: content::Meta) -> Self {
        Metadata_ {
            id: value.id,
            hash: value.hash,
            size: Size_ { width: value.w, height: value.h },
            pages: value.n
        }
    }
}
impl From<content::Content> for Content_<f64> {
    fn from(value: content::Content) -> Self {
        Content_{
            questions: HashMap::from_iter(value.q.into_iter().map(|(s,q)| (s,q.into()))),
            pointers: value.mk.into(),
            qr_code: value.qr.into(),
            metadata: value.md.into()
        }
    }
}
