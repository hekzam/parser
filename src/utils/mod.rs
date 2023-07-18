

mod angle;
mod structs;
mod transforms;
mod trig;

pub use angle::*;
pub use structs::*;
pub use transforms::*;
pub use trig::*;


use num_traits::float::Float;

pub use opencv::core::{Point_, Rect_, Size_, VecN};
use opencv::{core, objdetect, prelude::*, Result};

/// Detects and decode a qr code
pub fn detect_qr(img: &Mat) -> Result<(QRDelimitor, QRData)> {
    let detector = objdetect::QRCodeDetector::default()?;

    let mut pt = Mat::default();
    let data = detector.detect_and_decode(img, &mut pt, &mut core::no_array())?;
    if pt.empty() {
        return Err(opencv::Error::new(core::StsNullPtr, "No QR Code was detected."));
    }

    Ok((
        pt.data_typed()?.to_owned().into(),
        data.try_into()
            .map_err(|_| opencv::Error::new(opencv::core::StsParseError, "not enough data in code!"))?,
    ))
}

/// Finds the index at which quartile q ([0;1]) is reached
/// This is for lists where list[i] is the number of elements with value i.
/// Otherwise, calculating quartiles is trivial
pub fn index_quartile<T: Float>(values: &[T], q: T) -> Result<usize> {
    let mut last = T::from(0.).unwrap();
    let cumsum: Vec<T> = values.iter().map(|v| {
        last = last + *v;
        last
    }).collect();

    // Reduce population to the correct quartile
    let top = *cumsum.last().ok_or(opencv::Error::new(opencv::core::StsVecLengthErr, "cannot sum over empty array."))? * q;

    for i in 0..cumsum.len() {
        if cumsum[i] >= top {
            return Ok(i);
        }
    }
    return Err(opencv::Error::new(opencv::core::StsError, "unkown error when trying to calculate median."));
} 
