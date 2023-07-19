use std::fmt::Debug;

use num_traits::float::Float;

use crate::utils::*;

/// This describes major discrepancies in shape: a square not being square-y, a rectangle not being a rectangle...
#[derive(Debug)]
pub enum ShapeError {
    /// The angle is not square!
    AngleNotRight,

    /// Nonspecific angle error
    WrongAngle,

    /// The shape is not orthogonal
    NotOrtho,

    /// The scale difference is not linear (ie. may be streched along one axis)
    NotLinearScale,
}

/// This can be used to display the error, and contains details about the error.
/*pub struct DetailedShapeError<T, U: Float> {
    error: ShapeError,
    expected: T,
    actual: T,
    senibility: U,
}*/

pub type ShapeResult<T> = Result<T, ShapeError>;

impl ShapeError {
    /// The leeway for angles.
    /// 
    /// (actual_angle / expected_angle) must be in
    /// [1/ANGLE_THRESHOLD;ANGLE_THRESHOLD]
    pub const ANGLE_THRESHOLD: f64 = 1.1;

    /// Compares angle to a standard right angle
    pub fn try_right_angle<T: Angle<U> + From<Deg<U>> + Copy, U: Float>(value: T) -> ShapeResult<T> {
        // Float to float is safe
        let right_angle = T::from(Deg(U::from(90.0).unwrap()));

        match ShapeError::is_angle(&value, &right_angle) {
            false => Err(ShapeError::AngleNotRight),
            true => Ok(value),
        }
    }

    /// Compares angle to another angle
    /// 
    /// Note: Variable threshold?
    pub fn is_angle<T: Angle<U> + From<Deg<U>> + Copy, U: Float>(value: &T, angle: &T) -> bool {
        // Float to float should never fail (something something famous last words..)
        let antr = U::from(Self::ANGLE_THRESHOLD).unwrap(); 

        value.value() / angle.value() < antr && angle.value() / value.value() < antr
    }

    /// Compares an angle to its expected value
    pub fn try_angle<T: Angle<U> + From<Deg<U>> + Copy, U: Float>(&self, value: T) -> ShapeResult<T> {
        match self {
            Self::AngleNotRight => Self::try_right_angle(value),
            _ => Err(Self::WrongAngle) // No default angle here
        }
    }
}

impl ShapeError {
    pub const LENGTH_THRESHOLD: f64 = 1.1;

    /// Tries to obtain a rectangle out of four points.
    pub fn as_ortho<T: Float>(value: &[Point_<T>; 4]) -> ShapeResult<OriRect2D<T>> {
        //trace!("{:?}", value);
        //*1- Find the first diagonal
        let a = value[0].distance(&value[1]);
        let b = value[0].distance(&value[2]);
        let c = value[0].distance(&value[3]);
        // Now that we know how far [0] is from [1,2,3], we find the longest distance. From that we can deduce our points.
        //*2- Order our points
        let (adj_a, adj_b, other);
        if a > b {
            if a > c {
                other = value[1];
                adj_a = value[2];
                adj_b = value[3];
            } else {
                other = value[3];
                adj_a = value[1];
                adj_b = value[2];
            }
        } else {
            if b > c {
                other = value[2];
                adj_a = value[1];
                adj_b = value[3];
            } else {
                other = value[3];
                adj_a = value[1];
                adj_b = value[2];
            }
        }
        // We find the two other points. A rect should be described with the points in anti-trigonometric (clockwise) order.
        //println!("{:?} {:?} {:?}", value[0], adj_a, other);
        //println!("{:?} {:?} {:?}", value[0], adj_b, other);
        //let angl_a = value[0].signed_direction(&[adj_a, other]);
        //let angl_b = value[0].signed_direction(&[adj_b, other]);
        //println!(">>> {:?} {:?}", angl_a, angl_b);
        // One will be positive and the other negative!
        let ordered;
        // Disabled code, the points returned by the QR finder are in the right order. Angle has been overhauled so we could maybe reenable this code?
        /*if angl_a > angl_b {
            ordered = [value[0], adj_a, other, adj_b];
        } else {
            ordered = [value[0], adj_b, other, adj_a];
        }*/
        ordered = [value[0], adj_a, other, adj_b];
        // We can now reorder our points

        //*3- Calculate rect's parameters rotation
        //println!("o: {:?}", ordered);
        //trace!("GAY: {:?}", ordered[0].signed_direction(&[ordered[1], ordered[3]]));
        let rot = VecN::from_points(ordered[0], ordered[1]);
        //println!("r: {:?}", rot.direction());
        let rot = rot.direction();
        // We could rotate the rectangle. As it turns out, we can be sure that it will be valid in [0;90] (by just rotating the points around!)
        let size = Size_ {
            width: ordered[0].distance(&ordered[1]),
            height: ordered[0].distance(&ordered[3])
        };
        let pos = ordered[0];

        // Having nailed down all our points, we get a ton of interesting properties, 
        // we check some of them to make sure our shape is close enough to a rectangle!
        //*4- Verify shape
        Self::try_right_angle(ordered[0].direction(&[ordered[1], ordered[3]]))?;
        Self::try_right_angle(ordered[2].direction(&[ordered[1], ordered[3]]))?;

        Ok(OriRect2D::new(rot, size, pos))
    }

    /// Returns wether or not the described size is square
    pub fn is_square<T: Float>(height: T, width: T) -> bool {
        // Float conversions never fail
        let sctr = T::from(Self::LENGTH_THRESHOLD).unwrap();
        width/height < sctr && height/width < sctr
    }

    /// Tries to fit a 2D scale difference into a linear scale (square)
    pub fn try_lin_scale<T: Float>(dx: T, dy: T) -> ShapeResult<T> {
        match Self::is_square(dx, dy) {
            true => Ok((dx + dy) / T::from(2.0).unwrap()),
            false => Err(Self::NotLinearScale)
        }
    }
}