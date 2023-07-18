use std::{f32::consts::PI, fmt::Debug, ops::{Sub, Add}};

use num_traits::{float::Float, NumCast};

pub use opencv::core::{Point_, Rect_, Size_, VecN};


#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Deg<T>(pub T);
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Rad<T>(pub T);

const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.;
const DEG_PER_RAD: f64 = 180. / std::f64::consts::PI;

/// A type that describes an angle. Bascially a periodic value from 0 to Self::circle
pub trait Angle<T: Float> where Self: Sized {
    /// The angle value associated with a full circle. 2PI for radians, 360 for degrees
    const CIRCLE: f64 = 1.;
    /// The angle's value
    fn value(&self) -> T;

    /// Create a new angle from the value
    fn new(value: T) -> Self;

    fn sin(&self) -> T {
        (self.to_portion() * T::from(PI * 2.).unwrap()).sin()
    }

    fn cos(&self) -> T {
        (self.to_portion() * T::from(PI * 2.).unwrap()).cos()
    }

    /// Returns the absolute value of an angle. This method is not conservative, you should use as_unsigned instead!
    fn abs(&self) -> Self {
        Self::new(self.value().abs())
    }

    /// Transforms angle, from [-circle/2;circle/2] to [0; circle] (by adding circle to negatice values, thus ensuring that the angle is conserved)
    /// Values are assumed in [-circle/2;circle/2]. If not, call clamp instead!
    fn as_unsigned(&self) -> Self {
        match self.value() >= T::zero() {
            true => Self::new(self.value()),
            false => Self::from_portion(self.to_portion() + T::one())
        }
    }

    /// Transforms angle, from [0;circle] to [-circle/2;circle/2] (ensuring that the angle is conserved)
    /// Values are assumed in [0;circle]. If not, call clamp first!
    fn as_signed(&self) -> Self {
        match self.to_portion() <= T::from(0.5).unwrap() {
            true => Self::new(self.value()),
            false => Self::from_portion(self.to_portion() - T::one())
        }
    }

    /// Transforms angle, from [-inf;inf] to [0;circle] (by doing modulo CIRCLE, preserving the underlying angle)
    fn clamp(&self) -> Self {
        let v = self.to_portion();
        let c = v % T::one();

        Self::from_portion(c)
    }

    /// Angle from a portion ([0;1])
    fn from_portion(value: T) -> Self {
        Self::new(value * T::from(Self::CIRCLE).unwrap())
    }

    /// Angle to a portion ([0;1])
    fn to_portion(&self) -> T {
        self.value() / T::from(Self::CIRCLE).unwrap()
    }
}
impl<T: Float> Angle<T> for Deg<T> {
    const CIRCLE: f64 = 360.;

    fn value(&self) -> T {
        self.0.clone()
    }

    fn new(value: T) -> Self {
        Deg(value)
    }
}
impl<T: Float> Angle<T> for Rad<T> {
    const CIRCLE: f64 = std::f64::consts::PI * 2.;
    fn value(&self) -> T {
        self.0.clone()
    }

    fn new(value: T) -> Self {
        Rad(value)
    }
}
impl<T: Float> From<Deg<T>> for Rad<T> {
    fn from(value: Deg<T>) -> Self {
        Rad(T::from(<f64 as NumCast>::from(value.value()).unwrap_or(f64::NAN) * RAD_PER_DEG).unwrap_or(T::zero()))
    }
}
impl<T: Float> From<Rad<T>> for Deg<T> {
    fn from(value: Rad<T>) -> Self {
        Deg(T::from(<f64 as NumCast>::from(value.value()).unwrap_or(f64::NAN) * DEG_PER_RAD).unwrap_or(T::zero()))
    }
}
//todo Redo these! Right, now, mixing negative and positive angles could be messy. Maybe we could call clamp!
impl<T: Sub<T, Output = T> + Float> Sub<Deg<T>> for Deg<T> {
    type Output = Deg<T>;
    fn sub(self, rhs: Deg<T>) -> Self::Output {
        Deg(self.value() - rhs.value())
    }
}
impl<T: Add<T, Output = T> + Float> Add<Deg<T>> for Deg<T> {
    type Output = Deg<T>;
    fn add(self, rhs: Deg<T>) -> Self::Output {
        Deg(self.value() + rhs.value())
    }
}
