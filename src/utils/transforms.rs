use std::ops::Mul;
use num_traits::{float::Float, NumCast, ToPrimitive};
pub use opencv::core::{Point_, Rect_, Size_, VecN};
use opencv::Result;
use crate::status::ShapeError;
use super::structs::Pointers;

pub trait Castable<T: ToPrimitive + Clone, U: NumCast + Clone> {
    type Target;

    /// Returns the generic components of Self as a vector
    fn unfold(&self) -> Vec<T>;

    /// Creates a new Self from generic components.
    fn fold(&self, with: Vec<U>) -> Self::Target;

    fn cast(&self) -> Self::Target {
        let x = self
            .unfold()
            .into_iter()
            .map(|t| U::from(t).expect("invalid cast"))
            .collect();
        return self.fold(x);
    }
}
pub trait Resizable<Sizer = Self> {
    type Scaler;

    fn rescale(&self, scale: Self::Scaler) -> Self;

    fn resize(&self, scale: Sizer) -> Self;
}
pub trait Sizable<Sizer = Self> {
    type Scaler;

    fn as_size(&self, other: &Self) -> Sizer;

    fn as_scale(&self, other: &Self) -> Result<Self::Scaler>;
}

impl<T: ToPrimitive + Clone, U: NumCast + Clone> Castable<T, U> for Size_<T> {
    type Target = Size_<U>;

    fn unfold(&self) -> Vec<T> {
        return vec![self.width.clone(), self.height.clone()];
    }
    fn fold(&self, with: Vec<U>) -> Self::Target {
        return Size_ {
            width: with[0].clone(),
            height: with[1].clone(),
        };
    }
}
impl<T: ToPrimitive + Clone, U: NumCast + Clone> Castable<T, U> for Point_<T> {
    type Target = Point_<U>;

    fn unfold(&self) -> Vec<T> {
        return vec![self.x.clone(), self.y.clone()];
    }
    fn fold(&self, with: Vec<U>) -> Self::Target {
        return Point_ {
            x: with[0].clone(),
            y: with[1].clone(),
        };
    }
}
impl<T: ToPrimitive + Clone, U: NumCast + Clone> Castable<T, U> for Rect_<T> {
    type Target = Rect_<U>;

    fn unfold(&self) -> Vec<T> {
        return vec![
            self.x.clone(),
            self.y.clone(),
            self.width.clone(),
            self.height.clone(),
        ];
    }
    fn fold(&self, with: Vec<U>) -> Self::Target {
        return Rect_ {
            x: with[0].clone(),
            y: with[1].clone(),
            width: with[2].clone(),
            height: with[3].clone(),
        };
    }
}
impl<T: ToPrimitive + Clone, U: NumCast + Clone> Castable<T, U> for Pointers<T> {
    type Target = Pointers<U>;

    fn unfold(&self) -> Vec<T> {
        vec![self.diameter.clone(), self.master.x.clone(), self.master.y.clone(), self.short.x.clone(), self.short.y.clone(), self.long.x.clone(), self.long.y.clone()]
    }
    fn fold(&self, with: Vec<U>) -> Self::Target {
        Pointers {
            diameter: with[0].clone(),
            master: Point_ {
                x: with[1].clone(),
                y: with[2].clone(),
            },
            short: Point_ {
                x: with[3].clone(),
                y: with[4].clone(),
            },
            long: Point_ {
                x: with[5].clone(),
                y: with[6].clone(),
            },
        }
    }
}
impl<T: Castable<U, V>, U: ToPrimitive + Clone, V: NumCast + Clone> Castable<U, V> for Vec<T>
where
    Vec<T>: FromIterator<T::Target>,
{
    type Target = Vec<T>;

    fn unfold(&self) -> Vec<U> {
        todo!(); // We implement cast directly!
    }

    fn fold(&self, _: Vec<V>) -> Self::Target {
        todo!(); // We implement cast directly!
    }

    fn cast(&self) -> Self::Target {
        self.iter().map(|v| v.cast()).collect()
    }
}


impl<T: Mul<Output = T> + Copy> Resizable for Size_<T> {
    type Scaler = T;

    fn rescale(&self, scale: Self::Scaler) -> Self {
        Size_ {
            width: self.width * scale,
            height: self.height * scale,
        }
    }

    fn resize(&self, scale: Self) -> Self {
        Size_ {
            width: self.width * scale.width,
            height: self.height * scale.height,
        }
    }
}
impl<T: Mul<Output = T> + Copy> Resizable for Point_<T> {
    type Scaler = T;

    fn rescale(&self, scale: Self::Scaler) -> Self {
        Point_ {
            x: self.x * scale,
            y: self.y * scale,
        }
    }

    fn resize(&self, scale: Self) -> Self {
        Point_ {
            x: self.x * scale.x,
            y: self.y * scale.y,
        }
    }
}
impl<T: Mul<Output = T> + Copy> Resizable<Size_<T>> for Rect_<T> {
    type Scaler = T;

    fn rescale(&self, scale: Self::Scaler) -> Self {
        Rect_ {
            x: self.x * scale,
            y: self.y * scale,
            width: self.width * scale,
            height: self.height * scale,
        }
    }

    fn resize(&self, scale: Size_<T>) -> Self {
        Rect_ {
            x: self.x * scale.width,
            y: self.y * scale.height,
            width: self.width * scale.width,
            height: self.height * scale.height,
        }
    }
}
impl<T: Mul<Output = T> + Copy> Resizable for Pointers<T> {
    type Scaler = T;

    fn rescale(&self, scale: Self::Scaler) -> Self {
        Pointers {
            diameter: self.diameter.clone() * scale.clone(),
            master: Point_ {
                x: self.master.x.clone() * scale.clone(),
                y: self.master.y.clone() * scale.clone(),
            },
            short: Point_ {
                x: self.short.x.clone() * scale.clone(),
                y: self.short.y.clone() * scale.clone(),
            },
            long: Point_ {
                x: self.long.x.clone() * scale.clone(),
                y: self.long.y.clone() * scale,
            },
        }
    }
    fn resize(&self, _: Self) -> Self {
        todo!()
    }
}
impl<T: Resizable<V, Scaler = U>, U: Clone, V: Clone> Resizable<V> for Vec<T> {
    type Scaler = U;

    fn rescale(&self, scale: Self::Scaler) -> Self {
        let mut r = Vec::new();
        for v in self {
            r.push(v.rescale(scale.clone()));
        }
        r
    }

    fn resize(&self, scale: V) -> Self {
        let mut r = Vec::new();
        for v in self {
            r.push(v.resize(scale.clone()));
        }
        r
    }
}


impl<T: Float> Sizable for Size_<T> {
    type Scaler = T;

    fn as_size(&self, other: &Self) -> Size_<T> {
        Size_ {
            width: self.width / other.width,
            height: self.height / other.height,
        }
    }
    fn as_scale(&self, other: &Self) -> Result<Self::Scaler> {
        ShapeError::try_lin_scale(self.width / other.width, self.height / other.height)
            .map_err(|_| opencv::Error::new(opencv::core::StsParseError, "Scale is not linear."))
    }
}
impl<T: Float> Sizable<Size_<T>> for Rect_<T> {
    type Scaler = T;

    fn as_size(&self, other: &Self) -> Size_<T> {
        Size_ {
            width: self.width / other.width,
            height: self.height / other.height,
        }
    }
    fn as_scale(&self, other: &Self) -> Result<Self::Scaler> {
        ShapeError::try_lin_scale(self.width / other.width, self.height / other.height)
            .map_err(|_| opencv::Error::new(opencv::core::StsParseError, "Scale error!"))
    }
}

