use super::angle::*;
use super::structs::*;

use std::{collections::BTreeMap, fmt::Debug};

use num_traits::{float::Float, Num, NumCast, ToPrimitive};

pub use opencv::core::{Point_, Rect_, Size_, VecN};

use crate::status::{ShapeError, ShapeResult};

fn euclidean_distance<T: Float>(from: &Point_<T>, to: &Point_<T>) -> T {
    ((from.x - to.x).powi(2) + (from.y - to.y).powi(2)).sqrt()
}

/// Objects with this trait are able to describe some kind of rectangle
pub trait OrientedOrthogonal {
    type Size;
    type Angle;
    type Position: Point;

    fn angle(&self) -> Self::Angle;
    fn size(&self) -> Self::Size;
    fn position(&self) -> Self::Position;

    fn new(rot: Self::Angle, len: Self::Size, pos: Self::Position) -> Self
    where
        Self: Sized;
}

/// Objects with this trait are able to describe some kind of location in space
pub trait Point {
    type Length: Num;
    type Angle;

    /// (Euclidean) distance between two points
    fn distance(&self, other: &Self) -> Self::Length;

    /// The angle between points
    fn direction(&self, other: &[Self]) -> Self::Angle
    where
        Self: Sized;

    /// Direction between -pi,pi
    fn signed_direction(&self, other: &[Self]) -> Self::Angle
    where
        Self: Sized;
}

/// Objects with this trait are able to describe (euclidean) geometric vectors
pub trait Vector<P> {
    type Length;
    type Angle;

    /// The length of a vector
    fn length(&self) -> Self::Length;

    /// The angle of a vector
    fn direction(&self) -> Self::Angle;

    /// Create a new vector from constituants
    fn from_constituants(len: Self::Length, dir: Self::Angle) -> Self;

    fn from_points(from: P, to: P) -> Self;

    fn to_point(&self, from: P) -> P;

    /// Create a vector with the same direction and a new length
    fn with_length(&self, len: Self::Length) -> Self
    where Self: Sized {
        Self::from_constituants(len, self.direction())
    }
    /// Create a vector with the same length and a new direction
    fn with_direction(&self, dir: Self::Angle) -> Self where Self: Sized {
        Self::from_constituants(self.length(), dir)
    }

    fn normalize(&self) -> Self;

    fn as_normal(a: &Self::Angle) -> Self;

    fn dot_product(&self, other: Self) -> Self::Length;
}

impl<T: Float> Point for Point_<T> {
    type Length = T;
    type Angle = Deg<T>;

    fn distance(&self, other: &Self) -> Self::Length {
        euclidean_distance(self, other)
    }

    fn direction(&self, other: &[Self]) -> Self::Angle
    where
        Self: Sized,
    {
        assert!(other.len() == 2);

        let a = VecN::from_points(*self, other[0]);
        let b = VecN::from_points(*self, other[1]);
        let v = a.dot_product(b) / (a.length() * b.length());
        //println!("{:?}", v.acos());

        Rad(v.acos()).into()
    }

    fn signed_direction(&self, other: &[Self]) -> Self::Angle {
        assert!(other.len() == 2);

        let a = VecN::from_points(*self, other[0]);
        let b = VecN::from_points(*self, other[1]);

        let a = a.direction().as_unsigned();
        let b = b.direction().as_unsigned();

        //println!("!!!{:?} {:?}!!!", a, b);

        (b - a).clamp().as_signed()
    }
}
impl<V: Float> Vector<Point_<V>> for VecN<V, 2> {
    type Angle = Deg<V>;
    type Length = V;

    fn length(&self) -> Self::Length {
        (self[0].powi(2) + self[1].powi(2)).sqrt()
    }

    /// Direction in -pi,pi
    fn direction(&self) -> Self::Angle {
        let norm = self.normalize();
        match norm[1].asin() >= V::zero() {
            // 0;pi angle
            true => Rad(norm[0].acos()).into(),
            // -pi;0 angle
            false => Rad(-norm[0].acos()).into(), //(V::from(2.).unwrap() - (norm[0] + V::from(1.).unwrap()) - V::from(1.).unwrap()).acos() - V::from(PI).unwrap()
        }
    }

    fn from_constituants(len: Self::Length, dir: Self::Angle) -> Self {
        let norm = VecN::as_normal(&dir);
        [norm[0] * len, norm[1] * len].into()
    }

    fn from_points(from: Point_<V>, to: Point_<V>) -> Self {
        [to.x - from.x, to.y - from.y].into()
    }

    fn to_point(&self, from: Point_<V>) -> Point_<V> {
        Point_ { x: from.x + self[0], y: from.y + self[1] }
    }

    fn normalize(&self) -> Self {
        let l = self.length();
        [self[0] / l, self[1] / l].into()
    }

    fn as_normal(a: &Self::Angle) -> Self {
        [a.cos(), a.sin()].into()
    }

    fn dot_product(&self, other: Self) -> V {
        self[0] * other[0] + self[1] * other[1]
    }
}
impl<T: Float> OrientedOrthogonal for OriRect2D<T> {
    type Angle = Deg<T>;
    type Position = Point_<T>;
    type Size = Size_<T>;

    fn new(rot: Self::Angle, len: Self::Size, pos: Self::Position) -> Self
    where
        Self: Sized,
    {
        OriRect2D {
            angle: rot,
            rect: Rect_ {
                x: pos.x,
                y: pos.y,
                width: len.width,
                height: len.height,
            },
        }
    }

    fn angle(&self) -> Self::Angle {
        self.angle
    }

    fn position(&self) -> Self::Position {
        Point_ {
            x: self.rect.x,
            y: self.rect.y,
        }
    }

    fn size(&self) -> Self::Size {
        self.rect.size()
    }
}

impl<T: Float> Pointers<T> {
    /// Distance from master to short
    fn width(&self) -> T {
        euclidean_distance(&self.master, &self.short)
    }

    /// Distance from master to long
    fn height(&self) -> T {
        euclidean_distance(&self.master, &self.long)
    }

    /// Distance from master to long
    fn hyp(&self) -> T {
        euclidean_distance(&self.short, &self.long)
    }

    pub fn rotate(&self, angle: Deg<T>, around: Point_<T>) -> Self {
        let rotation = |p| {
            let v = VecN::from_points(around, p);
            let r = VecN::from_constituants(v.length(), v.direction() + angle);
            r.to_point(around)
        };
        Pointers {
            diameter: self.diameter,
            master: rotation(self.master),
            short: rotation(self.short),
            long: rotation(self.long)
        }
    }
}

impl<T: Copy> From<&Pointers<T>> for [Point_<T>; 3] {
    fn from(value: &Pointers<T>) -> Self {
        [value.master, value.short, value.long]
    }
}
impl<T> From<Pointers<T>> for [Point_<T>; 3] {
    fn from(value: Pointers<T>) -> Self {
        [value.master, value.short, value.long]
    }
}

impl<T: Float> TryFrom<Pointers<T>> for Size_<T> {
    type Error = ShapeError;
    /// We also make angle checks to make sure the transform is mostly linear.
    fn try_from(value: Pointers<T>) -> ShapeResult<Self> {
        let (w, h, d) = (value.width(), value.height(), value.hyp());
        let calc_hyp = (w.powi(2) + h.powi(2)).sqrt();
        match ShapeError::is_square(calc_hyp, d) {
            true => Ok(Size_ {
                width: w,
                height: h,
            }),
            false => Err(ShapeError::NotLinearScale)
        }
    }
}

impl<T: Float + Default + Debug> Pointers<T> {
    pub fn as_computed<V: Ord + NumCast + ToPrimitive + Copy>(
        &self,
        value: &Vec<(Point_<T>, T)>,
    ) -> Self {
        // For now this simple implementation should work
        let mut value = value.clone();
        let px: [Point_<T>; 3] = self.into();
        let mut res = [Point_::default(); 3];

        //todo: diameter checks?
        let mut d = T::from(0.).unwrap();
        for p in 0..3 {
            // Rank every possible point against the expected position (ie: the best choice is the nearest)
            let mut rankings = BTreeMap::new();
            for a in 0..value.len() {
                let d = px[p].distance(&value[a].0);
                // We have to cast here, because there is no primitive both float and ord
                rankings.insert(V::from(d).unwrap(), (a, d));
            }
            //Take the best one
            let r = rankings.pop_first().unwrap();
            //Check that there isnt another option that comes close to this
            if !rankings.is_empty() {
                assert!(rankings.first_key_value().unwrap().1 .1 / r.1 .1 > T::from(2.).unwrap());
            }
            let (v, dv) = value.remove(r.1 .0);
            res[p] = v;
            d = d + (dv / T::from(3.).unwrap());
        }
        Pointers {
            diameter: d,
            master: res[0],
            short: res[1],
            long: res[2],
        }
    }
}
impl TryFrom<Vec<u8>> for QRData {
    type Error = ();
    fn try_from(mut value: Vec<u8>) -> std::result::Result<Self, Self::Error> {
        let page = value.pop().ok_or(())?;
        let exid: u8 = value.pop().ok_or(())?;
        Ok(QRData{hash:value, id:exid, page:page})
    }
}

impl TryFrom<QRDelimitor> for OriRect2D<f32> {
    type Error = ShapeError;
    fn try_from(value: QRDelimitor) -> ShapeResult<OriRect2D<f32>> {
        let v = [value[0], value[1], value[2], value[3]];
        //println!("v: {:?}", v);
        ShapeError::as_ortho(&v) //I'm sorry
    }
}
