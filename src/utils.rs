use std::{collections::{BTreeMap, HashMap}, f32::consts::PI, fmt::Debug, ops::{Mul, Sub}};

use num_traits::{float::Float, Num, NumCast, ToPrimitive, Zero};

pub use opencv::core::{Point_, Rect_, Size_, VecN};
use opencv::{core, objdetect, prelude::*, Result};

use crate::status::{ShapeError, ShapeResult};

fn euclidean_distance<T: Float>(from: &Point_<T>, to: &Point_<T>) -> T {
    ((from.x - to.x).powi(2) + (from.y - to.y).powi(2)).sqrt()
}

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

#[derive(Default, Clone, Debug)]
pub struct Question_<T> {
    pub id: String,
    pub page: u8,
    pub rect: Rect_<T>
}

#[derive(Clone, Debug)]
pub enum Answer {
    Binary,
}

#[derive(Default, Clone, Debug)]
pub struct Metadata_<T> {
    pub id: u8,
    pub hash: Vec<u8>,
    pub size: Size_<T>,
    /// The page count
    pub pages: u8,
}

#[derive(Default, Clone, Debug)]
pub struct Content_<T> {
    pub questions: HashMap<String, Question_<T>>,
    pub pointers: Pointers<T>,
    pub qr_code: Rect_<T>,
    pub metadata: Metadata_<T>,
}


#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Deg<T>(pub T);
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Rad<T>(pub T);

const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.;
const DEG_PER_RAD: f64 = 180. / std::f64::consts::PI;

/// A type that describes an agle
pub trait Angle<T> {
    /// The angle's value
    fn value(&self) -> T;

    fn sin(&self) -> T where T: Float;

    fn cos(&self) -> T where T: Float;
}
impl<T: Clone> Angle<T> for Deg<T> {
    fn value(&self) -> T {
        self.0.clone()
    }

    fn sin(&self) -> T where T: Float {
        self.value().sin()
    }

    fn cos(&self) -> T where T: Float {
        self.value().cos()
    }
}
impl<T: Clone> Angle<T> for Rad<T> {
    fn value(&self) -> T {
        self.0.clone()
    }

    fn sin(&self) -> T where T: Float {
        self.value().sin()
    }

    fn cos(&self) -> T where T: Float {
        self.value().cos()
    }
}
impl<T: Clone + NumCast + Zero> From<Deg<T>> for Rad<T> {
    fn from(value: Deg<T>) -> Self {
        Rad(T::from(<f64 as NumCast>::from(value.value()).unwrap_or(f64::NAN) * RAD_PER_DEG).unwrap_or(T::zero()))
    }
}
impl<T: Clone + NumCast + Zero> From<Rad<T>> for Deg<T> {
    fn from(value: Rad<T>) -> Self {
        Deg(T::from(<f64 as NumCast>::from(value.value()).unwrap_or(f64::NAN) * DEG_PER_RAD).unwrap_or(T::zero()))
    }
}
impl<T: Sub<T, Output = T> + Clone> Sub<Deg<T>> for Deg<T> {
    type Output = Deg<T>;
    fn sub(self, rhs: Deg<T>) -> Self::Output {
        Deg(self.value() - rhs.value())
    }
}

/// Objects with this trait are able to describe some kind of rectangle
pub trait OrientedOrthogonal {
    type Size: Size;
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

pub trait Size {
    type Length: Num;

    //fn new<T: Vector<Length = Self::Length>>(vec: T) -> Self;
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
    fn from_constituants(len: Self::Length, dir: Self::Angle) -> Self
    where
        Self: Sized;

    fn from_points(from: P, to: P) -> Self
    where
        Self: Sized;

    /// Create a vector with the same direction and a new length
    fn with_length(&self, len: Self::Length) -> Self
    where
        Self: Sized,
    {
        Self::from_constituants(len, self.direction())
    }
    /// Create a vector with the same length and a new direction
    fn with_direction(&self, dir: Self::Angle) -> Self
    where
        Self: Sized,
    {
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

        Rad(v.acos()).into()
    }

    fn signed_direction(&self, other: &[Self]) -> Self::Angle {
        assert!(other.len() == 2);

        let a = VecN::from_points(*self, other[0]);
        let b = VecN::from_points(*self, other[1]);

        a.direction() - b.direction()
    }
}
impl<V: Float> Vector<Point_<V>> for VecN<V, 2> {
    type Angle = Deg<V>;
    type Length = V;

    fn length(&self) -> Self::Length {
        (self[0].powi(2) + self[1].powi(2)).sqrt()
    }

    fn direction(&self) -> Self::Angle {
        let norm = self.normalize();
        match norm[1].asin() > V::from(0.).unwrap() {
            true => Rad(norm[0].acos()).into(),
            false => Rad((V::from(2.).unwrap()
                - (norm[0] + V::from(1.).unwrap())
                - V::from(1.).unwrap())
            .acos()
                - V::from(PI).unwrap())
            .into(),
        }
    }

    fn from_constituants(len: Self::Length, dir: Self::Angle) -> Self
    where
        Self: Sized,
    {
        let norm = VecN::as_normal(&dir);
        [norm[0] * len, norm[1] * len].into()
    }

    fn from_points(from: Point_<V>, to: Point_<V>) -> Self
    where
        Self: Sized,
    {
        [to.x - from.x, to.y - from.y].into()
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
impl<T> Size for Size_<T> {
    type Length = f32;
    /*fn new<T: Vector<Length = Self::Length>>(vec: T) -> Self {
        Size_(())
    }*/
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
            .map_err(|_| opencv::Error::new(opencv::core::StsParseError, ""))
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

impl<T: Float + Default> Pointers<T> {
    pub fn as_computed<V: Ord + NumCast + ToPrimitive + Copy>(
        &self,
        value: &Vec<(Point_<T>, T)>,
    ) -> Self {
        // For now this simple implementation should work
        let mut value = value.clone();
        let px: [Point_<T>; 3] = self.into();
        let mut res = [Point_::default(); 3];

        //todo: diameter checks
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
