

type Float = f64;
type Int = i64;
type Text = String;

/// Position and general information about a document.
pub struct Content {
    q: Vec<Question>,
    mk: Markers,
    qr: Rectangle,
    p: Meta,
}

/// Document metadata
struct Meta {
    w: Float,
    h: Float,
    n: Int,
}
/// Struct describing a rectangle
struct Rectangle {
    x: Float,
    y: Float,
    dx: Float,
    dy: Float,
}
/// The markers on a document
struct Markers {
    m: Circle, 
    l: Circle,
    s: Circle,
}
/// A circle!
struct Circle {
    x: Float,
    y: Float,
    d: Float,
}
/// A question box
struct Question {
    id: Text,
    p: Int,
    x: Float,
    y: Float,
    dx: Float,
    dy: Float,
}