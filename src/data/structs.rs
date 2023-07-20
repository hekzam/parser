/*
 * Content structs
*/

pub mod content {
    use std::collections::HashMap;
    use serde::Deserialize;

    type Float = f64;
    type Int = u8;
    type Text = String;
    /// Position and general information about a document.
    #[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
    pub enum Kind {
        Binary,
        Numeric,
        Text,
    }
    /// Position and general information about a document.
    #[derive(Deserialize, Debug)]
    pub struct Content {
        pub q: HashMap<Text, Question>,
        pub mk: Markers,
        pub qr: Rectangle,
        pub md: Meta,
    }
    /// Document metadata
    #[derive(Deserialize, Debug)]
    pub struct Meta {
        pub id: Int,
        pub hash: Vec<Int>,
        pub w: Float,
        pub h: Float,
        pub n: Int,
    }
    /// Struct describing a rectangle
    #[derive(Deserialize, Debug)]
    pub struct Rectangle {
        pub x: Float,
        pub y: Float,
        pub dx: Float,
        pub dy: Float,
    }
    /// The markers on a document
    #[derive(Deserialize, Debug)]
    pub struct Markers {
        pub d: Float,
        pub m: Position, 
        pub l: Position,
        pub s: Position,
    }
    /// A position
    #[derive(Deserialize, Debug)]
    pub struct Position {
        pub x: Float,
        pub y: Float,
    }
    /// A question box
    #[derive(Deserialize, Debug)]
    pub struct Question {
        pub t: Kind,
        pub p: Int,
        pub at: Rectangle,
    }
}
/*
 * Model structs
*/
pub mod model {
    use std::collections::HashMap;
    use serde::Deserialize;

    type Float = f64;
    type Int = u8;
    type Text = String;
    /// Describes the model information
    #[derive(Deserialize, Debug)]
    pub struct Model {
        pub md: Meta,
        pub ex: HashMap<Text, Exercise>,
    }
    /// Describe an exercise
    #[derive(Deserialize, Debug)]
    pub struct Exercise {
        pub max: Float,
        pub min: Float,
        pub q: HashMap<Text, Question>,
    }
    /// Describes a question
    #[derive(Deserialize, Debug)]
    pub struct Question {
        #[serde(flatten)]
        pub kind: Kind,
        pub max: Float,
        pub min: Float,
    }
    /// Describes a (binary) answer
    #[derive(Deserialize, Debug)]
    pub struct Answer {
        /// The number of points awarded when true
        pub score: Float,
    }
    /// Describes a (MultipleTF) answer
    #[derive(Deserialize, Debug)]
    pub struct AnswerTF {
        /// The number of points awarded when true
        pub score: (Float, Float),
    }
    impl AnswerTF {
        pub fn as_true(&self) -> Answer {
            Answer { score: self.score.0 } 
        }
        pub fn as_false(&self) -> Answer {
            Answer { score: self.score.1 } 
        }
    }
    /// Document metadata
    #[derive(Deserialize, Debug)]
    pub struct Meta {
        pub id: Int,
        pub hash: Vec<Int>,
    }
    /// Position and general information about a document.
    #[derive(Deserialize, Debug)]
    #[serde(tag = "kind", content = "a")]
    pub enum Kind {
        MCQ(HashMap<Text, Answer>),
        OneInN(HashMap<Text, Answer>),
        MultipleTF(HashMap<Text, AnswerTF>),
    }
}