use std::{collections::HashMap, f32::consts::PI, fmt::{Write, Display}, path::PathBuf, time, ops::{Add, AddAssign}};

use clap::Parser;
use log::{debug, error, info, log_enabled, trace, warn, Level};

use opencv::{
    core::{self, CV_8UC1},
    features2d::{SimpleBlobDetector, SimpleBlobDetector_Params},
    imgcodecs, imgproc,
    prelude::*,
    types, Result,
};
/*opencv::opencv_branch_4! {
    use opencv::core::AccessFlag::ACCESS_READ;
}*/
opencv::not_opencv_branch_4! {
    use opencv::core::ACCESS_READ;
}

mod cli;
mod data;
mod status;
mod utils;
use utils::*;
use data::structs::model;

use crate::status::ShapeError;
/*
 * Convenience Functions *
 not *just* for convenience, but mostly so
*/
/// Makes some platform checks
fn boot_cl() -> Result<()> {
    let opencl_have = core::have_opencl()?;
    if opencl_have {
        core::set_use_opencl(true)?;
        if log_enabled!(Level::Info) {
            // Platform information is cool, but not always useful
            let mut platforms = types::VectorOfPlatformInfo::new();
            core::get_platfoms_info(&mut platforms)?;
            for (platf_num, platform) in platforms.into_iter().enumerate() {
                info!("Platform #{}: {}", platf_num, platform.name()?);
                for dev_num in 0..platform.device_number()? {
                    let mut dev = core::Device::default();
                    platform.get_device(&mut dev, dev_num)?;
                    info!("  OpenCL device #{}: {}", dev_num, dev.name()?);
                    info!("    vendor:  {}", dev.vendor_name()?);
                    info!("    version: {}", dev.version()?);
                }
            }
        }
    }
    let opencl_use = core::use_opencl()?;
    info!(
        "OpenCL is {} and {}",
        if opencl_have {
            "available"
        } else {
            "not available"
        },
        if opencl_use { "enabled" } else { "disabled" },
    );

    match opencl_have && opencl_use {
        false => Err(opencv::Error::new(
            core::StsInternal,
            "Empty image from reading file.",
        )),
        true => Ok(()),
    }
}

/// Write an image only useful when tracing
fn image_trace(img: &Mat, name: &str) -> Result<bool> {
    const TRACE_FOLDER: &str = "parser/trace/";
    const TRACE_EXTENSION: &str = ".jpg";

    if log_enabled!(Level::Trace) {
        let fname = TRACE_FOLDER.to_owned() + name + TRACE_EXTENSION;
        trace!("Writing image trace: \"{}\"", fname);
        imgcodecs::imwrite(&fname, img, &core::Vector::default())
    } else {
        Ok(false)
    }
}

/// When tracing, creates an order-stable iteration from an unstable collection
fn ssitr<S: Iterator<Item = U>, T: IntoIterator<IntoIter = S, Item = U>, U, Z: FnMut(&U,&U) -> std::cmp::Ordering>(unstable_collection: T, stabilizer: Z) -> std::vec::IntoIter<U> {
    // Use a order-stable iteration when tracing
    if log_enabled!(Level::Trace) {
        let mut vtr = unstable_collection.into_iter().collect::<Vec<U>>();
        vtr.sort_by(stabilizer);
        vtr.into_iter()
    } else {
        // Note: not really efficient, would be great if we didnt have to go trhu a collect :(
        unstable_collection.into_iter().collect::<Vec<U>>().into_iter()
    }
}

/*
 * Display Functions *
 just pretty stuff that can often be useful
*/
/// Creates a visual representation of a histogram
fn disp_hist(hist: &Mat, maxh: i32, barw: i32) -> Result<Mat> {
    let imgs = Size_ {
        width: barw * 256,
        height: maxh,
    };
    let img = Mat::new_size_with_default(imgs, CV_8UC1, core::Scalar::default())?;

    let mut max_val = 0.;
    core::min_max_loc(
        &hist,
        None,
        Some(&mut max_val),
        None,
        None,
        &core::no_array(),
    )?;


    for i in 0..256 {
        /*trace!("{:?}", Rect_ {
            x: i * barw,
            y: 0,
            width: barw,
            height: ((maxh as f32) * ((hist.at::<f32>(i)? + 1.).ln() / max_val as f32)) as i32,
        });*/
        let mut clip = Mat::roi(&img, Rect_ {
            x: i * barw,
            y: 0,
            width: barw,
            height: ((maxh as f32) * (hist.at::<f32>(i)? / max_val as f32)) as i32,
        })?;

        clip.set(core::Scalar::new(255., 0., 0., 0.))?;
        /*imgproc::rectangle(
            &mut img,
            Rect_ {
                x: i * barw,
                y: 0,
                width: barw,
                height: ((maxh as f32) * (hist.at::<f32>(i)?.ln() / max_val as f32)) as i32,
            },
            core::Scalar::new(255., 0., 0., 0.),
            1,
            imgproc::LINE_8,
            0,
        )?;*/
    }

    //imgcodecs::imwrite("YAAAAA.jpg", &img, &core::Vector::new())?;
    Ok(img)
}

/// Returns a string describing an image
fn image_description_string(img: &Mat) -> Result<String> {
    let f = |e: std::fmt::Error| opencv::Error::new(core::StsError, e.to_string());
    let mut r = String::new();

    let sz = img.size()?;
    let chn = match img.channels() {
        1 => "gray",
        3 => "rgb",
        4 => "rgba",
        _ => "unknown",
    };
    let de = match img.depth() {
        core::CV_8U => "8 bits",
        core::CV_8S => "8 bits (signed)",
        core::CV_16U => "16 bits",
        core::CV_16S => "16 bits (signed)",
        core::CV_32S => "32 bits (signed)",
        core::CV_32F => "floating (32)",
        core::CV_64F => "floating (64)",
        _ => "unknown",
    };
    write!(&mut r, "image({}x{}, {} {})", sz.width, sz.height, chn, de).map_err(f)?;
    Ok(r)
}

/// Returs a string describing a hash
fn pretty_hash(hash: &Vec<u8>) -> Result<String> {
    let f = |e: std::fmt::Error| opencv::Error::new(core::StsError, e.to_string());

    let mut r = String::new();
    for v in hash {
        write!(&mut r, "{:02x}", v).map_err(f)?;
    }

    Ok(r)
}

/// Returs a short string describing a hash
fn short_pretty_hash(hash: &Vec<u8>) -> Result<String> {
    let f = |e: std::fmt::Error| opencv::Error::new(core::StsError, e.to_string());

    let s = pretty_hash(hash)?;

    let mut r = String::new();

    write!(&mut r, "{}[{}]", &s[0..6], s.len()).map_err(f)?;

    Ok(r)
}

/// Returns a string describing QR Data
fn qr_description_string(qr: &QRData) -> Result<String> {
    let f = |e: std::fmt::Error| opencv::Error::new(core::StsError, e.to_string());
    let mut r = String::new();

    write!(
        &mut r,
        "qr_data({}, exam ID {}, page {})",
        pretty_hash(&qr.hash)?,
        qr.id,
        qr.page
    )
    .map_err(f)?;
    Ok(r)
}

/// Returns a string describing markers
fn markers_description_string(markers: &Pointers<f32>) -> Result<String> {
    let f = |e: std::fmt::Error| opencv::Error::new(core::StsError, e.to_string());
    let mut r = String::new();

    write!(
        &mut r,
        "markers({}:{}-l, {}:{}-m, {}:{}-s, ø{})",
        markers.long.x.round(),
        markers.long.y.round(),
        markers.master.x.round(),
        markers.master.y.round(),
        markers.short.x.round(),
        markers.short.y.round(),
        markers.diameter
    )
    .map_err(f)?;
    Ok(r)
}

/// Returns a string describing a 2x3 transform matrix
fn transform_description(tr: &Mat) -> Result<String> {
    let f = |e: std::fmt::Error| opencv::Error::new(core::StsError, e.to_string());
    let mut r = String::new();

    let values: &[f64] = tr.data_typed()?;
    if values.len() != 6 {
        error!(
            "Transform matrix has wrong size! Should contain 6 elements, has {}",
            values.len()
        );
        return Err(opencv::Error::new(
            core::StsBadSize,
            "Wrong size for transform matrix.",
        ));
    }

    write!(
        &mut r,
        "[{:.3} {:.3} {:.3} | {:.3} {:.3} {:.3}]",
        values[0], values[1], values[2], values[3], values[4], values[5]
    )
    .map_err(f)?;

    Ok(r)
}

/// Returns a string describing a rectangle
fn rect_description(rect: &Rect_<f32>) -> Result<String> {
    let f = |e: std::fmt::Error| opencv::Error::new(core::StsError, e.to_string());
    let mut r = String::new();

    write!(
        &mut r,
        "r({:.2}:{:.2} - {:.2}:{:.2})",
        rect.tl().x,
        rect.tl().y,
        rect.br().x,
        rect.br().y
    )
    .map_err(f)?;

    Ok(r)
}

/// Returns a string describing a standard-size rectangle
fn standard_rect_description(rect: &Rect_<i32>) -> Result<String> {
    let f = |e: std::fmt::Error| opencv::Error::new(core::StsError, e.to_string());
    let mut r = String::new();

    write!(&mut r, "r({}:{})", rect.tl().x, rect.tl().y).map_err(f)?;

    Ok(r)
}

type HistSeries = (Mat, [i32; 4]);
/// Creates the image buffer to use in display
fn startup_hist_serie(
    histogram_height: i32,
    bar_width: i32,
    histogram_count: i32,
) -> Result<HistSeries> {
    Ok((
        Mat::new_rows_cols_with_default(
            histogram_height * histogram_count,
            histogram_height + bar_width * 256,
            core::CV_8UC1,
            core::Scalar::default(),
        )?,
        [0, histogram_count, histogram_height, bar_width],
    ))
}

/// Displays multiple histograms next to their source, in the same image, one after the other
/// The source will be a square image of the size of the histogram's height
fn disp_hist_serie(hist_series: &mut HistSeries, imgsource: &Mat, histogram: &Mat) -> Result<()> {
    let count = hist_series.1[1];
    let height = hist_series.1[2];
    let barw = hist_series.1[3];
    let i = hist_series.1.get_mut(0).unwrap();
    if *i >= count {
        return Err(opencv::Error::new(core::StsBadSize, "Histogram buffer is already full, tried to add too many!"));
    }
    // We make a clip of the area to write to
    let mut imgs_clip = Mat::roi(&hist_series.0, Rect_ { x: 0, y: *i * height, width: height, height })?;

    // Standardize the image size
    imgproc::resize(imgsource, &mut imgs_clip, Size_ { width: height, height }, 0., 0., imgproc::INTER_NEAREST)?;

    let mut hist_clip = Mat::roi(&hist_series.0, Rect_ { x: height, y: *i * height, width: barw * 256, height })?;

    // We write our histogram
    disp_hist(histogram, height, barw)?.copy_to(&mut hist_clip)?;

    i.add_assign(1);

    Ok(())
}

/*
 * Core functions *
 the collection of them makes up the bulk of the app
*/
/// Reads an image from a file and applies a slight filter cleanup
fn read_image(path: &str) -> Result<Mat> {
    // Get the image
    debug!("Reading image at: \"{}\"", path);
    let img = imgcodecs::imread(&path, imgcodecs::IMREAD_GRAYSCALE)?; //Open & convert to grayscale
    info!("Image: {}", image_description_string(&img)?);
    if img.empty() {
        error!("Empty image (searched at {})! This is likely because reading failed (ie: there was no file to read).", path);
        return Err(opencv::Error::new(
            core::BadImageSize,
            "Empty image from reading file.",
        ));
    }
    let resol: Size_<i32> = img.size()?;
    // it may be nice to check aspect ratios at this point?
    //todo rescale image if too big?  (>600 dpi, probs) (to make the next operations less costly!)
    if resol.width.max(resol.height) > 5000 {
        // Not a fail state!
        warn!("Your image appears to be very big! ({} by {}) This could have an impact on performance!", resol.width, resol.height)
    }
    // Image cleanup
    let mut dst = Mat::default();
    // Clean the image (this one gives very good results!)
    imgproc::bilateral_filter(&img, &mut dst, 5, 50., 50., core::BORDER_DEFAULT)?;
    image_trace(&dst, "cleaned")?;
    Ok(dst)
}

/// Calculates the standard resolution
fn standard_size(image: &Mat, standard: &Size_<f32>) -> Result<f32> {
    let resol = image.size()?;

    let res = resol.cast().as_scale(&standard)?;

    info!("Resolution: {} px/cm", res);

    Ok(res)
}

/// Detects, decodes, and checks a QR code
fn check_qr<T>(
    image: &Mat,
    res: f32,
    standard_qr: &Rect_<f32>,
    meta: &Metadata_<T>,
) -> Result<(OriRect2D<f32>, QRData)> {
    const LEN_DX: f32 = ShapeError::LENGTH_THRESHOLD as f32;
    // Detect QR:
    //todo! threshold pour eviter d'avoir des pbs de transparance (eheh trans-parance)
    let qr = detect_qr(&image)?;

    let qr_pos: Vec<Point_<f32>> = qr.0.into_iter().map(|v| v.rescale(1. / res)).collect();

    let qr_pos = OriRect2D::try_from(qr_pos).unwrap(); // note: point error?
    debug!("QR Position: {} deg({:.2})", rect_description(&qr_pos.rect)?, qr_pos.angle.value());
    let sz = qr_pos.rect.as_scale(&standard_qr.cast()).unwrap(); //todo hande the error better
    if sz > LEN_DX || sz < 1. / LEN_DX {
        error!("Detected QR Code is too {}. Resizing by a factor greater than {:.0}% could cause too great of an error, and is therefore disabled.",
            if sz > LEN_DX { "big" } else { "small" }, (1. - LEN_DX) * 100.);
        return Err(opencv::Error::new(core::StsBadSize, "Invalid QR code size."));
    }
    // We could use the rotation as a hint for mirroring detection: 
    //We can calculate the expected aspect ratio, and match it.
    //If the AR is reversed (we expect portrait but have landscape)
    //  and the QR code is rotated by ±90°,
    //  the image is just rotated by that same amount. We can apply the inverse
    //  rotation accordingly.
    //If the AR is as expected (we expected and got portrait)
    //  and the QR code is still rotated by ±90°,
    //  the image is mirrored! We can apply an inverse mirror operation,
    //  (trivial, cf: https://docs.opencv.org/4.x/d1/da0/tutorial_remap.html)
    //  And the resulting image will have a qr code rotated by 180° or 0° 
    //  (because mirror operations cancel each other with a rotation)
    //(Use the same logic for the cases of reversed AR with r in {180°,0°}
    //  (means the image is mirrored),
    //  and expected AR with r in {180°,0°} (means the image is rotated)

    info!("QR Data: {}", qr_description_string(&qr.1)?);
    // To chain multiple errors
    let mut r = Ok((qr_pos, qr.1.clone()));
    if meta.hash != qr.1.hash {
        error!(
            "Detected QR Code has hash {}, expected {}",
            short_pretty_hash(&qr.1.hash)?,
            short_pretty_hash(&meta.hash)?
        );
        r = Err(opencv::Error::new(core::StsBadArg, "Invalid QR Code data."));
    } else {
        debug!("QR Code contains the expected hash");
    }
    if meta.id != qr.1.id {
        error!(
            "Detected QR Code has exam ID {}, expected {}",
            qr.1.id, meta.id
        );
        r = Err(opencv::Error::new(core::StsBadArg, "Invalid QR Code data."));
    } else {
        debug!("QR Code contains the expected ID");
    }
    if meta.pages < qr.1.page || qr.1.page == 0 {
        error!(
            "Detected QR Code has page number {} which is not in [1;{}]",
            qr.1.page, meta.pages
        );
        r = Err(opencv::Error::new(core::StsBadArg, "Invalid QR Code data."));
    } else {
        debug!("QR Code contains a valid page number");
    }

    r
}

/// Detects markers within an image
fn detect_markers(
    image: &Mat,
    actual_qr: &(OriRect2D<f32>, QRData),
    standard_qr: &Rect_<f32>,
    standard_markers: &Pointers<f32>,
) -> Result<Pointers<f32>> {
    const LEN_DX: f32 = ShapeError::LENGTH_THRESHOLD as f32;

    // Either lower the size range (its 20% right now!), or we dont need sz :)
    let sz = actual_qr.0.rect.as_scale(standard_qr).unwrap();
    let dotsize: f32 = sz * standard_markers.diameter;
    let marker_radius = dotsize / 2.;

    // Now we find the 3 lowest blobs!
    // We only use a blob detector. It may be wise to cross-reference blob detection and template matching, to ensure better detection (removing some false positives, and maybe allowing us to lower our thresholds on critical parts, removing some false negatives?).
    // Template matching would be used as: for each matched blob, look at the template matched buffer, and check that we have a prositive result from it.
    let size_mul = (1. / LEN_DX, LEN_DX);
    let mut detector = SimpleBlobDetector::create(SimpleBlobDetector_Params {
        threshold_step: 10.,
        min_threshold: 0.,
        max_threshold: 180.,
        min_repeatability: 2,
        min_dist_between_blobs: 0.1, // Markers will be on the corners, with a wide margin of error. however, we dont use it because the best detected blob may not be in the corners (if our markers are not very good..). //IMG_FORM_DEFAULT.width() * res * sz * 0.1
        filter_by_color: true,
        blob_color: 0, // Filter for black markers
        filter_by_area: true,
        min_area: (marker_radius * size_mul.0).powi(2) * PI, // Filter for the right marker size, we can afford to be sensitive here.
        max_area: (marker_radius * size_mul.1).powi(2) * PI,
        filter_by_circularity: true,
        min_circularity: 0.8, // Filter for circular markers
        max_circularity: f32::MAX,
        filter_by_inertia: true,
        min_inertia_ratio: 0.6, //I have no idea what inertia is.
        max_inertia_ratio: f32::MAX,
        filter_by_convexity: true,
        min_convexity: 0.95,
        max_convexity: f32::MAX,
        collect_contours: false,
    })?;
    let mut keyp: core::Vector<core::KeyPoint> = core::Vector::new();
    detector.detect(&image, &mut keyp, &core::no_array())?;
    //note: Do a dichotomic search?
    let mut kp = Vec::new();
    for k in keyp {
        let s = k.pt();
        debug!("point(@{}:{}, ø{})", s.x.round(), s.y.round(), k.size());
        kp.push((s, k.size()));
    }
    // We use the rotation information from our qr code to rotate the standard markers (and thus, provide a rotation-aware fitting)
    // Right now we do need the qr code. IDEALLY, we would be able to do some minimal rotation detection even without, i think.
    // We could also apply the size hint obtained from the QR code to our standards markers!
    let center = image.size()?.cast().rescale(0.5).into();
    let rotated_markers = standard_markers.rotate(actual_qr.0.angle, center);

    info!("standard: {}", markers_description_string(&standard_markers)?);
    info!("rotated:  {}", markers_description_string(&rotated_markers)?);
    // We hopefully have our points, now fit them!
    // We have to force it as an int, because f32 is not ord!
    // We rescale it because the cast may make us loose too much information
    let actual_form = rotated_markers.as_computed::<i64>(&kp)?;
    info!("actual:   {}", markers_description_string(&actual_form)?);
    Ok(actual_form)
}

/// Fixes an image (applies transforms to ensure a standardized position)
fn fix_image(
    image: &Mat,
    standard_markers: &Pointers<f32>,
    actual_markers: &Pointers<f32>,
) -> Result<Mat> {
    let f = |_| opencv::Error::new(opencv::core::StsParseError, "Scale error!");
    // Calculate the fixed image's resolution
    //if we want to avoid rescales, we need to apply an inverse scale transform on b
    let size_a = Size_::try_from(standard_markers.to_owned()).map_err(f)?;
    let size_b = Size_::try_from(actual_markers.to_owned()).map_err(f)?;
    let new_scale: f32 = size_a.as_scale(&size_b)?;
    //let new_res: Size_<i32> = image.size()?.cast().rescale(new_scale).cast();
    // Now that we have the new res, we do the inverse scale transform (which transforms our points as to ensure the resize element of the transform is 1)!
    //todo
    debug!("Rescale by a factor of: {}", new_scale);

    // Create a transform map to go from original image to fixed image
    let a: [Point_<f32>; 3] = standard_markers.into();
    let b: [Point_<f32>; 3] = actual_markers.into();

    let scalemap = imgproc::get_affine_transform_slice(b.as_slice(), a.as_slice())?;
    debug!("transform matrix: {}", transform_description(&scalemap)?);

    // Fix the image & write it
    let mut resized = Mat::default();
    imgproc::warp_affine(
        &image,
        &mut resized,
        &scalemap,
        image.size()?,
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        core::Scalar::default(),
    )?;
    image_trace(&resized, "resized")?;
    Ok(resized)
}

fn histogram(image: &Mat) -> Result<Mat> {
    let mut himg: core::Vector<Mat> = core::Vector::new();
    himg.push(image.clone());
    let mut hist = Mat::new_nd_with_default(&[256], core::CV_32FC1, core::Scalar::default())?;
    imgproc::calc_hist(
        &himg,
        &core::Vector::from(vec![0]),
        &core::no_array(),
        &mut hist,
        &core::Vector::from(vec![256]),
        &core::Vector::new(),
        false,
    )?;
    Ok(hist)
}

fn resolve_boxes(
    image: &Mat,
    boxes: &HashMap<String, Question_<f64>>,
    page: u8,
    resolution: f32,
) -> Result<HashMap<String, Answer>> {
    let mut r = HashMap::new();
    let mut inner_hist_series;
    // Traces the boxes
    if log_enabled!(Level::Trace) {
        inner_hist_series = startup_hist_serie(100, 1, boxes.iter().filter(|v| v.1.page == page && v.1.kind == data::Kind::Binary).count() as i32)?;
    } else {
        // Otherwise rust yells at us
        inner_hist_series = (Mat::default(), [0,0,0,0])
    }

    // Calculate the background light
    let total_hist = histogram(image)?;
    if log_enabled!(Level::Trace) {
        image_trace(&disp_hist(&total_hist, 100, 1)?, "histogram")?;
    }
    // We could probably bump it to 75?
    //let total_hist: &[f32] = total_hist.data_typed()?;
    //let bg = index_quartile(total_hist, 0.5);
    //then we could use out background light as a calibration.
    // Even better, we could find Q1-Q3, and with it learn something about the standard distribution of light? (and thus, we'd be able to guess wether or not it deviates from it significantly)

    let itr = ssitr(boxes, |a, b| a.0.cmp(b.0));
    for (id, b) in itr {
        if b.page != page {
            continue;
        }
        // Extract the actual content!
        let contour = b.rect.rescale(resolution as f64).cast();
        trace!("box \"{}\": {}", id, standard_rect_description(&contour)?);
        let crop = Mat::roi(&image, contour)?;
        // Analyse the content dependent on type
        match b.kind {
            data::Kind::Binary => {
                let hists;
                if log_enabled!(Level::Trace) {
                    hists = Some(&mut inner_hist_series);
                } else {
                    hists = None;
                }
                r.insert(id.to_owned(), binary_box_analysis(&crop, hists)?)
            },
            _ => todo!(),
        };
    }
    image_trace(&inner_hist_series.0, "captures")?;
    Ok(r)
}

/// Analyses a binary box, returns the answer
fn binary_box_analysis(crop: &Mat, histser: Option<&mut HistSeries>) -> Result<Answer> {
    //note We could use the median color of the paper as a tool to scale our threshold
    // Calculate a hist to see how everythin' is distributed
    let hist = histogram(crop)?;
    if let Some(h) = histser {
        disp_hist_serie(h, crop, &hist)?;
    }
    let hist: &[f32] = hist.data_typed()?;
    let i = index_quartile(hist, 0.5)?;
    let a;
    if i <= 142 {
        a = Answer::Binary(true);
    } else {
        a = Answer::Binary(false);
    }
    trace!("=> {:?} ({})", a, i);
    Ok(a)
    //disp_hist(&hist, 200, 2)?;
}


/*
 * Notation functions *
 used for the notation process.
*/
/// Stores the calculated grades
#[derive(Debug, Clone)]
enum GradeCollector {
    Number(BinaryAnswerScore),
    Digits(Vec<BinaryAnswerScore>),
}
impl GradeCollector {
    fn clamp(self, min: f64, max: f64) -> Self {
        match self {
            GradeCollector::Number(x) => GradeCollector::Number(x.clamp(min, max)),
            GradeCollector::Digits(_) => self,
        }
    }
}
impl Add<TypedAnswer> for GradeCollector {
    type Output = Option<GradeCollector>;
    fn add(self, rhs: TypedAnswer) -> Self::Output {
        match self {
            GradeCollector::Number(x) => match rhs.1 {
                model::Grader::Number => Some(GradeCollector::Number(x+rhs.0)),
                _ => None,
            },
            GradeCollector::Digits(v) => match rhs.1 {
                model::Grader::Digits(i) => {
                    let mut w = v.clone();
                    let a = w.get_mut(i as usize)?;
                    *a = *a + rhs.0;
                    Some(GradeCollector::Digits(w))
                },
                _ => None,
            }
        }
    }
}
impl Add for GradeCollector {
    type Output = Option<GradeCollector>;
    fn add(self, rhs: Self) -> Self::Output {
        match self {
            GradeCollector::Number(x) => match rhs {
                GradeCollector::Number(y) => Some(GradeCollector::Number(x+y)),
                _ => None,
            },
            GradeCollector::Digits(v) => match rhs {
                GradeCollector::Digits(w) => {
                    if v.len() != w.len() {
                        return None;
                    }
                    let mut i = 0;
                    Some(GradeCollector::Digits(v.into_iter().map(|a| {
                        let r = a + w[i];
                        i+=1;
                        r
                    }).collect()))
                },
                _ => None,
            }
        }
    }
}
impl Display for GradeCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GradeCollector::Number(x) => write!(f, "{}", x),
            GradeCollector::Digits(v) => {
                for x in v {
                    write!(f, "{}", x.string("_"))?;
                }
                Ok(())
            },
        }
    }
}
impl From<model::Grader> for GradeCollector {
    fn from(value: model::Grader) -> Self {
        match value {
            model::Grader::Number => GradeCollector::Number(BinaryAnswerScore::Blank),
            model::Grader::Digits(n) => GradeCollector::Digits(
                (0..n).into_iter()
                    .map(|_| BinaryAnswerScore::Blank).collect())
        }
    }
}
type Grades = HashMap<String, GradeCollector>;
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
enum BinaryAnswerScore {
    #[default]
    Blank,
    Answered(f64),
}
impl BinaryAnswerScore {
    /// Whenver it's safe to assume that unanswered will mean 0 points (ie: MCQs)
    fn value(&self) -> f64 {
        match self {
            BinaryAnswerScore::Blank => 0.,
            BinaryAnswerScore::Answered(x) => *x,
        }
    }
    fn answered(&self) -> bool {
        match self {
            BinaryAnswerScore::Blank => false,
            BinaryAnswerScore::Answered(_) => true,
        }
    }
    fn clamp(self, min: f64, max: f64) -> Self {
        match self {
            BinaryAnswerScore::Blank => self,
            BinaryAnswerScore::Answered(x) => BinaryAnswerScore::Answered(x.clamp(min, max)),
        }
    }
    fn string(&self, blank: &str) -> String {
        match self {
            BinaryAnswerScore::Blank => blank.to_owned(),
            BinaryAnswerScore::Answered(x) => x.to_string(),
        }
    }
}
impl From<f64> for BinaryAnswerScore {
    fn from(value: f64) -> Self {
        if value.is_nan() {
            BinaryAnswerScore::Blank
        } else {
            BinaryAnswerScore::Answered(value)
        }
    }
}
impl Add for BinaryAnswerScore {
    type Output = BinaryAnswerScore;
    fn add(self, rhs: Self) -> Self::Output {
        match self {
            // Either kept blank, or assigned an answer
            BinaryAnswerScore::Blank => rhs,
            // Merge the anwer values
            BinaryAnswerScore::Answered(x) => BinaryAnswerScore::Answered(x + rhs.value()),
        }
    }
}
impl Display for BinaryAnswerScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryAnswerScore::Blank => write!(f, "blank"),
            BinaryAnswerScore::Answered(x) => write!(f, "{}", x),
        }
    }
}
type TypedAnswer = (BinaryAnswerScore, model::Grader);
/// Notes an individual binary answer
fn note_bin_answer(_a_id: &String, model: &model::Answer, actual: &Answer) -> Option<BinaryAnswerScore> {
    match actual {
        Answer::Binary(v) => match v {
            true => Some(model.score.into()),
            false => Some(f64::NAN.into()),
        }
    }
}
/// Notes a question. Since MTF (MTF hehe) is technically multiple questions, we return a collection of answer scores.
fn note_question(_q_id: &String, model: &model::Question, actual: &HashMap<String, Answer>) -> Option<Vec<BinaryAnswerScore>> {
    let mut grade_table = Vec::new();
    match &model.kind {
        // The simplest case. We just test each answer!
        model::Kind::MCQ(answers) => {
            let mut note = BinaryAnswerScore::Blank;

            let itr = ssitr(answers, |a, b| a.0.cmp(b.0));
            for (id, a) in itr {
                if let Some(v) = actual.get(id) {
                    let n = note_bin_answer(&id, &a, v)?;
                    if n.answered() {
                        trace!("answer {} filled, granting {} points.", id, n);
                    }
                    note = note + n;
                } else {
                    error!("Discrepancy detected: {} does not exist in the position file.", id);
                }
            }
            let note = note.clamp(model.min, model.max);
            if note.answered() {
                debug!("Note for question {}: {}", _q_id, note);
            } else {
                warn!("Question {} was not answered!", _q_id);
            }
            grade_table.push(note);
        },
        // W
        model::Kind::OneInN(answers) => {
            let mut note = BinaryAnswerScore::Blank;
            
            let itr = ssitr(answers, |a, b| a.0.cmp(b.0));
            for (id, a) in itr {
                if let Some(v) = actual.get(id) {
                    let n = note_bin_answer(id, a, v)?;
                    if !note.answered() {
                        // Not yet found the actual answer
                        if n.answered() {
                            debug!("Question {} has answer {}, granting {} points.", _q_id, id, n);
                        }
                        note = note + n;
                    } else if n.answered() {
                        // More than one answer found
                        error!("Multiple answers were chosen for this OneInN question: {}.", _q_id);
                        return None;
                    }
                } else {
                    error!("Discrepancy detected: {} does not exist in the position file.", id);
                }
            }
            if !note.answered() {
                warn!("Question {} was not answered!", _q_id);
            }
            // We PROBABLY dont need to clamp here?
            let note = note.clamp(model.min, model.max);
            grade_table.push(note);
        },
        // Like MCQ mixed with OneInN (because it's closer to a series of anser than a series of questions)
        model::Kind::MultipleTF(answers) => {            
            let itr = ssitr(answers, |a, b| a.0.cmp(b.0));
            for (id, a) in itr {
                // Check both true and false
                let id_t = id.to_owned() + "T";
                let t;
                if let Some(v) = actual.get(&id_t) {
                    t = Some(note_bin_answer(&id_t, &a.as_true(), v)?);
                } else {
                    t = None
                }
                let id_f = id.to_owned() + "F";
                let f;
                if let Some(v) = actual.get(&id_f) {
                    f = Some(note_bin_answer(&id_f, &a.as_false(), v)?);
                } else {
                    f = None
                }
                // Check that either both or none of them were found AND that only one was ticked.
                if f.is_some() && t.is_some() {
                    let f = f.unwrap();
                    let t = t.unwrap();
                    if !(f.answered() && t.answered()) {
                        let a = f+t;
                        if a.answered() {
                            debug!("{} was answered {}, granting {} points.", id, if t.answered() { "true" } else { "false" }, (a));
                        } else {
                            warn!("{} was not answered!", _q_id); 
                        }
                        grade_table.push(a);
                    } else {
                        error!("TrueFalse question has both answers detected as filled: {}", id);
                        return None;
                    }
                } else if !(t.is_none() && f.is_none()) {
                    // Only one of the two does not exist, perplexing...
                    error!("Could not find answer box \"{}\" in TrueFalse element: {}.", if t.is_some() { "true" } else { "false" }, id);
                    return None;
                } else {
                    error!("Discrepancy detected: {} does not exist in the position file.", id);
                }
            }
        }
    };

    Some(grade_table)
}
/// Notes a whole exercise. Compiles the individual scores into grades as it goes
fn note_exercise(_ex_id: &String, model: &model::Exercise, actual: &HashMap<String, Answer>, initial_grader: Grades) -> Option<Grades> {
    let mut grades = initial_grader;

    let itr = ssitr(&model.q, |a, b| a.0.cmp(b.0));
    for (id, q) in itr {
        let note = note_question(id, q, actual)?;
        let grade = grades.entry(q.gd.0.clone()).or_insert(q.gd.1.clone().into());
        let note = note.into_iter().reduce(|acc, e| acc + e)?.clamp(q.min, q.max);
        *grade = (grade.clone() + (note, q.gd.1.clone()))?;
    }
    for (_, v) in grades.iter_mut() {
        *v = v.clone().clamp(model.min, model.max);
    }
    //let note = note.clamp(model.min, model.max);
    //info!("Note for exercise {}: {}", _ex_id, note);
    Some(grades)
}
/// Notes the exam according to the model
fn note_exam(model: &model::Model, answers: &HashMap<String, Answer>) -> Option<Grades> {
    let mut grades = Grades::new();
    for (s, gd) in &model.grading {
        grades.insert(s.to_owned(), gd.clone().into());
    }
    // The gader's initial state
    let initial_grader = grades.clone();

    let itr = ssitr(&model.ex, |a, b| a.0.cmp(b.0));
    for (id, e) in itr {
        let e_grade = note_exercise(id, e, answers, initial_grader.clone())?;
        // Merge the grades
        for (k, v) in grades.iter_mut() {
            if let Some(w) = e_grade.get(k) {
                *v = (v.clone() + w.clone())?;
            }
        }
    }
    for (_, v) in grades.iter_mut() {
        *v = v.clone().clamp(model.min, model.max);
    }
    Some(grades)
}

/*
 * Meta functions *
 they describe whole processes.
*/

/// Reads an image, and resolves the boxes according to the pointed content.
fn resolve(image_file: &String, content: &Content_<f64>) -> Result<(HashMap<String, Answer>, u8)> {
    // Get the image
    let image = read_image(image_file)?;

    let res = standard_size(&image, &content.metadata.size.cast())?;
    // Edge detection (find the little guys)
    /*let mut blurred = Mat::default();
    imgproc::gaussian_blur(&img, &mut blurred, core::Size::new(7, 7), 0., 0., core::BORDER_DEFAULT)?;
    imgcodecs::imwrite(&("cpu_blur-".to_owned() + &img_file), &blurred, &v)?;

    //Reveal edges
    let mut edges = Mat::default();
    imgproc::canny(&blurred, &mut edges, 25., 50., 3, false)?;
    imgcodecs::imwrite(&("cpu_edges-".to_owned() + &img_file), &edges, &v)?;

    // Find contours
    let mut contours: Vector<Vector<core::Point>> = core::Vector::new();
    //let mut hierarchy = core::Vector::new();
    imgproc::find_contours(&edges, &mut contours, imgproc::RETR_TREE, imgproc::CHAIN_APPROX_TC89_L1, core::Point::default())?;

    // Draw contours
    let mut dr= Mat::zeros_size(img.size()?, core::CV_8UC3)?.to_mat()?;
    imgproc::draw_contours(&mut dr, &contours, -1, core::Scalar::new(255., 255., 255., 0.), 1, imgproc::LINE_8, &core::no_array(), i32::MAX, core::Point::default())?;
    imgcodecs::imwrite(&("cpu_contours-".to_owned() + &img_file), &dr, &v)?;*/

    // Detect QR:
    //todo! threshold pour eviter d'avoir des pbs de transparance (eheh trans-parance)
    let qr = check_qr(&image, res, &content.qr_code.cast(), &content.metadata)?;

    let markers = detect_markers(&image, &qr, &content.qr_code.cast(), &content.pointers.cast().rescale(res))?;

    /*
    let size = Size_ { width: dotsize, height: dotsize };
    let size: Size_<i32> = size.rescale(res).cast();
    let size: Size_<i32> = Size_ { width: size.width + 2, height: size.width + 2 };
    println!("{:?}", size.width);

    //imgproc::rectangle(&mut qr, qr_pos.rect.rescale(res).cast(), core::Scalar::from_array([0.0, 255.0, 255.0, 255.0]), 1, imgproc::LINE_8, 0)?;
    //imgproc::circle(&mut qr, qr_pos.rect.tl().rescale(res).cast(), (qr_pos.rect.width * res) as i32, core::Scalar::from_array([0.0, 0.0, 0.0, 0.0]), 1, imgproc::LINE_4, 0)?;
    //imgcodecs::imwrite("HOMO-SEX.jpg", &qr, &v)?;

    // Create synthetic fiducial using the scale from the qr code as an indicator
    let mut template_marker = Mat::new_size_with_default(size, core::CV_8UC1, core::Scalar::new(255., 0., 0., 0.))?;
    //println!("{:?}", template_marker);
    //draw circle
    let pos = Point_ { x: dotsize/2., y: dotsize/2. };
    let pos: Point_<i32> = pos.rescale(res).cast();
    let pos: Point_<i32> = Point_ { x: pos.x + 1, y: pos.y + 1 };
    //imgproc::circle(&mut template_marker, pos, marker_radius as i32, core::Scalar::new(0., 0., 0., 0.), 1, imgproc::LINE_8, 0)?;
    imgproc::circle(&mut template_marker, pos, marker_radius as i32, core::Scalar::new(0., 0., 0., 0.), imgproc::FILLED, imgproc::FILLED, 0)?;
    imgcodecs::imwrite("T-GAY-SEX.jpg", &template_marker, &v)?;
    // 👍

    // Run the search; and image correction
    let mut markersdct = Mat::new_size_with_default(Size_ { width: resol.width - size.width + 1, height: resol.height - size.height + 1 }, core::CV_32FC1, core::Scalar::default())?;
    imgproc::match_template(&img, &template_marker, &mut markersdct, imgproc::TM_SQDIFF, &core::no_array())?;
    //let mut normalker = Mat::default();
    //core::normalize(&markersdct, &mut normalker, 0., 1., core::NORM_MINMAX, core::CV_32FC1, &core::no_array())?;
    //println!("{:?}", normalker);
    let mut marker_disp = Mat::default();
    core::normalize(&markersdct, &mut marker_disp, 0., 255., core::NORM_MINMAX, core::CV_8UC1, &core::no_array())?;
    */
    // 👍👍

    // 👍👍👍

    let resized = fix_image(&image, &content.pointers.cast().rescale(res), &markers)?;

    Ok((resolve_boxes(&resized, &content.questions, qr.1.page, res)?, qr.1.page))
}

/// Notes the resolved content according to the model
fn grade<T>(answers: &HashMap<String, Answer>, model: &model::Model, content_meta: &Metadata_<T>) -> Result<Grades> {
    if model.md.id != content_meta.id {
        error!("Inconsitent exam id between the model and the position files. Are you sure you have the right files?");
        return Err(opencv::Error::new(core::StsBadArg, "Inconsistent exam id."));
    }
    if model.md.hash != content_meta.hash {
        error!("Inconsitent exam hash between the model and the position files. Are you sure your files are from the same compilation?");
        warn!("This error can be recovered from manually! Try to cross-reference your JSON files!");
        return Err(opencv::Error::new(core::StsBadArg, "Inconsistent exam hash."));
    }

    note_exam(model, answers)
        .ok_or(opencv::Error::new(core::StsParseError, "Answer parsing failed."))
}

fn main() -> Result<()> {
    // Activate the logger (the default lever should not be higher than warn, as warnings should not be disregarded)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let args = cli::Cli::parse();

    // Start opencl
    boot_cl()?;

    let start = time::Instant::now();

    let ct: Content_<f64> = data::read_content(PathBuf::from(args.positions))?;

    // Run resolve for each image, accumulate the answers.
    let mut answers = HashMap::new();
    let mut i = 0;
    let mut page_counter = vec![];
    for image in args.images {
        info!("Processing image #{}", i);
        let (ans, p) = resolve(&image, &ct)?;
        answers.extend(ans);
        page_counter.push(p);
        i += 1;
    }
    info!("Processing finished, now grading.");
    // Check that we have all the pages
    page_counter.sort();
    let orig_ct = page_counter.len();
    page_counter.dedup();
    // These two are errors, but since they are *Technically* recoverable, we don't fail. should we?
    if page_counter.len() < orig_ct {
        error!("Some of the resoloved images were duplicated! This could mean you have mixed up two different copies!");
    }
    if page_counter.len() < ct.metadata.pages as usize {
        error!("It seems you have too few pages! This means that grading will not output the right result, if not outright fail!");
    }
    if page_counter.len() > ct.metadata.pages as usize {
        warn!("It seems you have too many pages?? Don't worry, they will be ignored during the grading process, but you should look into that!");
    }

    let model = data::read_model(PathBuf::from(args.model))?;

    let note = grade(&answers, &model, &ct.metadata)?;

    info!("Done: {:#?}", start.elapsed());

    for (k, v) in note {
        print!("{}: {}\n", k, v);
    }

    Ok(())
}
