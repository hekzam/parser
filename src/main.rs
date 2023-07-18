use std::{f32::consts::PI, fmt::Write, path::PathBuf, time, collections::HashMap};

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
    const TRACE_FOLDER: &str = "trace/";
    const TRACE_EXTENSION: &str = ".jpg";

    if log_enabled!(Level::Trace) {
        let fname = TRACE_FOLDER.to_owned() + name + TRACE_EXTENSION;
        trace!("Writing image trace: \"{}\"", fname);
        imgcodecs::imwrite(&fname, img, &core::Vector::default())
    } else {
        Ok(false)
    }
}

/*
 * Display Functions *
 just pretty stuff that can often be useful
*/
/// Display a histogram as a bar chart
fn disp_hist(hist: &Mat, maxh: i32, barw: i32) -> Result<()> {
    let imgs = Size_ {
        width: barw * 256,
        height: maxh,
    };
    let mut img = Mat::new_size_with_default(imgs, CV_8UC1, core::Scalar::default())?;

    let mut max_val = 0.;
    core::min_max_loc(
        &hist,
        None,
        Some(&mut max_val),
        None,
        None,
        &core::no_array(),
    )?;
    max_val = max_val.ln();

    for i in 0..256 {
        imgproc::rectangle(
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
        )?;
    }

    imgcodecs::imwrite("YAAAAA.jpg", &img, &core::Vector::new())?;
    Ok(())
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

    write!(&mut r,
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
        error!("Transform matrix has wrong size! Should contain 6 elements, has {}", values.len());
        return Err(opencv::Error::new(
            core::StsBadSize,
            "Wrong size for transform matrix.",
        ));
    }

    write!(&mut r, "[{:.3} {:.3} {:.3} | {:.3} {:.3} {:.3}]", values[0], values[1], values[2], values[3], values[4], values[5]).map_err(f)?;

    Ok(r)
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
    debug!("Image: {}", image_description_string(&img)?);
    if img.empty() {
        error!("Empty image (searched at {})! This is likely because reading failed (ie: there was no file to read).", path);
        return Err(opencv::Error::new(
            core::BadImageSize,
            "Empty image from reading file.",
        ));
    }
    let resol: Size_<i32> = img.size()?;

    //todo rescale image if too big?  (>600 dpi, probs) (to make the next operations less costly!)
    if resol.width.max(resol.height) > 5000 {
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

    debug!("Resolution: {} px/cm", res);

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
    debug!("QR Position{:?}", qr_pos);
    let sz = qr_pos.rect.as_scale(&standard_qr.cast()).unwrap(); //todo hande the error better
    if sz > LEN_DX || sz < 1. / LEN_DX {
        warn!(
            "Detected QR Code is too {}.",
            if sz > LEN_DX { "big" } else { "small" }
        );
    }

    debug!("QR Data: {}", qr_description_string(&qr.1)?);
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
        trace!("QR Code contains the expected hash");
    }
    if meta.id != qr.1.id {
        error!(
            "Detected QR Code has exam ID {}, expected {}",
            qr.1.id, meta.id
        );
        r = Err(opencv::Error::new(core::StsBadArg, "Invalid QR Code data."));
    } else {
        trace!("QR Code contains the expected ID");
    }
    if meta.pages < qr.1.page || qr.1.page == 0 {
        error!(
            "Detected QR Code has page number {} which is not in [1;{}]",
            qr.1.page, meta.pages
        );
        r = Err(opencv::Error::new(core::StsBadArg, "Invalid QR Code data."));
    } else {
        trace!("QR Code contains a valid page number");
    }

    r
}

/// Detects markers within an image
fn detect_markers(
    image: &Mat,
    qr_sz: f32,
    standard_markers: &Pointers<f32>,
) -> Result<Pointers<f32>> {
    const LEN_DX: f32 = ShapeError::LENGTH_THRESHOLD as f32;

    // Either lower the size range (its 20% right now!), or we dont need sz :)
    let dotsize: f32 = qr_sz * standard_markers.diameter;
    let marker_radius = dotsize / 2.;

    // Now we find the 3 lowest blobs!
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
        min_circularity: 0.85, // Filter for circular markers
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
        trace!("point(@{}:{}, ø{})", s.x.round(), s.y.round(), k.size());
        kp.push((s, k.size()));
    }
    // We hopefully have our points, now fit them!
    // We have to force it as an int, because f32 is not ord!
    // We rescale it because the cast may make us loose too much information
    let actual_form = standard_markers.as_computed::<i64>(&kp);
    debug!("standard: {}", markers_description_string(&standard_markers)?);
    debug!("actual:   {}", markers_description_string(&actual_form)?);
    Ok(actual_form)
}

/// Fixes an image (applies transforms to ensure a standardized position)
fn fix_image(image: &Mat, standard_markers: &Pointers<f32>, actual_markers: &Pointers<f32>) -> Result<Mat> {
    let f = |_| opencv::Error::new(opencv::core::StsParseError, "Scale error!");
    // Calculate the fixed image's resolution
    //if we want to avoid rescales, we need to apply an inverse scale transform on b
    let size_a = Size_::try_from(standard_markers.to_owned()).map_err(f)?;
    let size_b = Size_::try_from(actual_markers.to_owned()).map_err(f)?;
    let new_scale: f32 = size_a.as_scale(&size_b)?;
    let new_res: Size_<i32> = image.size()?.cast().rescale(new_scale).cast();
    // Now that we have the new res, we do the inverse scale transform (which transforms our points as to ensure the resize element of the transform is 1)!
    //todo
    debug!("Rescale by a factor of: {}", new_scale);

    // Create a transform map to go from original image to fixed image
    let a: [Point_<f32>; 3] = standard_markers.into();
    let b: [Point_<f32>; 3] = actual_markers.into();
    //println!("{:?} {:?}", a, b);
    let scalemap = imgproc::get_affine_transform_slice(b.as_slice(), a.as_slice())?;
    trace!("transform matrix: {}", transform_description(&scalemap)?);

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

fn resolve_boxes<T: Into<i32>>(image: &Mat, boxes: HashMap<String, Question_<T>>) -> Result<HashMap<String, Answer>> {

    for b in boxes {
        
    }

    todo!()
}

fn main() -> Result<()> {
    // Activate the logger (the default lever should not be higher than warn, as warnings should not be disregarded)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let args = cli::Cli::parse();

    // Start opencl
    boot_cl()?;

    let start = time::Instant::now();

    let ct = data::read(PathBuf::from(args.positions))?;

    // Get the image
    let image = read_image(&args.image)?;

    let res = standard_size(&image, &ct.metadata.size.cast())?;
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
    let qr = check_qr(&image, res, &ct.qr_code.cast(), &ct.metadata)?;

    // The size differrence between expected and actual sizes
    let sz = qr.0.rect.as_scale(&ct.qr_code.cast()).unwrap(); //todo hande the error better

    let markers = detect_markers(&image, sz, &ct.pointers.cast().rescale(res))?;

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
    // Calculate a hist to see how everythin' is distributed
    //let mut himg: core::Vector<Mat> = core::Vector::new();
    //himg.push(marker_disp.clone());
    //let mut hist = Mat::new_nd_with_default(&[256], core::CV_32FC1, core::Scalar::default())?;
    //imgproc::calc_hist(&himg, &core::Vector::from(vec![0]), &core::no_array(), &mut hist, &core::Vector::from(vec![256]), &core::Vector::new(), false)?;
    //disp_hist(&hist, 200, 2)?;

    //imgcodecs::imwrite("GAY-SEX.jpg", &marker_disp, &v)?;
    // 👍👍

    // 👍👍👍

    let resized = fix_image(&image, &ct.pointers.cast().rescale(res), &markers)?;
    
    // Crop the desired answer from the fixed image
    //let r = resolve(BOX, IMG_SIZE, resol);
    //let crop = Mat::roi(&resized, r)?;
    //let mut crop = Mat::copy(&resized)?;
    //imgproc::rectangle(&mut crop, r, core::Scalar::from_array([0.0, 255.0, 255.0, 0.0]), 5, imgproc::LINE_8, 0)?;
    //imgcodecs::imwrite(&("cpu_crop_fixed-".to_owned() + &img_file), &crop, &v)?;

    info!("Done: {:#?}", start.elapsed());
    Ok(())
}
