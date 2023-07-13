use std::{time, f32::consts::PI, path::PathBuf};

use clap::Parser;
use log::{debug, error, log_enabled, info, trace, warn, Level};

use opencv::{
	core::{self, CV_8UC1},
	imgcodecs, imgproc,
	prelude::*,
	types, Result,
    features2d::{SimpleBlobDetector, SimpleBlobDetector_Params},
};
opencv::opencv_branch_4! {
	use opencv::core::AccessFlag::ACCESS_READ;
}
opencv::not_opencv_branch_4! {
	use opencv::core::ACCESS_READ;
}

mod cli;
mod utils;
mod data;
mod status;
use utils::*;

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
			"\x1b[32mavailable\x1b[39m"
		} else {
			"\x1b[31mnot available\x1b[39m"
		},
		if opencl_use {
			"\x1b[32menabled\x1b[39m"
		} else {
			"\x1b[31mdisabled\x1b[39m"
		},
	);

    match opencl_have && opencl_use  {
        false => Err(opencv::Error::new(core::StsInternal, "Empty image from reading file.")),
        true => Ok(())
    }
}

/// Write an image only useful when tracing
fn image_trace(img: &Mat, name: &str) -> Result<bool> {
    const TRACE_FOLDER: &str = "trace/";

    if log_enabled!(Level::Trace) {
        trace!("Written Image trace: {}", name);
        imgcodecs::imwrite(&(TRACE_FOLDER.to_owned() + name), img, &core::Vector::default())
    } else {
        Ok(false)
    }
}

fn disp_hist(hist: &Mat, maxh: i32, barw: i32) -> Result<()> {
    let imgs = Size_ {width: barw * 256, height: maxh};
    let mut img = Mat::new_size_with_default(imgs, CV_8UC1, core::Scalar::default())?;

    let mut max_val = 0.;
    core::min_max_loc(&hist, None, Some(&mut max_val), None, None, &core::no_array())?;
    max_val = max_val.ln();

    for i in 0..256 {
        imgproc::rectangle(&mut img, Rect_ { x: i * barw, y: 0, width: barw, height: ((maxh as f32) * (hist.at::<f32>(i)?.ln() / max_val as f32)) as i32 }, core::Scalar::new(255., 0., 0., 0.), 1, imgproc::LINE_8, 0)?;
    }
    
    imgcodecs::imwrite("YAAAAA.jpg", &img, &core::Vector::new())?;
    Ok(())
}

/// Reads an image from a file and applies a slight filter cleanup
fn read_image(path: &str) -> Result<Mat> {
    // Get the image
	let img = imgcodecs::imread(&path, imgcodecs::IMREAD_GRAYSCALE)?; //Open & convert to grayscale
    debug!("Image: {:?}", img);
    if img.empty() {
        error!("Empty image (searched at {})! This is likely because reading failed (ie: there was no file to read).", path);
        return Err(opencv::Error::new(core::BadImageSize, "Empty image from reading file."));
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

fn main() -> Result<()> {

    // Activate the logger (the default lever should not be higher than warn, as warnings should not be disregarded)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

	let args = cli::Cli::parse();

    // Start opencl
	boot_cl()?;

    let start = time::Instant::now();

    let v: core::Vector<i32> = core::Vector::new();

    let ct = data::read(PathBuf::from(args.positions))?;

    // Get the image
	let image = read_image(&args.image)?;

    let resol = image.size()?;

    let res = resol.cast().as_scale(&ct.metadata.size)? as f32; //scale(&img.size()?.cast(), &IMG_SIZE);
    //let im_size = actual_size(RESOLUTION_DEFAULT, img.size()?.cast().cast());
    println!("res: {}", res);
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
    let (qr_pos, _) = detect_qr(&image)?;
    //let mut qr = Mat::copy(&img)?;
    //imgproc::circle(&mut qr, qr_pos[0].cast(), 2, core::Scalar::from_array([0.0, 0.0, 0.0, 0.0]), 1, imgproc::LINE_4, 0)?;
    //imgproc::circle(&mut qr, qr_pos[1].cast(), 2, core::Scalar::from_array([0.0, 0.0, 0.0, 0.0]), 1, imgproc::LINE_4, 0)?;
    //imgproc::circle(&mut qr, qr_pos[2].cast(), 2, core::Scalar::from_array([0.0, 0.0, 0.0, 0.0]), 1, imgproc::LINE_4, 0)?;
    //imgproc::circle(&mut qr, qr_pos[3].cast(), 2, core::Scalar::from_array([0.0, 0.0, 0.0, 0.0]), 1, imgproc::LINE_4, 0)?;
    //println!("{:?}", qr_pos);

    let qr_pos: Vec<Point_<f32>> = qr_pos.into_iter()
        .map(|v| v.rescale(1. / res))
        .collect();


    let qr_pos = OriRect2D::try_from(qr_pos).unwrap(); // note: point error?
    println!("{:?}", qr_pos);
    let sz = qr_pos.rect.as_scale(&ct.qr_code.cast()).unwrap(); //todo hande the error better
    assert!(sz < 1.1 && 0.8 < sz, "Detected QR Code is too small or too big.");

    let pt: Pointers<f32> = ct.pointers.cast();
    let dotsize: f32 = sz * pt.diameter; //todo add a pixel or two of border probsly.. and AA at the circle border
    let marker_radius = dotsize / 2. * res;
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
    // Calculate a hist to see how everythin is distributed
    //let mut himg: core::Vector<Mat> = core::Vector::new();
    //himg.push(marker_disp.clone());
    //let mut hist = Mat::new_nd_with_default(&[256], core::CV_32FC1, core::Scalar::default())?;
    //imgproc::calc_hist(&himg, &core::Vector::from(vec![0]), &core::no_array(), &mut hist, &core::Vector::from(vec![256]), &core::Vector::new(), false)?;
    //disp_hist(&hist, 200, 2)?;

    //imgcodecs::imwrite("GAY-SEX.jpg", &marker_disp, &v)?;
    // 👍👍

    // Now we find the 3 lowest blobs!
    let size_mul = (0.9 * sz, 1.1 * sz);
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
    //Do a dichotomic search!!
    //let border = Point_::new(size.width/2, size.height/2).cast();
    let mut kp = Vec::new();
    for k in keyp { 
        let s = k.pt();
        println!("{:?} {:?}", s, k.size());
        kp.push((s, k.size()));
    }
    // We hopefully have our points, now fit them!
    // We have to force it as an int, because f32 is not ord!
    // We rescale it because the cast may make us loose too much information
    let actual_form = ct.pointers.cast().rescale(res).as_computed::<i64>(&kp);
    println!("{:?}", actual_form);
    // 👍👍👍

    /*println!("\x1b[32m{:?} {:?}", IMG_FORM_DEFAULT.short.rescale(res), POINTER_DIAMETER);
    println!("\x1b[33m{:?} {:?}", kp[0].0, kp[0].1 / res);
    println!("\x1b[32m{:?} {:?}", IMG_FORM_DEFAULT.master.rescale(res), POINTER_DIAMETER);
    println!("\x1b[33m{:?} {:?}", kp[1].0, kp[1].1 / res);
    println!("\x1b[32m{:?} {:?}", IMG_FORM_DEFAULT.long.rescale(res), POINTER_DIAMETER);
    println!("\x1b[33m{:?} {:?}\x1b[39m", kp[2].0, kp[2].1 / res);*/

    // Calculate the fixed image's resolution
    //let resol: Size_<i32> = img.size()?; //match_resolution(IMG_FORM_DEFAULT, IMG_FORM_ACTUAL, img.size()?.cast()).cast(); //if we want to avoid rescales, we need to apply an inverse scale transform on b
    //println!("{:?}", resol);

    // Create a transform map to go from original image to fixed image
    let a: [Point_<f32>; 3] = ct.pointers.cast().rescale(res).into();
    let b: [Point_<f32>; 3] = actual_form.into();
    //println!("{:?} {:?}", a, b);
    let scalemap = imgproc::get_affine_transform_slice(b.as_slice(), a.as_slice())?;
    //println!("{:?}", scalemap);

    // Fix the image & write it
    let mut resized = Mat::default();
    imgproc::warp_affine(&image, &mut resized, &scalemap, resol, imgproc::INTER_LINEAR, core::BORDER_CONSTANT, core::Scalar::default())?;
    imgcodecs::imwrite("FIXED.jpg", &resized, &v)?;
    // Crop the desired answer from the fixed image
    //let r = resolve(BOX, IMG_SIZE, resol);
    //let crop = Mat::roi(&resized, r)?;
    //let mut crop = Mat::copy(&resized)?;
    //imgproc::rectangle(&mut crop, r, core::Scalar::from_array([0.0, 255.0, 255.0, 0.0]), 5, imgproc::LINE_8, 0)?;
    //imgcodecs::imwrite(&("cpu_crop_fixed-".to_owned() + &img_file), &crop, &v)?;

    /*println!("Timing CPU implementation... ");
	let start = time::Instant::now();
	for _ in 0..ITERATIONS {
		let mut gray = Mat::default();
		imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
		let mut blurred = Mat::default();
		imgproc::gaussian_blur(&gray, &mut blurred, core::Size::new(7, 7), 1.5, 0., core::BORDER_DEFAULT)?;
		let mut edges = Mat::default();
		imgproc::canny(&blurred, &mut edges, 0., 50., 3, false)?;
	}
	println!("{:#?}", start.elapsed());
	if opencl_use {
		println!("Timing OpenCL implementation... ");
		let mat = imgcodecs::imread(&img_file, imgcodecs::IMREAD_COLOR)?;
		let img = mat.get_umat(ACCESS_READ, UMatUsageFlags::USAGE_DEFAULT)?;
		let start = time::Instant::now();
		for _ in 0..ITERATIONS {
			let mut gray = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
			imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
			let mut blurred = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
			imgproc::gaussian_blur(&gray, &mut blurred, core::Size::new(7, 7), 1.5, 0., core::BORDER_DEFAULT)?;
			let mut edges = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
			imgproc::canny(&blurred, &mut edges, 0., 50., 3, false)?;
		}
		println!("{:#?}", start.elapsed());
	}*/
    println!("done! {:#?}", start.elapsed());
	Ok(())
}