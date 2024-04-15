This branch contains a prototype of a exam copy parser in Rust.
The Rust variant of this code is **unmaintained**, as the main development has moved to C++ in the `main` branch.

Rationale for moving to C++
===========================
The main dependency of this code is OpenCV.

Unfortunately, as I write these lines (2024-04-15), OpenCV's rust binding is risky to use (experimental, "use at your own risk") and annoying to package (their cargo setup calls llvm to generate the binding).

Additionnally, Rust benefits for this code are not obvious, as our code is very small and simple compared to OpenCV's.

Moving to C++ to reduce technical debt and for simplicity's sake (no binding maintenance overhead, no packaging shenanigans).
