extern crate core;
#[cfg(feature = "gpu-cuda")]
pub mod cudagraph;
pub mod dualgraph;
#[cfg(feature = "gpu-wgpu")]
pub mod gpugraph;
mod util;

pub use dualgraph::*;
#[cfg(feature = "gpu-wgpu")]
pub use gpugraph::*;
pub use ndarray_rand::rand;
