extern crate core;
#[cfg(feature = "gpu-cuda")]
pub mod cudagraph;
pub mod dualgraph;
#[cfg(feature = "gpu-wgpu")]
pub mod gpugraph;
mod util;

#[cfg(feature = "gpu-cuda")]
pub use cudagraph::*;
#[cfg(feature = "gpu-cuda")]
pub use cudarc::*;
pub use dualgraph::*;
#[cfg(feature = "gpu-wgpu")]
pub use gpugraph::*;
pub use ndarray_rand::rand;
