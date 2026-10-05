extern crate core;

pub use ndarray;

#[cfg(feature = "gpu-cuda")]
pub mod cudagraph;
pub mod dualgraph;
mod util;

#[cfg(feature = "gpu-cuda")]
pub use cudagraph::*;
#[cfg(feature = "gpu-cuda")]
pub use cudarc::*;
pub use dualgraph::*;
pub use ndarray_rand::rand;
