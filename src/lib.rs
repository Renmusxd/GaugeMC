pub mod dualgraph;
#[cfg(feature = "gpu-wgpu")]
pub mod gpugraph;

pub use dualgraph::*;
#[cfg(feature = "gpu-wgpu")]
pub use gpugraph::*;
pub use ndarray_rand::rand;
