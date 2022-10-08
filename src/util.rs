use rayon::prelude::*;

pub struct MeshIterator<const N: usize> {
    bounds: [usize; N],
}

impl<const N: usize> MeshIterator<N> {
    pub fn new(bounds: [usize; N]) -> Self {
        Self { bounds }
    }

    fn multiindex(i: usize, incs: &[usize; N], bounds: &[usize; N]) -> [usize; N] {
        let mut g = [i; N];
        g.iter_mut()
            .zip(incs.iter().copied().zip(bounds.iter().copied()))
            .for_each(|(x, (inc, bound))| {
                *x /= inc;
                *x %= bound;
            });
        g
    }

    pub fn build_iterator(&self) -> impl Iterator<Item = [usize; N]> {
        let prod_cumulative = self.build_index_incs();
        let bounds = self.bounds;

        let p = bounds.iter().product();
        (0..p).map(move |i| Self::multiindex(i, &prod_cumulative, &bounds))
    }

    pub fn build_parallel_iterator(&self) -> impl ParallelIterator<Item = [usize; N]> {
        let prod_cumulative = self.build_index_incs();
        let bounds = self.bounds;

        let p = bounds.iter().product();
        (0..p)
            .into_par_iter()
            .map(move |i| Self::multiindex(i, &prod_cumulative, &bounds))
    }

    fn build_index_incs(&self) -> [usize; N] {
        let mut prod_cumulative = [1usize; N];
        for (i, x) in self.bounds.iter().copied().enumerate() {
            prod_cumulative[..i].iter_mut().for_each(|v| {
                *v *= x;
            });
        }
        prod_cumulative
    }
}

#[cfg(test)]
mod util_tests {
    use crate::util::MeshIterator;

    #[test]
    fn test_prod_cumulative() {
        let (x, y, z) = (2, 3, 5);
        let m = MeshIterator::new([x, y, z]);
        let mut it = m.build_iterator();
        for ix in 0..x {
            for iy in 0..y {
                for iz in 0..z {
                    assert_eq!(it.next(), Some([ix, iy, iz]));
                }
            }
        }
    }
}
