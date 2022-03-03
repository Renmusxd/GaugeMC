use ndarray::{Array4, Array5};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::{rand, RandomExt};
use num_traits::float::Float;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Default, Clone, Eq, PartialEq, Debug, Hash)]
pub struct SiteIndex {
    pub t: usize,
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl SiteIndex {
    pub fn get(&self, dim: Dimension) -> usize {
        match dim {
            Dimension::T => self.t,
            Dimension::X => self.x,
            Dimension::Y => self.y,
            Dimension::Z => self.z,
        }
    }

    pub fn set(&mut self, dim: Dimension, val: usize) {
        match dim {
            Dimension::T => self.t = val,
            Dimension::X => self.x = val,
            Dimension::Y => self.y = val,
            Dimension::Z => self.z = val,
        }
    }

    pub fn modify(&mut self, dim: UnitDirection, delta: usize, bounds: &Self) {
        let (edit, bound) = match dim {
            (_, Dimension::T) => (&mut self.t, bounds.t),
            (_, Dimension::X) => (&mut self.x, bounds.x),
            (_, Dimension::Y) => (&mut self.y, bounds.y),
            (_, Dimension::Z) => (&mut self.z, bounds.z),
        };
        match dim {
            (Sign::Positive, _) => {
                *edit += delta;
                *edit %= bound;
            }
            (Sign::Negative, _) => {
                let delta = bound - (delta % bound);
                *edit += delta;
                *edit %= bound;
            }
        }
    }

    pub fn site_in_dir(&self, dim: UnitDirection, delta: usize, bounds: &Self) -> Self {
        let mut other = self.clone();
        other.modify(dim, delta, bounds);
        other
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub enum Dimension {
    T,
    X,
    Y,
    Z,
}

impl PartialOrd for Dimension {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (a, b) if a == b => Some(Ordering::Equal),
            (Self::T, _) => Some(Ordering::Less),
            (_, Self::T) => Some(Ordering::Greater),
            (Self::X, _) => Some(Ordering::Less),
            (_, Self::X) => Some(Ordering::Greater),
            (Self::Y, _) => Some(Ordering::Less),
            (_, Self::Y) => Some(Ordering::Greater),
            // Since type system doesn't know how equality works, wants this one.
            (_, _) => Some(Ordering::Equal),
        }
    }
}

impl Dimension {
    pub fn in_order(self, other: Self) -> bool {
        match (self, other) {
            (Self::T, _) => true,
            (_, Self::T) => false,
            (Self::X, _) => true,
            (_, Self::X) => false,
            (Self::Y, _) => true,
            (_, Self::Y) => false,
            (Self::Z, Self::Z) => true,
        }
    }
}

impl From<usize> for Dimension {
    fn from(dim: usize) -> Self {
        match dim {
            0 => Self::T,
            1 => Self::X,
            2 => Self::Y,
            3 => Self::Z,
            _ => panic!("Not a valid dimension."),
        }
    }
}

impl From<Dimension> for usize {
    fn from(d: Dimension) -> Self {
        match d {
            Dimension::T => 0,
            Dimension::X => 1,
            Dimension::Y => 2,
            Dimension::Z => 3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Sign {
    Positive,
    Negative,
}

impl Sign {
    pub fn flip(self) -> Self {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
        }
    }
}

type UnitDirection = (Sign, Dimension);

pub struct NDDualGraph {
    bounds: SiteIndex,
    np: Array5<i32>,
    vn: Vec<f64>,
    currents: HashMap<(SiteIndex, Dimension), i32>,
}

impl NDDualGraph {
    pub fn new<V>(t: usize, x: usize, y: usize, z: usize, vnp: V) -> Self
    where
        V: IntoIterator<Item = f64>,
    {
        Self {
            bounds: SiteIndex { t, x, y, z },
            np: Array5::zeros((t, x, y, z, 6)),
            vn: vnp.into_iter().collect(),
            currents: Default::default(),
        }
    }

    /// Add flux to a plaquette and place current along boundaries.
    pub fn add_flux(&mut self, site: SiteIndex, p: usize, delta: i32) {
        if delta != 0 {
            let (first, second) = Self::plaquette_subindex_to_dim_dim(p);
            self.np[(site.t, site.x, site.y, site.z, p)] += delta;

            let origin = self.currents.entry((site.clone(), first)).or_default();
            *origin -= delta;
            if *origin == 0 {
                self.currents.remove(&(site.clone(), first));
            }

            let fsite = site.site_in_dir((Sign::Positive, first), 1, &self.bounds);
            let second_entry = self.currents.entry((fsite.clone(), second)).or_default();
            *second_entry -= delta;
            if *second_entry == 0 {
                self.currents.remove(&(fsite, second));
            }

            let ssite = site.site_in_dir((Sign::Positive, second), 1, &self.bounds);
            let third_entry = self.currents.entry((ssite.clone(), first)).or_default();
            *third_entry += delta;
            if *third_entry == 0 {
                self.currents.remove(&(ssite, first));
            }

            let fourth_entry = self.currents.entry((site.clone(), second)).or_default();
            *fourth_entry += delta;
            if *fourth_entry == 0 {
                self.currents.remove(&(site, second));
            }

            debug_assert_eq!(self.get_edges_with_violations(), vec![]);
        }
    }

    pub fn plaquettes_next_to_edge(
        site: &SiteIndex,
        d: Dimension,
        bounds: &SiteIndex,
    ) -> ([(SiteIndex, usize); 3], [(SiteIndex, usize); 3]) {
        let mut poss: [(SiteIndex, usize); 3] = Default::default();
        let mut negs = poss.clone();

        (0..4usize)
            .map(Dimension::from)
            .filter(|ad| *ad != d)
            .zip(poss.iter_mut().zip(negs.iter_mut()))
            .for_each(|(ad, ((ps, pp), (ns, np)))| {
                let p = Self::dim_dim_to_plaquette_subindex(d, ad);
                *pp = p;
                *np = p;
                let other_site = site.site_in_dir((Sign::Negative, ad), 1, bounds);
                if d.in_order(ad) {
                    *ps = site.clone();
                    *ns = other_site;
                } else {
                    *ps = other_site;
                    *ns = site.clone();
                }
            });
        (poss, negs)
    }

    pub fn get_edges_with_violations(&self) -> Vec<(SiteIndex, Dimension)> {
        let edge_iterator = (0..self.bounds.t).flat_map(|t| {
            (0..self.bounds.x).flat_map(move |x| {
                (0..self.bounds.y).flat_map(move |y| {
                    (0..self.bounds.z).flat_map(move |z| {
                        (0..4usize)
                            .map(Dimension::from)
                            .map(move |d| (SiteIndex { t, x, y, z }, d))
                    })
                })
            })
        });
        edge_iterator
            .par_bridge()
            .filter(|(s, d)| {
                let (poss, negs) = Self::plaquettes_next_to_edge(s, *d, &self.bounds);
                let sum = poss
                    .iter()
                    .cloned()
                    .map(|p| (p, 1))
                    .chain(negs.iter().cloned().map(|n| (n, -1)))
                    .map(|((site, p), mult)| self.np[(site.t, site.x, site.y, site.z, p)] * mult)
                    .sum::<i32>();
                let key = (s.clone(), *d);
                let current = self.currents.get(&key).cloned().unwrap_or(0);
                let sum = sum + current;
                sum != 0
            })
            .collect()
    }

    fn dim_dim_to_plaquette_subindex(first: Dimension, second: Dimension) -> usize {
        match (first, second) {
            (Dimension::T, Dimension::X) => 0,
            (Dimension::T, Dimension::Y) => 1,
            (Dimension::T, Dimension::Z) => 2,
            (Dimension::X, Dimension::Y) => 3,
            (Dimension::X, Dimension::Z) => 4,
            (Dimension::Y, Dimension::Z) => 5,
            (first, second) if first != second => {
                Self::dim_dim_to_plaquette_subindex(second, first)
            }
            (_, _) => panic!("Dimensions may not be equal, no plaquette defined."),
        }
    }

    fn plaquette_subindex_to_dim_dim(p: usize) -> (Dimension, Dimension) {
        match p {
            0 => (Dimension::T, Dimension::X),
            1 => (Dimension::T, Dimension::Y),
            2 => (Dimension::T, Dimension::Z),
            3 => (Dimension::X, Dimension::Y),
            4 => (Dimension::X, Dimension::Z),
            5 => (Dimension::Y, Dimension::Z),
            _ => panic!("Invalid plaquette subindex"),
        }
    }

    fn get_np_indices_for_cube(
        cube_index: [usize; 4],
        dims: &[Dimension; 3],
        leftover: Dimension,
        off: usize,
        bounds: &SiteIndex,
    ) -> ([(SiteIndex, usize); 3], [(SiteIndex, usize); 3]) {
        let [rho, mu, nu, sigma] = cube_index;
        // Convert mu-rho to t-z
        let mut site = SiteIndex::default();
        site.set(leftover, rho);
        site.set(dims[0], mu);
        site.set(dims[1], nu);
        let parity = (mu + nu + off) % 2;
        site.set(dims[2], sigma * 2 + parity);

        // (t,x,y,z) corresponds to the "lowest" ordered corner.
        // Get the plaquette indices and lookup values.
        let ps = [
            NDDualGraph::dim_dim_to_plaquette_subindex(dims[1], dims[2]),
            NDDualGraph::dim_dim_to_plaquette_subindex(dims[0], dims[2]),
            NDDualGraph::dim_dim_to_plaquette_subindex(dims[0], dims[1]),
        ];
        let mut poss = [
            (site.clone(), ps[0]),
            (site.clone(), ps[1]),
            (site.clone(), ps[2]),
        ];
        let mut negs = poss.clone();
        negs.iter_mut()
            .zip(dims.iter().cloned().enumerate())
            .for_each(|(entry, (i, d))| {
                let site = site.site_in_dir((Sign::Positive, d), 1, bounds);
                *entry = (site, ps[i])
            });
        // Swap entry 1 to make signs correct.
        let t = poss[1].clone();
        poss[1] = negs[1].clone();
        negs[1] = t;

        debug_assert_ne!(poss, negs);
        (poss, negs)
    }

    fn get_cube_choice(
        &self,
        cube_index: [usize; 4],
        dims: &[Dimension; 3],
        leftover: Dimension,
        off: usize,
        rand_num: f64,
    ) -> Option<i32> {
        let (poss, negs) =
            Self::get_np_indices_for_cube(cube_index, dims, leftover, off, &self.bounds);

        // Copy over the plaquette values at the appropriate positions for use later.
        let f = |(s, p): &(SiteIndex, usize)| (s.t, s.x, s.y, s.z, *p);
        let pos_vals = [
            self.np[f(&poss[0])],
            self.np[f(&poss[1])],
            self.np[f(&poss[2])],
        ];
        let neg_vals = [
            self.np[f(&negs[0])],
            self.np[f(&negs[1])],
            self.np[f(&negs[2])],
        ];

        // Now we have all our plaquette indices.
        // Add all the boltzman weights for all changes we could make to this cube
        // let moves_and_weights = (1i32..2 * self.vn.len() as i32)
        let moves_and_weights = (1i32..self.vn.len() as i32 - 1i32)
            .flat_map(|inc_dec| [inc_dec, -inc_dec])
            // Get boltzman weights for increasing and decreasing by value.
            .map(|delta| {
                // Add delta to each pos_vals, subtract from each neg_vals
                let pos_sum = pos_vals
                    .iter()
                    .cloned()
                    .map(|v| {
                        let new_val = (v + delta).abs() as usize;
                        if new_val >= self.vn.len() {
                            f64::infinity()
                        } else {
                            self.vn[new_val] - self.vn[v.abs() as usize]
                        }
                    })
                    .sum::<f64>();
                let neg_sum = neg_vals
                    .iter()
                    .cloned()
                    .map(|v| {
                        let new_val = (v - delta).abs() as usize;
                        if new_val >= self.vn.len() {
                            f64::infinity()
                        } else {
                            self.vn[new_val] - self.vn[v.abs() as usize]
                        }
                    })
                    .sum::<f64>();
                // Sum of energy changes
                (delta, pos_sum + neg_sum)
            })
            // Convert to boltzman weights.
            .map(|(delta, delta_energy)| (delta, (-delta_energy).exp()))
            .collect::<Vec<_>>();

        // Find which edit to perform if any.
        let total_nontrivial_weight = moves_and_weights.iter().map(|(_, w)| *w).sum::<f64>();
        // The total weight is 1+this stuff.

        let rand_num = rand_num * (1.0 + total_nontrivial_weight);
        if rand_num <= 1.0 {
            // Do nothing!
            None
        } else {
            // Find the change to perform.
            let rand_num = rand_num - 1.0;
            let delta = moves_and_weights
                .into_iter()
                .try_fold(rand_num, |r, (delta, w)| {
                    let r = r - w;
                    if r <= 0.0 {
                        Err(delta)
                    } else {
                        Ok(r)
                    }
                })
                .unwrap_err();
            Some(delta)
        }
    }

    fn get_cube_index(
        index: [usize; 4],
        p: usize,
        dims: &[Dimension; 3],
        leftover: Dimension,
        off: usize,
        bounds: &SiteIndex,
    ) -> Option<((usize, usize, usize, usize), Sign)> {
        // Strategy is to check of the dimension normal to the plaquette
        // (in 3D of cube) has the correct offset. If it doesn't, plaquette belongs
        // to the adjacent cube instead.
        let [t, x, y, z] = index;
        let (p_one, p_two) = Self::plaquette_subindex_to_dim_dim(p);
        if p_one == leftover || p_two == leftover {
            None
        } else {
            let p_normal = dims
                .iter()
                .cloned()
                .find(|d| *d != p_one && *d != p_two)
                .unwrap();

            let mut site = SiteIndex { t, x, y, z };

            let parity_check = (dims.iter().map(|d| site.get(*d)).sum::<usize>() + off) % 2 == 0;
            let sign = if !parity_check {
                site.modify((Sign::Negative, p_normal), 1, bounds);
                Sign::Negative
            } else {
                Sign::Positive
            };
            let sign = if p_normal == dims[1] {
                sign.flip()
            } else {
                sign
            };
            let cube_index = (
                site.get(leftover),
                site.get(dims[0]),
                site.get(dims[1]),
                site.get(dims[2]) / 2,
            );

            let cube_bounds = Self::get_cube_update_shape(bounds, dims, leftover);
            debug_assert!(cube_index.0 < cube_bounds.0);
            debug_assert!(cube_index.1 < cube_bounds.1);
            debug_assert!(cube_index.2 < cube_bounds.2);
            debug_assert!(cube_index.3 < cube_bounds.3);

            // Now we know the cube location, time to lookup.
            // Only nontrivial value is the collapsed dimension, which is 2*x + 0/1
            Some((cube_index, sign))
        }
    }

    pub fn get_cube_update_shape(
        bounds: &SiteIndex,
        dims: &[Dimension; 3],
        leftover: Dimension,
    ) -> (usize, usize, usize, usize) {
        (
            bounds.get(leftover),
            bounds.get(dims[0]),
            bounds.get(dims[1]),
            bounds.get(dims[2]) / 2,
        )
    }

    pub fn apply_cube_updates(
        cube_choices: &Array4<i32>,
        dims: &[Dimension; 3],
        leftover: Dimension,
        off: usize,
        np: &mut Array5<i32>,
        bounds: &SiteIndex,
    ) {
        np.indexed_iter_mut()
            .par_bridge()
            .for_each(|((t, x, y, z, p), np)| {
                let cube_index = Self::get_cube_index([t, x, y, z], p, dims, leftover, off, bounds);
                if let Some((cube_index, sign)) = cube_index {
                    let cube_choice = cube_choices[cube_index];
                    match sign {
                        Sign::Positive => *np += cube_choice,
                        Sign::Negative => *np -= cube_choice,
                    }
                }
            });
    }

    pub fn graph_state(&self) -> &Array5<i32> {
        &self.np
    }
    pub fn graph_state_mut(&mut self) -> &mut Array5<i32> {
        &mut self.np
    }

    pub fn get_bounds(&self) -> SiteIndex {
        self.bounds.clone()
    }

    fn update_cubes_for_dim<R>(&mut self, dims: &[Dimension; 3], offset: bool, rng: Option<&mut R>)
    where
        R: Rng,
    {
        // Go through each cube and read off the relevant plaquettes.
        // Then make a choice over all possible increments and decrements.
        let off = if offset { 1 } else { 0 };

        let leftover = Self::get_leftover_dim(dims);
        let shape = Self::get_cube_update_shape(&self.bounds, dims, leftover);
        let mut cube_choices = Array4::<i32>::zeros(shape);

        // Pick all cube deltas, either using a single rng or a thread specific one
        if let Some(rng) = rng {
            let rands = Array4::random_using(shape, Uniform::new(0.0, 1.0), rng);
            cube_choices
                .indexed_iter_mut()
                .par_bridge()
                .for_each(|((rho, mu, nu, sigma), c)| {
                    let rand_num = rands[(rho, mu, nu, sigma)];
                    let cube_choice =
                        self.get_cube_choice([rho, mu, nu, sigma], dims, leftover, off, rand_num);
                    if let Some(delta) = cube_choice {
                        *c = delta;
                    }
                });
        } else {
            cube_choices
                .indexed_iter_mut()
                .par_bridge()
                .for_each(|((rho, mu, nu, sigma), c)| {
                    let rand_num = rand::thread_rng().gen();
                    let cube_choice =
                        self.get_cube_choice([rho, mu, nu, sigma], dims, leftover, off, rand_num);
                    if let Some(delta) = cube_choice {
                        *c = delta;
                    }
                });
        }

        // Apply all cube deltas.
        Self::apply_cube_updates(
            &cube_choices,
            dims,
            leftover,
            off,
            &mut self.np,
            &self.bounds,
        );

        // Debug check that no edges have violations.
        debug_assert_eq!(self.get_edges_with_violations(), vec![]);
    }

    pub fn get_cube_dim_iterator() -> impl Iterator<Item = [Dimension; 3]> {
        (0..4usize)
            .flat_map(move |mu| {
                (1 + mu..4usize)
                    .flat_map(move |nu| (1 + nu..4usize).map(move |sigma| (mu, nu, sigma)))
            })
            .map(|(mu, nu, sigma)| {
                [
                    Dimension::from(mu),
                    Dimension::from(nu),
                    Dimension::from(sigma),
                ]
            })
    }

    pub fn get_leftover_dim(dims: &[Dimension; 3]) -> Dimension {
        (0..4usize)
            .map(Dimension::from)
            .find(|rho| dims.iter().all(|d| d != rho))
            .unwrap()
    }

    pub fn get_cube_dim_and_offset_iterator() -> impl Iterator<Item = ([Dimension; 3], bool)> {
        Self::get_cube_dim_iterator().flat_map(|d| [(d, false), (d, true)])
    }

    pub fn local_update_sweep<R: Rng>(&mut self, rng: Option<&mut R>) {
        // Go through all possible cube
        if let Some(rng) = rng {
            Self::get_cube_dim_and_offset_iterator()
                .for_each(|(dims, offset)| self.update_cubes_for_dim(&dims, offset, Some(rng)));
        } else {
            Self::get_cube_dim_and_offset_iterator()
                .for_each(|(dims, offset)| self.update_cubes_for_dim::<R>(&dims, offset, None));
        }
    }

    pub fn global_update_sweep<R: Rng>(&mut self, rng: Option<&mut R>) {
        // We can update all planes at once, nothing overlaps.
        // There are 6 planar dims, for each the number of planes is the product of remaining two
        // dimensions: # = y*z + x*z + x*y + t*z + t*y + t*x
        //               = t*(x+y+z) + x*(y+z) + y*z
        let (t, x, y, z) = (self.bounds.t, self.bounds.x, self.bounds.y, self.bounds.z);
        let tnums = t * (x + y + z);
        let xnums = x * (y + z);
        let yz = y * z;
        let num_planes = tnums + xnums + yz;
        let mut choices = vec![0_i8; num_planes];
        choices
            .iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(plane, choice)| {
                let rand_num = 1.0;
                let (plane_dims, mu, nu) = match plane {
                    p if p < t * x => (
                        [Dimension::Y, Dimension::Z], // Remaining [Dimension::T, Dimension::X],
                        p / x,
                        p % x,
                    ),
                    p if p < t * (x + y) => {
                        let p = p - t * x;
                        (
                            [Dimension::X, Dimension::Z], // Remaining [Dimension::T, Dimension::Y],
                            p / y,
                            p % y,
                        )
                    }
                    p if p < tnums => {
                        let p = p - t * (x + y);
                        (
                            [Dimension::X, Dimension::Y], // Remaining [Dimension::T, Dimension::Z],
                            p / z,
                            p % z,
                        )
                    }
                    p if p < tnums + x * y => {
                        let p = p - tnums;
                        (
                            [Dimension::T, Dimension::Z], // Remaining [Dimension::X, Dimension::Y],
                            p / y,
                            p % y,
                        )
                    }
                    p if p < tnums + x * (y + z) => {
                        let p = p - (tnums + x * y);
                        (
                            [Dimension::T, Dimension::Y], // Remaining [Dimension::X, Dimension::Z],
                            p / z,
                            p % z,
                        )
                    }
                    p => {
                        let p = p - (tnums + xnums);
                        (
                            [Dimension::T, Dimension::X], // Remaining [Dimension::Y, Dimension::Z],
                            p / z,
                            p % z,
                        )
                    }
                };
                let p = Self::dim_dim_to_plaquette_subindex(plane_dims[0], plane_dims[1]);
                // Iterate over plaquettes in plane (mu, nu, [p])
                // sum up energy costs for +1 and -1, then decide fate.
                let (e_sub_one, e_add_one) = self
                    .np
                    .axis_iter(ndarray::Axis(plane_dims[1].into()))
                    .fold((0.0, 0.0), |acc, np| {
                        np.axis_iter(ndarray::Axis(plane_dims[0].into())).fold(
                            acc,
                            |(sub, add), np| {
                                let int_n = np.get((mu, nu, p)).unwrap();
                                let d_sub = self.vn[(int_n - 1).abs() as usize]
                                    - self.vn[int_n.abs() as usize];
                                let d_add = self.vn[(int_n + 1).abs() as usize]
                                    - self.vn[int_n.abs() as usize];
                                (sub + d_sub, add + d_add)
                            },
                        )
                    });
                // Find which edit to perform if any.
                let w_sub_one = e_sub_one.exp();
                let w_add_one = e_add_one.exp();
                let total_nontrivial_weight = w_sub_one + w_add_one;
                // The total weight is 1+this stuff.
                let rand_num = rand_num * (1.0 + total_nontrivial_weight);
                if rand_num <= 1.0 {
                    // Do nothing!
                    return;
                }
                let rand_num = rand_num - 1.0;
                // Now choose between the remaining.
                *choice = if rand_num < w_sub_one { -1 } else { 1 };
            });
        self.np
            .indexed_iter_mut()
            .par_bridge()
            .for_each(|((t, x, y, z, p), n)| {
                // TODO find the appropriate choice
            });
    }

    pub fn clone_graph(&self) -> Array5<i32> {
        self.np.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand::prelude::*;

    const fn simple_bounds() -> SiteIndex {
        SiteIndex {
            t: 2,
            x: 4,
            y: 6,
            z: 8,
        }
    }

    #[test]
    fn test_modify_simple() {
        let bounds = simple_bounds();
        let mut site = SiteIndex {
            t: 0,
            x: 1,
            y: 2,
            z: 3,
        };

        (0..4).for_each(|i| {
            let d = Dimension::from(i);
            site.modify((Sign::Positive, d), 1, &bounds);
            assert_eq!(
                site,
                SiteIndex {
                    t: 0 + if i == 0 { 1 } else { 0 },
                    x: 1 + if i == 1 { 1 } else { 0 },
                    y: 2 + if i == 2 { 1 } else { 0 },
                    z: 3 + if i == 3 { 1 } else { 0 },
                }
            );
            site.modify((Sign::Negative, d), 1, &bounds);
            assert_eq!(
                site,
                SiteIndex {
                    t: 0,
                    x: 1,
                    y: 2,
                    z: 3,
                }
            )
        });
    }

    #[test]
    fn test_modify_wrap() {
        let bounds = simple_bounds();
        let mut site = SiteIndex {
            t: 0,
            x: 0,
            y: 0,
            z: 0,
        };
        (0..4).for_each(|i| {
            let d = Dimension::from(i);
            site.modify((Sign::Negative, d), 1, &bounds);
            assert_eq!(
                site,
                SiteIndex {
                    t: if i == 0 { bounds.t - 1 } else { 0 },
                    x: if i == 1 { bounds.x - 1 } else { 0 },
                    y: if i == 2 { bounds.y - 1 } else { 0 },
                    z: if i == 3 { bounds.z - 1 } else { 0 },
                }
            );
            site.modify((Sign::Positive, d), 1, &bounds);
            assert_eq!(
                site,
                SiteIndex {
                    t: 0,
                    x: 0,
                    y: 0,
                    z: 0,
                }
            );
        });
    }

    #[test]
    fn test_in_order() {
        (0..4usize)
            .flat_map(move |mu| (1 + mu..4usize).map(move |nu| (mu, nu)))
            .map(|(mu, nu)| (Dimension::from(mu), Dimension::from(nu)))
            .for_each(|(mu, nu)| assert!(mu.in_order(nu)));
    }

    #[test]
    fn test_flux_additions() {
        let bounds = simple_bounds();
        let mut graph = NDDualGraph::new(bounds.t, bounds.x, bounds.y, bounds.z, []);
        assert_eq!(
            graph.np.iter().map(|n| n.abs()).sum::<i32>(),
            0,
            "Unexpected sum from entries {:?}",
            graph
                .np
                .indexed_iter()
                .filter(|(_, n)| n.abs() != 0)
                .collect::<Vec<_>>()
        );
        assert_eq!(graph.get_edges_with_violations(), vec![]);
        graph.add_flux(
            SiteIndex {
                t: 0,
                x: 0,
                y: 0,
                z: 0,
            },
            0,
            1,
        );
        assert_eq!(
            graph.np.iter().map(|n| *n).sum::<i32>(),
            1,
            "Unexpected sum from entries {:?}",
            graph
                .np
                .indexed_iter()
                .filter(|(_, n)| n.abs() != 0)
                .collect::<Vec<_>>()
        );
        assert_eq!(graph.get_edges_with_violations(), vec![]);
        assert_eq!(graph.currents.iter().map(|(_, c)| c.abs()).sum::<i32>(), 4);
        graph.add_flux(
            SiteIndex {
                t: 0,
                x: 1,
                y: 0,
                z: 0,
            },
            0,
            1,
        );
        assert_eq!(
            graph.np.iter().map(|n| *n).sum::<i32>(),
            2,
            "Unexpected sum from entries {:?}",
            graph
                .np
                .indexed_iter()
                .filter(|(_, n)| n.abs() != 0)
                .collect::<Vec<_>>()
        );
        assert_eq!(graph.get_edges_with_violations(), vec![]);
        assert_eq!(graph.currents.iter().map(|(_, c)| c.abs()).sum::<i32>(), 6);
    }

    #[test]
    fn test_cube_update_violations() {
        let bounds = simple_bounds();
        let mut graph = NDDualGraph::new(bounds.t, bounds.x, bounds.y, bounds.z, []);

        for (dims, offset) in NDDualGraph::get_cube_dim_and_offset_iterator() {
            let leftover = NDDualGraph::get_leftover_dim(&dims);
            let off = if offset { 1 } else { 0 };
            let shape = NDDualGraph::get_cube_update_shape(&graph.bounds, &dims, leftover);
            let mut cube_choices = Array4::<i32>::zeros(shape);
            // Go through each possible cube update, do it and undo it, check expected behavior.
            let cube_locations = (0..shape.0).flat_map(|mu| {
                (0..shape.1).flat_map(move |nu| {
                    (0..shape.2)
                        .flat_map(move |sigma| (0..shape.3).map(move |rho| (mu, nu, sigma, rho)))
                })
            });
            for cube_site in cube_locations {
                cube_choices[cube_site] = 1;
                NDDualGraph::apply_cube_updates(
                    &cube_choices,
                    &dims,
                    leftover,
                    off,
                    &mut graph.np,
                    &bounds,
                );
                assert_eq!(
                    graph.np.iter().map(|n| n.abs()).sum::<i32>(),
                    6,
                    "Unexpected sum from entries {:?}",
                    graph
                        .np
                        .indexed_iter()
                        .filter(|(_, n)| n.abs() != 0)
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    graph.np.iter().map(|n| *n).sum::<i32>(),
                    0,
                    "Unexpected sum from entries {:?}",
                    graph
                        .np
                        .indexed_iter()
                        .filter(|(_, n)| n.abs() != 0)
                        .collect::<Vec<_>>()
                );
                assert_eq!(graph.get_edges_with_violations(), vec![]);
                cube_choices[cube_site] = -1;
                NDDualGraph::apply_cube_updates(
                    &cube_choices,
                    &dims,
                    leftover,
                    off,
                    &mut graph.np,
                    &bounds,
                );
                assert_eq!(
                    graph.np.iter().map(|n| n.abs()).sum::<i32>(),
                    0,
                    "Unexpected sum from entries {:?}",
                    graph
                        .np
                        .indexed_iter()
                        .filter(|(_, n)| n.abs() != 0)
                        .collect::<Vec<_>>()
                );
                assert_eq!(graph.get_edges_with_violations(), vec![]);
                cube_choices[cube_site] = 0;
            }
        }
    }

    #[test]
    fn test_cube_side_counts() {
        let bounds = simple_bounds();
        let mut graph = NDDualGraph::new(bounds.t, bounds.x, bounds.y, bounds.z, []);

        for (dims, offset) in NDDualGraph::get_cube_dim_and_offset_iterator() {
            let leftover = NDDualGraph::get_leftover_dim(&dims);
            let off = if offset { 1 } else { 0 };
            let site_indices = (0..bounds.t).flat_map(move |a| {
                (0..bounds.x).flat_map(move |b| {
                    (0..bounds.y).flat_map(move |c| (0..bounds.z).map(move |d| (a, b, c, d)))
                })
            });
            for (t, x, y, z) in site_indices {
                let ncubes = (0..6)
                    .map(|p| {
                        NDDualGraph::get_cube_index([t, x, y, z], p, &dims, leftover, off, &bounds)
                    })
                    .filter(Option::is_some)
                    .count();
                assert_eq!(ncubes, 3);
            }
        }
    }

    #[test]
    fn test_cube_locations() {
        let bounds = simple_bounds();

        let dims = NDDualGraph::get_cube_dim_and_offset_iterator()
            .map(|(d, b)| (d, if b { 1 } else { 0 }));
        for (dims, off) in dims {
            let leftover = NDDualGraph::get_leftover_dim(&dims);
            let bb = &bounds;
            let cube_indices = (0..bb.get(leftover)).flat_map(move |a| {
                (0..bb.get(dims[0])).flat_map(move |b| {
                    (0..bb.get(dims[1]))
                        .flat_map(move |c| (0..bb.get(dims[2]) / 2).map(move |d| (a, b, c, d)))
                })
            });
            for (c_t, c_x, c_y, c_z) in cube_indices {
                let cube_index = [c_t, c_x, c_y, c_z];
                let (poss, negs) =
                    NDDualGraph::get_np_indices_for_cube(cube_index, &dims, leftover, off, &bounds);
                poss.into_iter()
                    .chain(negs.into_iter())
                    .for_each(|test_loc| {
                        let (a_s, a_p) = test_loc;
                        let (a_t, a_x, a_y, a_z) = (a_s.t, a_s.x, a_s.y, a_s.z);
                        let test_cube_index = NDDualGraph::get_cube_index(
                            [a_t, a_x, a_y, a_z],
                            a_p,
                            &dims,
                            leftover,
                            off,
                            &bounds,
                        );
                        let cube_index = (c_t, c_x, c_y, c_z);
                        assert_eq!(
                            test_cube_index.map(|(s,_)| s),
                            Some(cube_index),
                            "Error mapping plaquette {:?} back to cube {:?} with dims {:?} and offset={}",
                            (a_s, a_p),
                            cube_index,
                            dims, off
                        )
                    })
            }
        }
    }

    #[test]
    fn test_cube_choice() {
        let bounds = simple_bounds();
        let mut graph = NDDualGraph::new(bounds.t, bounds.x, bounds.y, bounds.z, [0.0, 0.1, 0.4]);
        let choice = graph.get_cube_choice(
            [0, 0, 0, 0],
            &[Dimension::T, Dimension::X, Dimension::Y],
            Dimension::Z,
            0,
            0.5,
        );
        assert_eq!(choice, Some(1))
    }

    #[test]
    fn test_local_update() {
        let bounds = simple_bounds();
        let mut graph = NDDualGraph::new(bounds.t, bounds.x, bounds.y, bounds.z, [0.0, 0.1, 0.4]);
        let mut rng = SmallRng::from_entropy();
        graph.local_update_sweep(&mut rng);
    }
}
