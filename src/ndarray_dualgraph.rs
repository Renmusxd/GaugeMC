use ndarray::{Array4, Array5};
use num_traits::float::Float;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Default, Clone, Eq, PartialEq, Debug)]
pub struct SiteIndex {
    t: usize,
    x: usize,
    y: usize,
    z: usize,
}

impl SiteIndex {
    fn get(&self, dim: Dimension) -> usize {
        match dim {
            Dimension::T => self.t,
            Dimension::X => self.x,
            Dimension::Y => self.y,
            Dimension::Z => self.z,
        }
    }

    fn set(&mut self, dim: Dimension, val: usize) {
        match dim {
            Dimension::T => self.t = val,
            Dimension::X => self.x = val,
            Dimension::Y => self.y = val,
            Dimension::Z => self.z = val,
        }
    }

    fn modify(&mut self, dim: UnitDirection, delta: usize, bounds: &Self) {
        let (edit, bound) = match dim {
            (_, Dimension::T) => (&mut self.t, bounds.t),
            (_, Dimension::X) => (&mut self.x, bounds.x),
            (_, Dimension::Y) => (&mut self.y, bounds.y),
            (_, Dimension::Z) => (&mut self.z, bounds.z),
        };
        match dim {
            (Sign::Positive, _) => {
                *edit += delta;
                *edit = *edit % bound;
            }
            (Sign::Negative, _) => {
                let delta = bound - (delta % bound);
                *edit += delta;
                *edit = *edit % bound;
            }
        }
    }

    fn site_in_dir(&self, dim: UnitDirection, delta: usize, bounds: &Self) -> Self {
        let mut other = self.clone();
        other.modify(dim, delta, bounds);
        other
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct PlaquetteIndex {
    t: usize,
    x: usize,
    y: usize,
    z: usize,
    p: usize,
}

impl PlaquetteIndex {
    fn from_site(site: SiteIndex, p: usize) -> Self {
        Self {
            t: site.t,
            x: site.x,
            y: site.y,
            z: site.z,
            p,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, Debug)]
enum Dimension {
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
    fn in_order(self, other: Self) -> bool {
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
enum Sign {
    Positive,
    Negative,
}

impl Sign {
    fn flip(self) -> Self {
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
        }
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

    pub fn site_to_plaquette_index(
        &self,
        site: &SiteIndex,
        first: UnitDirection,
        second: UnitDirection,
    ) -> (PlaquetteIndex, Sign) {
        let mut site = site.clone();
        let mut sign = Sign::Positive;
        let first = match first {
            (Sign::Negative, dim) => {
                site.modify((Sign::Negative, dim), 1, &self.bounds);
                sign = sign.flip();
                dim
            }
            (_, dim) => dim,
        };
        let second = match second {
            (Sign::Negative, dim) => {
                site.modify((Sign::Negative, dim), 1, &self.bounds);
                sign = sign.flip();
                dim
            }
            (_, dim) => dim,
        };
        if !first.in_order(second) {
            sign = sign.flip()
        };
        let p = Self::dim_dim_to_plaquette_subindex(first, second);
        (PlaquetteIndex::from_site(site, p), sign)
    }

    pub fn local_update_sweep<R: Rng>(&mut self, beta: f64, rng: &mut R) {
        // Go through all possible cube
        (0..4usize)
            .flat_map(move |mu| {
                (1 + mu..4usize).flat_map(move |nu| {
                    (1 + nu..4usize)
                        .flat_map(move |sigma| [(mu, nu, sigma, false), (mu, nu, sigma, true)])
                })
            })
            .map(|(mu, nu, sigma, offset)| {
                let dims = [
                    Dimension::from(mu),
                    Dimension::from(nu),
                    Dimension::from(sigma),
                ];
                (dims, offset)
            })
            .for_each(|(dims, offset)| {
                // Go through each cube and read off the relevant plaquettes.
                // Then make a choice over all possible increments and decrements.
                let off = if offset { 1 } else { 0 };

                let leftover = (0..4usize)
                    .map(|i| Dimension::from(i))
                    .find(|rho| dims.iter().all(|d| d != rho))
                    .unwrap();

                let mut cube_choices = Array4::<i32>::zeros((
                    self.np.shape()[usize::from(leftover)],
                    self.np.shape()[usize::from(dims[0])],
                    self.np.shape()[usize::from(dims[1])],
                    self.np.shape()[usize::from(dims[2])] / 2,
                ));

                // Pick all cube deltas.
                cube_choices.indexed_iter_mut().par_bridge().for_each(
                    |((rho, mu, nu, sigma), c)| {
                        // TODO get a random number
                        let rand_num: f64 = rand::thread_rng().gen();

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
                        let poss = [
                            (site.t, site.x, site.y, site.z, ps[0]),
                            (site.t, site.x, site.y, site.z, ps[1]),
                            (site.t, site.x, site.y, site.z, ps[2]),
                        ];
                        let mut negs = poss.clone();
                        negs.iter_mut()
                            .zip(dims.iter().cloned().enumerate())
                            .for_each(|(entry, (i, d))| {
                                let site = site.site_in_dir((Sign::Positive, d), 1, &self.bounds);
                                *entry = (site.t, site.x, site.y, site.z, ps[i])
                            });

                        // Copy over the plaquette values at the apropriate positions for use later.
                        let pos_vals = [self.np[poss[0]], self.np[poss[1]], self.np[poss[2]]];
                        let neg_vals = [self.np[negs[0]], self.np[negs[1]], self.np[negs[2]]];

                        // Now we have all our plaquette indices.
                        // Add all the boltzman weights for all changes we could make to this cube
                        let moves_and_weights = (1i32..2 * self.vn.len() as i32)
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
                                            self.vn[new_val] - self.vn[v as usize]
                                        }
                                    })
                                    .sum::<f64>();
                                let neg_sum = neg_vals
                                    .iter()
                                    .cloned()
                                    .map(|v| {
                                        let new_val = (v - delta).abs() as usize;
                                        if new_val as usize >= self.vn.len() {
                                            f64::infinity()
                                        } else {
                                            self.vn[new_val] - self.vn[v as usize]
                                        }
                                    })
                                    .sum::<f64>();
                                // Sum of energy changes
                                (delta, pos_sum + neg_sum)
                            })
                            // Convert to boltzman weights.
                            .map(|(delta, delta_energy)| (delta, (-beta * delta_energy).exp()))
                            .collect::<Vec<_>>();

                        // Find which edit to perform if any.
                        let total_nontrivial_weight =
                            moves_and_weights.iter().map(|(_, w)| *w).sum::<f64>();
                        // The total weight is 1+this stuff.
                        let rand_num = rand_num * (1.0 + total_nontrivial_weight);
                        if rand_num <= 1.0 {
                            // Do nothing!
                            return;
                        }

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

                        // Set the chosen cube delta.
                        *c = delta;
                    },
                );

                // Apply all cube deltas.
                self.np
                    .indexed_iter_mut()
                    .par_bridge()
                    .for_each(|((t, x, y, z, p), np)| {
                        // Strategy is to check of the dimension normal to the plaquette
                        // (in 3D of cube) has the correct offset. If it doesn't, plaquette belongs
                        // to the adjacent cube instead.
                        let (p_one, p_two) = Self::plaquette_subindex_to_dim_dim(p);
                        if p_one == leftover || p_two == leftover {
                            return;
                        }
                        let p_normal = dims
                            .iter()
                            .cloned()
                            .find(|d| *d != p_one && *d != p_two)
                            .unwrap();

                        let mut site = SiteIndex { t, x, y, z };

                        let parity_check =
                            (dims.iter().map(|d| site.get(*d)).sum::<usize>() + off) % 2 == 0;
                        if !parity_check {
                            site.modify((Sign::Negative, p_normal), 1, &self.bounds);
                        }

                        // Now we know the cube location, time to lookup.
                        // Only nontrivial value is the collapsed dimension, which is 2*x + 0/1
                        let cube_choice = cube_choices[(
                            site.get(leftover),
                            site.get(dims[0]),
                            site.get(dims[1]),
                            site.get(dims[2]) / 2,
                        )];
                        *np += cube_choice;
                    });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn simple_bounds() -> SiteIndex {
        SiteIndex {
            t: 4,
            x: 8,
            y: 16,
            z: 32,
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
        let dims = (0..4usize)
            .flat_map(move |mu| (1 + mu..4usize).map(move |nu| (mu, nu)))
            .map(|(mu, nu)| (Dimension::from(mu), Dimension::from(nu)))
            .for_each(|(mu, nu)| assert!(mu.in_order(nu)));
    }
}
