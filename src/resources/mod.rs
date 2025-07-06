use crate::physics;
use avian3d::math::Scalar;
use avian3d::math::Vector;
use bevy::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

#[derive(Resource, Deref, DerefMut, Debug, Clone, PartialEq)]
pub struct SharedRng(pub ChaCha8Rng);

impl SharedRng {
    pub fn from_seed(seed: u64) -> Self {
        Self(ChaCha8Rng::seed_from_u64(seed))
    }

    pub fn from_optional_seed(seed: Option<u64>) -> Self {
        match seed {
            Some(seed) => Self::from_seed(seed),
            None => Self::default(),
        }
    }
}

impl Default for SharedRng {
    fn default() -> Self {
        Self(ChaCha8Rng::from_rng(&mut rand::rng()))
    }
}

#[derive(Resource, Deref, DerefMut, Copy, Clone, PartialEq, Debug)]
pub struct GravitationalConstant(pub Scalar);

impl Default for GravitationalConstant {
    fn default() -> Self {
        Self(1e1)
    }
}

#[derive(Resource, Deref, DerefMut, Copy, Clone, PartialEq, Debug)]
pub struct BodyCount(pub usize);

impl Default for BodyCount {
    fn default() -> Self {
        Self(100)
    }
}

#[derive(Resource, Deref, DerefMut, Copy, Clone, Default, PartialEq, Debug)]
pub struct Barycenter(pub Option<Vector>);

#[derive(Resource, Deref, DerefMut, Debug)]
pub struct GravitationalOctree(pub physics::octree::OptimizedOctree);

impl GravitationalOctree {
    pub fn new(octree: physics::octree::OptimizedOctree) -> Self {
        Self(octree)
    }
}

#[derive(Resource, Default)]
pub struct OctreeVisualizationSettings {
    pub enabled: bool,
    pub max_depth: Option<usize>, // None means show all levels
}

#[derive(Resource, Default)]
pub struct BarycenterGizmoVisibility {
    pub enabled: bool,
}

#[derive(Resource, Default)]
pub struct LoadingProgress {
    pub progress: f32, // 0.0 to 1.0
    pub current_message: String,
}

#[derive(Resource)]
pub struct BodySpawningProgress {
    pub bodies_spawned: usize,
    pub total_bodies: usize,
    pub batch_size: usize,
}

#[derive(Resource, Deref, DerefMut)]
pub struct LoadingTimer(pub Timer);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_shared_rng_deterministic_with_seed() {
        let seed = 12345u64;
        let mut rng1 = SharedRng::from_seed(seed);
        let mut rng2 = SharedRng::from_seed(seed);

        // Generate some random numbers from both RNGs
        let values1: Vec<f64> = (0..10).map(|_| rng1.random_range(0.0..1.0)).collect();
        let values2: Vec<f64> = (0..10).map(|_| rng2.random_range(0.0..1.0)).collect();

        // They should be identical since they use the same seed
        assert_eq!(values1, values2);
    }

    #[test]
    fn test_shared_rng_from_optional_seed() {
        let seed = 54321u64;
        let mut rng_with_seed = SharedRng::from_optional_seed(Some(seed));
        let mut rng_with_same_seed = SharedRng::from_seed(seed);

        // Generate some random numbers from both RNGs
        let value1: f64 = rng_with_seed.random_range(0.0..1.0);
        let value2: f64 = rng_with_same_seed.random_range(0.0..1.0);

        // They should be identical
        assert_eq!(value1, value2);
    }

    #[test]
    fn test_shared_rng_from_optional_seed_none() {
        let mut rng1 = SharedRng::from_optional_seed(None);
        let mut rng2 = SharedRng::from_optional_seed(None);

        // Generate some random numbers from both RNGs
        let value1: f64 = rng1.random_range(0.0..1.0);
        let value2: f64 = rng2.random_range(0.0..1.0);

        // They should be different since they use random seeds
        assert_ne!(value1, value2);
    }
}
