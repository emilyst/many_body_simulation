use avian3d::math::Vector;
use bevy::prelude::Entity;
use criterion::BenchmarkId;
use criterion::Criterion;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

use stardrift::physics;

fn generate_test_bodies(count: usize, seed: u64) -> Vec<physics::octree::OctreeBody> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bodies = Vec::with_capacity(count);

    let radius = 200.0;

    for i in 0..count {
        let theta = rng.random_range(0.0..2.0 * std::f64::consts::PI);
        let phi = rng.random_range(0.0..std::f64::consts::PI);
        let r = rng.random_range(0.0..radius);

        let x = r * phi.sin() * theta.cos();
        let y = r * phi.sin() * theta.sin();
        let z = r * phi.cos();

        let position = Vector::new(x, y, z);
        let mass = rng.random_range(1.0..100.0);

        bodies.push(physics::octree::OctreeBody {
            entity: Entity::from_raw(i as u32),
            position,
            mass,
        });
    }

    bodies
}

fn bench_octree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_construction");

    let body_counts = [10, 50, 100, 500, 1000, 2_000, 5_000];
    let theta = 0.5;
    let min_distance = 10.0;
    let max_force = 1e4;

    for &count in &body_counts {
        let bodies = generate_test_bodies(count, 42);

        group.bench_with_input(BenchmarkId::new("bodies", count), &count, |b, _| {
            b.iter(|| {
                let mut octree = physics::octree::LegacyOctree::new(theta, min_distance, max_force);
                octree.build(black_box(bodies.clone()));
                black_box(octree);
            });
        });
    }

    group.finish();
}

fn bench_octree_construction_leaf_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_construction");

    let leaf_thresholds = [1, 5, 10, 20, 50, 100];
    let body_count = 1000;
    let theta = 0.5;
    let min_distance = 10.0;
    let max_force = 1e4;
    let bodies = generate_test_bodies(body_count, 42);

    for &threshold in &leaf_thresholds {
        group.bench_with_input(
            BenchmarkId::new("leaf_threshold", threshold),
            &threshold,
            |b, &threshold_val| {
                b.iter(|| {
                    let mut octree =
                        physics::octree::LegacyOctree::new(theta, min_distance, max_force)
                            .with_leaf_threshold(threshold_val);
                    octree.build(black_box(bodies.clone()));
                    black_box(octree);
                });
            },
        );
    }

    group.finish();
}

fn bench_octree_force_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_force_calculation");

    let body_counts = [10, 50, 100, 500, 1_000, 2_000];
    let theta = 0.5;
    let min_distance = 10.0;
    let max_force = 1e4;
    let g = 1e1;

    for &count in &body_counts {
        let bodies = generate_test_bodies(count, 42);
        let mut octree = physics::octree::LegacyOctree::new(theta, min_distance, max_force);
        octree.build(bodies.clone());

        group.bench_with_input(BenchmarkId::new("bodies", count), &count, |b, _| {
            b.iter(|| {
                let mut total_force = Vector::ZERO;
                for body in &bodies {
                    let force = octree.calculate_force(black_box(body), None, g);
                    total_force += force;
                }
                black_box(total_force);
            });
        });
    }

    group.finish();
}

fn bench_octree_force_calculation_theta(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_force_calculation");

    let theta_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0];
    let body_count = 5_000;
    let min_distance = 10.0;
    let max_force = 1e4;
    let g = 1e1;
    let bodies = generate_test_bodies(body_count, 42);

    for &theta in &theta_values {
        let mut octree = physics::octree::LegacyOctree::new(theta, min_distance, max_force);
        octree.build(bodies.clone());

        group.bench_with_input(
            BenchmarkId::new("theta", (theta * 100.0) as u32),
            &theta,
            |b, _| {
                b.iter(|| {
                    let mut total_force = Vector::ZERO;
                    for body in &bodies {
                        let force = octree.calculate_force(black_box(body), None, g);
                        total_force += force;
                    }
                    black_box(total_force);
                });
            },
        );
    }

    group.finish();
}

fn bench_complete_physics_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_physics_cycle");

    let body_counts = [50, 100, 250, 500, 1_000];
    let theta = 0.5;
    let min_distance = 10.0;
    let max_force = 1e4;
    let g = 1e1;

    for &count in &body_counts {
        let bodies = generate_test_bodies(count, 42);

        group.bench_with_input(BenchmarkId::new("bodies", count), &count, |b, _| {
            b.iter(|| {
                let mut octree = physics::octree::LegacyOctree::new(theta, min_distance, max_force);
                octree.build(black_box(bodies.clone()));

                let mut total_force = Vector::ZERO;
                for body in &bodies {
                    let force = octree.calculate_force(black_box(body), None, g);
                    total_force += force;
                }

                black_box(total_force);
            });
        });
    }

    group.finish();
}

fn bench_complete_physics_cycle_extreme_body_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_physics_cycle");
    group.measurement_time(std::time::Duration::from_secs(20));

    let body_counts = [5_000, 10_000, 20_000, 100_000, 500_000, 1_000_000];
    let theta = 0.5;
    let min_distance = 10.0;
    let max_force = 1e4;
    let g = 1e1;

    for &count in &body_counts {
        let bodies = generate_test_bodies(count, 42);

        group.bench_with_input(BenchmarkId::new("bodies", count), &count, |b, _| {
            b.iter(|| {
                let mut octree = physics::octree::LegacyOctree::new(theta, min_distance, max_force);
                octree.build(black_box(bodies.clone()));

                let mut total_force = Vector::ZERO;
                for body in &bodies {
                    let force = octree.calculate_force(black_box(body), None, g);
                    total_force += force;
                }

                black_box(total_force);
            });
        });
    }

    group.finish();
}

fn bench_octree_pool_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_pool_reuse");

    let body_counts = [100, 500, 1_000, 2_000, 5_000];
    let theta = 0.5;
    let min_distance = 10.0;
    let max_force = 1e4;

    for &count in &body_counts {
        let bodies = generate_test_bodies(count, 100);

        // Benchmark without pool (new octree each time)
        group.bench_with_input(BenchmarkId::new("without_pool", count), &count, |b, _| {
            b.iter(|| {
                let mut octree1 =
                    physics::octree::LegacyOctree::new(theta, min_distance, max_force);
                octree1.build(black_box(bodies.clone()));
                black_box(&octree1);

                let mut octree2 =
                    physics::octree::LegacyOctree::new(theta, min_distance, max_force);
                octree2.build(black_box(bodies.clone()));
                black_box(&octree2);

                let mut octree3 =
                    physics::octree::LegacyOctree::new(theta, min_distance, max_force);
                octree3.build(black_box(bodies.clone()));
                black_box(&octree3);
            });
        });

        // Benchmark with pool (reuse same octree instance)
        group.bench_with_input(BenchmarkId::new("with_pool", count), &count, |b, _| {
            b.iter(|| {
                let mut octree = physics::octree::LegacyOctree::with_pool_capacity(
                    theta,
                    min_distance,
                    max_force,
                    50,
                    100,
                );

                octree.build(black_box(bodies.clone()));
                black_box(&octree);

                octree.build(black_box(bodies.clone()));
                black_box(&octree);

                octree.build(black_box(bodies.clone()));
                black_box(&octree);
            });
        });
    }

    group.finish();
}

criterion::criterion_group!(
    benches,
    bench_octree_construction,
    bench_octree_construction_leaf_threshold,
    bench_octree_force_calculation,
    bench_octree_force_calculation_theta,
    bench_complete_physics_cycle,
    bench_complete_physics_cycle_extreme_body_counts,
    bench_octree_pool_reuse,
);

criterion::criterion_main!(benches);
