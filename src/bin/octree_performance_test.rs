use avian3d::math::Vector;
use bevy::prelude::Entity;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stardrift::physics::octree::{LegacyOctree, OctreeBody, OptimizedOctree};
use std::time::Instant;

fn generate_test_bodies(count: usize, seed: u64) -> Vec<OctreeBody> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bodies = Vec::with_capacity(count);

    for i in 0..count {
        let position = Vector::new(
            rng.random_range(-100.0..100.0),
            rng.random_range(-100.0..100.0),
            rng.random_range(-100.0..100.0),
        );
        let mass = rng.random_range(1.0..10.0);

        bodies.push(OctreeBody {
            entity: Entity::from_raw(i as u32),
            position,
            mass,
        });
    }

    bodies
}

fn benchmark_build_performance() {
    println!("=== Octree Build Performance Comparison ===");

    for &body_count in &[100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000] {
        let bodies = generate_test_bodies(body_count, 42);

        // Benchmark original octree
        let start = Instant::now();
        for _ in 0..10 {
            let mut octree = LegacyOctree::new(0.5, 0.1, 1000.0);
            octree.build(bodies.clone());
        }
        let original_time = start.elapsed();

        // Benchmark optimized octree
        let start = Instant::now();
        for _ in 0..10 {
            let mut octree = OptimizedOctree::new(0.5, 0.1, 1000.0);
            octree.build(bodies.clone());
        }
        let optimized_time = start.elapsed();

        let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

        println!(
            "Bodies: {:5} | Original: {:8.2}ms | Optimized: {:8.2}ms | Speedup: {:.2}x",
            body_count,
            original_time.as_secs_f64() * 1000.0 / 10.0,
            optimized_time.as_secs_f64() * 1000.0 / 10.0,
            speedup
        );
    }
}

fn benchmark_force_calculation_performance() {
    println!("\n=== Force Calculation Performance Comparison ===");

    for &body_count in &[100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000] {
        let bodies = generate_test_bodies(body_count, 42);
        let test_body = &bodies[0];

        // Setup octrees
        let mut original_octree = LegacyOctree::new(0.5, 0.1, 1000.0);
        original_octree.build(bodies.clone());

        let mut optimized_octree = OptimizedOctree::new(0.5, 0.1, 1000.0);
        optimized_octree.build(bodies.clone());

        // Benchmark original octree force calculation
        let start = Instant::now();
        for _ in 0..1000 {
            let _force = original_octree.calculate_force(
                test_body,
                original_octree.root.as_ref(),
                6.67430e-11,
            );
        }
        let original_time = start.elapsed();

        // Benchmark optimized octree force calculation
        let start = Instant::now();
        for _ in 0..1000 {
            let _force = optimized_octree.calculate_force(test_body, 6.67430e-11);
        }
        let optimized_time = start.elapsed();

        let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

        println!(
            "Bodies: {:5} | Original: {:8.2}μs | Optimized: {:8.2}μs | Speedup: {:.2}x",
            body_count,
            original_time.as_nanos() as f64 / 1000.0 / 1000.0,
            optimized_time.as_nanos() as f64 / 1000.0 / 1000.0,
            speedup
        );
    }
}

fn benchmark_memory_usage() {
    println!("\n=== Memory Usage Comparison ===");

    let body_count = 1000;
    let bodies = generate_test_bodies(body_count, 42);

    // Test original octree
    let mut original_octree = LegacyOctree::new(0.5, 0.1, 1000.0);
    original_octree.build(bodies.clone());
    let original_stats = original_octree.octree_stats();
    let original_pool_stats = original_octree.pool_stats();

    // Test optimized octree
    let mut optimized_octree = OptimizedOctree::new(0.5, 0.1, 1000.0);
    optimized_octree.build(bodies.clone());
    let optimized_stats = optimized_octree.octree_stats();
    let optimized_pool_stats = optimized_octree.pool_stats();

    println!("Original Octree:");
    println!(
        "  Nodes: {}, Bodies: {}",
        original_stats.node_count, original_stats.body_count
    );
    println!(
        "  Pool: {} internal, {} external",
        original_pool_stats.0, original_pool_stats.1
    );

    println!("Optimized Octree:");
    println!(
        "  Nodes: {}, Bodies: {}",
        optimized_stats.node_count, optimized_stats.body_count
    );
    println!(
        "  Pool: {} allocated, {} free",
        optimized_pool_stats.0, optimized_pool_stats.1
    );
}

fn test_correctness() {
    println!("\n=== Correctness Test ===");

    let bodies = generate_test_bodies(100, 42);
    let test_body = &bodies[0];

    // Setup octrees
    let mut original_octree = LegacyOctree::new(0.5, 0.1, 1000.0);
    original_octree.build(bodies.clone());

    let mut optimized_octree = OptimizedOctree::new(0.5, 0.1, 1000.0);
    optimized_octree.build(bodies.clone());

    // Calculate forces
    let original_force =
        original_octree.calculate_force(test_body, original_octree.root.as_ref(), 6.67430e-11);
    let optimized_force = optimized_octree.calculate_force(test_body, 6.67430e-11);

    let difference = (original_force - optimized_force).length();
    let relative_error = difference / original_force.length();

    println!("Original force:  {original_force:?}");
    println!("Optimized force: {optimized_force:?}");
    println!("Difference:      {difference:.2e}");
    println!("Relative error:  {relative_error:.2e}");

    if relative_error < 1e-10 {
        println!("✅ Results match within acceptable tolerance");
    } else {
        println!("❌ Results differ significantly");
    }
}

fn main() {
    println!("Octree Memory Optimization Performance Test");
    println!("==========================================");

    test_correctness();
    benchmark_build_performance();
    benchmark_force_calculation_performance();
    benchmark_memory_usage();

    println!("\n=== Summary ===");
    println!("The optimized octree implementation includes:");
    println!("• Struct-of-Arrays (SoA) layout for better cache locality");
    println!("• Memory pool with index-based storage instead of pointers");
    println!("• Separated hot/cold data for improved cache performance");
    println!("• Explicit memory layout control with #[repr(C)]");
    println!("• Reduced heap fragmentation and allocation overhead");
}
