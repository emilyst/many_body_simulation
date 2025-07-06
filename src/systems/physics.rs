use crate::config;
use crate::physics;
use crate::resources;
use crate::utils;
use avian3d::math::Scalar;
use avian3d::math::Vector;
use avian3d::prelude::*;
use bevy::prelude::*;
use rand::Rng;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PhysicsSet {
    BuildOctree,
    ApplyForces,
}

pub fn spawn_simulation_bodies(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    rng: &mut ResMut<resources::SharedRng>,
    body_count: usize,
    config: &config::SimulationConfig,
) {
    let body_distribution_sphere_radius = utils::math::min_sphere_radius_for_surface_distribution(
        body_count,
        config.physics.body_distribution_sphere_radius_multiplier,
        config.physics.body_distribution_min_distance,
    );
    let min_temp = config.rendering.min_temperature;
    let max_temp = config.rendering.max_temperature;
    let min_radius = config.physics.min_body_radius;
    let max_radius = config.physics.max_body_radius;
    let bloom_intensity = config.rendering.bloom_intensity;
    let saturation_intensity = config.rendering.saturation_intensity;

    let mut spawn_data = Vec::with_capacity(body_count);

    for _ in 0..body_count {
        let position = utils::math::random_unit_vector(rng) * body_distribution_sphere_radius;
        let transform = Transform::from_translation(position.as_vec3());
        let radius =
            rng.random_range(config.physics.min_body_radius..=config.physics.max_body_radius);
        let mesh = meshes.add(Sphere::new(radius as f32));

        let temperature =
            min_temp + (max_temp - min_temp) * (max_radius - radius) / (max_radius - min_radius);
        let material = utils::color::emissive_material_for_temp(
            materials,
            temperature,
            bloom_intensity,
            saturation_intensity,
        );

        spawn_data.push((
            transform,
            Collider::sphere(radius),
            GravityScale(0.0),
            RigidBody::Dynamic,
            ExternalForce::ZERO,
            MeshMaterial3d(material),
            Mesh3d(mesh),
        ));
    }

    commands.spawn_batch(spawn_data);
}

#[allow(clippy::type_complexity)]
pub fn rebuild_octree(
    bodies: Query<(Entity, &Transform, &ComputedMass), (With<RigidBody>, Changed<Transform>)>,
    mut octree: ResMut<resources::GravitationalOctree>,
) {
    if bodies.is_empty() {
        return;
    }

    octree.build(
        bodies
            .iter()
            .map(|(entity, transform, mass)| physics::octree::OctreeBody {
                entity,
                position: Vector::from(transform.translation),
                mass: mass.value(),
            }),
    );
}

#[allow(clippy::type_complexity)]
pub fn apply_gravitation_octree(
    g: Res<resources::GravitationalConstant>,
    octree: Res<resources::GravitationalOctree>,
    mut bodies: Query<
        (Entity, &Transform, &ComputedMass, &mut ExternalForce),
        (With<RigidBody>, Changed<Transform>),
    >,
) {
    bodies
        .par_iter_mut()
        .for_each(|(entity, transform, mass, mut external_force)| {
            external_force.set_force(octree.calculate_force(
                &physics::octree::OctreeBody {
                    entity,
                    position: Vector::from(transform.translation),
                    mass: mass.value(),
                },
                **g,
            ));
        });
}

pub fn counteract_barycentric_drift(
    mut bodies: Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
    mut barycenter: ResMut<resources::Barycenter>,
) {
    let (weighted_positions, total_mass): (Vector, Scalar) = bodies
        .iter()
        .map(|(transform, mass)| (Vector::from(transform.translation), mass.value()))
        .fold((Vector::ZERO, 0.0), |(pos_acc, mass_acc), (pos, mass)| {
            (pos_acc + pos * mass, mass_acc + mass)
        });

    let updated_barycenter = weighted_positions / total_mass;

    if total_mass.abs() <= Scalar::EPSILON {
        return;
    }

    if !updated_barycenter.is_finite() {
        return;
    }

    let Some(barycenter) = **barycenter else {
        **barycenter = Some(updated_barycenter);
        return;
    };

    let barycentric_drift = updated_barycenter - barycenter;

    if barycentric_drift.length_squared().abs() <= Scalar::EPSILON {
        return;
    }

    bodies.iter_mut().for_each(|(mut transform, _)| {
        transform.translation += -barycentric_drift.as_vec3();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::SystemState;

    fn create_test_world() -> World {
        let mut world = World::new();
        world.insert_resource(resources::Barycenter::default());
        world
    }

    fn create_test_body_with_mass_and_position(
        commands: &mut Commands,
        mass: f64,
        position: Vec3,
    ) -> Entity {
        commands
            .spawn((
                Transform::from_translation(position),
                ComputedMass::new(mass),
                RigidBody::Dynamic,
            ))
            .id()
    }

    #[test]
    fn test_counteract_barycentric_drift_initial_barycenter_setting() {
        let mut world = create_test_world();
        let mut system_state: SystemState<(
            Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
            ResMut<resources::Barycenter>,
        )> = SystemState::new(&mut world);

        // Create a single body at position (1, 2, 3) with mass 5.0
        {
            let mut commands = world.commands();
            create_test_body_with_mass_and_position(&mut commands, 5.0, Vec3::new(1.0, 2.0, 3.0));
        }
        world.flush();

        // Get barycenter and ensure it's not set
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        assert!(barycenter.is_none());

        // Run the system
        counteract_barycentric_drift(bodies, barycenter);

        // Check that barycenter was set to the body's position
        let barycenter = world.resource::<resources::Barycenter>();
        assert!(barycenter.is_some());
        let barycenter_pos = barycenter.unwrap();
        assert!((barycenter_pos.x - 1.0).abs() < Scalar::EPSILON);
        assert!((barycenter_pos.y - 2.0).abs() < Scalar::EPSILON);
        assert!((barycenter_pos.z - 3.0).abs() < Scalar::EPSILON);
    }

    #[test]
    fn test_counteract_barycentric_drift_multiple_bodies() {
        let mut world = create_test_world();
        let mut system_state: SystemState<(
            Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
            ResMut<resources::Barycenter>,
        )> = SystemState::new(&mut world);

        // Create two bodies: mass 2.0 at (0,0,0) and mass 4.0 at (3,0,0)
        // Expected barycenter: (2*0 + 4*3)/(2+4) = 12/6 = 2.0 on x-axis
        {
            let mut commands = world.commands();
            create_test_body_with_mass_and_position(&mut commands, 2.0, Vec3::new(0.0, 0.0, 0.0));
            create_test_body_with_mass_and_position(&mut commands, 4.0, Vec3::new(3.0, 0.0, 0.0));
        }
        world.flush();

        // Get barycenter and ensure it's not set
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        assert!(barycenter.is_none());

        // Run the system
        counteract_barycentric_drift(bodies, barycenter);

        // Verify initial barycenter is set correctly
        let barycenter = world.resource::<resources::Barycenter>();
        assert!(barycenter.is_some());
        let initial_barycenter = barycenter.unwrap();
        assert!((initial_barycenter.x - 2.0).abs() < Scalar::EPSILON);
        assert!((initial_barycenter.y - 0.0).abs() < Scalar::EPSILON);
        assert!((initial_barycenter.z - 0.0).abs() < Scalar::EPSILON);
    }

    #[test]
    fn test_counteract_barycentric_drift_correction() {
        let mut world = create_test_world();
        let mut system_state: SystemState<(
            Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
            ResMut<resources::Barycenter>,
        )> = SystemState::new(&mut world);

        // Create a body and set initial barycenter
        {
            let mut commands = world.commands();
            create_test_body_with_mass_and_position(&mut commands, 1.0, Vec3::new(0.0, 0.0, 0.0));
        }
        world.flush();

        // Set an initial barycenter manually
        world.resource_mut::<resources::Barycenter>().0 = Some(Vector::new(0.0, 0.0, 0.0));

        // Move the body to create drift
        {
            let mut query = world.query::<&mut Transform>();
            for mut transform in query.iter_mut(&mut world) {
                transform.translation = Vec3::new(2.0, 0.0, 0.0);
            }
        }

        // Store original position for comparison
        let _original_positions: Vec<Vec3> = {
            let mut query = world.query::<&Transform>();
            query.iter(&world).map(|t| t.translation).collect()
        };

        // Run the system to correct drift
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        counteract_barycentric_drift(bodies, barycenter);

        // Check that bodies were moved back to counteract drift
        let corrected_positions: Vec<Vec3> = {
            let mut query = world.query::<&Transform>();
            query.iter(&world).map(|t| t.translation).collect()
        };

        // The body should have been moved back by the drift amount
        // Original barycenter was (0,0,0), new would be (2,0,0), so drift is (2,0,0)
        // Body should be moved by -drift = (-2,0,0), so final position should be (0,0,0)
        assert!((corrected_positions[0].x - 0.0).abs() < f32::EPSILON);
        assert!((corrected_positions[0].y - 0.0).abs() < f32::EPSILON);
        assert!((corrected_positions[0].z - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_counteract_barycentric_drift_zero_mass() {
        let mut world = create_test_world();
        let mut system_state: SystemState<(
            Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
            ResMut<resources::Barycenter>,
        )> = SystemState::new(&mut world);

        // Create a body with zero mass
        {
            let mut commands = world.commands();
            create_test_body_with_mass_and_position(&mut commands, 0.0, Vec3::new(1.0, 2.0, 3.0));
        }
        world.flush();

        let original_barycenter = **world.resource::<resources::Barycenter>();

        // Run the system
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        counteract_barycentric_drift(bodies, barycenter);

        // Barycenter should remain unchanged due to zero mass early return
        let final_barycenter = **world.resource::<resources::Barycenter>();
        assert_eq!(original_barycenter, final_barycenter);
    }

    #[test]
    fn test_counteract_barycentric_drift_no_bodies() {
        let mut world = create_test_world();
        let mut system_state: SystemState<(
            Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
            ResMut<resources::Barycenter>,
        )> = SystemState::new(&mut world);

        let original_barycenter = **world.resource::<resources::Barycenter>();

        // Run the system with no bodies
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        counteract_barycentric_drift(bodies, barycenter);

        // Barycenter should remain unchanged
        let final_barycenter = **world.resource::<resources::Barycenter>();
        assert_eq!(original_barycenter, final_barycenter);
    }

    #[test]
    fn test_counteract_barycentric_drift_small_drift() {
        let mut world = create_test_world();
        let mut system_state: SystemState<(
            Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
            ResMut<resources::Barycenter>,
        )> = SystemState::new(&mut world);

        // Create a body
        {
            let mut commands = world.commands();
            create_test_body_with_mass_and_position(&mut commands, 1.0, Vec3::new(0.0, 0.0, 0.0));
        }
        world.flush();

        // Set initial barycenter
        world.resource_mut::<resources::Barycenter>().0 = Some(Vector::new(0.0, 0.0, 0.0));

        // Move body by a very small amount (less than epsilon threshold)
        let tiny_offset = Scalar::EPSILON.sqrt() * 0.5; // Much smaller than epsilon
        {
            let mut query = world.query::<&mut Transform>();
            for mut transform in query.iter_mut(&mut world) {
                transform.translation = Vec3::new(tiny_offset as f32, 0.0, 0.0);
            }
        }

        let original_positions: Vec<Vec3> = {
            let mut query = world.query::<&Transform>();
            query.iter(&world).map(|t| t.translation).collect()
        };

        // Run the system
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        counteract_barycentric_drift(bodies, barycenter);

        // Positions should remain unchanged due to small drift threshold
        let final_positions: Vec<Vec3> = {
            let mut query = world.query::<&Transform>();
            query.iter(&world).map(|t| t.translation).collect()
        };

        assert_eq!(original_positions, final_positions);
    }

    #[test]
    fn test_counteract_barycentric_drift_non_finite_barycenter() {
        let mut world = create_test_world();
        let mut system_state: SystemState<(
            Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
            ResMut<resources::Barycenter>,
        )> = SystemState::new(&mut world);

        // Create a body at a position that would create non-finite barycenter when divided by zero
        // This is tricky to test directly, but we can test the is_finite check by creating
        // a scenario where the calculation might produce NaN or infinity
        {
            let mut commands = world.commands();
            create_test_body_with_mass_and_position(
                &mut commands,
                f64::INFINITY,
                Vec3::new(1.0, 0.0, 0.0),
            );
        }
        world.flush();

        let _original_barycenter = **world.resource::<resources::Barycenter>();

        // Run the system
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        counteract_barycentric_drift(bodies, barycenter);

        // The system should handle non-finite values gracefully
        // In this case, it should either set a finite barycenter or leave it unchanged
        let final_barycenter = **world.resource::<resources::Barycenter>();
        if let Some(barycenter_val) = final_barycenter {
            assert!(barycenter_val.is_finite(), "Barycenter should be finite");
        }
    }

    #[test]
    fn test_counteract_barycentric_drift_complex_scenario() {
        let mut world = create_test_world();
        let mut system_state: SystemState<(
            Query<(&mut Transform, &ComputedMass), With<RigidBody>>,
            ResMut<resources::Barycenter>,
        )> = SystemState::new(&mut world);

        // Create multiple bodies with different masses and positions
        {
            let mut commands = world.commands();
            create_test_body_with_mass_and_position(&mut commands, 1.0, Vec3::new(-2.0, 0.0, 0.0));
            create_test_body_with_mass_and_position(&mut commands, 2.0, Vec3::new(1.0, 0.0, 0.0));
            create_test_body_with_mass_and_position(&mut commands, 3.0, Vec3::new(2.0, 1.0, -1.0));
        }
        world.flush();

        // Run system first time to establish initial barycenter
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        counteract_barycentric_drift(bodies, barycenter);

        let initial_barycenter = world.resource::<resources::Barycenter>().unwrap();

        // Move all bodies by the same offset to create uniform drift
        let drift_offset = Vec3::new(0.5, -0.3, 0.2);
        {
            let mut query = world.query::<&mut Transform>();
            for mut transform in query.iter_mut(&mut world) {
                transform.translation += drift_offset;
            }
        }

        // Run system again to correct the drift
        let (bodies, barycenter) = system_state.get_mut(&mut world);
        counteract_barycentric_drift(bodies, barycenter);

        // After correction, the barycenter resource should still be the original barycenter
        // because the system corrects drift by moving bodies back
        let final_barycenter = world.resource::<resources::Barycenter>().unwrap();

        // The barycenter resource should remain the original barycenter
        let barycenter_diff = (final_barycenter - initial_barycenter).length();
        assert!(
            barycenter_diff < Scalar::EPSILON * 10.0,
            "Barycenter resource should remain close to original position after drift correction"
        );
    }
}
