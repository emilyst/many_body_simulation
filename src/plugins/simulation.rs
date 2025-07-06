use crate::config;
use crate::physics;
use crate::resources;
use crate::states;
use crate::systems;
#[cfg(feature = "diagnostics")]
use bevy::ecs::schedule::LogLevel;
#[cfg(feature = "diagnostics")]
use bevy::ecs::schedule::ScheduleBuildSettings;
use bevy::prelude::*;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum SimulationSet {
    Loading,
    Input,
    UI,
    Camera,
    Visualization,
    Configuration,
}

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        let config = config::SimulationConfig::load_from_user_config();

        app.insert_resource(config.clone());
        app.insert_resource(resources::SharedRng::from_optional_seed(
            config.physics.initial_seed,
        ));
        app.insert_resource(resources::GravitationalConstant(
            config.physics.gravitational_constant,
        ));
        app.insert_resource(resources::BodyCount(config.physics.body_count));
        app.init_resource::<resources::Barycenter>();
        app.insert_resource(resources::GravitationalOctree::new(
            physics::octree::OptimizedOctree::new(
                config.physics.octree_theta,
                config.physics.force_calculation_min_distance,
                config.physics.force_calculation_max_force,
            )
            .with_leaf_threshold(config.physics.octree_leaf_threshold),
        ));
        app.insert_resource(resources::OctreeVisualizationSettings {
            enabled: false,
            ..default()
        });
        app.init_resource::<resources::BarycenterGizmoVisibility>();
        app.init_resource::<resources::LoadingProgress>();

        app.add_event::<systems::simulation_actions::RestartSimulationEvent>();
        app.add_event::<systems::simulation_actions::ToggleOctreeVisualizationEvent>();
        app.add_event::<systems::simulation_actions::ToggleBarycenterGizmoVisibilityEvent>();
        app.add_event::<systems::simulation_actions::TogglePauseSimulationEvent>();

        #[cfg(feature = "diagnostics")]
        app.edit_schedule(FixedUpdate, |schedule| {
            schedule.set_build_settings(ScheduleBuildSettings {
                ambiguity_detection: LogLevel::Warn,
                ..default()
            });
        });

        app.configure_sets(
            FixedUpdate,
            (
                systems::physics::PhysicsSet::BuildOctree,
                systems::physics::PhysicsSet::ApplyForces,
            )
                .chain(),
        );

        app.configure_sets(
            Update,
            (
                SimulationSet::Loading,
                (
                    SimulationSet::Input,
                    SimulationSet::UI,
                    SimulationSet::Visualization,
                ), // These can run in parallel
                SimulationSet::Camera, // Camera follows after input/UI for responsiveness
            )
                .chain(),
        );

        app.add_systems(Startup, systems::camera::spawn_camera);
        app.add_systems(
            OnEnter(states::AppState::Loading),
            (
                systems::loading::setup_loading_screen,
                systems::loading::start_loading_process,
            ),
        );
        app.add_systems(
            FixedUpdate,
            (
                systems::physics::rebuild_octree
                    .in_set(systems::physics::PhysicsSet::BuildOctree)
                    .run_if(
                        in_state(states::AppState::Running).or(in_state(states::AppState::Paused)),
                    ),
                systems::physics::apply_gravitation_octree
                    .in_set(systems::physics::PhysicsSet::ApplyForces)
                    .run_if(in_state(states::AppState::Running)),
                systems::physics::counteract_barycentric_drift.run_if(
                    in_state(states::AppState::Running).or(in_state(states::AppState::Paused)),
                ),
            )
                .chain(),
        );
        app.add_systems(
            Update,
            (
                systems::loading::update_loading_progress
                    .run_if(in_state(states::AppState::Loading)),
                systems::loading::advance_loading_step
                    .run_if(in_state(states::LoadingState::InitializingConfig)),
                systems::loading::spawn_bodies_async
                    .run_if(in_state(states::LoadingState::SpawningBodies)),
                systems::loading::finalize_loading
                    .run_if(in_state(states::LoadingState::BuildingOctree)),
                systems::loading::setup_ui_after_loading
                    .run_if(in_state(states::LoadingState::SettingUpUI)),
                systems::loading::complete_loading.run_if(in_state(states::AppState::Running)),
            )
                .in_set(SimulationSet::Loading),
        );
        app.add_systems(
            Update,
            systems::camera::draw_barycenter_gizmo
                .in_set(SimulationSet::Camera)
                .run_if(in_state(states::AppState::Running).or(in_state(states::AppState::Paused))),
        );
        app.add_systems(
            Update,
            (
                systems::input::pause_physics_on_space,
                systems::input::restart_simulation_on_n,
                systems::input::toggle_barycenter_gizmo_visibility_on_c,
                systems::input::toggle_octree_visualization,
                systems::simulation_actions::handle_restart_simulation_event,
                systems::simulation_actions::handle_toggle_octree_visualization_event,
                systems::simulation_actions::handle_toggle_barycenter_gizmo_visibility_event,
                systems::simulation_actions::handle_toggle_pause_simulation_event,
            )
                .in_set(SimulationSet::Input)
                .run_if(in_state(states::AppState::Running).or(in_state(states::AppState::Paused))),
        );
        app.add_systems(
            Update,
            (
                systems::ui::handle_barycenter_gizmo_button,
                systems::ui::handle_octree_button,
                systems::ui::handle_pause_button,
                systems::ui::handle_restart_button,
                systems::ui::update_barycenter_gizmo_button_text,
                systems::ui::update_octree_button_text,
                systems::ui::update_pause_button_text,
            )
                .in_set(SimulationSet::UI)
                .run_if(in_state(states::AppState::Running).or(in_state(states::AppState::Paused))),
        );
        app.add_systems(
            Update,
            systems::visualization::visualize_octree
                .in_set(SimulationSet::Visualization)
                .run_if(in_state(states::AppState::Running).or(in_state(states::AppState::Paused))),
        );

        app.add_systems(
            Update,
            systems::input::quit_on_escape.in_set(SimulationSet::Input),
        );
    }
}
