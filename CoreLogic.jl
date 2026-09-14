# ============================================================================
# CoreLogic.jl - הגדרות בסיס, לוגיקה ואלגוריתמי הניווט של הרובוטים
# ============================================================================

using LinearAlgebra
using Random

# ======================== Configuration ========================
const GRID_WIDTH = 20.0
const GRID_HEIGHT = 20.0
const STEP_SIZE = 0.3
const COLLISION_RADIUS = 0.3
const TOWER_REACH_DIST = 0.5
const ROBOT_SIZE = 0.7
const SENSOR_RANGE = COLLISION_RADIUS * 1.1

# ── Enemy (Adversary) Tower Configuration ──
const ENEMY_DUTY_CYCLE = 1 / 4
const ENEMY_STRENGTH = 1.0
const ENEMY_CYCLE_BASE = 40

const ANOMALY_THRESHOLD = 2.0
const ENEMY_CONFIRM_STEPS = 3
const INTENSITY_EPS = 1e-8

# ======================== Data Structures ========================
mutable struct Vec2
    x::Float64
    y::Float64
end

struct Obstacle
    xmin::Float64
    ymin::Float64
    xmax::Float64
    ymax::Float64
end

struct SimulationEnv
    width::Float64
    height::Float64
    tower::Vec2
    enemy_tower::Vec2
    obstacles::Vector{Obstacle}
end

mutable struct Robot
    pos::Vec2
    angle::Float64
    state::Symbol
    hit_intensity::Float64
end

mutable struct FollowState
    i_H::Float64
    high_at_hit::Float64
    local_max_i::Float64
    steps_since_max::Int
    has_climbed::Bool
    active::Bool
end

struct TrajectoryPoint
    x::Float64
    y::Float64
    step::Int
    state::Symbol
    angle::Float64
    intensity_val::Float64
    front_sensor::Bool
    left_sensor::Bool
    enemy_active::Bool
    high_intensity_memory::Float64
    anti_enemy_mode::Symbol
    reached::Bool
    confirm_counter::Int
    action::Symbol
end

mutable struct RobotState
    robot::Robot
    strategy::Symbol
    follow::FollowState
    high_intensity_memory::Float64
    prev_perceived::Float64
    anti_enemy::Symbol
    frozen_angle::Float64
    frozen_state::Symbol
    confirm_counter::Int
    reached::Bool
    reached_step::Int
    needs_orientation::Bool
    last_action::Symbol
end

# ======================== Utility Functions ========================
function distance(a::Vec2, b::Vec2)
    sqrt((a.x - b.x)^2 + (a.y - b.y)^2)
end

function normalize_angle(a::Float64)
    while a < 0
        a += 2π
    end
    while a >= 2π
        a -= 2π
    end
    return a
end

function intensity(pos::Vec2, tower::Vec2)
    d = distance(pos, tower)
    return d < 0.01 ? Inf : 1.0 / d
end

function is_enemy_active(step::Int)
    if step < 20
        return false
    end
    active_steps = round(Int, ENEMY_DUTY_CYCLE * ENEMY_CYCLE_BASE)
    return ((step - 20) % ENEMY_CYCLE_BASE) < active_steps
end

function perceived_intensity(pos::Vec2, env::SimulationEnv, step::Int)
    target_signal = intensity(pos, env.tower)
    enemy_signal = 0.0
    if is_enemy_active(step)
        enemy_signal = intensity(pos, env.enemy_tower) * ENEMY_STRENGTH
    end
    return target_signal + enemy_signal
end

# ======================== Collision & Sensors ========================
function collides_with_obstacle(pos::Vec2, env::SimulationEnv)
    for obs in env.obstacles
        cx = clamp(pos.x, obs.xmin, obs.xmax)
        cy = clamp(pos.y, obs.ymin, obs.ymax)
        dx = pos.x - cx
        dy = pos.y - cy
        if dx * dx + dy * dy < COLLISION_RADIUS^2
            return true
        end
    end
    return false
end

function collides_with_boundary(pos::Vec2, env::SimulationEnv)
    pos.x < COLLISION_RADIUS || pos.x > env.width - COLLISION_RADIUS ||
        pos.y < COLLISION_RADIUS || pos.y > env.height - COLLISION_RADIUS
end

function is_valid_pos(pos::Vec2, env::SimulationEnv)
    !collides_with_obstacle(pos, env) && !collides_with_boundary(pos, env)
end

function check_front_sensor(robot::Robot, env::SimulationEnv)
    px = robot.pos.x + SENSOR_RANGE * cos(robot.angle)
    py = robot.pos.y + SENSOR_RANGE * sin(robot.angle)
    return !is_valid_pos(Vec2(px, py), env)
end

function check_left_sensor(robot::Robot, env::SimulationEnv)
    la = robot.angle + π / 2
    px = robot.pos.x + SENSOR_RANGE * cos(la)
    py = robot.pos.y + SENSOR_RANGE * sin(la)
    return !is_valid_pos(Vec2(px, py), env)
end

# ======================== Algorithm Primitives ========================
function u_ori!(robot::Robot, env::SimulationEnv, step::Int=0)
    # Orient purely toward the REAL tower — ignore enemy tower entirely
    # so that the robot always heads precisely toward the correct target.
    dx = env.tower.x - robot.pos.x
    dy = env.tower.y - robot.pos.y
    robot.angle = normalize_angle(atan(dy, dx))
end

function u_fwd!(robot::Robot, env::SimulationEnv)
    nx = robot.pos.x + STEP_SIZE * cos(robot.angle)
    ny = robot.pos.y + STEP_SIZE * sin(robot.angle)
    new_pos = Vec2(nx, ny)
    if is_valid_pos(new_pos, env)
        robot.pos = new_pos
        return true
    end
    return false
end

const FOLLOW_TURN_STEP = deg2rad(5.0)

function next_position(
    robot::Robot,
    angle::Float64
)::Vec2
    return Vec2(
        robot.pos.x + STEP_SIZE * cos(angle),
        robot.pos.y + STEP_SIZE * sin(angle)
    )
end

function can_move_at(
    robot::Robot,
    angle::Float64,
    env::SimulationEnv
)::Bool
    return is_valid_pos(
        next_position(robot, angle),
        env
    )
end

function obstacle_on_left(
    pos::Vec2,
    heading::Float64,
    env::SimulationEnv
)::Bool
    # Several probes approximate a finite-width left bumper.
    for offset_deg in (60.0, 75.0, 90.0, 105.0, 120.0)
        probe_angle = heading + deg2rad(offset_deg)

        probe = Vec2(
            pos.x + SENSOR_RANGE * cos(probe_angle),
            pos.y + SENSOR_RANGE * sin(probe_angle)
        )

        if !is_valid_pos(probe, env)
            return true
        end
    end

    return false
end

function u_fol!(
    robot::Robot,
    env::SimulationEnv
)::Symbol
    original_angle = robot.angle

    # Look for the leftmost valid movement that still keeps
    # an obstacle boundary on the robot's left.
    #
    # Search starts 90 degrees to the left and progresses
    # clockwise. The search is intentionally less than 360
    # degrees to prevent arbitrary direction reversal.
    for i in 0:54
        candidate_angle = normalize_angle(
            original_angle +
            π / 2 -
            i * FOLLOW_TURN_STEP
        )

        if !can_move_at(
            robot,
            candidate_angle,
            env
        )
            continue
        end

        candidate_pos =
            next_position(robot, candidate_angle)

        if obstacle_on_left(
            candidate_pos,
            candidate_angle,
            env
        )
            robot.angle = candidate_angle
            robot.pos = candidate_pos
            return :Moved
        end
    end

    # No movement can currently preserve left contact.
    # Rotate clockwise in place to pass an inward corner.
    robot.angle = normalize_angle(
        original_angle - FOLLOW_TURN_STEP
    )

    return :Turned
end

function update_follow_peak!(
    rs::RobotState,
    cur_i::Float64
)::Bool
    # Record that the robot climbed above the intensity
    # measured when boundary following began.
    if cur_i >
       rs.follow.high_at_hit + INTENSITY_EPS

        rs.follow.has_climbed = true
    end

    # Update the local maximum only after a real movement.
    if cur_i >
       rs.follow.local_max_i + INTENSITY_EPS

        rs.follow.local_max_i = cur_i
        rs.follow.steps_since_max = 0

    elseif cur_i <
           rs.follow.local_max_i - INTENSITY_EPS

        rs.follow.steps_since_max += 1
    end

    has_passed_hit_intensity =
        rs.follow.has_climbed &&
        rs.follow.local_max_i >
            rs.follow.high_at_hit + INTENSITY_EPS

    is_descending =
        rs.follow.steps_since_max >= 1

    return has_passed_hit_intensity &&
           is_descending
end

# ======================== Anomaly Detection ========================
function max_natural_intensity_change(prev_intensity::Float64)
    if prev_intensity < 0.001
        return 0.01
    end
    d = 1.0 / prev_intensity
    if d <= STEP_SIZE * 1.5
        return 1.5
    end
    val = STEP_SIZE / (d * (d - STEP_SIZE))
    return min(val, 1.5)
end

function is_spike_anomaly(cur_i::Float64, prev_i::Float64)
    delta = cur_i - prev_i
    delta > 0 || return false
    return delta > ANOMALY_THRESHOLD * max_natural_intensity_change(prev_i)
end

function is_drop_anomaly(cur_i::Float64, prev_i::Float64)
    delta = prev_i - cur_i
    delta > 0 || return false
    if prev_i < 0.001
        return false
    end
    is_massive_drop = (delta / prev_i) > 0.15
    return is_massive_drop || (delta > 1.2 * max_natural_intensity_change(cur_i))
end

# ======================== Robot Step Logic ========================
function do_normal_step!(
    rs::RobotState,
    env::SimulationEnv,
    step::Int
)
    robot = rs.robot

    if robot.state == :MoveToTower
        if rs.needs_orientation
            u_ori!(robot, env, step)
            rs.needs_orientation = false
        end

        if u_fwd!(robot, env)
            rs.last_action = :u_fwd

        else
            # The target-directed motion encountered an obstacle.
            if rs.strategy == :Naive
                measured_hit_i = perceived_intensity(
                    robot.pos,
                    env,
                    step
                )
            else
                measured_hit_i = intensity(
                    robot.pos,
                    env.tower
                )
            end

            # If MoveToTower began from a detected boundary maximum,
            # robot.hit_intensity contains that consumed maximum.
            #
            # The new threshold must not fall back to the lower
            # post-maximum sample where the collision was detected.
            new_i_H = max(
                measured_hit_i,
                robot.hit_intensity
            )

            robot.hit_intensity = new_i_H
            robot.state = :FollowObstacle

            # Turn right so that the contacted obstacle is on the left.
            robot.angle = normalize_angle(
                robot.angle - π / 2
            )

            rs.follow = FollowState(
                new_i_H,  # i_H: maximum already consumed
                new_i_H,  # threshold for the new follow episode
                measured_hit_i,
                0,
                false,
                true
            )

            rs.last_action = :u_fol_turn
        end

    elseif robot.state == :FollowObstacle
        follow_result = u_fol!(
            robot,
            env
        )

        if follow_result == :Turned
            rs.last_action = :u_fol_turn

            # Rotation is not a new boundary sample.
            return
        end

        rs.last_action = :u_fol

        if rs.strategy == :Naive
            cur_i = perceived_intensity(
                robot.pos,
                env,
                step
            )

            rs.high_intensity_memory = max(
                rs.high_intensity_memory,
                cur_i
            )
        else
            cur_i = intensity(
                robot.pos,
                env.tower
            )
        end

        detected_peak = update_follow_peak!(
            rs,
            cur_i
        )

        if detected_peak
            # Preserve the actual detected maximum before u_ori.
            # The robot may currently be slightly past the maximum
            # because discrete detection requires a descending sample.
            robot.hit_intensity = rs.follow.local_max_i

            robot.state = :MoveToTower
            rs.follow.active = false

            rs.high_intensity_memory = max(
                rs.high_intensity_memory,
                rs.follow.local_max_i
            )

            u_ori!(
                robot,
                env,
                step
            )

            rs.needs_orientation = false
            rs.last_action = :u_ori
        end
    end
end

function update_cautious_step!(
    rs::RobotState,
    env::SimulationEnv,
    step::Int,
    cur_perceived::Float64
)
    if rs.anti_enemy == :Normal

        if is_spike_anomaly(cur_perceived, rs.prev_perceived)
            rs.anti_enemy = :Paused
            rs.frozen_angle = rs.robot.angle
            rs.frozen_state = rs.robot.state
            rs.confirm_counter = 0
            rs.last_action = :Stopped
            return
        end

        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            real_i = intensity(rs.robot.pos, env.tower)

            rs.high_intensity_memory = max(
                rs.high_intensity_memory,
                real_i
            )
        end

        do_normal_step!(rs, env, step)

    elseif rs.anti_enemy == :Paused

        # Freeze the complete navigation state.
        # Do not move and do not modify FollowState.
        rs.last_action = :Stopped

        if cur_perceived < rs.prev_perceived - 0.005
            rs.confirm_counter = 1

        elseif is_spike_anomaly(
            cur_perceived,
            rs.prev_perceived
        )
            rs.confirm_counter = 0

        elseif rs.confirm_counter > 0
            rs.confirm_counter += 1

            if rs.confirm_counter >= ENEMY_CONFIRM_STEPS
                rs.anti_enemy = :Normal
                rs.confirm_counter = 0

                # Position did not change, but restore the explicitly
                # frozen values for consistency.
                rs.robot.state = rs.frozen_state
                rs.robot.angle = rs.frozen_angle

                # Do not modify:
                # rs.follow.high_at_hit
                # rs.follow.local_max_i
                # rs.follow.has_climbed
                # rs.follow.steps_since_max
                # rs.high_intensity_memory
                # rs.needs_orientation

                rs.last_action = :Resumed
            end
        end
    end
end

function perform_inertial_motion!(
    rs::RobotState,
    env::SimulationEnv,
    step::Int
)
    if rs.robot.state == :MoveToTower

        # Continue in the current direction without reorientation.
        moved = u_fwd!(
            rs.robot,
            env
        )

        if moved
            rs.last_action = :u_fwd
        else
            # The inertial trajectory reached an obstacle.
            # Initialize ordinary boundary following.
            hit_i = intensity(
                rs.robot.pos,
                env.tower
            )

            rs.robot.hit_intensity = hit_i
            rs.robot.state = :FollowObstacle
            rs.robot.angle = normalize_angle(
                rs.robot.angle - π / 2
            )

            rs.follow = FollowState(
                hit_i,
                hit_i,
                hit_i,
                0,
                false,
                true
            )

            rs.last_action = :u_fol_turn
        end

    elseif rs.robot.state == :FollowObstacle

        follow_result = u_fol!(
            rs.robot,
            env
        )

        if follow_result == :Turned
            rs.last_action = :u_fol_turn
            return
        end

        rs.last_action = :u_fol

        # The inertial controller ignores the corrupted enemy
        # measurement for local-maximum tracking.
        cur_natural_i = intensity(
            rs.robot.pos,
            env.tower
        )

        detected_peak = update_follow_peak!(
            rs,
            cur_natural_i
        )

        if detected_peak
            rs.anti_enemy = :Normal
            rs.robot.state = :MoveToTower
            rs.follow.active = false

            rs.high_intensity_memory = max(
                rs.high_intensity_memory,
                rs.follow.local_max_i
            )

            u_ori!(
                rs.robot,
                env,
                step
            )

            rs.needs_orientation = false
            rs.last_action = :u_ori
        end
    end
end



function update_inertial_step!(
    rs::RobotState,
    env::SimulationEnv,
    step::Int,
    cur_perceived::Float64
)
    if rs.anti_enemy == :Normal

        if is_spike_anomaly(
            cur_perceived,
            rs.prev_perceived
        )
            rs.anti_enemy = :InertialMode
            rs.frozen_angle = rs.robot.angle
            rs.frozen_state = rs.robot.state

            # Perform one inertial control tick immediately.
            perform_inertial_motion!(
                rs,
                env,
                step
            )

            return
        end

        if is_drop_anomaly(
            cur_perceived,
            rs.prev_perceived
        )
            real_i = intensity(
                rs.robot.pos,
                env.tower
            )

            rs.high_intensity_memory = max(
                rs.high_intensity_memory,
                real_i
            )
        end

        do_normal_step!(
            rs,
            env,
            step
        )

    elseif rs.anti_enemy == :InertialMode

        if is_drop_anomaly(
            cur_perceived,
            rs.prev_perceived
        )
            rs.anti_enemy = :Normal

            # Do not restore frozen_state here.
            # The robot moved during InertialMode, so its current
            # navigation state is the relevant state.

            # Do not reset FollowState.
            # Continue from the state developed during inertial motion.

            real_i = intensity(
                rs.robot.pos,
                env.tower
            )

            rs.high_intensity_memory = max(
                rs.high_intensity_memory,
                real_i
            )

            rs.last_action = :Resumed
            return
        end

        perform_inertial_motion!(
            rs,
            env,
            step
        )
    end
end

function update_robot_step!(rs::RobotState, env::SimulationEnv, step::Int)
    rs.reached && return
    if distance(rs.robot.pos, env.tower) < TOWER_REACH_DIST
        rs.reached = true
        rs.reached_step = step
        rs.last_action = :Arrived
        return
    end
    cur_perceived = perceived_intensity(rs.robot.pos, env, step)

    if rs.strategy == :Naive
        if cur_perceived > rs.high_intensity_memory
            rs.high_intensity_memory = cur_perceived
        end
        do_normal_step!(rs, env, step)
    elseif rs.strategy == :Cautious
        update_cautious_step!(rs, env, step, cur_perceived)
    elseif rs.strategy == :Inertial
        update_inertial_step!(rs, env, step, cur_perceived)
    end

    if rs.strategy != :Naive && rs.anti_enemy == :Normal
        # Use real tower intensity only — never let enemy signal enter high_intensity_memory
        real_i = intensity(rs.robot.pos, env.tower)
        if real_i > rs.high_intensity_memory
            rs.high_intensity_memory = real_i
        end
    end
    rs.prev_perceived = perceived_intensity(rs.robot.pos, env, step)
end

function make_traj_point(rs::RobotState, env::SimulationEnv, step::Int)
    current_action = rs.reached ? :Arrived : rs.last_action

    TrajectoryPoint(
        rs.robot.pos.x, rs.robot.pos.y, step,
        rs.robot.state, rs.robot.angle,
        perceived_intensity(rs.robot.pos, env, step),
        check_front_sensor(rs.robot, env),
        check_left_sensor(rs.robot, env),
        is_enemy_active(step),
        rs.high_intensity_memory,
        rs.anti_enemy,
        rs.reached,
        rs.confirm_counter,
        current_action
    )
end

function generate_random_env()
    tower = Vec2(18.0, 18.0)
    robot_start = Vec2(2.0, 1.0)
    obstacles = Obstacle[]
    for _ in 1:rand(3:9)
        w = rand() * 3.0 + 1.5
        h = rand() * 3.0 + 1.5
        x = rand() * 10.0 + 4.0
        y = rand() * 10.0 + 4.0
        push!(obstacles, Obstacle(x, y, x + w, y + h))
    end
    enemy_pos = Vec2(0.0, 0.0)
    temp_env = SimulationEnv(GRID_WIDTH, GRID_HEIGHT, tower, Vec2(0.0, 0.0), obstacles)
    while true
        cand = Vec2(rand() * (GRID_WIDTH - 2 * COLLISION_RADIUS) + COLLISION_RADIUS,
            rand() * (GRID_HEIGHT - 2 * COLLISION_RADIUS) + COLLISION_RADIUS)
        if is_valid_pos(cand, temp_env) && distance(cand, robot_start) > 6.0 && distance(cand, tower) > 8.0
            enemy_pos = cand
            break
        end
    end
    return SimulationEnv(GRID_WIDTH, GRID_HEIGHT, tower, enemy_pos, obstacles)
end