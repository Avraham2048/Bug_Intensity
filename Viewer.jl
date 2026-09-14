# ============================================================================
# Viewer.jl - מערכת תצוגה גרפית אינטראקטיבית ו-Replay
# ============================================================================

using GLMakie
using GLMakie: xlims!, ylims!
using Printf
using DataFrames
using CSV
using Random

function state_display_label(t::TrajectoryPoint)
    if t.reached
        return "★ Arrived"
    elseif t.anti_enemy_mode == :Paused
        return t.confirm_counter > 0 ? @sprintf("⏸ Confirm %d/%d", t.confirm_counter, ENEMY_CONFIRM_STEPS) : "⏸ Paused"
    elseif t.anti_enemy_mode == :InertialMode
        return "⤇ Inertial"
    elseif t.state == :MoveToTower
        return "→ Moving"
    else
        return "~ Following"
    end
end

function run_viewer_simulation(env::SimulationEnv, max_steps::Int=300)
    strategies = [:Naive, :Cautious, :Inertial]
    states = RobotState[]

    for strat in strategies
        robot = Robot(Vec2(2.0, 1.0), 0.0, :MoveToTower, 0.0)
        u_ori!(robot, env, 0)
        init_i = perceived_intensity(robot.pos, env, 0)
        push!(states, RobotState(
            robot, strat, FollowState(init_i, init_i, init_i, 0, false, false),
            init_i, init_i, :Normal, 0.0, :MoveToTower, 0, false, 0, false, :u_ori
        ))
    end

    trajectories = [TrajectoryPoint[] for _ in 1:3]
    step = 0
    for (i, rs) in enumerate(states)
        push!(trajectories[i], make_traj_point(rs, env, step))
    end

    while step < max_steps && !all(rs -> rs.reached, states)
        step += 1
        for (i, rs) in enumerate(states)
            update_robot_step!(rs, env, step)
            push!(trajectories[i], make_traj_point(rs, env, step))
        end
    end
    return trajectories, [rs.reached for rs in states], [rs.reached_step for rs in states]
end

function launch_viewer(env::SimulationEnv, trajs, reached_flags, reached_steps)
    n_frames = length(trajs[1])
    strategy_names = ["R1 Naive", "R2 Cautious", "R3 Inertial"]
    robot_colors = [RGBf(0.3, 0.79, 0.94), RGBf(1.0, 0.6, 0.2), RGBf(0.3, 0.9, 0.3)]

    all_pts = [[Point2f(t.x, t.y) for t in traj] for traj in trajs]

    fig = Figure(size=(1050, 1050), backgroundcolor=RGBf(0.059, 0.059, 0.137))
    Label(fig[1, 1], " 3-Robot Adversary Tower Comparative Replay", fontsize=20, color=RGBf(0.92, 0.94, 0.96), font=:bold)

    ax = Axis(fig[3, 1], aspect=DataAspect(), backgroundcolor=RGBf(0.086, 0.094, 0.165),
        xgridvisible=false, ygridvisible=false, xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false)
    xlims!(ax, -1.0, 21.0)
    ylims!(ax, -1.0, 21.0)
    deactivate_interaction!(ax, :scrollzoom)
    deactivate_interaction!(ax, :rectanglezoom)

    controls = GridLayout(fig[4, 1], tellwidth=false)
    sl = Slider(fig[5, 1], range=1:n_frames, startvalue=1, color_active=RGBf(0.26, 0.36, 0.90), linewidth=14)
    frame_idx = sl.value
    speed_val = Observable(1.0)
    is_playing = Observable(false)
    timer_ref = Ref{Union{Timer,Nothing}}(nothing)
    programmatic_update = Ref(false)

    info_panel = GridLayout(fig[2, 1], tellwidth=false)
    common_info = @lift @sprintf("Step %d / %d   |   Adversary: %s", trajs[1][$frame_idx].step, n_frames - 1, trajs[1][$frame_idx].enemy_active ? "ON" : "OFF")
    Label(info_panel[1, 1], common_info, fontsize=12, color=RGBf(0.60, 0.64, 0.76))

    btn_prev = Button(controls[1, 1], label="◀ -1 Step", width=90)
    btn_play = Button(controls[1, 2], label=@lift($is_playing ? "|| Pause" : "> Play"), width=90)
    btn_next = Button(controls[1, 3], label="+1 Step ▶", width=90)
    btn_1x = Button(controls[1, 4], label="1x", width=50)
    btn_5x = Button(controls[1, 5], label="5x", width=50)

    poly!(ax, Rect(0, 0, 20.0, 20.0), color=RGBf(0.086, 0.094, 0.165), strokecolor=RGBf(0.30, 0.32, 0.45), strokewidth=2)
    for obs in env.obstacles
        poly!(ax, Rect(obs.xmin, obs.ymin, obs.xmax - obs.xmin, obs.ymax - obs.ymin), color=RGBf(0.27, 0.29, 0.40))
    end

    scatter!(ax, [env.tower.x], [env.tower.y], marker=:star5, markersize=26, color=RGBf(1.0, 0.40, 0.40))
    scatter!(ax, [env.enemy_tower.x], [env.enemy_tower.y], marker=:star5, markersize=26, color=RGBf(0.0, 1.0, 0.4))
    scatter!(ax, [env.enemy_tower.x], [env.enemy_tower.y], markersize=80, marker=:circle, color=RGBAf(0.0, 1.0, 0.4, 0.18), visible=@lift(trajs[1][$frame_idx].enemy_active))

    # --- 1. Draw Trajectories ---
    for i in 1:3
        let pts = all_pts[i], col = robot_colors[i]
            lines!(ax, @lift(pts[1:$frame_idx]), color=RGBAf(col.r, col.g, col.b, 0.35), linewidth=2)
        end
    end

    # --- 2. Draw Text Info, Robot Body, and Sensors ---
    for i in 1:3
        let traj = trajs[i], col = robot_colors[i], name = strategy_names[i], idx = i, r_size = ROBOT_SIZE

            # --- A. Text Info Panel ---
            info_obs = @lift begin
                t = traj[$frame_idx]
                reached_str = reached_flags[idx] && t.reached ? "   ✓ Step $(reached_steps[idx])" : ""

                combat_mode = t.anti_enemy_mode == :Normal ? "Regular" : String(t.anti_enemy_mode)
                prim_action = String(t.action)

                @sprintf("%s | Mode: %-8s | Action: %-7s | I = %.3f | Hi = %.3f %s",
                    name, combat_mode, prim_action, min(t.intensity_val, 99.9), min(t.high_intensity_memory, 99.9), reached_str)
            end
            Label(info_panel[1+i, 1], info_obs, fontsize=13, color=col, font=:bold)

            # --- B. Robot Aura ---
            scatter!(ax, @lift([traj[$frame_idx].x]), @lift([traj[$frame_idx].y]), markersize=28,
                color=@lift(RGBAf(col.r, col.g, col.b, traj[$frame_idx].reached ? 0.05 : 0.15)))

            # --- C. Robot Body (Square) ---
            robot_verts = @lift begin
                t = traj[$frame_idx]
                h = r_size / 2
                ca, sa = cos(t.angle), sin(t.angle)
                Point2f[(t.x + cx * ca - cy * sa, t.y + cx * sa + cy * ca) for (cx, cy) in ((-h, -h), (h, -h), (h, h), (-h, h))]
            end
            body_col = @lift begin
                if traj[$frame_idx].reached
                    RGBAf(col.r, col.g, col.b, 0.35)
                elseif String(traj[$frame_idx].anti_enemy_mode) == "Paused"
                    RGBAf(col.r, col.g, col.b, 0.55)
                else
                    RGBAf(col.r, col.g, col.b, 1.0)
                end
            end
            poly!(ax, robot_verts, color=body_col, strokecolor=:white, strokewidth=1.5)

            # --- D. Direction Line (Nose) ---
            lines!(ax, @lift([traj[$frame_idx].x, traj[$frame_idx].x + (r_size * 0.55) * cos(traj[$frame_idx].angle)]),
                @lift([traj[$frame_idx].y, traj[$frame_idx].y + (r_size * 0.55) * sin(traj[$frame_idx].angle)]), color=:white, linewidth=1.5)

            # --- E. Front Sensor ---
            scatter!(ax, @lift([traj[$frame_idx].x + (r_size / 2) * cos(traj[$frame_idx].angle)]),
                @lift([traj[$frame_idx].y + (r_size / 2) * sin(traj[$frame_idx].angle)]),
                markersize=7, color=@lift(traj[$frame_idx].front_sensor ? RGBf(0.10, 0.92, 0.30) : RGBf(0.40, 0.42, 0.48)), strokecolor=:white, strokewidth=0.5)

            # --- F. Left Sensor ---
            scatter!(ax, @lift([traj[$frame_idx].x - (r_size / 2) * sin(traj[$frame_idx].angle)]),
                @lift([traj[$frame_idx].y + (r_size / 2) * cos(traj[$frame_idx].angle)]),
                markersize=7, color=@lift(traj[$frame_idx].left_sensor ? RGBf(0.10, 0.92, 0.30) : RGBf(0.40, 0.42, 0.48)), strokecolor=:white, strokewidth=0.5)
        end
    end

    banner_vis = @lift $frame_idx >= n_frames
    text!(ax, Point2f(10.0, 12.8), text="Simulation Complete", fontsize=22, color=RGBAf(1.0, 0.84, 0.0, 0.92), align=(:center, :center), font=:bold, visible=banner_vis)

    function stop_p!()
        if timer_ref[] !== nothing
            close(timer_ref[])
            timer_ref[] = nothing
        end
    end

    function start_p!()
        stop_p!()
        curr_f = Float64(frame_idx[])
        timer_ref[] = Timer(0.033; interval=0.033) do t
            curr_f = min(curr_f + speed_val[], n_frames)
            nxt = round(Int, curr_f)
            if nxt > frame_idx[]
                programmatic_update[] = true
                set_close_to!(sl, nxt)
                programmatic_update[] = false
            end
            if nxt >= n_frames
                is_playing[] = false
                stop_p!()
            end
        end
    end

    function step_by!(delta::Int)
        is_playing[] = false
        stop_p!()
        new_val = clamp(frame_idx[] + delta, 1, n_frames)
        programmatic_update[] = true
        set_close_to!(sl, new_val)
        programmatic_update[] = false
    end

    on(btn_play.clicks) do _
        if is_playing[]
            is_playing[] = false
            stop_p!()
        else
            if frame_idx[] >= n_frames
                programmatic_update[] = true
                set_close_to!(sl, 1)
                programmatic_update[] = false
            end
            is_playing[] = true
            start_p!()
        end
    end

    on(btn_prev.clicks) do _
        step_by!(-1)
    end

    on(btn_next.clicks) do _
        step_by!(1)
    end

    on(btn_1x.clicks) do _
        speed_val[] = 1.0
    end
    on(btn_5x.clicks) do _
        speed_val[] = 5.0
    end
    on(frame_idx) do _
        if !programmatic_update[] && is_playing[]
            is_playing[] = false
            stop_p!()
        end
    end

    # control with left and right keyboard Button
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            if event.key == Keyboard.right
                step_by!(1)
            elseif event.key == Keyboard.left
                step_by!(-1)
            elseif event.key == Keyboard.space
                btn_play.clicks[] = btn_play.clicks[] + 1
            end
        end
    end

    rowsize!(fig.layout, 1, Fixed(32))
    rowsize!(fig.layout, 2, Fixed(100))
    screen = display(fig)
    wait(screen)
    return fig
end

function watch_replay(run_id::Int, filename::String="simulation_results.csv")
    if !isfile(filename)
        println("Error: CSV file not found. Run batch simulation first.")
        return
    end
    df = CSV.read(filename, DataFrame; types=Dict(:Seed => String, :RunID => Int))
    target_rows = df[df.RunID.==run_id, :]
    if nrow(target_rows) == 0
        println("Error: RunID $run_id not found in $filename.")
        return
    end

    target_seed = parse(UInt64, target_rows.Seed[1])
    println("Found RunID $run_id. Reconstructing environment from Seed...")

    Random.seed!(target_seed)
    env = generate_random_env()

    println("Running Simulation Viewer...")
    trajs, reached_flags, reached_steps = run_viewer_simulation(env, 400)

    Random.seed!()
    launch_viewer(env, trajs, reached_flags, reached_steps)
end

function run_direct_simulation()
    println("Generating random environment and running simulation viewer...")
    env = generate_random_env()
    trajs, reached_flags, reached_steps = run_viewer_simulation(env, 400)
    launch_viewer(env, trajs, reached_flags, reached_steps)
end
