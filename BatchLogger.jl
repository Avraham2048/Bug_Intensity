# ============================================================================
# BatchLogger.jl - מערכת הרצות ושמירת נתונים (ללא תצוגה)
# ============================================================================

using DataFrames
using CSV
using Random

function run_single_batch_sim(env::SimulationEnv, max_steps::Int=400)
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

    step = 0
    while step < max_steps && !all(rs -> rs.reached, states)
        step += 1
        for rs in states
            update_robot_step!(rs, env, step)
        end
    end

    results = Dict{Symbol,NamedTuple}()
    for rs in states
        results[rs.strategy] = (success=rs.reached, steps=rs.reached ? rs.reached_step : max_steps)
    end
    return results
end

function run_batch_simulations(n_runs::Int=100, filename::String="simulation_results.csv")
    println("Starting fast batch simulation of $n_runs environments (No GUI)...")
    df = DataFrame(RunID=Int[], Seed=String[], Strategy=String[], Success=Bool[], Steps=Int[])

    for run_id in 1:n_runs
        if run_id % 20 == 0
            println("  Completed $run_id / $n_runs...")
        end
        current_seed = rand(UInt64)
        Random.seed!(current_seed)
        env = generate_random_env()
        sim_results = run_single_batch_sim(env)
        for (strat, res) in sim_results
            push!(df, (run_id, string(current_seed), String(strat), res.success, res.steps))
        end
        Random.seed!()
    end

    CSV.write(filename, df)
    println("Batch simulation finished. Results saved to $filename")
end

# ============================================================================
# export_run_details for Debugging
# ============================================================================
function export_run_details(run_id::Int, input_csv::String="simulation_results.csv", output_csv::String="debug_run_$(run_id).csv"; max_steps::Int=300)
    if !isfile(input_csv)
        println("Error: $input_csv not found. Run run_batch_simulations first.")
        return
    end

    df = CSV.read(input_csv, DataFrame; types=Dict(:Seed => String, :RunID => Int))
    target_rows = df[df.RunID.==run_id, :]

    if nrow(target_rows) == 0
        println("Error: RunID $run_id not found in $input_csv.")
        return
    end

    seed = parse(UInt64, target_rows.Seed[1])
    println("Exporting RunID $run_id (Seed: $seed) to $output_csv...")

    Random.seed!(seed)
    env = generate_random_env()

    # הרצת הסימולציה עם מעקב מלא
    trajs, reached_flags, reached_steps = run_viewer_simulation(env, max_steps)
    Random.seed!()

    strategy_names = ["Naive", "Cautious", "Inertial"]

    detailed_df = DataFrame(
        Step=Int[],
        RobotID=Int[],
        Strategy=String[],
        X=Float64[],
        Y=Float64[],
        AngleDeg=Float64[],
        State=String[],
        Action=String[],
        Intensity=Float64[],
        HighIntensityMem=Float64[],
        EnemyActive=Bool[],
        AntiEnemyMode=String[],
        ConfirmCounter=Int[],
        FrontSensor=Bool[],
        LeftSensor=Bool[],
        Reached=Bool[]
    )

    for (r_idx, traj) in enumerate(trajs)
        for pt in traj
            push!(detailed_df, (
                pt.step,
                r_idx,
                strategy_names[r_idx],
                round(pt.x, digits=4),
                round(pt.y, digits=4),
                round(rad2deg(pt.angle), digits=1),
                String(pt.state),
                String(pt.action),
                round(pt.intensity_val, digits=4),
                round(pt.high_intensity_memory, digits=4),
                pt.enemy_active,
                String(pt.anti_enemy_mode),
                pt.confirm_counter,
                pt.front_sensor,
                pt.left_sensor,
                pt.reached
            ))
        end
    end

    CSV.write(output_csv, detailed_df)
    println("✓ Detailed trajectory successfully saved to: $output_csv")
end
