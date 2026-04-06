module FEMBaBBridge

using LinearAlgebra
using Random
using DynamicPolynomials
using MultivariatePolynomials
using TSSOS
using JuMP
import Ipopt
import MathOptInterface as MOI

include("fem_lp_loader.jl")
using .FEMLPLoader: FEMPolyInstance, box_bound_polynomials, tssos_pop, scale_instance_to_unit_box

export PreparedBaBInput,
       FixationEvent,
       BranchEvent,
       BaBNode,
       BaBResult,
       prepare_bab_input,
       to_original_coordinates,
       to_internal_coordinates,
       node_pop,
       solve_node,
       root_relaxation,
       branchable_coords,
       fixable_coords,
       select_coords_to_fix,
       select_branch_direction_from_scores,
       select_direction,
       select_cut_value,
       snap_to_boundary,
       make_child_bounds,
       build_branch_polynomial,
       branch_once_smoke,
       branch_and_bound,
       branch_and_bound_instance,
       branch_and_bound_fem_instance,
       bab_summary_string,
       node_summary_string,
       path_summary_string,
       print_bab_summary,
       construct_CDKmarg_SVD_local,
       next_step_CDK_local,
       solve_quadratic_inequality

Base.@kwdef struct PreparedBaBInput
    name::String
    vars
    var_names::Vector{String}
    var_lookup::Dict{Any, Int}
    objective
    base_ineqs
    base_eqs
    root_pop
    numeq::Int
    ilb_root::Vector{Float64}
    iub_root::Vector{Float64}
    original_ilb_root::Vector{Float64} = Float64[]
    original_iub_root::Vector{Float64} = Float64[]
    coord_shift::Vector{Float64} = Float64[]
    coord_scale::Vector{Float64} = Float64[]
    scaled_to_unit_box::Bool = false
    branch_idx::Vector{Int}
    epigraph_idx::Union{Nothing, Int}
    bound_encoding::Symbol
end

Base.@kwdef struct FixationEvent
    iteration::Int
    node_id::Int
    var_idx::Int
    var_name::String
    old_lb::Float64
    old_ub::Float64
    pseudo_moment::Float64
    fixed_value::Float64
    marginal_rank::Int = 0
    polynomial
end

Base.@kwdef struct BranchEvent
    iteration::Int
    parent_node_id::Int
    child_side::Symbol
    var_idx::Int
    var_name::String
    parent_lb::Float64
    parent_ub::Float64
    cut_value::Float64
    child_lb::Float64
    child_ub::Float64
    polynomial
end

Base.@kwdef mutable struct BaBNode
    id::Int
    parent_id::Union{Nothing, Int}
    depth::Int
    last_side::Union{Nothing, Symbol} = nothing
    raw_best_lb::Float64 = Inf
    best_lb::Float64 = Inf
    best_ub::Float64 = Inf
    rel_gap::Float64 = Inf
    ub_candidate::Float64 = Inf
    ub_valid::Bool = false
    pop_length::Int = 0
    numeq::Int = 0
    moment_matrix = nothing
    solution::Vector{Float64} = Float64[]
    tssos_normalized::Bool = false
    tssos_objective_scale::Float64 = 1.0
    tssos_min_constraint_scale::Float64 = 1.0
    tssos_max_constraint_scale::Float64 = 1.0
    tssos_report = nothing
    tssos_candidate_report = nothing
    tssos_lb_state::Symbol = :valid
    tssos_lb_reason::String = ""
    tssos_call_seconds::Float64 = 0.0
    tssos_sdp_assemble_seconds::Float64 = 0.0
    tssos_sdp_solve_seconds::Float64 = 0.0
    incumbent_source::Symbol = :none
    fallback_attempted::Bool = false
    fallback_attempts::Int = 0
    ilb::Vector{Float64} = Float64[]
    iub::Vector{Float64} = Float64[]
    branch_history::Vector{BranchEvent} = BranchEvent[]
    fixation_history::Vector{FixationEvent} = FixationEvent[]
    status::Symbol = :active
    status_reason::String = ""
    selected_cut_direction::Union{Nothing, Int} = nothing
    selected_cut_value::Union{Nothing, Float64} = nothing
    selected_coords_to_fix::Vector{Int} = Int[]
    selected_cut_points::Vector{Float64} = Float64[]
    selected_interval_lengths::Vector{Float64} = Float64[]
    feasibility_report = nothing
end

Base.@kwdef struct BaBResult
    name::String
    best_lower_bound::Float64
    best_upper_bound::Float64
    iterations::Int
    lb_hist::Vector{Float64}
    ub_hist::Vector{Float64}
    gap_hist::Vector{Float64}
    nodes::Vector{BaBNode}
    active_node_ids::Vector{Int}
    frontier_node_ids::Vector{Int}
    max_iter_reached::Bool
    time_limit_reached::Bool = false
    elapsed_seconds::Float64 = 0.0
    wall_clock_seconds::Float64 = 0.0
    tssos_sdp_assemble_seconds::Float64 = 0.0
    metadata::Dict{String, Any}
end

function _typed_vector(items::AbstractVector)
    isempty(items) && return Any[]
    T = foldl(promote_type, map(typeof, items))
    return T[item for item in items]
end

function _poly_value(poly, var_lookup::Dict{Any, Int}, point::AbstractVector{<:Real})
    value = 0.0
    for (coeff, mon) in zip(coefficients(poly), monomials(poly))
        term = Float64(coeff)
        for var in variables(mon)
            deg = degree(mon, var)
            deg == 0 && continue
            term *= Float64(point[var_lookup[var]])^deg
        end
        value += term
    end
    return value
end

function _evaluate_polynomial(poly, prob::PreparedBaBInput, point::AbstractVector{<:Real})
    length(prob.vars) == length(point) || error("Variable and point lengths do not match.")
    return _poly_value(poly, prob.var_lookup, point)
end

function to_original_coordinates(prob::PreparedBaBInput, point::AbstractVector{<:Real})
    length(point) == length(prob.vars) || error("Point length does not match number of variables.")
    if !prob.scaled_to_unit_box
        return Float64.(point)
    end
    return prob.coord_shift .+ prob.coord_scale .* Float64.(point)
end

function to_internal_coordinates(prob::PreparedBaBInput, point::AbstractVector{<:Real})
    length(point) == length(prob.vars) || error("Point length does not match number of variables.")
    if !prob.scaled_to_unit_box
        return Float64.(point)
    end

    internal = zeros(Float64, length(point))
    for i in eachindex(point)
        if iszero(prob.coord_scale[i])
            internal[i] = 0.0
        else
            internal[i] = (Float64(point[i]) - prob.coord_shift[i]) / prob.coord_scale[i]
        end
    end
    return internal
end

function prepare_bab_input(inst::FEMPolyInstance;
                           include_box_bounds::Bool = true,
                           bound_encoding::Symbol = :quadratic,
                           scale_to_unit_box::Bool = true)
    working_inst = inst
    coord_shift = copy(inst.lb)
    coord_scale = ones(Float64, length(inst.lb))

    if scale_to_unit_box
        working_inst, coord_shift, coord_scale = scale_instance_to_unit_box(inst)
    end

    root_pop, numeq = tssos_pop(
        working_inst;
        include_box_bounds = include_box_bounds,
        bound_encoding = bound_encoding,
    )
    typed_vars = _typed_vector(working_inst.vars)
    typed_ineqs = _typed_vector(working_inst.inequalities)
    typed_eqs = _typed_vector(working_inst.equalities)
    typed_root_pop = _typed_vector(root_pop)
    var_lookup = Dict{Any, Int}(typed_vars[i] => i for i in eachindex(typed_vars))
    return PreparedBaBInput(
        name = working_inst.name,
        vars = typed_vars,
        var_names = working_inst.var_names,
        var_lookup = var_lookup,
        objective = working_inst.objective,
        base_ineqs = typed_ineqs,
        base_eqs = typed_eqs,
        root_pop = typed_root_pop,
        numeq = numeq,
        ilb_root = copy(working_inst.lb),
        iub_root = copy(working_inst.ub),
        original_ilb_root = copy(inst.lb),
        original_iub_root = copy(inst.ub),
        coord_shift = Float64.(coord_shift),
        coord_scale = Float64.(coord_scale),
        scaled_to_unit_box = scale_to_unit_box,
        branch_idx = copy(working_inst.branch_idx),
        epigraph_idx = working_inst.epigraph_idx,
        bound_encoding = bound_encoding,
    )
end

function node_pop(prob::PreparedBaBInput;
                  ilb::AbstractVector{<:Real} = prob.ilb_root,
                  iub::AbstractVector{<:Real} = prob.iub_root,
                  extra_ineqs::AbstractVector = Any[],
                  extra_eqs::AbstractVector = Any[],
                  include_box_bounds::Bool = true,
                  bound_encoding::Symbol = prob.bound_encoding,
                  branch_encoding::Symbol = bound_encoding)
    length(ilb) == length(prob.vars) || error("ilb has wrong length.")
    length(iub) == length(prob.vars) || error("iub has wrong length.")

    polys = Any[prob.objective]
    append!(polys, prob.base_ineqs)
    append!(polys, extra_ineqs)

    root_bound_eqs = Any[]
    branch_bound_eqs = Any[]
    if include_box_bounds
        root_bound_ineqs, root_bound_eqs = box_bound_polynomials(
            prob.vars,
            prob.ilb_root,
            prob.iub_root;
            encoding = bound_encoding,
        )
        append!(polys, root_bound_ineqs)
        append!(polys, root_bound_eqs)

        tightened_idx = [
            i for i in eachindex(prob.vars)
            if abs(Float64(ilb[i]) - prob.ilb_root[i]) > 1e-10 ||
               abs(Float64(iub[i]) - prob.iub_root[i]) > 1e-10
        ]
        if !isempty(tightened_idx)
            branch_bound_ineqs, branch_bound_eqs = box_bound_polynomials(
                prob.vars[tightened_idx],
                Float64.(ilb[tightened_idx]),
                Float64.(iub[tightened_idx]);
                encoding = branch_encoding,
            )
            append!(polys, branch_bound_ineqs)
            append!(polys, branch_bound_eqs)
        end
    end

    append!(polys, prob.base_eqs)
    append!(polys, extra_eqs)

    return _typed_vector(polys), length(prob.base_eqs) + length(extra_eqs) +
           (include_box_bounds ? length(root_bound_eqs) + length(branch_bound_eqs) : 0)
end

function _objective_value(prob::PreparedBaBInput, sol::AbstractVector{<:Real})
    if !isnothing(prob.epigraph_idx) && prob.objective == prob.vars[prob.epigraph_idx]
        return Float64(sol[prob.epigraph_idx])
    end
    return _evaluate_polynomial(prob.objective, prob, sol)
end

function _poly_scale(poly)
    coeffs = coefficients(poly)
    isempty(coeffs) && return 1.0
    max_abs = maximum(abs.(Float64.(coeffs)))
    return (!isfinite(max_abs) || iszero(max_abs)) ? 1.0 : max_abs
end

function _normalize_polynomials(polys::AbstractVector; normalize::Bool = true)
    isempty(polys) && return (_typed_vector(Any[]), 1.0, 1.0, 1.0)
    if !normalize
        return (_typed_vector(collect(polys)), 1.0, 1.0, 1.0)
    end

    scales = Float64[_poly_scale(poly) for poly in polys]
    normalized = Any[polys[i] / scales[i] for i in eachindex(polys)]
    constraint_scales = length(scales) > 1 ? scales[2:end] : Float64[]
    min_constraint_scale = isempty(constraint_scales) ? 1.0 : minimum(constraint_scales)
    max_constraint_scale = isempty(constraint_scales) ? 1.0 : maximum(constraint_scales)

    return (
        _typed_vector(normalized),
        scales[1],
        min_constraint_scale,
        max_constraint_scale,
    )
end

function _capture_text_output_local(f)
    path = tempname()
    result_ref = Ref{Any}(nothing)
    started_at = time_ns()
    text = open(path, "w+") do io
        redirect_stderr(io) do
            redirect_stdout(io) do
                result_ref[] = f()
            end
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
    rm(path; force = true)
    return result_ref[], text, (time_ns() - started_at) / 1.0e9
end

function _parse_tssos_timing_report(text::AbstractString)
    assemble_seconds = 0.0
    solve_seconds = 0.0

    for m in eachmatch(r"SDP assembling time:\s*([0-9eE+\-\.]+)\s*seconds\.", text)
        parsed = tryparse(Float64, strip(m.captures[1]))
        parsed !== nothing && (assemble_seconds += parsed)
    end
    for m in eachmatch(r"SDP solving time:\s*([0-9eE+\-\.]+)\s*seconds\.", text)
        parsed = tryparse(Float64, strip(m.captures[1]))
        parsed !== nothing && (solve_seconds += parsed)
    end

    return (
        assemble_seconds = assemble_seconds,
        solve_seconds = solve_seconds,
        has_explicit_timing = (assemble_seconds > 0.0) || (solve_seconds > 0.0),
    )
end

function _tssos_text_for_display(text::AbstractString; QUIET::Bool = true)
    !QUIET && return text
    keep = String[]
    for line in split(text, '\n')
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "*********************************** TSSOS") ||
           startswith(s, "TSSOS is launching") ||
           occursin(r"^\-+$", s) ||
           startswith(s, "The clique sizes of varibles:") ||
           occursin(r"^\[[^\]]+\]$", s) ||
           startswith(s, "Assembling the SDP") ||
           startswith(s, "SDP assembling time:") ||
           startswith(s, "Solving the SDP") ||
           startswith(s, "SDP solving time:") ||
           startswith(s, "termination status:") ||
           startswith(s, "solution status:") ||
           startswith(s, "optimum =") ||
           startswith(s, "Found a locally optimal solution by Ipopt") ||
           startswith(s, "The local solver failed refining the solution!") ||
           startswith(s, "The relative optimality gap is:")
            push!(keep, line)
        end
    end
    return isempty(keep) ? text : join(keep, "\n") * "\n"
end

function _parse_tssos_local_report(text::AbstractString)
    failed = occursin("The local solver failed refining the solution!", text)
    succeeded = occursin("Found a locally optimal solution by Ipopt", text)

    reported_ub = nothing
    ub_match = match(r"Found a locally optimal solution by Ipopt, giving an upper bound:\s*([^\.\r\n]+(?:\.[0-9eE+\-]+)?)", text)
    if ub_match !== nothing
        parsed = tryparse(Float64, strip(ub_match.captures[1]))
        parsed !== nothing && (reported_ub = parsed)
    end

    reported_gap = nothing
    gap_match = match(r"The relative optimality gap is:\s*([^\%\r\n]+)%", text)
    if gap_match !== nothing
        parsed = tryparse(Float64, strip(gap_match.captures[1]))
        parsed !== nothing && (reported_gap = parsed / 100.0)
    end

    termination_status = nothing
    term_match = match(r"termination status:\s*([^\r\n]+)", text)
    if term_match !== nothing
        termination_status = strip(term_match.captures[1])
    end

    solution_status = nothing
    sol_match = match(r"solution status:\s*([^\r\n]+)", text)
    if sol_match !== nothing
        solution_status = strip(sol_match.captures[1])
    end

    return (
        local_refine_success = succeeded,
        local_refine_failed = failed,
        reported_upper_bound = reported_ub,
        reported_relative_gap = reported_gap,
        has_local_refine_message = succeeded || failed,
        termination_status = termination_status,
        solution_status = solution_status,
    )
end

function _normalized_status_string(x)
    isnothing(x) && return ""
    return uppercase(strip(String(x)))
end

function _tssos_lb_state(report)
    report === nothing && return (:valid, "usable relaxation lower bound")

    term = hasproperty(report, :termination_status) ? _normalized_status_string(report.termination_status) : ""
    sol = hasproperty(report, :solution_status) ? _normalized_status_string(report.solution_status) : ""

    if term == "DUAL_INFEASIBLE"
        return (:unreliable, "dual infeasible relaxation status")
    elseif term == "INFEASIBLE_OR_UNBOUNDED"
        return (:unreliable, "infeasible-or-unbounded relaxation status")
    elseif term in ("INFEASIBLE", "PRIMAL_INFEASIBLE", "LOCALLY_INFEASIBLE", "ALMOST_INFEASIBLE")
        return (:relaxation_infeasible, "relaxation infeasible")
    elseif occursin("INFEASIBILITY_CERTIFICATE", sol)
        return (:relaxation_infeasible, "relaxation infeasibility certificate")
    end

    # Explicit non-optimal statuses are treated as approximate: keep the raw
    # SDP objective for diagnostics, but do not use it to decrease/fathom the
    # certified global lower bound in descendants.
    if !isempty(term) || !isempty(sol)
        trusted_terms = Set(["OPTIMAL", "NEAR_OPTIMAL", "ALMOST_OPTIMAL"])
        trusted_solutions = Set(["OPTIMAL", "NEAR_OPTIMAL", "ALMOST_OPTIMAL"])
        if !(term in trusted_terms) || !(isempty(sol) || sol in trusted_solutions)
            return (:approximate, "non-optimal relaxation status ($(isempty(term) ? "unknown" : term) / $(isempty(sol) ? "unknown" : sol))")
        end
    end

    return (:valid, "usable relaxation lower bound")
end

function _tssos_lb_summary(node::BaBNode)
    if node.tssos_lb_state == :valid
        return "usable (stored LB = $(_fmt(node.best_lb)))"
    elseif node.tssos_lb_state == :approximate
        return "approximate ($(node.tssos_lb_reason)); stored LB = $(_fmt(node.best_lb)), raw SDP value = $(_fmt(node.raw_best_lb))"
    elseif node.tssos_lb_state == :relaxation_infeasible
        return "relaxation infeasible"
    end

    if isfinite(node.raw_best_lb)
        return "$(node.tssos_lb_reason); stored LB = $(_fmt(node.best_lb)), raw SDP value = $(_fmt(node.raw_best_lb))"
    end
    return "$(node.tssos_lb_reason); stored LB = $(_fmt(node.best_lb))"
end

function _recompute_best_lower_bound(nodes::Vector{BaBNode},
                                     frontier_ids::Vector{Int},
                                     current_best_lb::Float64,
                                     best_upper_bound::Float64,
                                     optimality_tol::Float64)
    if !isempty(frontier_ids)
        return minimum(nodes[id].best_lb for id in frontier_ids)
    end

    if any(node -> node.status == :stopped_unreliable, nodes)
        return isfinite(current_best_lb) ? current_best_lb : -Inf
    end

    if isfinite(best_upper_bound)
        eps_abs = optimality_tol * max(abs(best_upper_bound), 1.0)
        return best_upper_bound - eps_abs
    end

    return current_best_lb
end

function _tssos_refine_summary(report)
    report === nothing && return "unknown"
    if hasproperty(report, :local_refine_success) && getproperty(report, :local_refine_success)
        ub = hasproperty(report, :reported_upper_bound) ? getproperty(report, :reported_upper_bound) : nothing
        ub_str = isnothing(ub) ? "reported success" : "reported UB = $(_fmt(ub))"
        return "success ($ub_str)"
    elseif hasproperty(report, :local_refine_failed) && getproperty(report, :local_refine_failed)
        return "reported local-refinement failure"
    elseif hasproperty(report, :has_local_refine_message) && getproperty(report, :has_local_refine_message)
        return "reported local-refinement status"
    end
    return "no local-refinement message seen"
end

function _poly_total_degree(mon)
    deg = 0
    for var in variables(mon)
        deg += degree(mon, var)
    end
    return deg
end

function _polys_are_quadratic(polys::AbstractVector)
    for poly in polys
        for mon in monomials(poly)
            _poly_total_degree(mon) <= 2 || return false
        end
    end
    return true
end

function _poly_to_jump_expr(poly,
                            jump_vars::AbstractVector,
                            var_lookup::Dict{Any, Int};
                            scale::Float64 = 1.0)
    scale = (!isfinite(scale) || iszero(scale)) ? 1.0 : scale
    expr = 0.0

    for (coeff, mon) in zip(coefficients(poly), monomials(poly))
        term = Float64(coeff) / scale
        mon_deg = _poly_total_degree(mon)
        mon_deg <= 2 || error("The NLP fallback currently supports only quadratic polynomials. Found total degree $mon_deg in $poly")

        for var in variables(mon)
            deg = degree(mon, var)
            idx = var_lookup[var]
            if deg == 1
                term *= jump_vars[idx]
            elseif deg == 2
                term *= jump_vars[idx]^2
            elseif deg != 0
                error("Unsupported monomial degree $deg in $poly")
            end
        end
        expr += term
    end

    return expr
end

function _initial_local_start(relaxation_sol::AbstractVector{<:Real},
                              ilb::AbstractVector{<:Real},
                              iub::AbstractVector{<:Real})
    length(ilb) == length(iub) || error("ilb and iub must have the same length.")
    n = length(ilb)
    has_relaxation = length(relaxation_sol) == n

    start = zeros(n)
    for i in 1:n
        li = Float64(ilb[i])
        ui = Float64(iub[i])
        mid = li == ui ? li : (li + ui) / 2
        candidate = has_relaxation && isfinite(relaxation_sol[i]) ? Float64(relaxation_sol[i]) : mid
        start[i] = clamp(candidate, li, ui)
    end
    return start
end

function _random_start(ilb::AbstractVector{<:Real},
                       iub::AbstractVector{<:Real},
                       rng::AbstractRNG)
    start = zeros(length(ilb))
    for i in eachindex(ilb)
        li = Float64(ilb[i])
        ui = Float64(iub[i])
        start[i] = li == ui ? li : li + rand(rng) * (ui - li)
    end
    return start
end

function _push_unique_start!(starts::Vector{Vector{Float64}},
                             candidate::AbstractVector{<:Real};
                             atol::Float64 = 1e-10)
    cand = Float64.(candidate)
    for existing in starts
        length(existing) == length(cand) || continue
        if all(abs.(existing .- cand) .<= atol)
            return starts
        end
    end
    push!(starts, cand)
    return starts
end

function _build_fallback_starts(relaxation_solution::AbstractVector{<:Real},
                                ilb::AbstractVector{<:Real},
                                iub::AbstractVector{<:Real},
                                rng::AbstractRNG;
                                random_starts::Int = 4)
    starts = Vector{Vector{Float64}}()
    midpoint = _initial_local_start(Float64[], ilb, iub)
    _push_unique_start!(starts, midpoint)

    if length(relaxation_solution) == length(ilb)
        _push_unique_start!(starts, _initial_local_start(relaxation_solution, ilb, iub))
        blended = 0.5 .* midpoint .+ 0.5 .* _initial_local_start(relaxation_solution, ilb, iub)
        _push_unique_start!(starts, blended)
    end

    for _ in 1:max(0, random_starts)
        _push_unique_start!(starts, _random_start(ilb, iub, rng))
    end

    return starts
end

function _report_violation_score(report)
    report === nothing && return Inf
    terms = Float64[]
    if hasproperty(report, :max_bound_violation) && hasproperty(report, :bound_tol) &&
       isfinite(report.max_bound_violation) && isfinite(report.bound_tol) && report.bound_tol > 0
        push!(terms, report.max_bound_violation / report.bound_tol)
    end
    if hasproperty(report, :max_ineq_violation) && hasproperty(report, :ineq_tol) &&
       isfinite(report.max_ineq_violation) && isfinite(report.ineq_tol) && report.ineq_tol > 0
        push!(terms, report.max_ineq_violation / report.ineq_tol)
    end
    if hasproperty(report, :max_eq_violation) && hasproperty(report, :eq_tol) &&
       isfinite(report.max_eq_violation) && isfinite(report.eq_tol) && report.eq_tol > 0
        push!(terms, report.max_eq_violation / report.eq_tol)
    end
    isempty(terms) && return Inf
    return maximum(terms)
end

function _solve_node_nlp(prob::PreparedBaBInput;
                         ilb::AbstractVector{<:Real},
                         iub::AbstractVector{<:Real},
                         extra_ineqs::AbstractVector = Any[],
                         extra_eqs::AbstractVector = Any[],
                         start_solution::AbstractVector{<:Real} = Float64[],
                         time_limit::Union{Nothing, Float64} = nothing,
                         normalize::Bool = true,
                         max_iter::Int = 2000,
                         local_solver::Symbol = :ipopt_nlp_fallback,
                         attempt_index::Int = 1,
                         ineq_tol::Float64 = 1e-6,
                         eq_tol::Float64 = 1e-6,
                         bound_tol::Float64 = 1e-6)
    n = length(prob.vars)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "sb", "yes")
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", min(1e-8, ineq_tol, eq_tol, bound_tol))
    set_optimizer_attribute(model, "constr_viol_tol", max(ineq_tol, eq_tol, bound_tol))
    if !isnothing(time_limit)
        set_optimizer_attribute(model, "max_cpu_time", max(1e-3, time_limit))
    end

    start = _initial_local_start(start_solution, ilb, iub)
    @variable(model, ilb[i] <= x[i = 1:n] <= iub[i], start = start[i])

    objective_scale = normalize ? _poly_scale(prob.objective) : 1.0
    objective_expr = _poly_to_jump_expr(prob.objective, x, prob.var_lookup; scale = objective_scale)
    @objective(model, Min, objective_expr)

    constraint_scales = Float64[]
    for poly in prob.base_ineqs
        scale = normalize ? _poly_scale(poly) : 1.0
        push!(constraint_scales, scale)
        @constraint(model, _poly_to_jump_expr(poly, x, prob.var_lookup; scale = scale) >= 0)
    end
    for poly in extra_ineqs
        scale = normalize ? _poly_scale(poly) : 1.0
        push!(constraint_scales, scale)
        @constraint(model, _poly_to_jump_expr(poly, x, prob.var_lookup; scale = scale) >= 0)
    end
    for poly in prob.base_eqs
        scale = normalize ? _poly_scale(poly) : 1.0
        push!(constraint_scales, scale)
        @constraint(model, _poly_to_jump_expr(poly, x, prob.var_lookup; scale = scale) == 0)
    end
    for poly in extra_eqs
        scale = normalize ? _poly_scale(poly) : 1.0
        push!(constraint_scales, scale)
        @constraint(model, _poly_to_jump_expr(poly, x, prob.var_lookup; scale = scale) == 0)
    end

    optimize!(model)

    term_status = JuMP.termination_status(model)
    primal_status = JuMP.primal_status(model)
    has_point = JuMP.has_values(model)
    sol = has_point ? Float64.(JuMP.value.(x)) : Float64[]

    feasibility_report = _assess_solution(
        prob,
        sol,
        ilb,
        iub,
        extra_ineqs,
        extra_eqs;
        ineq_tol = ineq_tol,
        eq_tol = eq_tol,
        bound_tol = bound_tol,
    )

    scale_summary = isempty(constraint_scales) ? (min = 1.0, max = 1.0) :
                    (min = minimum(constraint_scales), max = maximum(constraint_scales))

    feasibility_report = merge(
        feasibility_report,
        (
            local_solver = local_solver,
            termination_status = term_status,
            primal_status = primal_status,
            nlp_has_values = has_point,
            normalization_enabled = normalize,
            objective_scale = objective_scale,
            min_constraint_scale = scale_summary.min,
            max_constraint_scale = scale_summary.max,
            attempt_index = attempt_index,
        ),
    )

    ub_candidate = feasibility_report.objective_candidate
    ub_valid = feasibility_report.valid
    ub = ub_valid ? ub_candidate : Inf

    return (
        best_ub = ub,
        ub_candidate = ub_candidate,
        ub_valid = ub_valid,
        solution = sol,
        feasibility_report = feasibility_report,
    )
end

function _tssos_candidate_result(prob::PreparedBaBInput,
                                 relaxation_solution::AbstractVector{<:Real},
                                 ilb::AbstractVector{<:Real},
                                 iub::AbstractVector{<:Real},
                                 extra_ineqs::AbstractVector,
                                 extra_eqs::AbstractVector,
                                 tssos_report;
                                 normalize::Bool = true,
                                 objective_scale::Float64 = 1.0,
                                 min_constraint_scale::Float64 = 1.0,
                                 max_constraint_scale::Float64 = 1.0,
                                 ineq_tol::Float64 = 1e-6,
                                 eq_tol::Float64 = 1e-6,
                                 bound_tol::Float64 = 1e-6)
    extracted_report = _assess_solution(
        prob,
        relaxation_solution,
        ilb,
        iub,
        extra_ineqs,
        extra_eqs;
        ineq_tol = ineq_tol,
        eq_tol = eq_tol,
        bound_tol = bound_tol,
    )

    reported_success = tssos_report !== nothing &&
                       hasproperty(tssos_report, :local_refine_success) &&
                       tssos_report.local_refine_success
    reported_upper_bound = (tssos_report !== nothing && hasproperty(tssos_report, :reported_upper_bound)) ?
                           tssos_report.reported_upper_bound : nothing
    term_status = reported_success ? :reported_success :
                  (tssos_report !== nothing && hasproperty(tssos_report, :local_refine_failed) &&
                   tssos_report.local_refine_failed ? :reported_failure : :no_message)
    primal_status = extracted_report.valid ? :FEASIBLE_POINT : :UNKNOWN_RESULT_STATUS

    extracted_report = merge(
        extracted_report,
        (
            local_solver = :tssos_refine,
            termination_status = term_status,
            primal_status = primal_status,
            normalization_enabled = normalize,
            objective_scale = objective_scale,
            min_constraint_scale = min_constraint_scale,
            max_constraint_scale = max_constraint_scale,
            reported_upper_bound = reported_upper_bound,
            evaluation_target = :tssos_extracted_solution,
        ),
    )

    trust_reported_upper_bound = reported_success && reported_upper_bound !== nothing && isfinite(reported_upper_bound)
    reported_ub_original_units = trust_reported_upper_bound ? Float64(reported_upper_bound) * objective_scale : nothing
    ub_candidate = trust_reported_upper_bound ? reported_ub_original_units : extracted_report.objective_candidate
    ub_valid = trust_reported_upper_bound || (reported_success && extracted_report.valid)
    ub = ub_valid ? ub_candidate : Inf

    feasibility_report = if trust_reported_upper_bound
        merge(
            extracted_report,
            (
                valid = true,
                objective_candidate = reported_ub_original_units,
                max_bound_violation = 0.0,
                max_ineq_violation = 0.0,
                max_eq_violation = 0.0,
                primal_status = :FEASIBLE_POINT,
                evaluation_target = :tssos_reported_local_solution,
                trusted_reported_upper_bound = true,
            ),
        )
    else
        extracted_report
    end

    return (
        best_ub = ub,
        ub_candidate = ub_candidate,
        ub_valid = ub_valid,
        solution = copy(relaxation_solution),
        feasibility_report = feasibility_report,
        extracted_report = extracted_report,
    )
end

function _solve_node_nlp_multistart(prob::PreparedBaBInput;
                                    ilb::AbstractVector{<:Real},
                                    iub::AbstractVector{<:Real},
                                    extra_ineqs::AbstractVector = Any[],
                                    extra_eqs::AbstractVector = Any[],
                                    relaxation_solution::AbstractVector{<:Real} = Float64[],
                                    time_limit::Union{Nothing, Float64} = nothing,
                                    normalize::Bool = true,
                                    max_iter::Int = 2000,
                                    rng::AbstractRNG = Random.GLOBAL_RNG,
                                    random_starts::Int = 4,
                                    max_attempts::Union{Nothing, Int} = nothing,
                                    ineq_tol::Float64 = 1e-6,
                                    eq_tol::Float64 = 1e-6,
                                    bound_tol::Float64 = 1e-6)
    starts = _build_fallback_starts(
        relaxation_solution,
        ilb,
        iub,
        rng;
        random_starts = random_starts,
    )
    if max_attempts !== nothing
        max_keep = max(0, Int(max_attempts))
        starts = starts[1:min(length(starts), max_keep)]
    end

    best_valid = nothing
    best_invalid = nothing
    attempts = 0
    started_at = time_ns()

    for (idx, start) in enumerate(starts)
        remaining = isnothing(time_limit) ? nothing : max(0.0, time_limit - _elapsed_seconds(started_at))
        if !isnothing(remaining) && remaining <= 1e-3
            break
        end
        per_attempt_limit = isnothing(remaining) ? nothing : max(1e-3, remaining / max(1, length(starts) - idx + 1))
        res = _solve_node_nlp(
            prob;
            ilb = ilb,
            iub = iub,
            extra_ineqs = extra_ineqs,
            extra_eqs = extra_eqs,
            start_solution = start,
            time_limit = per_attempt_limit,
            normalize = normalize,
            max_iter = max_iter,
            local_solver = :ipopt_nlp_fallback,
            attempt_index = idx,
            ineq_tol = ineq_tol,
            eq_tol = eq_tol,
            bound_tol = bound_tol,
        )
        attempts += 1

        if res.ub_valid
            if best_valid === nothing || res.best_ub < best_valid.best_ub
                best_valid = res
            end
        else
            if best_invalid === nothing ||
               _report_violation_score(res.feasibility_report) < _report_violation_score(best_invalid.feasibility_report)
                best_invalid = res
            end
        end
    end

    chosen = best_valid !== nothing ? best_valid :
             (best_invalid !== nothing ? best_invalid : (
                 best_ub = Inf,
                 ub_candidate = Inf,
                 ub_valid = false,
                 solution = Float64[],
                 feasibility_report = (
                     valid = false,
                     objective_candidate = Inf,
                     max_bound_violation = Inf,
                     max_ineq_violation = Inf,
                     max_eq_violation = Inf,
                     ineq_tol = ineq_tol,
                     eq_tol = eq_tol,
                     bound_tol = bound_tol,
                     local_solver = :ipopt_nlp_fallback,
                     termination_status = :not_attempted,
                     primal_status = :NO_SOLUTION,
                     nlp_has_values = false,
                     normalization_enabled = normalize,
                     objective_scale = 1.0,
                     min_constraint_scale = 1.0,
                     max_constraint_scale = 1.0,
                     attempt_index = 0,
                 ),
             ))

    report = merge(
        chosen.feasibility_report,
        (
            fallback_attempted = attempts > 0,
            fallback_attempts = attempts,
            fallback_recovered = best_valid !== nothing,
        ),
    )

    return (
        best_ub = chosen.best_ub,
        ub_candidate = chosen.ub_candidate,
        ub_valid = chosen.ub_valid,
        solution = chosen.solution,
        feasibility_report = report,
        attempts = attempts,
    )
end

function _feasibility_diagnostic(report)
    report === nothing && return "no feasibility report"

    if report.max_bound_violation <= report.bound_tol &&
       report.max_ineq_violation <= report.ineq_tol &&
       report.max_eq_violation <= report.eq_tol
        return "feasible"
    end

    if report.max_ineq_violation <= report.ineq_tol &&
       report.max_eq_violation <= report.eq_tol &&
       report.max_bound_violation > report.bound_tol
        return "near-feasible; rejected only by box-bound tolerance"
    end

    if report.max_bound_violation <= 10 * report.bound_tol &&
       report.max_ineq_violation <= 10 * report.ineq_tol &&
       report.max_eq_violation <= 10 * report.eq_tol
        return "near-feasible; small residual violations remain"
    end

    return "invalid local candidate; structural constraints still violated"
end

function _local_solver_summary(report)
    report === nothing && return nothing
    hasproperty(report, :local_solver) || return nothing

    solver = getproperty(report, :local_solver)
    term = hasproperty(report, :termination_status) ? getproperty(report, :termination_status) : missing
    primal = hasproperty(report, :primal_status) ? getproperty(report, :primal_status) : missing
    return "solver=$(solver), termination=$(term), primal=$(primal)"
end

function _fallback_summary(node::BaBNode)
    if !node.fallback_attempted
        return "not attempted"
    elseif node.incumbent_source == :nlp_fallback && node.ub_valid
        return "attempted with $(node.fallback_attempts) start(s), recovered a valid incumbent"
    end
    return "attempted with $(node.fallback_attempts) start(s), no valid incumbent recovered"
end

function _tssos_extracted_point_summary(node::BaBNode)
    node.tssos_candidate_report === nothing && return nothing
    reported_success = node.tssos_report !== nothing &&
                       hasproperty(node.tssos_report, :local_refine_success) &&
                       node.tssos_report.local_refine_success
    if node.incumbent_source == :tssos_refine && reported_success
        return "diagnostic only; reported TSSOS Ipopt UB trusted"
    end
    return node.tssos_candidate_report.valid ? "passes external feasibility screening" :
           _feasibility_diagnostic(node.tssos_candidate_report)
end

function _assess_solution(prob::PreparedBaBInput,
                          sol::AbstractVector{<:Real},
                          ilb::AbstractVector{<:Real},
                          iub::AbstractVector{<:Real},
                          extra_ineqs::AbstractVector,
                          extra_eqs::AbstractVector;
                          ineq_tol::Float64 = 1e-6,
                          eq_tol::Float64 = 1e-6,
                          bound_tol::Float64 = 1e-8)
    if length(sol) != length(prob.vars) || !all(isfinite, sol)
        return (
            valid = false,
            objective_candidate = Inf,
            max_bound_violation = Inf,
            max_ineq_violation = Inf,
            max_eq_violation = Inf,
            ineq_tol = ineq_tol,
            eq_tol = eq_tol,
            bound_tol = bound_tol,
        )
    end

    objective_candidate = _objective_value(prob, sol)

    max_bound_violation = 0.0
    for i in eachindex(sol)
        max_bound_violation = max(max_bound_violation, ilb[i] - sol[i], sol[i] - iub[i], 0.0)
    end

    max_ineq_violation = 0.0
    for poly in prob.base_ineqs
        max_ineq_violation = max(max_ineq_violation, -_evaluate_polynomial(poly, prob, sol))
    end
    for poly in extra_ineqs
        max_ineq_violation = max(max_ineq_violation, -_evaluate_polynomial(poly, prob, sol))
    end

    max_eq_violation = 0.0
    for poly in prob.base_eqs
        max_eq_violation = max(max_eq_violation, abs(_evaluate_polynomial(poly, prob, sol)))
    end
    for poly in extra_eqs
        max_eq_violation = max(max_eq_violation, abs(_evaluate_polynomial(poly, prob, sol)))
    end

    valid = isfinite(objective_candidate) &&
            max_bound_violation <= bound_tol &&
            max_ineq_violation <= ineq_tol &&
            max_eq_violation <= eq_tol

    return (
        valid = valid,
        objective_candidate = objective_candidate,
        max_bound_violation = max_bound_violation,
        max_ineq_violation = max_ineq_violation,
        max_eq_violation = max_eq_violation,
        ineq_tol = ineq_tol,
        eq_tol = eq_tol,
        bound_tol = bound_tol,
    )
end

function solve_node(prob::PreparedBaBInput;
                    ilb::AbstractVector{<:Real},
                    iub::AbstractVector{<:Real},
                    extra_ineqs::AbstractVector = eltype(prob.base_ineqs)[],
                    extra_eqs::AbstractVector = eltype(prob.base_eqs)[],
                    bound_encoding::Symbol = prob.bound_encoding,
                    branch_encoding::Symbol = bound_encoding,
                    d::Int = 1,
                    seed::Union{Nothing, Int} = nothing,
                    TS::Bool = false,
                    CS::Bool = false,
                    solution::Bool = true,
                    QUIET::Bool = true,
                    show_tssos_output::Bool = true,
                    tssos_normalize::Bool = true,
                    local_solver_time_limit::Union{Nothing, Float64} = nothing,
                    nlp_normalize::Bool = true,
                    nlp_max_iter::Int = 2000,
                    nlp_fallback_random_starts::Int = 4,
                    nlp_fallback_max_attempts::Union{Nothing, Int} = 2,
                    feasibility_ineq_tol::Float64 = 1e-6,
                    feasibility_eq_tol::Float64 = 1e-6,
                    feasibility_bound_tol::Float64 = 1e-6)
    seed !== nothing && Random.seed!(seed)
    pop, numeq = node_pop(
        prob;
        ilb = ilb,
        iub = iub,
        extra_ineqs = extra_ineqs,
        extra_eqs = extra_eqs,
        include_box_bounds = true,
        bound_encoding = bound_encoding,
        branch_encoding = branch_encoding,
    )
    tssos_pop, tssos_objective_scale, tssos_min_constraint_scale, tssos_max_constraint_scale =
        _normalize_polynomials(pop; normalize = tssos_normalize)

    tssos_result, tssos_text, tssos_call_seconds = _capture_text_output_local(() -> cs_tssos_first(
        tssos_pop,
        prob.vars,
        d;
        numeq = numeq,
        TS = TS,
        CS = CS,
        solution = solution,
        QUIET = false,
    ))
    if show_tssos_output
        print(_tssos_text_for_display(tssos_text; QUIET = QUIET))
    end
    opt, sol, data = tssos_result
    tssos_report = _parse_tssos_local_report(tssos_text)
    tssos_timing = _parse_tssos_timing_report(tssos_text)

    relaxation_solution = (sol !== nothing && length(sol) == length(prob.vars)) ? Float64.(sol) : Float64[]
    tssos_candidate = _tssos_candidate_result(
        prob,
        relaxation_solution,
        ilb,
        iub,
        extra_ineqs,
        extra_eqs,
        tssos_report;
        normalize = tssos_normalize,
        objective_scale = tssos_objective_scale,
        min_constraint_scale = tssos_min_constraint_scale,
        max_constraint_scale = tssos_max_constraint_scale,
        ineq_tol = feasibility_ineq_tol,
        eq_tol = feasibility_eq_tol,
        bound_tol = feasibility_bound_tol,
    )

    raw_best_lb = opt * tssos_objective_scale
    lb_state, lb_reason = _tssos_lb_state(tssos_report)

    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)
    run_fallback = !tssos_candidate.ub_valid &&
                   lb_state in (:valid, :approximate) &&
                   _polys_are_quadratic(pop)
    fallback_res = nothing
    if run_fallback
        fallback_res = _solve_node_nlp_multistart(
            prob;
            ilb = ilb,
            iub = iub,
            extra_ineqs = extra_ineqs,
            extra_eqs = extra_eqs,
            relaxation_solution = relaxation_solution,
            time_limit = local_solver_time_limit,
            normalize = nlp_normalize,
            max_iter = nlp_max_iter,
            rng = rng,
            random_starts = nlp_fallback_random_starts,
            max_attempts = nlp_fallback_max_attempts,
            ineq_tol = feasibility_ineq_tol,
            eq_tol = feasibility_eq_tol,
            bound_tol = feasibility_bound_tol,
        )
    end

    chosen = if tssos_candidate.ub_valid
        tssos_candidate
    elseif fallback_res !== nothing && fallback_res.ub_valid
        fallback_res
    elseif fallback_res !== nothing
        fallback_res
    else
        tssos_candidate
    end

    incumbent_source = if tssos_candidate.ub_valid
        :tssos_refine
    elseif fallback_res !== nothing && fallback_res.ub_valid
        :nlp_fallback
    else
        :none
    end

    fallback_attempted = fallback_res !== nothing
    fallback_attempts = fallback_res === nothing ? 0 : fallback_res.attempts

    feasibility_report = chosen.feasibility_report
    ub_candidate = chosen.ub_candidate
    ub_valid = chosen.ub_valid
    ub = ub_valid ? ub_candidate : Inf
    best_lb = if lb_state in (:valid, :approximate)
        raw_best_lb
    elseif lb_state == :relaxation_infeasible
        Inf
    else
        -Inf
    end
    rel_gap = isfinite(ub) ? (ub - best_lb) / max(abs(ub), 1.0) : Inf
    solution_vec = incumbent_source == :tssos_refine ? relaxation_solution :
                   (isempty(chosen.solution) ? relaxation_solution : chosen.solution)

    return (
        raw_best_lb = raw_best_lb,
        best_lb = best_lb,
        best_ub = ub,
        ub_candidate = ub_candidate,
        ub_valid = ub_valid,
        rel_gap = rel_gap,
        solution = solution_vec,
        moment_matrix = data.moment[1],
        pop = pop,
        tssos_pop = tssos_pop,
        numeq = numeq,
        data = data,
        tssos_normalized = tssos_normalize,
        tssos_objective_scale = tssos_objective_scale,
        tssos_min_constraint_scale = tssos_min_constraint_scale,
        tssos_max_constraint_scale = tssos_max_constraint_scale,
        tssos_report = tssos_report,
        tssos_candidate_report = tssos_candidate.extracted_report,
        tssos_lb_state = lb_state,
        tssos_lb_reason = lb_reason,
        tssos_call_seconds = tssos_call_seconds,
        tssos_sdp_assemble_seconds = tssos_timing.assemble_seconds,
        tssos_sdp_solve_seconds = tssos_timing.has_explicit_timing ? tssos_timing.solve_seconds : tssos_call_seconds,
        incumbent_source = incumbent_source,
        fallback_attempted = fallback_attempted,
        fallback_attempts = fallback_attempts,
        feasibility_report = feasibility_report,
    )
end

function root_relaxation(prob::PreparedBaBInput;
                         d::Int = 1,
                         seed::Union{Nothing, Int} = nothing,
                         TS::Bool = false,
                         CS::Bool = false,
                         solution::Bool = true,
                         QUIET::Bool = true,
                         show_tssos_output::Bool = true,
                         tssos_normalize::Bool = true,
                         branch_encoding::Symbol = prob.bound_encoding,
                         local_solver_time_limit::Union{Nothing, Float64} = nothing,
                         nlp_normalize::Bool = true,
                         nlp_max_iter::Int = 2000,
                         nlp_fallback_random_starts::Int = 4,
                         nlp_fallback_max_attempts::Union{Nothing, Int} = 2,
                         feasibility_ineq_tol::Float64 = 1e-6,
                         feasibility_eq_tol::Float64 = 1e-6,
                         feasibility_bound_tol::Float64 = 1e-6)
    return solve_node(
        prob;
        ilb = prob.ilb_root,
        iub = prob.iub_root,
        d = d,
        seed = seed,
        TS = TS,
        CS = CS,
        solution = solution,
        QUIET = QUIET,
        show_tssos_output = show_tssos_output,
        tssos_normalize = tssos_normalize,
        branch_encoding = branch_encoding,
        local_solver_time_limit = local_solver_time_limit,
        nlp_normalize = nlp_normalize,
        nlp_max_iter = nlp_max_iter,
        nlp_fallback_random_starts = nlp_fallback_random_starts,
        nlp_fallback_max_attempts = nlp_fallback_max_attempts,
        feasibility_ineq_tol = feasibility_ineq_tol,
        feasibility_eq_tol = feasibility_eq_tol,
        feasibility_bound_tol = feasibility_bound_tol,
    )
end

function branchable_coords(prob::PreparedBaBInput;
                           excluded::AbstractVector{<:Integer} = Int[],
                           ilb::AbstractVector{<:Real} = prob.ilb_root,
                           iub::AbstractVector{<:Real} = prob.iub_root,
                           width_tol::Float64 = 1e-10)
    excluded_set = Set(Int.(excluded))
    return [
        i for i in prob.branch_idx
        if !(i in excluded_set) && Float64(iub[i]) - Float64(ilb[i]) > width_tol
    ]
end

function fixable_coords(coords_to_fix_CDK::AbstractVector{<:Integer},
                        prob::PreparedBaBInput;
                        ilb::AbstractVector{<:Real} = prob.ilb_root,
                        iub::AbstractVector{<:Real} = prob.iub_root,
                        width_tol::Float64 = 1e-10)
    branchable = Set(branchable_coords(prob; ilb = ilb, iub = iub, width_tol = width_tol))
    return [i for i in Int.(coords_to_fix_CDK) if i in branchable]
end

select_coords_to_fix(fix_CDK::Bool,
                     coords_to_fix_CDK::AbstractVector{<:Integer},
                     prob::PreparedBaBInput;
                     ilb::AbstractVector{<:Real} = prob.ilb_root,
                     iub::AbstractVector{<:Real} = prob.iub_root,
                     width_tol::Float64 = 1e-10) =
    fix_CDK ? fixable_coords(coords_to_fix_CDK, prob; ilb = ilb, iub = iub, width_tol = width_tol) : Int[]

function marginal_moment_rank_local(M, k::Int; threshold::Float64 = 1e-4)
    vals = eigvals(Symmetric(M[[1, k + 1], [1, k + 1]]))
    return count(abs.(Float64.(vals)) .> threshold)
end

function select_branch_direction_from_scores(scores::AbstractVector{<:Real},
                                             prob::PreparedBaBInput;
                                             excluded::AbstractVector{<:Integer} = Int[],
                                             ilb::AbstractVector{<:Real} = prob.ilb_root,
                                             iub::AbstractVector{<:Real} = prob.iub_root,
                                             width_tol::Float64 = 1e-10)
    allowed = branchable_coords(
        prob;
        excluded = excluded,
        ilb = ilb,
        iub = iub,
        width_tol = width_tol,
    )
    isempty(allowed) && return nothing
    local_scores = [Float64(scores[i]) for i in allowed]
    return allowed[argmax(local_scores)]
end

function _midpoint_cut_points(ilb::AbstractVector{<:Real},
                              iub::AbstractVector{<:Real})
    [Float64(ilb[i]) == Float64(iub[i]) ? Float64(ilb[i]) :
     (Float64(ilb[i]) + Float64(iub[i])) / 2 for i in eachindex(ilb)]
end

function _width_scores(ilb::AbstractVector{<:Real},
                       iub::AbstractVector{<:Real})
    [max(0.0, Float64(iub[i]) - Float64(ilb[i])) for i in eachindex(ilb)]
end

function select_direction(dir_strategy::Symbol,
                          cut_points::AbstractVector{<:Real},
                          interval_lengths::AbstractVector{<:Real},
                          cut_direction_CDK::Union{Nothing, Int},
                          prob::PreparedBaBInput;
                          excluded::AbstractVector{<:Integer} = Int[],
                          ilb::AbstractVector{<:Real} = prob.ilb_root,
                          iub::AbstractVector{<:Real} = prob.iub_root,
                          rng::AbstractRNG = Random.GLOBAL_RNG,
                          width_tol::Float64 = 1e-10)
    allowed = branchable_coords(
        prob;
        excluded = excluded,
        ilb = ilb,
        iub = iub,
        width_tol = width_tol,
    )
    isempty(allowed) && return nothing

    if dir_strategy == :cdk
        direction = select_branch_direction_from_scores(
            interval_lengths,
            prob;
            excluded = excluded,
            ilb = ilb,
            iub = iub,
            width_tol = width_tol,
        )
        direction !== nothing && return direction
        if cut_direction_CDK !== nothing && cut_direction_CDK in allowed
            return cut_direction_CDK
        end
        return nothing
    elseif dir_strategy == :random
        return rand(rng, allowed)
    else
        error("Unsupported dir_strategy: $dir_strategy. Use :cdk or :random.")
    end
end

function select_cut_value(cut_points::AbstractVector{<:Real},
                          cut_direction::Int,
                          ilb::AbstractVector{<:Real},
                          iub::AbstractVector{<:Real};
                          strategy::Symbol = :mid,
                          rng::AbstractRNG = Random.GLOBAL_RNG,
                          round_digits::Union{Nothing, Int} = 5)
    a = Float64(ilb[cut_direction])
    b = Float64(iub[cut_direction])
    width = b - a
    width <= 1e-10 && return (a + b) / 2

    function _stabilize_cut(raw::Float64)
        eps = min(width / 3, max(1e-6 * width, 1e-12))
        val = clamp(raw, a + eps, b - eps)
        if round_digits !== nothing
            mid = (a + b) / 2
            tau = max(10.0^(-round_digits), 1e-8 * max(width, 1.0))
            abs(val - mid) <= tau && (val = mid)
            val = round(val; digits = round_digits)
            val = clamp(val, a + eps, b - eps)
        end
        return val
    end

    if strategy == :mid
        return _stabilize_cut((a + b) / 2)
    elseif strategy == :cdk
        raw = Float64(cut_points[cut_direction])
        return _stabilize_cut(raw)
    elseif strategy == :random
        return _stabilize_cut(a + rand(rng) * width)
    else
        error("Unsupported cut strategy: $strategy. Use :cdk, :random, or :mid.")
    end
end

function snap_to_boundary(m::Float64, a::Float64, b::Float64;
                          rel::Float64 = 1e-3,
                          abs::Float64 = 1e-8)
    w = b - a
    if w <= abs
        return true, (a + b) / 2
    end
    tau = max(abs, rel * w)
    if m <= a + tau
        return true, a
    elseif m >= b - tau
        return true, b
    end
    return false, m
end

function _stabilize_fix_value(m::Float64, a::Float64, b::Float64;
                              rel::Float64 = 1e-8,
                              abs::Float64 = 1e-8,
                              round_digits::Union{Nothing, Int} = 5)
    val = clamp(m, a, b)
    snapped, val = snap_to_boundary(val, a, b; rel = rel, abs = abs)
    if snapped
        return val
    end

    if round_digits !== nothing
        val = round(val; digits = round_digits)
        if Base.abs(val) <= 10.0^(-round_digits)
            val = 0.0
        end
        val = clamp(val, a, b)
        _, val = snap_to_boundary(val, a, b; rel = rel, abs = abs)
    end

    return val
end

function _cut_near_bound(cut::Float64, a::Float64, b::Float64;
                         rel_tol::Float64 = 1e-6,
                         abs_tol::Float64 = 1e-5)
    w = b - a
    w <= 0.0 && return true
    tau = max(abs_tol, rel_tol * w)
    return cut <= a + tau || cut >= b - tau
end

function make_child_bounds(ilb::AbstractVector{<:Real},
                           iub::AbstractVector{<:Real},
                           cut_direction::Int,
                           cut_value::Real)
    length(ilb) == length(iub) || error("ilb and iub must have the same length.")

    ilb_left = Float64.(ilb)
    iub_left = Float64.(iub)
    ilb_right = Float64.(ilb)
    iub_right = Float64.(iub)

    cut = Float64(cut_value)
    iub_left[cut_direction] = cut
    ilb_right[cut_direction] = cut

    return (
        left = (ilb = ilb_left, iub = iub_left),
        right = (ilb = ilb_right, iub = iub_right),
    )
end

function build_branch_polynomial(var, lb::Real, ub::Real; encoding::Symbol = :quadratic)
    ineqs, eqs = box_bound_polynomials([var], [Float64(lb)], [Float64(ub)]; encoding = encoding)
    return isempty(ineqs) ? eqs[1] : ineqs[1]
end

function branch_once_smoke(prob::PreparedBaBInput;
                           d::Int = 1,
                           cut_strategy::Symbol = :mid,
                           QUIET::Bool = true,
                           branch_encoding::Symbol = prob.bound_encoding,
                           cdk_level::Union{Symbol, Real} = :marginal_dim,
                           rank_tol::Float64 = 1e-4,
                           unreliable_relaxation_policy::Symbol = :width_fallback,
                           exclude_near_bound_cuts::Bool = true,
                           cut_near_bound_rel_tol::Float64 = 1e-6,
                           cut_near_bound_abs_tol::Float64 = 1e-5,
                           cut_round_digits::Union{Nothing, Int} = 5,
                           tssos_normalize::Bool = true,
                           nlp_normalize::Bool = true,
                           nlp_max_iter::Int = 2000,
                           nlp_fallback_random_starts::Int = 4,
                           nlp_fallback_max_attempts::Union{Nothing, Int} = 2,
                           feasibility_ineq_tol::Float64 = 1e-6,
                           feasibility_eq_tol::Float64 = 1e-6,
                           feasibility_bound_tol::Float64 = 1e-6)
    root = root_relaxation(
        prob;
        d = d,
        QUIET = QUIET,
        tssos_normalize = tssos_normalize,
        branch_encoding = branch_encoding,
        nlp_normalize = nlp_normalize,
        nlp_max_iter = nlp_max_iter,
        nlp_fallback_random_starts = nlp_fallback_random_starts,
        nlp_fallback_max_attempts = nlp_fallback_max_attempts,
        feasibility_ineq_tol = feasibility_ineq_tol,
        feasibility_eq_tol = feasibility_eq_tol,
        feasibility_bound_tol = feasibility_bound_tol,
    )
    cdk_guidance_allowed = root.tssos_lb_state == :valid
    cut_points, interval_lengths, coords_to_fix, cut_direction = if cdk_guidance_allowed
        next_step_CDK_local(
            root.moment_matrix,
            prob.vars;
            branch_idx = prob.branch_idx,
            ilb = prob.ilb_root,
            iub = prob.iub_root,
            rank_tol = rank_tol,
            cdk_level = cdk_level,
            exclude_near_bound = exclude_near_bound_cuts,
            cut_near_bound_rel_tol = cut_near_bound_rel_tol,
            cut_near_bound_abs_tol = cut_near_bound_abs_tol,
        )
    elseif unreliable_relaxation_policy == :width_fallback
        width_scores = _width_scores(prob.ilb_root, prob.iub_root)
        midpoint_cuts = _midpoint_cut_points(prob.ilb_root, prob.iub_root)
        midpoint_cuts, width_scores, Int[], nothing
    else
        error("Root relaxation is $(root.tssos_lb_reason); branch_once_smoke stops under unreliable_relaxation_policy=$(unreliable_relaxation_policy).")
    end

    cut_direction === nothing && error("No branchable direction found at the root node.")

    cut_value = select_cut_value(
        cut_points,
        cut_direction,
        prob.ilb_root,
        prob.iub_root;
        strategy = cut_strategy,
        round_digits = cut_round_digits,
    )
    children = make_child_bounds(prob.ilb_root, prob.iub_root, cut_direction, cut_value)
    left = solve_node(
        prob;
        ilb = children.left.ilb,
        iub = children.left.iub,
        d = d,
        QUIET = QUIET,
        tssos_normalize = tssos_normalize,
        branch_encoding = branch_encoding,
        nlp_normalize = nlp_normalize,
        nlp_max_iter = nlp_max_iter,
        nlp_fallback_random_starts = nlp_fallback_random_starts,
        nlp_fallback_max_attempts = nlp_fallback_max_attempts,
        feasibility_ineq_tol = feasibility_ineq_tol,
        feasibility_eq_tol = feasibility_eq_tol,
        feasibility_bound_tol = feasibility_bound_tol,
    )
    right = solve_node(
        prob;
        ilb = children.right.ilb,
        iub = children.right.iub,
        d = d,
        QUIET = QUIET,
        tssos_normalize = tssos_normalize,
        branch_encoding = branch_encoding,
        nlp_normalize = nlp_normalize,
        nlp_max_iter = nlp_max_iter,
        nlp_fallback_random_starts = nlp_fallback_random_starts,
        nlp_fallback_max_attempts = nlp_fallback_max_attempts,
        feasibility_ineq_tol = feasibility_ineq_tol,
        feasibility_eq_tol = feasibility_eq_tol,
        feasibility_bound_tol = feasibility_bound_tol,
    )

    return (
        root = root,
        left = left,
        right = right,
        cut_direction = cut_direction,
        cut_name = prob.var_names[cut_direction],
        cut_value = cut_value,
        coords_to_fix = coords_to_fix,
        interval_lengths = interval_lengths,
        cut_points = cut_points,
        children = children,
    )
end

function solve_quadratic_inequality(a::Float64, b::Float64, c::Float64, inequality::String)
    discriminant = b^2 - 4 * a * c

    if discriminant > 0
        x1 = (-b + sqrt(discriminant)) / (2 * a)
        x2 = (-b - sqrt(discriminant)) / (2 * a)
        roots = sort([x1, x2])
    elseif discriminant == 0
        roots = [-b / (2 * a)]
    else
        roots = Float64[]
    end

    intervals = Tuple{Float64, Float64}[]

    if inequality == "<=" || inequality == "<"
        if length(roots) == 2
            x1, x2 = roots
            if a > 0
                push!(intervals, (x1, x2))
            else
                push!(intervals, (-Inf, x1))
                push!(intervals, (x2, Inf))
            end
        elseif length(roots) == 1
            x1 = roots[1]
            if a > 0
                push!(intervals, (-Inf, x1))
                push!(intervals, (x1, Inf))
            else
                push!(intervals, (-Inf, Inf))
            end
        else
            push!(intervals, (-Inf, Inf))
        end
    elseif inequality == ">=" || inequality == ">"
        if length(roots) == 2
            x1, x2 = roots
            if a > 0
                push!(intervals, (-Inf, x1))
                push!(intervals, (x2, Inf))
            else
                push!(intervals, (x1, x2))
            end
        elseif length(roots) == 1
            x1 = roots[1]
            if a > 0
                push!(intervals, (x1, Inf))
            else
                push!(intervals, (-Inf, x1))
            end
        else
            push!(intervals, (-Inf, Inf))
        end
    else
        error("Unsupported inequality type. Use <=, <, >=, or >.")
    end

    return intervals
end

function _interval_overlap(interval::Tuple{Float64, Float64},
                           lb::Float64,
                           ub::Float64)
    lo = max(interval[1], lb)
    hi = min(interval[2], ub)
    return hi > lo ? (lo, hi) : nothing
end

function _local_cdk_interval(intervals::Vector{Tuple{Float64, Float64}},
                             lb::Float64,
                             ub::Float64,
                             anchor::Float64)
    clipped = Tuple{Float64, Float64}[]
    for interval in intervals
        overlap = _interval_overlap(interval, lb, ub)
        overlap === nothing || push!(clipped, overlap)
    end
    isempty(clipped) && return nothing

    for interval in clipped
        if interval[1] <= anchor <= interval[2]
            return interval
        end
    end

    lengths = [interval[2] - interval[1] for interval in clipped]
    return clipped[argmax(lengths)]
end

function construct_CDKmarg_SVD_local(Mm, k::Int, vars; threshold::Float64 = 1e-4)
    eigen_result = eigen(Mm[[1, k + 1], [1, k + 1]])
    Qvec = eigen_result.vectors
    Qval = eigen_result.values

    p_alpha_squared = Any[]
    for j in 1:length(Qval)
        push!(p_alpha_squared, (vcat(1, vars[k])' * Qvec[:, j])^2)
    end

    positiveEV = count(ev -> abs(ev) > threshold, Qval)
    negativeEV = length(Qval) - positiveEV

    if positiveEV == 0
        cdk = zero(p_alpha_squared[1])
    elseif positiveEV == length(Qval)
        cdk = sum(p_alpha_squared[i] / Qval[i] for i in 1:length(Qval))
    else
        cdk = sum(p_alpha_squared[i] / Qval[i] for i in (length(Qval) - positiveEV + 1):length(Qval))
    end

    return cdk, p_alpha_squared[1:negativeEV], positiveEV, negativeEV, minimum(Qval)
end

function _cdk_level_value(cdk_level::Symbol, M, k::Int)
    if cdk_level == :full_dim
        return Float64(size(M, 1))
    elseif cdk_level == :marginal_dim
        return 2.0
    else
        error("Unsupported cdk_level symbol: $(cdk_level). Use :full_dim or :marginal_dim.")
    end
end

_cdk_level_value(cdk_level::Real, M, k::Int) = Float64(cdk_level)

function next_step_CDK_local(M, vars;
                             branch_idx::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                             ilb::Union{Nothing, AbstractVector{<:Real}} = nothing,
                             iub::Union{Nothing, AbstractVector{<:Real}} = nothing,
                             rank_tol::Float64 = 1e-4,
                             cdk_threshold::Float64 = 1e-4,
                             cdk_level::Union{Symbol, Real} = :full_dim,
                             exclude_near_bound::Bool = true,
                             cut_near_bound_rel_tol::Float64 = 1e-6,
                             cut_near_bound_abs_tol::Float64 = 1e-5,
                             width_tol::Float64 = 1e-10)
    n = length(vars)
    branch_set = isnothing(branch_idx) ? Set(1:n) : Set(Int.(branch_idx))
    if xor(isnothing(ilb), isnothing(iub))
        error("Pass both ilb and iub, or neither.")
    end
    if !isnothing(ilb)
        length(ilb) == n || error("ilb has wrong length.")
        length(iub) == n || error("iub has wrong length.")
    end

    interval_lengths = Float64[]
    cut_points = Float64[]
    coords_to_fix = Int[]

    for k in 1:n
        cdk_res = construct_CDKmarg_SVD_local(M, k, vars; threshold = cdk_threshold)
        marginal_rank = marginal_moment_rank_local(M, k; threshold = rank_tol)
        if marginal_rank == 1 && (k in branch_set)
            push!(coords_to_fix, k)
        end

        cdk_k = cdk_res[1]
        level = _cdk_level_value(cdk_level, M, k)
        a, b, c = coefficients(level - cdk_k)
        roots_k = solve_quadratic_inequality(a, b, c, ">=")
        cut_point = Float64(M[1, k + 1])

        if isnothing(ilb)
            local_interval = isempty(roots_k) ? nothing : roots_k[1]
        else
            lbk = Float64(ilb[k])
            ubk = Float64(iub[k])
            local_interval = _local_cdk_interval(roots_k, lbk, ubk, cut_point)
        end

        interval_length = if local_interval === nothing
            0.0
        else
            max(0.0, local_interval[2] - local_interval[1])
        end

        if !isnothing(ilb) && exclude_near_bound
            lbk = Float64(ilb[k])
            ubk = Float64(iub[k])
            if _cut_near_bound(
                cut_point,
                lbk,
                ubk;
                rel_tol = cut_near_bound_rel_tol,
                abs_tol = cut_near_bound_abs_tol,
            )
                interval_length = 0.0
            end
        end
        push!(interval_lengths, interval_length)
        push!(cut_points, cut_point)
    end

    for k in 1:n
        if !(k in branch_set) || (!isnothing(ilb) && Float64(iub[k]) - Float64(ilb[k]) <= width_tol)
            interval_lengths[k] = 0.0
        end
    end

    for i in coords_to_fix
        interval_lengths[i] = 0.0
    end

    if all(iszero, interval_lengths)
        cut_direction = nothing
    else
        cut_direction = findfirst(==(maximum(interval_lengths)), interval_lengths)
    end

    return cut_points, interval_lengths, coords_to_fix, cut_direction
end

function _fmt(x::Real)
    if !isfinite(x)
        return string(x)
    elseif abs(Float64(x)) >= 1e4 || (abs(Float64(x)) > 0 && abs(Float64(x)) < 1e-4)
        return string(round(Float64(x); sigdigits = 6))
    else
        return string(round(Float64(x); digits = 6))
    end
end

_fmt(x) = string(x)

function _global_gap(best_lb::Real, best_ub::Real)
    isfinite(best_ub) || return Inf
    return (Float64(best_ub) - Float64(best_lb)) / max(abs(Float64(best_ub)), 1.0)
end

_elapsed_seconds(start_time_ns::UInt64) = (time_ns() - start_time_ns) / 1.0e9

function _time_limit_hit(start_time_ns::UInt64, time_limit::Union{Nothing, Float64})
    !isnothing(time_limit) && _elapsed_seconds(start_time_ns) >= time_limit
end

function _remaining_time(start_time_ns::UInt64, time_limit::Union{Nothing, Float64})
    isnothing(time_limit) && return nothing
    return max(0.0, time_limit - _elapsed_seconds(start_time_ns))
end

function _active_priority(node::BaBNode, global_ub::Float64)
    if isfinite(global_ub)
        return (global_ub - node.best_lb) / max(abs(global_ub), 1.0)
    end
    return -node.best_lb
end

function _lb_state_priority(node::BaBNode)
    if node.tssos_lb_state == :valid
        return 2
    elseif node.tssos_lb_state == :approximate
        return 1
    end
    return 0
end

function _tightened_bounds_summary(prob::PreparedBaBInput,
                                   ilb::AbstractVector{<:Real},
                                   iub::AbstractVector{<:Real};
                                   max_items::Int = 8,
                                   atol::Float64 = 1e-10)
    rows = String[]
    for i in eachindex(prob.vars)
        if abs(ilb[i] - prob.ilb_root[i]) > atol || abs(iub[i] - prob.iub_root[i]) > atol
            push!(rows, "$(prob.var_names[i]) in [$(_fmt(ilb[i])), $(_fmt(iub[i]))]")
        end
    end
    if isempty(rows)
        return ["(root box)"]
    elseif length(rows) > max_items
        shown = rows[1:max_items]
        push!(shown, "... ($(length(rows) - max_items) more)")
        return shown
    end
    return rows
end

function _fixed_bounds_summary(prob::PreparedBaBInput,
                               ilb::AbstractVector{<:Real},
                               iub::AbstractVector{<:Real};
                               max_items::Int = 8,
                               atol::Float64 = 1e-10)
    rows = String[]
    for i in eachindex(prob.vars)
        if abs(iub[i] - ilb[i]) <= atol
            push!(rows, "$(prob.var_names[i]) = $(_fmt(ilb[i]))")
        end
    end
    if isempty(rows)
        return ["(none)"]
    elseif length(rows) > max_items
        shown = rows[1:max_items]
        push!(shown, "... ($(length(rows) - max_items) more)")
        return shown
    end
    return rows
end

function _model_fixed_bounds_summary(prob::PreparedBaBInput;
                                     max_items::Int = 8,
                                     atol::Float64 = 1e-10)
    return _fixed_bounds_summary(prob, prob.ilb_root, prob.iub_root; max_items = max_items, atol = atol)
end

function _cdk_fixed_bounds_summary(prob::PreparedBaBInput,
                                   ilb::AbstractVector{<:Real},
                                   iub::AbstractVector{<:Real};
                                   max_items::Int = 8,
                                   atol::Float64 = 1e-10)
    rows = String[]
    for i in eachindex(prob.vars)
        root_fixed = abs(prob.iub_root[i] - prob.ilb_root[i]) <= atol
        node_fixed = abs(iub[i] - ilb[i]) <= atol
        if node_fixed && !root_fixed
            push!(rows, "$(prob.var_names[i]) = $(_fmt(ilb[i]))")
        end
    end
    if isempty(rows)
        return ["(none)"]
    elseif length(rows) > max_items
        shown = rows[1:max_items]
        push!(shown, "... ($(length(rows) - max_items) more)")
        return shown
    end
    return rows
end

function _fixation_summary(events::Vector{FixationEvent}; max_items::Int = 8)
    isempty(events) && return ["(none)"]
    rows = [
        "$(evt.var_name) = $(_fmt(evt.fixed_value)) from rank-$(evt.marginal_rank) marginal moment with pseudo-moment $(_fmt(evt.pseudo_moment)) in [$(_fmt(evt.old_lb)), $(_fmt(evt.old_ub))]"
        for evt in events
    ]
    if length(rows) > max_items
        shown = rows[1:max_items]
        push!(shown, "... ($(length(rows) - max_items) more)")
        return shown
    end
    return rows
end

function _branch_path_summary(events::Vector{BranchEvent}; max_items::Int = 8)
    isempty(events) && return ["(root)"]
    rows = [
        "$(evt.child_side): $(evt.var_name) in [$(_fmt(evt.child_lb)), $(_fmt(evt.child_ub))] (cut=$(_fmt(evt.cut_value)))"
        for evt in events
    ]
    if length(rows) > max_items
        shown = rows[1:max_items]
        push!(shown, "... ($(length(rows) - max_items) more)")
        return shown
    end
    return rows
end

function _top_branch_rows(prob::PreparedBaBInput,
                          cut_points::AbstractVector{<:Real},
                          interval_lengths::AbstractVector{<:Real},
                          ilb::AbstractVector{<:Real},
                          iub::AbstractVector{<:Real};
                          top::Int = 5)
    rows = [(
        idx = i,
        name = prob.var_names[i],
        lb = Float64(ilb[i]),
        ub = Float64(iub[i]),
        cut_point = Float64(cut_points[i]),
        interval = Float64(interval_lengths[i]),
    ) for i in branchable_coords(prob; ilb = ilb, iub = iub)]
    sort!(rows; by = row -> -row.interval)
    return rows[1:min(top, length(rows))]
end

function _child_status(best_lb::Float64,
                       best_upper_bound::Float64,
                       tol::Float64,
                       optimality_tol::Float64)
    if !isfinite(best_lb)
        return :pruned_nonfinite, "non-finite lower bound"
    end

    if isfinite(best_upper_bound)
        eps_abs = optimality_tol * max(abs(best_upper_bound), 1.0)
        if best_lb >= best_upper_bound - eps_abs
            return :pruned_fathomed, "fathomed by bound"
        end

        gap = (best_upper_bound - best_lb) / max(abs(best_upper_bound), 1.0)
        if gap <= tol
            return :closed, "gap <= tol"
        end
    end

    return :active, "kept active"
end

function _decorate_child_reason(reason::String, solve_res)
    solve_res.ub_valid && return reason
    return string(reason, "; no valid incumbent (", _feasibility_diagnostic(solve_res.feasibility_report), ", not a proof of infeasibility)")
end

function _unique_fixation_events(nodes::Vector{BaBNode})
    seen = Set{Tuple{Int, Int, Int}}()
    unique_events = FixationEvent[]
    for node in nodes
        for evt in node.fixation_history
            key = (evt.node_id, evt.var_idx, evt.iteration)
            if !(key in seen)
                push!(seen, key)
                push!(unique_events, evt)
            end
        end
    end
    return unique_events
end

function _prune_and_update_gaps!(nodes::Vector{BaBNode},
                                 active_ids::Vector{Int},
                                 frontier_ids::Vector{Int},
                                 best_upper_bound::Float64,
                                 tol::Float64,
                                 optimality_tol::Float64)
    isempty(active_ids) && return (pruned_ids = Int[], closed_ids = Int[])

    pruned_ids = Int[]
    closed_ids = Int[]
    refreshed_active = Int[]

    for id in active_ids
        node = nodes[id]
        status, reason = _child_status(node.best_lb, best_upper_bound, tol, optimality_tol)

        if status == :active
            node.status = :active
            if isempty(node.status_reason) || startswith(node.status_reason, "kept active") ||
               startswith(node.status_reason, "gap <= tol") || startswith(node.status_reason, "fathomed by bound")
                node.status_reason = _decorate_child_reason(reason, node)
            end
            push!(refreshed_active, id)
        elseif status == :closed
            node.status = :closed
            node.status_reason = _decorate_child_reason(reason, node)
            push!(closed_ids, id)
        else
            node.status = status
            node.status_reason = _decorate_child_reason(reason, node)
            push!(pruned_ids, id)
        end
    end

    empty!(active_ids)
    append!(active_ids, refreshed_active)

    if !isempty(pruned_ids)
        pruned_set = Set(pruned_ids)
        filter!(id -> !(id in pruned_set), frontier_ids)
    end

    return (pruned_ids = pruned_ids, closed_ids = closed_ids)
end

function _node_from_solve(node_id::Int,
                          parent_id::Union{Nothing, Int},
                          depth::Int,
                          side::Union{Nothing, Symbol},
                          solve_res,
                          ilb::Vector{Float64},
                          iub::Vector{Float64},
                          branch_history::Vector{BranchEvent},
                          fixation_history::Vector{FixationEvent};
                          status::Symbol = :active,
                          status_reason::String = "")
    return BaBNode(
        id = node_id,
        parent_id = parent_id,
        depth = depth,
        last_side = side,
        raw_best_lb = solve_res.raw_best_lb,
        best_lb = solve_res.best_lb,
        best_ub = solve_res.best_ub,
        rel_gap = solve_res.rel_gap,
        ub_candidate = solve_res.ub_candidate,
        ub_valid = solve_res.ub_valid,
        pop_length = length(solve_res.pop),
        numeq = solve_res.numeq,
        moment_matrix = solve_res.moment_matrix,
        solution = copy(solve_res.solution),
        tssos_normalized = solve_res.tssos_normalized,
        tssos_objective_scale = solve_res.tssos_objective_scale,
        tssos_min_constraint_scale = solve_res.tssos_min_constraint_scale,
        tssos_max_constraint_scale = solve_res.tssos_max_constraint_scale,
        tssos_report = solve_res.tssos_report,
        tssos_candidate_report = solve_res.tssos_candidate_report,
        tssos_lb_state = solve_res.tssos_lb_state,
        tssos_lb_reason = solve_res.tssos_lb_reason,
        tssos_call_seconds = solve_res.tssos_call_seconds,
        tssos_sdp_assemble_seconds = solve_res.tssos_sdp_assemble_seconds,
        tssos_sdp_solve_seconds = solve_res.tssos_sdp_solve_seconds,
        incumbent_source = solve_res.incumbent_source,
        fallback_attempted = solve_res.fallback_attempted,
        fallback_attempts = solve_res.fallback_attempts,
        ilb = copy(ilb),
        iub = copy(iub),
        branch_history = copy(branch_history),
        fixation_history = copy(fixation_history),
        status = status,
        status_reason = status_reason,
        feasibility_report = solve_res.feasibility_report,
    )
end

function _print_iteration_header(iteration::Int,
                                 active_count::Int,
                                 frontier_count::Int,
                                 best_lb::Float64,
                                 best_ub::Float64)
    println()
    println(repeat("=", 88))
    println("ITERATION $iteration")
    println(repeat("=", 88))
    println("Active nodes   : $active_count")
    println("Frontier nodes : $frontier_count")
    println("Global LB      : $(_fmt(best_lb))")
    println("Global UB      : $(_fmt(best_ub))")
    println("Global gap     : $(_fmt(_global_gap(best_lb, best_ub)))")
end

function _print_node_state(prob::PreparedBaBInput, node::BaBNode, global_ub::Float64)
    println("Selected node  : id=$(node.id), depth=$(node.depth), status=$(node.status)")
    println("Node LB / UB   : $(_fmt(node.best_lb)) / $(_fmt(node.best_ub))")
    println("Node gap       : $(_fmt(node.rel_gap))")
    println("Gap to global UB : $(_fmt(_global_gap(node.best_lb, global_ub)))")
    println("Node path      : ", join(_branch_path_summary(node.branch_history), " | "))
    println("Local bounds   : ", join(_tightened_bounds_summary(prob, node.ilb, node.iub), " ; "))
    println("TSSOS scaling  : normalized=$(node.tssos_normalized), objective=$(_fmt(node.tssos_objective_scale)), constraints=[$(_fmt(node.tssos_min_constraint_scale)), $(_fmt(node.tssos_max_constraint_scale))]")
    println("TSSOS times    : solve=$(_fmt(node.tssos_sdp_solve_seconds)), assemble=$(_fmt(node.tssos_sdp_assemble_seconds)), call=$(_fmt(node.tssos_call_seconds))")
    println("TSSOS refine   : ", _tssos_refine_summary(node.tssos_report))
    println("TSSOS LB       : ", _tssos_lb_summary(node))
    if node.tssos_candidate_report !== nothing
        println("TSSOS extracted pt: ", _tssos_extracted_point_summary(node))
    end
    println("Incumbent src  : ", node.incumbent_source)
    println("Fallback NLP   : ", _fallback_summary(node))
    println("Model-fixed vars : ", join(_model_fixed_bounds_summary(prob), " ; "))
    println("CDK-fixed vars   : ", join(_cdk_fixed_bounds_summary(prob, node.ilb, node.iub), " ; "))
    if node.ub_valid
        println("Chosen incumbent : valid feasible point")
    else
        println("Chosen incumbent : no valid feasible point found, candidate UB = $(_fmt(node.ub_candidate))")
        report = node.feasibility_report
        solver_summary = _local_solver_summary(report)
        solver_summary !== nothing && println("  local solver : ", solver_summary)
        if report !== nothing
            println("  diagnosis    : ", _feasibility_diagnostic(report))
            println("  note         : invalid incumbent is not a proof that the node region is infeasible")
            println("  violations   : bound=$(_fmt(report.max_bound_violation)), ineq=$(_fmt(report.max_ineq_violation)), eq=$(_fmt(report.max_eq_violation))")
        end
    end
end

function _print_cdk_rows(rows; title::String = "Top CDK candidates:")
    println(title)
    println(rpad("idx", 6), rpad("name", 12), rpad("lb", 14), rpad("ub", 14), rpad("cut_point", 16), "interval")
    for row in rows
        println(
            rpad(string(row.idx), 6),
            rpad(row.name, 12),
            rpad(_fmt(row.lb), 14),
            rpad(_fmt(row.ub), 14),
            rpad(_fmt(row.cut_point), 16),
            _fmt(row.interval),
        )
    end
end

function branch_and_bound_instance(prob::PreparedBaBInput;
                                   dir_strategy::Symbol = :cdk,
                                   cut_strategy::Symbol = :cdk,
                                   fix_CDK::Bool = true,
                                   max_iter::Int = 100,
                                   time_limit::Union{Nothing, Real} = nothing,
                                   tol::Float64 = 0.005,
                                   optimality_tol::Float64 = 1e-5,
                                   d::Int = 1,
                                   seed::Union{Nothing, Int} = nothing,
                                   TS::Bool = false,
                                   CS::Bool = false,
                                   QUIET::Bool = true,
                                   verbose::Bool = true,
                                   tssos_normalize::Bool = true,
                                   branch_encoding::Symbol = prob.bound_encoding,
                                   nlp_normalize::Bool = true,
                                   nlp_max_iter::Int = 2000,
                                   nlp_fallback_random_starts::Int = 4,
                                   nlp_fallback_max_attempts::Union{Nothing, Int} = 2,
                                   cdk_level::Union{Symbol, Real} = :marginal_dim,
                                   cdk_top::Int = 5,
                                   rank_tol::Float64 = 1e-4,
                                   unreliable_relaxation_policy::Symbol = :width_fallback,
                                   exclude_near_bound_cuts::Bool = true,
                                   cut_near_bound_rel_tol::Float64 = 1e-6,
                                   cut_near_bound_abs_tol::Float64 = 1e-5,
                                   fix_rel_tol::Float64 = 1e-8,
                                   fix_abs_tol::Float64 = 1e-8,
                                   fix_round_digits::Union{Nothing, Int} = 5,
                                   cut_round_digits::Union{Nothing, Int} = 5,
                                   width_tol::Float64 = 1e-10,
                                   feasibility_ineq_tol::Float64 = 1e-6,
                                   feasibility_eq_tol::Float64 = 1e-6,
                                   feasibility_bound_tol::Float64 = 1e-6)
    unreliable_relaxation_policy in (:stop_node, :width_fallback) ||
        error("Unsupported unreliable_relaxation_policy: $(unreliable_relaxation_policy). Use :stop_node or :width_fallback.")
    start_time_ns = time_ns()
    time_limit_sec = isnothing(time_limit) ? nothing : max(0.0, Float64(time_limit))
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    verbose && println("Solving root relaxation for $(prob.name)")
    root = root_relaxation(
        prob;
        d = d,
        seed = seed,
        TS = TS,
        CS = CS,
        QUIET = QUIET,
        tssos_normalize = tssos_normalize,
        branch_encoding = branch_encoding,
        local_solver_time_limit = _remaining_time(start_time_ns, time_limit_sec),
        nlp_normalize = nlp_normalize,
        nlp_max_iter = nlp_max_iter,
        nlp_fallback_random_starts = nlp_fallback_random_starts,
        nlp_fallback_max_attempts = nlp_fallback_max_attempts,
        feasibility_ineq_tol = feasibility_ineq_tol,
        feasibility_eq_tol = feasibility_eq_tol,
        feasibility_bound_tol = feasibility_bound_tol,
    )

    nodes = BaBNode[]
    root_node = _node_from_solve(
        1,
        nothing,
        0,
        nothing,
        root,
        prob.ilb_root,
        prob.iub_root,
        BranchEvent[],
        FixationEvent[];
        status = root.tssos_lb_state == :relaxation_infeasible ? :pruned_relaxation_infeasible : :active,
        status_reason = root.tssos_lb_state == :relaxation_infeasible ? "root relaxation infeasible" :
                        (root.tssos_lb_state == :unreliable && unreliable_relaxation_policy == :stop_node ?
                         "root unreliable relaxation stopped" :
                         (root.tssos_lb_state == :unreliable ? "root relaxation LB not trusted" :
                          (root.tssos_lb_state == :approximate ? "root relaxation LB approximate" : "root"))),
    )
    if root.tssos_lb_state == :unreliable && unreliable_relaxation_policy == :stop_node
        root_node.status = :stopped_unreliable
    end
    push!(nodes, root_node)

    frontier_ids = root.tssos_lb_state == :relaxation_infeasible ? Int[] : [root_node.id]
    active_ids = if root.tssos_lb_state == :relaxation_infeasible
        Int[]
    elseif root.tssos_lb_state == :unreliable && unreliable_relaxation_policy == :stop_node
        Int[]
    else
        [root_node.id]
    end

    best_lower_bound = root.best_lb
    best_upper_bound = root.best_ub
    root_refresh = _prune_and_update_gaps!(
        nodes,
        active_ids,
        frontier_ids,
        best_upper_bound,
        tol,
        optimality_tol,
    )
        if isempty(active_ids)
            best_lower_bound = _recompute_best_lower_bound(
                nodes,
                frontier_ids,
                best_lower_bound,
                best_upper_bound,
                optimality_tol,
            )
            if verbose && (!isempty(root_refresh.pruned_ids) || !isempty(root_refresh.closed_ids))
                println("Root queue refresh: pruned $(length(root_refresh.pruned_ids)) node(s), closed $(length(root_refresh.closed_ids)) node(s).")
            end
        end
    lb_hist = Float64[best_lower_bound]
    ub_hist = Float64[best_upper_bound]
    gap_hist = Float64[_global_gap(best_lower_bound, best_upper_bound)]

    iteration = 0
    max_iter_reached = false
    time_limit_reached = false

    while !isempty(active_ids)
        if _time_limit_hit(start_time_ns, time_limit_sec)
            time_limit_reached = true
            verbose && println("Time limit reached.")
            break
        end

        if iteration >= max_iter
            max_iter_reached = true
            verbose && println("Maximum iteration limit reached.")
            break
        end

        refresh = _prune_and_update_gaps!(
            nodes,
            active_ids,
            frontier_ids,
            best_upper_bound,
            tol,
            optimality_tol,
        )
        if verbose && (!isempty(refresh.pruned_ids) || !isempty(refresh.closed_ids))
            println("Queue refresh under current global UB: pruned $(length(refresh.pruned_ids)) node(s), closed $(length(refresh.closed_ids)) node(s).")
        end
        if isempty(active_ids)
            if verbose
                if any(nodes[id].status == :stopped_unreliable for id in frontier_ids)
                    println("No active nodes remain; unresolved nodes stopped by unreliable_relaxation_policy=$(unreliable_relaxation_policy) remain on the frontier.")
                else
                    println("No active nodes remain after pruning and gap updates.")
                end
            end
            break
        end

        iteration += 1
        best_lower_bound = _recompute_best_lower_bound(
            nodes,
            frontier_ids,
            best_lower_bound,
            best_upper_bound,
            optimality_tol,
        )
        verbose && _print_iteration_header(iteration, length(active_ids), length(frontier_ids), best_lower_bound, best_upper_bound)

        current_id = if isfinite(best_upper_bound)
            priorities = [(_lb_state_priority(nodes[id]), _active_priority(nodes[id], best_upper_bound)) for id in active_ids]
            active_ids[argmax(priorities)]
        else
            priorities = [(_lb_state_priority(nodes[id]), -nodes[id].best_lb) for id in active_ids]
            active_ids[argmax(priorities)]
        end
        filter!(id -> id != current_id, active_ids)
        filter!(id -> id != current_id, frontier_ids)

        current_node = nodes[current_id]
        verbose && _print_node_state(prob, current_node, best_upper_bound)

        ilb = copy(current_node.ilb)
        iub = copy(current_node.iub)

        cdk_guidance_available = current_node.tssos_lb_state == :valid
        cut_points, interval_lengths, coords_to_fix_CDK, cut_direction_CDK =
            if cdk_guidance_available
                next_step_CDK_local(
                    current_node.moment_matrix,
                    prob.vars;
                    branch_idx = prob.branch_idx,
                    ilb = ilb,
                    iub = iub,
                    rank_tol = rank_tol,
                    cdk_level = cdk_level,
                    exclude_near_bound = exclude_near_bound_cuts,
                    cut_near_bound_rel_tol = cut_near_bound_rel_tol,
                    cut_near_bound_abs_tol = cut_near_bound_abs_tol,
                    width_tol = width_tol,
                )
            elseif unreliable_relaxation_policy == :width_fallback
                midpoint_cuts = _midpoint_cut_points(ilb, iub)
                width_scores = _width_scores(ilb, iub)
                midpoint_cuts, width_scores, Int[], nothing
            else
                zeros(length(ilb)), zeros(length(ilb)), Int[], nothing
            end

        coords_to_fix = select_coords_to_fix(
            fix_CDK,
            coords_to_fix_CDK,
            prob;
            ilb = ilb,
            iub = iub,
            width_tol = width_tol,
        )

        if !cdk_guidance_available && unreliable_relaxation_policy == :stop_node
            current_node.status = :stopped_unreliable
            current_node.status_reason = "stopped by unreliable_relaxation_policy ($(current_node.tssos_lb_reason))"
            current_node.selected_coords_to_fix = Int[]
            current_node.selected_cut_points = Float64.(cut_points)
            current_node.selected_interval_lengths = Float64.(interval_lengths)
            current_node.ilb = copy(ilb)
            current_node.iub = copy(iub)
            best_lower_bound = _recompute_best_lower_bound(
                nodes,
                frontier_ids,
                best_lower_bound,
                best_upper_bound,
                optimality_tol,
            )
            push!(lb_hist, best_lower_bound)
            push!(ub_hist, best_upper_bound)
            push!(gap_hist, _global_gap(best_lower_bound, best_upper_bound))
            if verbose
                println("Applied fixations:")
                println("  (none)")
                println("Effective bounds after fixation:")
                println("  ", join(_tightened_bounds_summary(prob, ilb, iub; max_items = 12), " ; "))
                println("Model-fixed vars after fixation:")
                println("  ", join(_model_fixed_bounds_summary(prob; max_items = 12), " ; "))
                println("CDK-fixed vars after fixation:")
                println("  ", join(_cdk_fixed_bounds_summary(prob, ilb, iub; max_items = 12), " ; "))
                println("Branch scoring mode:")
                println("  stop node (TSSOS moment matrix not trusted for CDK: $(current_node.tssos_lb_reason))")
                println("Node $current_id is not branched further under unreliable_relaxation_policy=$(unreliable_relaxation_policy).")
            end
            continue
        end

        new_fixations = FixationEvent[]
        for idx in coords_to_fix
            a = ilb[idx]
            b = iub[idx]
            m = cut_points[idx]
            val = _stabilize_fix_value(
                Float64(m),
                Float64(a),
                Float64(b);
                rel = fix_rel_tol,
                abs = fix_abs_tol,
                round_digits = fix_round_digits,
            )
            if abs(a - val) <= width_tol && abs(b - val) <= width_tol
                continue
            end
            ilb[idx] = val
            iub[idx] = val
            push!(new_fixations, FixationEvent(
                iteration = iteration,
                node_id = current_id,
                var_idx = idx,
                var_name = prob.var_names[idx],
                old_lb = Float64(a),
                old_ub = Float64(b),
                pseudo_moment = Float64(m),
                fixed_value = Float64(val),
                marginal_rank = marginal_moment_rank_local(current_node.moment_matrix, idx; threshold = rank_tol),
                polynomial = build_branch_polynomial(prob.vars[idx], val, val; encoding = branch_encoding),
            ))
        end

        updated_fixations = vcat(current_node.fixation_history, new_fixations)
        current_node.fixation_history = updated_fixations
        current_node.selected_coords_to_fix = Int.(coords_to_fix)
        current_node.selected_cut_points = Float64.(cut_points)
        current_node.selected_interval_lengths = Float64.(interval_lengths)
        current_node.ilb = copy(ilb)
        current_node.iub = copy(iub)

        if verbose
            println("Applied fixations:")
            for row in _fixation_summary(new_fixations)
                println("  ", row)
            end
            println("Effective bounds after fixation:")
            println("  ", join(_tightened_bounds_summary(prob, ilb, iub; max_items = 12), " ; "))
            println("Model-fixed vars after fixation:")
            println("  ", join(_model_fixed_bounds_summary(prob; max_items = 12), " ; "))
            println("CDK-fixed vars after fixation:")
            println("  ", join(_cdk_fixed_bounds_summary(prob, ilb, iub; max_items = 12), " ; "))
            if !cdk_guidance_available
                println("Branch scoring mode:")
                println("  width fallback (TSSOS moment matrix not trusted for CDK: $(current_node.tssos_lb_reason))")
            end
            row_title = cdk_guidance_available ? "Top CDK candidates:" : "Top width-fallback candidates:"
            _print_cdk_rows(_top_branch_rows(prob, cut_points, interval_lengths, ilb, iub; top = cdk_top); title = row_title)
        end

        cut_direction = select_direction(
            dir_strategy,
            cut_points,
            interval_lengths,
            cut_direction_CDK,
            prob;
            excluded = Int.(coords_to_fix),
            ilb = ilb,
            iub = iub,
            rng = rng,
            width_tol = width_tol,
        )

        if cut_direction === nothing
            current_node.status = :terminal
            current_node.status_reason = "No branchable direction after fixation."
            push!(frontier_ids, current_id)
            best_lower_bound = _recompute_best_lower_bound(
                nodes,
                frontier_ids,
                best_lower_bound,
                best_upper_bound,
                optimality_tol,
            )
            push!(lb_hist, best_lower_bound)
            push!(ub_hist, best_upper_bound)
            push!(gap_hist, _global_gap(best_lower_bound, best_upper_bound))
            verbose && println("No branchable direction remains. Node $current_id becomes terminal.")
            continue
        end

        cut_value = select_cut_value(
            cut_points,
            cut_direction,
            ilb,
            iub;
            strategy = cut_strategy,
            rng = rng,
            round_digits = cut_round_digits,
        )

        current_node.selected_cut_direction = cut_direction
        current_node.selected_cut_value = cut_value
        current_node.status = :branched
        current_node.status_reason = "Branched on $(prob.var_names[cut_direction])"

        if verbose
            println("Selected branch variable : $(prob.var_names[cut_direction]) [idx=$(cut_direction)]")
            println("Selected cut value       : $(_fmt(cut_value))")
            println("Branch interval before   : [$(_fmt(ilb[cut_direction])), $(_fmt(iub[cut_direction]))]")
        end

        children = make_child_bounds(ilb, iub, cut_direction, cut_value)
        child_specs = (left = children.left, right = children.right)

        for side in (:left, :right)
            child_bounds = getfield(child_specs, side)
            child_poly = build_branch_polynomial(
                prob.vars[cut_direction],
                child_bounds.ilb[cut_direction],
                child_bounds.iub[cut_direction];
                encoding = branch_encoding,
            )
            branch_event = BranchEvent(
                iteration = iteration,
                parent_node_id = current_id,
                child_side = side,
                var_idx = cut_direction,
                var_name = prob.var_names[cut_direction],
                parent_lb = Float64(ilb[cut_direction]),
                parent_ub = Float64(iub[cut_direction]),
                cut_value = Float64(cut_value),
                child_lb = Float64(child_bounds.ilb[cut_direction]),
                child_ub = Float64(child_bounds.iub[cut_direction]),
                polynomial = child_poly,
            )

            solve_res = solve_node(
                prob;
                ilb = child_bounds.ilb,
                iub = child_bounds.iub,
                d = d,
                TS = TS,
                CS = CS,
                QUIET = QUIET,
                tssos_normalize = tssos_normalize,
                branch_encoding = branch_encoding,
                local_solver_time_limit = _remaining_time(start_time_ns, time_limit_sec),
                nlp_normalize = nlp_normalize,
                nlp_max_iter = nlp_max_iter,
                nlp_fallback_random_starts = nlp_fallback_random_starts,
                nlp_fallback_max_attempts = nlp_fallback_max_attempts,
                feasibility_ineq_tol = feasibility_ineq_tol,
                feasibility_eq_tol = feasibility_eq_tol,
                feasibility_bound_tol = feasibility_bound_tol,
            )

            if _time_limit_hit(start_time_ns, time_limit_sec)
                time_limit_reached = true
            end

            if solve_res.ub_valid && solve_res.best_ub < best_upper_bound
                best_upper_bound = solve_res.best_ub
                verbose && println("Updated global UB from $(side) child: $(_fmt(best_upper_bound))")
                refresh = _prune_and_update_gaps!(
                    nodes,
                    active_ids,
                    frontier_ids,
                    best_upper_bound,
                    tol,
                    optimality_tol,
                )
                if verbose && (!isempty(refresh.pruned_ids) || !isempty(refresh.closed_ids))
                    println("Queue refresh after incumbent update: pruned $(length(refresh.pruned_ids)) node(s), closed $(length(refresh.closed_ids)) node(s).")
                end
            end

            effective_solve_res = if solve_res.tssos_lb_state == :unreliable
                inherited_lb = current_node.best_lb
                inherited_gap = isfinite(solve_res.best_ub) ? (solve_res.best_ub - inherited_lb) / max(abs(solve_res.best_ub), 1.0) : Inf
                merge(solve_res, (best_lb = inherited_lb, rel_gap = inherited_gap))
            elseif solve_res.tssos_lb_state == :approximate
                inherited_lb = current_node.best_lb
                inherited_gap = isfinite(solve_res.best_ub) ? (solve_res.best_ub - inherited_lb) / max(abs(solve_res.best_ub), 1.0) : Inf
                merge(solve_res, (best_lb = inherited_lb, rel_gap = inherited_gap))
            else
                solve_res
            end

            if effective_solve_res.tssos_lb_state == :relaxation_infeasible
                child_status = :pruned_relaxation_infeasible
                child_reason = "relaxation infeasible"
            elseif effective_solve_res.tssos_lb_state == :unreliable &&
                   unreliable_relaxation_policy == :stop_node
                child_status = :stopped_unreliable
                child_reason = "stopped by unreliable_relaxation_policy ($(effective_solve_res.tssos_lb_reason)); inherited parent LB"
            else
                child_status, child_reason = _child_status(
                    effective_solve_res.best_lb,
                    best_upper_bound,
                    tol,
                    optimality_tol,
                )
                if effective_solve_res.tssos_lb_state == :unreliable
                    child_reason = string(
                        child_reason,
                        "; TSSOS LB not trusted (",
                        effective_solve_res.tssos_lb_reason,
                        "), inherited parent LB"
                    )
                elseif effective_solve_res.tssos_lb_state == :approximate
                    child_reason = string(
                        child_reason,
                        "; TSSOS LB approximate (",
                        effective_solve_res.tssos_lb_reason,
                        "), inherited parent LB"
                    )
                end
            end
            child_reason = _decorate_child_reason(child_reason, effective_solve_res)

            child_node = _node_from_solve(
                length(nodes) + 1,
                current_id,
                current_node.depth + 1,
                side,
                effective_solve_res,
                child_bounds.ilb,
                child_bounds.iub,
                vcat(current_node.branch_history, [branch_event]),
                updated_fixations;
                status = child_status,
                status_reason = child_reason,
            )

            push!(nodes, child_node)

            if child_status != :pruned_nonfinite &&
               child_status != :pruned_fathomed &&
               child_status != :pruned_relaxation_infeasible &&
               child_status != :stopped_unreliable
                push!(frontier_ids, child_node.id)
                if child_status == :active
                    push!(active_ids, child_node.id)
                end
            end

            if verbose
                println()
                println("$(uppercase(String(side))) child branch polynomial:")
                println("  ", child_poly)
                println("$(uppercase(String(side))) child interval:")
                println("  $(prob.var_names[cut_direction]) in [$(_fmt(child_bounds.ilb[cut_direction])), $(_fmt(child_bounds.iub[cut_direction]))]")
                println("$(uppercase(String(side))) child result:")
                println("  node id      : $(child_node.id)")
                println("  LB / UB      : $(_fmt(child_node.best_lb)) / $(_fmt(child_node.best_ub))")
                println("  UB candidate : $(_fmt(child_node.ub_candidate)) (valid=$(child_node.ub_valid))")
                println("  gap          : $(_fmt(child_node.rel_gap))")
                println("  gap to global UB : $(_fmt(_global_gap(child_node.best_lb, best_upper_bound)))")
                println("  status       : $(child_node.status) ($(child_node.status_reason))")
                println("  TSSOS scaling: normalized=$(child_node.tssos_normalized), objective=$(_fmt(child_node.tssos_objective_scale)), constraints=[$(_fmt(child_node.tssos_min_constraint_scale)), $(_fmt(child_node.tssos_max_constraint_scale))]")
                println("  TSSOS times  : solve=$(_fmt(child_node.tssos_sdp_solve_seconds)), assemble=$(_fmt(child_node.tssos_sdp_assemble_seconds)), call=$(_fmt(child_node.tssos_call_seconds))")
                println("  TSSOS refine : $(_tssos_refine_summary(child_node.tssos_report))")
                println("  TSSOS LB     : $(_tssos_lb_summary(child_node))")
                if child_node.tssos_candidate_report !== nothing
                    println("  TSSOS extracted pt : ", _tssos_extracted_point_summary(child_node))
                end
                println("  incumbent src : $(child_node.incumbent_source)")
                println("  fallback NLP  : $(_fallback_summary(child_node))")
                println("  local bounds : ", join(_tightened_bounds_summary(prob, child_node.ilb, child_node.iub; max_items = 12), " ; "))
                println("  model-fixed vars : ", join(_model_fixed_bounds_summary(prob; max_items = 12), " ; "))
                println("  CDK-fixed vars   : ", join(_cdk_fixed_bounds_summary(prob, child_node.ilb, child_node.iub; max_items = 12), " ; "))
                if !child_node.ub_valid && child_node.feasibility_report !== nothing
                    rep = child_node.feasibility_report
                    solver_summary = _local_solver_summary(rep)
                    solver_summary !== nothing && println("  local solver : ", solver_summary)
                    println("  diagnosis    : ", _feasibility_diagnostic(rep))
                    println("  note         : invalid incumbent is not a proof that the node region is infeasible")
                    println("  violations   : bound=$(_fmt(rep.max_bound_violation)), ineq=$(_fmt(rep.max_ineq_violation)), eq=$(_fmt(rep.max_eq_violation))")
                end
            end
        end

        if time_limit_reached
            verbose && println("Time limit reached.")
            break
        end

        best_lower_bound = _recompute_best_lower_bound(
            nodes,
            frontier_ids,
            best_lower_bound,
            best_upper_bound,
            optimality_tol,
        )

        push!(lb_hist, best_lower_bound)
        push!(ub_hist, best_upper_bound)
        push!(gap_hist, _global_gap(best_lower_bound, best_upper_bound))

        if verbose
            println()
            println("End-of-iteration summary:")
            println("  active nodes   : $(length(active_ids))")
            println("  frontier nodes : $(length(frontier_ids))")
            println("  global LB / UB : $(_fmt(best_lower_bound)) / $(_fmt(best_upper_bound))")
            println("  global gap     : $(_fmt(_global_gap(best_lower_bound, best_upper_bound)))")
        end
    end

    best_lower_bound = _recompute_best_lower_bound(
        nodes,
        frontier_ids,
        best_lower_bound,
        best_upper_bound,
        optimality_tol,
    )

    metadata = Dict{String, Any}(
        "dir_strategy" => dir_strategy,
        "cut_strategy" => cut_strategy,
        "fix_CDK" => fix_CDK,
        "bound_encoding" => prob.bound_encoding,
        "branch_encoding" => branch_encoding,
        "scaled_to_unit_box" => prob.scaled_to_unit_box,
        "relaxation_order" => d,
        "tssos_normalize" => tssos_normalize,
        "local_ub_solver" => "tssos_refine_with_nlp_fallback",
        "nlp_normalize" => nlp_normalize,
        "nlp_max_iter" => nlp_max_iter,
        "nlp_fallback_random_starts" => nlp_fallback_random_starts,
        "nlp_fallback_max_attempts" => nlp_fallback_max_attempts,
        "rank_tol" => rank_tol,
        "unreliable_relaxation_policy" => unreliable_relaxation_policy,
        "exclude_near_bound_cuts" => exclude_near_bound_cuts,
        "cut_near_bound_rel_tol" => cut_near_bound_rel_tol,
        "cut_near_bound_abs_tol" => cut_near_bound_abs_tol,
        "fix_rel_tol" => fix_rel_tol,
        "fix_abs_tol" => fix_abs_tol,
        "fix_round_digits" => fix_round_digits,
        "cut_round_digits" => cut_round_digits,
        "tol" => tol,
        "optimality_tol" => optimality_tol,
        "seed" => seed,
        "time_limit" => time_limit_sec,
        "feasibility_ineq_tol" => feasibility_ineq_tol,
        "feasibility_eq_tol" => feasibility_eq_tol,
        "feasibility_bound_tol" => feasibility_bound_tol,
        "elapsed_metric" => "sum_tssos_sdp_solve_seconds",
    )

    total_tssos_sdp_solve_seconds = sum(node.tssos_sdp_solve_seconds for node in nodes)
    total_tssos_sdp_assemble_seconds = sum(node.tssos_sdp_assemble_seconds for node in nodes)
    wall_clock_seconds = _elapsed_seconds(start_time_ns)

    return BaBResult(
        name = prob.name,
        best_lower_bound = best_lower_bound,
        best_upper_bound = best_upper_bound,
        iterations = iteration,
        lb_hist = lb_hist,
        ub_hist = ub_hist,
        gap_hist = gap_hist,
        nodes = nodes,
        active_node_ids = active_ids,
        frontier_node_ids = frontier_ids,
        max_iter_reached = max_iter_reached,
        time_limit_reached = time_limit_reached,
        elapsed_seconds = total_tssos_sdp_solve_seconds,
        wall_clock_seconds = wall_clock_seconds,
        tssos_sdp_assemble_seconds = total_tssos_sdp_assemble_seconds,
        metadata = metadata,
    )
end

function branch_and_bound(inst::FEMPolyInstance;
                          bound_encoding::Symbol = :quadratic,
                          branch_encoding::Symbol = bound_encoding,
                          scale_to_unit_box::Bool = true,
                          kwargs...)
    prob = prepare_bab_input(inst; bound_encoding = bound_encoding, scale_to_unit_box = scale_to_unit_box)
    return branch_and_bound_instance(prob; branch_encoding = branch_encoding, kwargs...)
end

function branch_and_bound_instance(inst::FEMPolyInstance;
                                   bound_encoding::Symbol = :quadratic,
                                   branch_encoding::Symbol = bound_encoding,
                                   scale_to_unit_box::Bool = true,
                                   kwargs...)
    return branch_and_bound(
        inst;
        bound_encoding = bound_encoding,
        branch_encoding = branch_encoding,
        scale_to_unit_box = scale_to_unit_box,
        kwargs...,
    )
end

function branch_and_bound_fem_instance(inst::FEMPolyInstance;
                                       bound_encoding::Symbol = :quadratic,
                                       branch_encoding::Symbol = bound_encoding,
                                       scale_to_unit_box::Bool = true,
                                       kwargs...)
    return branch_and_bound(
        inst;
        bound_encoding = bound_encoding,
        branch_encoding = branch_encoding,
        scale_to_unit_box = scale_to_unit_box,
        kwargs...,
    )
end

function bab_summary_string(result::BaBResult; max_nodes::Int = 20)
    lines = String[]
    unique_fix_events = _unique_fixation_events(result.nodes)
    fixing_node_ids = sort!(unique([evt.node_id for evt in unique_fix_events]))
    fixed_var_names = sort!(unique(evt.var_name for evt in unique_fix_events))
    inheriting_fix_nodes = count(node -> !isempty(node.fixation_history), result.nodes)
    tssos_refine_successes = count(node -> node.tssos_report !== nothing &&
                                          hasproperty(node.tssos_report, :local_refine_success) &&
                                          node.tssos_report.local_refine_success, result.nodes)
    tssos_refine_failures = count(node -> node.tssos_report !== nothing &&
                                         hasproperty(node.tssos_report, :local_refine_failed) &&
                                         node.tssos_report.local_refine_failed, result.nodes)
    tssos_incumbents_used = count(node -> node.incumbent_source == :tssos_refine, result.nodes)
    fallback_attempted = count(node -> node.fallback_attempted, result.nodes)
    fallback_recovered = count(node -> node.incumbent_source == :nlp_fallback, result.nodes)
    fallback_failed = count(node -> node.fallback_attempted && node.incumbent_source != :nlp_fallback, result.nodes)
    tssos_lb_approximate = count(node -> node.tssos_lb_state == :approximate, result.nodes)
    tssos_lb_unreliable = count(node -> node.tssos_lb_state == :unreliable, result.nodes)
    tssos_relax_infeasible = count(node -> node.tssos_lb_state == :relaxation_infeasible, result.nodes)
    stopped_unreliable = count(node -> node.status == :stopped_unreliable, result.nodes)
    stopped_unreliable_frontier = count(id -> result.nodes[id].status == :stopped_unreliable, result.frontier_node_ids)
    time_limit = get(result.metadata, "time_limit", nothing)
    fixed_var_summary = isempty(fixed_var_names) ? "(none)" : string(fixed_var_names)
    push!(lines, repeat("=", 88))
    push!(lines, "BaB SUMMARY FOR $(result.name)")
    push!(lines, repeat("=", 88))
    push!(lines, "Iterations          : $(result.iterations)")
    push!(lines, "Elapsed seconds     : $(_fmt(result.elapsed_seconds))")
    push!(lines, "Wall clock seconds  : $(_fmt(result.wall_clock_seconds))")
    push!(lines, "SDP assemble sec    : $(_fmt(result.tssos_sdp_assemble_seconds))")
    if !isnothing(time_limit)
        push!(lines, "Time limit (sec)    : $(_fmt(time_limit))")
    end
    push!(lines, "TSSOS normalize     : $(get(result.metadata, "tssos_normalize", missing))")
    if haskey(result.metadata, "local_ub_solver")
        push!(lines, "Local UB solver     : $(result.metadata["local_ub_solver"])")
        push!(lines, "NLP normalize       : $(get(result.metadata, "nlp_normalize", missing))")
        push!(lines, "NLP fallback starts : $(get(result.metadata, "nlp_fallback_random_starts", missing))")
        push!(lines, "NLP fallback cap    : $(get(result.metadata, "nlp_fallback_max_attempts", missing))")
    end
    if haskey(result.metadata, "elapsed_metric")
        push!(lines, "Elapsed metric      : $(get(result.metadata, "elapsed_metric", missing))")
    end
    if haskey(result.metadata, "rank_tol")
        push!(lines, "CDK rank tol        : $(result.metadata["rank_tol"])")
    end
    if haskey(result.metadata, "unreliable_relaxation_policy")
        push!(lines, "Unreliable policy   : $(result.metadata["unreliable_relaxation_policy"])")
    end
    if haskey(result.metadata, "scaled_to_unit_box")
        push!(lines, "Scaled to unit box  : $(result.metadata["scaled_to_unit_box"])")
    end
    if haskey(result.metadata, "branch_encoding")
        push!(lines, "Branch encoding     : $(result.metadata["branch_encoding"])")
    end
    if haskey(result.metadata, "fix_round_digits")
        push!(lines, "CDK fix rounding    : digits=$(result.metadata["fix_round_digits"]), tol=($(get(result.metadata, "fix_rel_tol", missing)), $(get(result.metadata, "fix_abs_tol", missing)))")
    end
    if haskey(result.metadata, "cut_round_digits")
        push!(lines, "CDK cut rounding    : digits=$(result.metadata["cut_round_digits"])")
    end
    if haskey(result.metadata, "exclude_near_bound_cuts")
        push!(
            lines,
            "CDK near-bound cuts : exclude=$(result.metadata["exclude_near_bound_cuts"]), tol=($(get(result.metadata, "cut_near_bound_rel_tol", missing)), $(get(result.metadata, "cut_near_bound_abs_tol", missing)))",
        )
    end
    push!(lines, "Best LB             : $(_fmt(result.best_lower_bound))")
    push!(lines, "Best UB             : $(_fmt(result.best_upper_bound))")
    push!(lines, "Final gap           : $(_fmt(_global_gap(result.best_lower_bound, result.best_upper_bound)))")
    push!(lines, "Max iter reached    : $(result.max_iter_reached)")
    push!(lines, "Time limit reached  : $(result.time_limit_reached)")
    push!(lines, "Total nodes         : $(length(result.nodes))")
    push!(lines, "Active nodes        : $(length(result.active_node_ids))")
    push!(lines, "Frontier nodes      : $(length(result.frontier_node_ids))")

    counts = Dict{Symbol, Int}()
    for node in result.nodes
        counts[node.status] = get(counts, node.status, 0) + 1
    end
    push!(lines, "Node status counts  : $(counts)")
    push!(lines, "CDK fix events      : $(length(unique_fix_events))")
    push!(lines, "Fixing nodes        : $(length(fixing_node_ids))")
    push!(lines, "Nodes with CDK fixes: $(inheriting_fix_nodes)")
    push!(lines, "CDK fixed vars      : $(fixed_var_summary)")
    push!(lines, "TSSOS refine ok     : $(tssos_refine_successes)")
    push!(lines, "TSSOS refine failed : $(tssos_refine_failures)")
    push!(lines, "TSSOS incumbents used : $(tssos_incumbents_used)")
    push!(lines, "TSSOS LB approximate  : $(tssos_lb_approximate)")
    push!(lines, "TSSOS LB unreliable   : $(tssos_lb_unreliable)")
    push!(lines, "Relaxation infeasible : $(tssos_relax_infeasible)")
    push!(lines, "Stopped unreliable    : $(stopped_unreliable)")
    push!(lines, "Stopped on frontier   : $(stopped_unreliable_frontier)")
    push!(lines, "Fallback NLP attempted: $(fallback_attempted)")
    push!(lines, "Fallback NLP recovered: $(fallback_recovered)")
    push!(lines, "Fallback NLP failed   : $(fallback_failed)")
    if stopped_unreliable_frontier > 0
        push!(lines, "Certification note    : unresolved unreliable-relaxation nodes remain on the frontier")
    end

    if !isempty(result.nodes)
        push!(lines, "")
        push!(lines, "First nodes:")
        push!(lines, rpad("id", 6) * rpad("parent", 8) * rpad("depth", 8) * rpad("status", 18) * rpad("LB", 14) * rpad("UB", 14) * "path")
        for node in result.nodes[1:min(max_nodes, length(result.nodes))]
            parent_label = isnothing(node.parent_id) ? "-" : string(node.parent_id)
            path_label = join(_branch_path_summary(node.branch_history; max_items = 2), " | ")
            push!(
                lines,
                rpad(string(node.id), 6) *
                rpad(parent_label, 8) *
                rpad(string(node.depth), 8) *
                rpad(string(node.status), 18) *
                rpad(_fmt(node.best_lb), 14) *
                rpad(_fmt(node.best_ub), 14) *
                path_label,
            )
        end
    end

    return join(lines, "\n")
end

function node_summary_string(node::BaBNode; max_events::Int = 20)
    lines = String[]
    push!(lines, "NODE $(node.id)")
    push!(lines, repeat("-", 60))
    push!(lines, "parent_id            : $(node.parent_id)")
    push!(lines, "depth                : $(node.depth)")
    push!(lines, "status               : $(node.status)")
    push!(lines, "status_reason        : $(node.status_reason)")
    push!(lines, "raw_best_lb          : $(_fmt(node.raw_best_lb))")
    push!(lines, "best_lb              : $(_fmt(node.best_lb))")
    push!(lines, "best_ub              : $(_fmt(node.best_ub))")
    push!(lines, "ub_candidate         : $(_fmt(node.ub_candidate))")
    push!(lines, "ub_valid             : $(node.ub_valid)")
    push!(lines, "rel_gap              : $(_fmt(node.rel_gap))")
    push!(lines, "tssos_normalized     : $(node.tssos_normalized)")
    push!(lines, "tssos_objective_scale: $(_fmt(node.tssos_objective_scale))")
    push!(lines, "tssos_constraint_rng : [$(_fmt(node.tssos_min_constraint_scale)), $(_fmt(node.tssos_max_constraint_scale))]")
    push!(lines, "tssos_sdp_solve_sec  : $(_fmt(node.tssos_sdp_solve_seconds))")
    push!(lines, "tssos_assemble_sec  : $(_fmt(node.tssos_sdp_assemble_seconds))")
    push!(lines, "tssos_call_sec       : $(_fmt(node.tssos_call_seconds))")
    push!(lines, "tssos_refine_status  : $(_tssos_refine_summary(node.tssos_report))")
    push!(lines, "tssos_lb_status      : $(_tssos_lb_summary(node))")
    if node.tssos_candidate_report !== nothing
        push!(lines, "tssos_extracted_diag : $(_tssos_extracted_point_summary(node))")
    end
    push!(lines, "incumbent_source     : $(node.incumbent_source)")
    push!(lines, "fallback_attempted   : $(node.fallback_attempted)")
    push!(lines, "fallback_attempts    : $(node.fallback_attempts)")
    push!(lines, "selected_cut_dir     : $(node.selected_cut_direction)")
    push!(lines, "selected_cut_value   : $(node.selected_cut_value)")
    push!(lines, "selected_coords_fix  : $(node.selected_coords_to_fix)")
    push!(lines, "branch history:")
    if isempty(node.branch_history)
        push!(lines, "  (root)")
    else
        for evt in node.branch_history[1:min(max_events, length(node.branch_history))]
            push!(lines, "  $(evt.child_side): $(evt.var_name) in [$(_fmt(evt.child_lb)), $(_fmt(evt.child_ub))] ; poly = $(evt.polynomial)")
        end
    end
    push!(lines, "fixation history:")
    if isempty(node.fixation_history)
        push!(lines, "  (none)")
    else
        for evt in node.fixation_history[1:min(max_events, length(node.fixation_history))]
            push!(lines, "  $(evt.var_name) = $(_fmt(evt.fixed_value)) ; rank = $(evt.marginal_rank) ; poly = $(evt.polynomial)")
        end
    end
    if node.feasibility_report !== nothing
        rep = node.feasibility_report
        push!(lines, "feasibility:")
        solver_summary = _local_solver_summary(rep)
        solver_summary !== nothing && push!(lines, "  local_solver        = $(solver_summary)")
        push!(lines, "  diagnosis           = $(_feasibility_diagnostic(rep))")
        push!(lines, "  max_bound_violation = $(_fmt(rep.max_bound_violation))")
        push!(lines, "  max_ineq_violation  = $(_fmt(rep.max_ineq_violation))")
        push!(lines, "  max_eq_violation    = $(_fmt(rep.max_eq_violation))")
        if hasproperty(rep, :normalization_enabled)
            push!(lines, "  normalization       = $(rep.normalization_enabled)")
            push!(lines, "  objective_scale     = $(_fmt(rep.objective_scale))")
            push!(lines, "  constraint_scales   = [$(_fmt(rep.min_constraint_scale)), $(_fmt(rep.max_constraint_scale))]")
        end
    end
    return join(lines, "\n")
end

function node_summary_string(prob::PreparedBaBInput, node::BaBNode; max_events::Int = 20, max_bounds::Int = 20)
    lines = split(node_summary_string(node; max_events = max_events), "\n")
    insert!(lines, 11, "local_bounds         : " * join(_tightened_bounds_summary(prob, node.ilb, node.iub; max_items = max_bounds), " ; "))
    insert!(lines, 12, "model_fixed_vars     : " * join(_model_fixed_bounds_summary(prob; max_items = max_bounds), " ; "))
    insert!(lines, 13, "cdk_fixed_vars       : " * join(_cdk_fixed_bounds_summary(prob, node.ilb, node.iub; max_items = max_bounds), " ; "))
    return join(lines, "\n")
end

function _node_chain(result::BaBResult, node_id::Int)
    1 <= node_id <= length(result.nodes) || error("node_id $(node_id) is out of range for result with $(length(result.nodes)) nodes.")
    chain = BaBNode[]
    current_id = node_id
    while true
        node = result.nodes[current_id]
        push!(chain, node)
        isnothing(node.parent_id) && break
        current_id = node.parent_id
    end
    reverse!(chain)
    return chain
end

function _last_branch_label(node::BaBNode)
    isempty(node.branch_history) && return "(root)"
    evt = node.branch_history[end]
    return "$(evt.child_side): $(evt.var_name) in [$(_fmt(evt.child_lb)), $(_fmt(evt.child_ub))] (cut=$(_fmt(evt.cut_value)))"
end

function _incoming_branch_label(node::BaBNode)
    isnothing(node.parent_id) && return "(root)"
    return _last_branch_label(node)
end

function path_summary_string(result::BaBResult, node_id::Int; max_children::Int = 8)
    chain = _node_chain(result, node_id)
    target = chain[end]
    lines = String[]

    push!(lines, "PATH TO NODE $(node_id) IN $(result.name)")
    push!(lines, repeat("-", 100))

    approx_idx = findfirst(node -> node.tssos_lb_state == :approximate, chain)
    unreliable_idx = findfirst(node -> node.tssos_lb_state == :unreliable, chain)
    approx_idx !== nothing && push!(lines, "First approximate LB : node $(chain[approx_idx].id)")
    unreliable_idx !== nothing && push!(lines, "First unreliable LB  : node $(chain[unreliable_idx].id)")
    isempty(lines[end]) || push!(lines, "")

    push!(
        lines,
        rpad("id", 6) *
        rpad("depth", 8) *
        rpad("status", 24) *
        rpad("tssos_lb", 14) *
        rpad("LB", 14) *
        rpad("UB", 14) *
        "last branch",
    )
    for node in chain
        push!(
            lines,
            rpad(string(node.id), 6) *
            rpad(string(node.depth), 8) *
            rpad(string(node.status), 24) *
            rpad(string(node.tssos_lb_state), 14) *
            rpad(_fmt(node.best_lb), 14) *
            rpad(_fmt(node.best_ub), 14) *
            _last_branch_label(node),
        )
    end

    notable = [node for node in chain if node.tssos_lb_state != :valid || !isempty(node.status_reason)]
    if !isempty(notable)
        push!(lines, "")
        push!(lines, "Notable states along the path:")
        for node in notable
            push!(lines, "  node $(node.id): TSSOS LB = $(_tssos_lb_summary(node))")
            !isempty(node.status_reason) && push!(lines, "    status_reason = $(node.status_reason)")
            if node.feasibility_report !== nothing && (!node.ub_valid || node.incumbent_source == :none)
                push!(lines, "    feasibility   = $(_feasibility_diagnostic(node.feasibility_report))")
            end
        end
    end

    children = sort([node for node in result.nodes if node.parent_id == target.id]; by = node -> node.id)
    if !isempty(children)
        push!(lines, "")
        push!(lines, "Immediate children of node $(target.id):")
        push!(
            lines,
            rpad("id", 6) *
            rpad("side", 8) *
            rpad("status", 24) *
            rpad("tssos_lb", 14) *
            rpad("LB", 14) *
            rpad("UB", 14) *
            "incoming branch",
        )
        for child in children[1:min(max_children, length(children))]
            push!(
                lines,
                rpad(string(child.id), 6) *
                rpad(string(child.last_side), 8) *
                rpad(string(child.status), 24) *
                rpad(string(child.tssos_lb_state), 14) *
                rpad(_fmt(child.best_lb), 14) *
                rpad(_fmt(child.best_ub), 14) *
                _incoming_branch_label(child),
            )
        end
        if length(children) > max_children
            push!(lines, "  ... ($(length(children) - max_children) more children)")
        end
    end

    return join(lines, "\n")
end

function Base.show(io::IO, ::MIME"text/plain", result::BaBResult)
    print(io, bab_summary_string(result; max_nodes = 12))
end

function print_bab_summary(result::BaBResult; max_nodes::Int = 20)
    println()
    println(bab_summary_string(result; max_nodes = max_nodes))
end

end
