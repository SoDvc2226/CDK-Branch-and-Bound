module FEMLPLoader

using JuMP
using DynamicPolynomials
using MultivariatePolynomials
using Downloads
import MathOptInterface as MOI

export FEMPolyInstance,
       extract_fem_instances,
       list_lp_files,
       load_lp_model,
       build_fem_instance,
       build_nl_instance,
       minlplib_target_instances,
       minlplib_instance_url,
       download_minlplib_instance,
       build_minlplib_instance,
       build_minlplib_ts_instance,
       scale_instance_to_unit_box,
       box_bound_polynomials,
       tssos_pop,
       instance_summary,
       validation_report,
       generic_instance_summary,
       generic_validation_report

Base.@kwdef struct FEMPolyInstance
    name::String
    source_path::String
    vars::Vector{Any}
    var_names::Vector{String}
    var_index::Dict{String, Int}
    objective::Any
    inequalities::Vector{Any}
    equalities::Vector{Any}
    lb::Vector{Float64}
    ub::Vector{Float64}
    branch_idx::Vector{Int}
    epigraph_idx::Union{Nothing, Int} = nothing
end

const MINLPLIB_SCREENSHOT_INSTANCES = [
    "pooling_adhya1pq",
    "pooling_adhya1stp",
    "pooling_adhya1tp",
    "pooling_adhya2pq",
    "pooling_adhya2stp",
    "pooling_adhya2tp",
    "pooling_adhya3pq",
    "pooling_adhya3stp",
    "pooling_adhya3tp",
    "pooling_adhya4pq",
    "pooling_adhya4stp",
    "pooling_adhya4tp",
    "pooling_bental4pq",
    "pooling_bental4stp",
    "pooling_bental4tp",
    "pooling_bental5pq",
    "pooling_bental5tp",
    "pooling_foulds2pq",
    "pooling_foulds2stp",
    "pooling_foulds2tp",
    "pooling_haverly1pq",
    "pooling_haverly1stp",
    "pooling_haverly1tp",
    "pooling_haverly2pq",
    "pooling_haverly2stp",
    "pooling_haverly2tp",
    "pooling_haverly3pq",
    "pooling_haverly3stp",
    "pooling_haverly3tp",
    "pooling_rt2pq",
    "pooling_rt2stp",
    "pooling_rt2tp",
    "wastewater02m1",
    "wastewater02m2",
    "wastewater04m1",
    "wastewater04m2",
    "wastewater05m1",
    "wastewater14m1",
    "wastewater15m1",
    "waterund01",
    "waterund08",
    "waterund11",
    "waterund17",
    "waterund18",
]

minlplib_target_instances() = copy(MINLPLIB_SCREENSHOT_INSTANCES)

function extract_fem_instances(zip_path::AbstractString;
                               outdir::Union{Nothing, AbstractString} = nothing,
                               overwrite::Bool = false)
    zip_path_abs = abspath(zip_path)
    outdir === nothing && (outdir = joinpath(dirname(zip_path_abs), "fem_instances_extracted"))
    outdir_abs = abspath(outdir)
    mkpath(outdir_abs)

    needs_extract = overwrite || isempty(list_lp_files(outdir_abs))
    if needs_extract
        if !Sys.iswindows()
            error("Automatic ZIP extraction is currently implemented for Windows only. Extract $zip_path_abs manually and call list_lp_files on the extracted folder.")
        end

        command = "Expand-Archive -LiteralPath '$(zip_path_abs)' -DestinationPath '$(outdir_abs)' -Force"
        run(Cmd(["powershell", "-NoProfile", "-Command", command]))
    end

    return list_lp_files(outdir_abs)
end

function list_lp_files(root::AbstractString)
    root_abs = abspath(root)
    if isfile(root_abs)
        endswith(lowercase(root_abs), ".lp") || error("Not an .lp file: $root_abs")
        return [root_abs]
    end

    files = String[]
    for (dir, _, names) in walkdir(root_abs)
        for name in names
            if endswith(lowercase(name), ".lp") &&
               !startswith(name, "._") &&
               !occursin("__MACOSX", dir)
                push!(files, joinpath(dir, name))
            end
        end
    end
    return sort!(files)
end

load_lp_model(lp_path::AbstractString) = read_from_file(abspath(lp_path))

function minlplib_instance_url(name::AbstractString; format::Symbol = :lp)
    fmt = lowercase(String(format))
    fmt == "lp" || error("Only format=:lp is currently supported.")
    bare = replace(basename(name), r"\.lp$" => "")
    return "https://www.minlplib.org/lp/$(bare).lp"
end

function download_minlplib_instance(name::AbstractString;
                                    outdir::AbstractString = joinpath(pwd(), "minlplib_instances"),
                                    overwrite::Bool = false)
    bare = replace(basename(name), r"\.lp$" => "")
    mkpath(outdir)
    lp_path = abspath(joinpath(outdir, bare * ".lp"))
    if !overwrite && isfile(lp_path)
        return lp_path
    end

    Downloads.download(minlplib_instance_url(bare), lp_path)
    return lp_path
end

function _collect_variables(model::JuMP.Model)
    moi_model = backend(model)
    moi_vars = collect(MOI.get(moi_model, MOI.ListOfVariableIndices()))
    sort!(moi_vars; by = v -> v.value)

    var_names = String[]
    for (i, vidx) in enumerate(moi_vars)
        name = MOI.get(moi_model, MOI.VariableName(), vidx)
        push!(var_names, isempty(name) ? "var_$i" : name)
    end

    var_pos = Dict{MOI.VariableIndex, Int}(vidx => i for (i, vidx) in enumerate(moi_vars))
    name_to_pos = Dict{String, Int}(name => i for (i, name) in enumerate(var_names))

    return moi_model, moi_vars, var_names, var_pos, name_to_pos
end

function _polyvars(n::Int)
    raw = DynamicPolynomials.@polyvar z[1:n]
    return raw isa Tuple ? collect(first(raw)) : collect(raw)
end

function _moi_function_to_poly(func, polyvars, var_pos)
    if func isa MOI.VariableIndex
        return polyvars[var_pos[func]]
    elseif func isa MOI.ScalarAffineFunction
        poly = func.constant
        for term in func.terms
            poly += term.coefficient * polyvars[var_pos[term.variable]]
        end
        return poly
    elseif func isa MOI.ScalarQuadraticFunction
        poly = func.constant
        for term in func.affine_terms
            poly += term.coefficient * polyvars[var_pos[term.variable]]
        end
        for term in func.quadratic_terms
            poly += term.coefficient *
                    polyvars[var_pos[term.variable_1]] *
                    polyvars[var_pos[term.variable_2]]
        end
        return poly
    else
        error("Unsupported MOI function type: $(typeof(func))")
    end
end

function _apply_bound!(lb::Vector{Float64}, ub::Vector{Float64},
                       pos::Int, sense::Symbol, value::Real)
    val = Float64(value)
    if sense == :lb
        lb[pos] = max(lb[pos], val)
    elseif sense == :ub
        ub[pos] = min(ub[pos], val)
    elseif sense == :eq
        lb[pos] = max(lb[pos], val)
        ub[pos] = min(ub[pos], val)
    else
        error("Unknown bound sense: $sense")
    end
    return nothing
end

function _capture_variable_bound!(lb::Vector{Float64}, ub::Vector{Float64},
                                  var_pos::Dict{MOI.VariableIndex, Int},
                                  func::MOI.VariableIndex, set)
    pos = var_pos[func]
    if set isa MOI.GreaterThan
        _apply_bound!(lb, ub, pos, :lb, set.lower)
    elseif set isa MOI.LessThan
        _apply_bound!(lb, ub, pos, :ub, set.upper)
    elseif set isa MOI.EqualTo
        _apply_bound!(lb, ub, pos, :eq, set.value)
    elseif set isa MOI.Interval
        _apply_bound!(lb, ub, pos, :lb, set.lower)
        _apply_bound!(lb, ub, pos, :ub, set.upper)
    elseif set isa MOI.ZeroOne
        _apply_bound!(lb, ub, pos, :lb, 0.0)
        _apply_bound!(lb, ub, pos, :ub, 1.0)
    elseif set isa MOI.Integer
        # No additional finite bounds are implied.
    else
        error("Unsupported variable set: $(typeof(set))")
    end
    return nothing
end

function _single_affine_bounds(func::MOI.ScalarAffineFunction, set)
    length(func.terms) == 1 || return nothing
    term = func.terms[1]
    coeff = term.coefficient
    iszero(coeff) && return nothing

    function affine_bound(rhs)
        return (rhs - func.constant) / coeff
    end

    if set isa MOI.LessThan
        bound = affine_bound(set.upper)
        return coeff > 0 ?
               [(term.variable, :ub, bound)] :
               [(term.variable, :lb, bound)]
    elseif set isa MOI.GreaterThan
        bound = affine_bound(set.lower)
        return coeff > 0 ?
               [(term.variable, :lb, bound)] :
               [(term.variable, :ub, bound)]
    elseif set isa MOI.EqualTo
        return [(term.variable, :eq, affine_bound(set.value))]
    elseif set isa MOI.Interval
        lower = affine_bound(set.lower)
        upper = affine_bound(set.upper)
        if coeff > 0
            return [(term.variable, :lb, lower), (term.variable, :ub, upper)]
        else
            return [(term.variable, :lb, upper), (term.variable, :ub, lower)]
        end
    end

    return nothing
end

function _constraint_polys(func, set, polyvars, var_pos)
    poly = _moi_function_to_poly(func, polyvars, var_pos)
    ineqs = Any[]
    eqs = Any[]

    if set isa MOI.LessThan
        push!(ineqs, set.upper - poly)
    elseif set isa MOI.GreaterThan
        push!(ineqs, poly - set.lower)
    elseif set isa MOI.EqualTo
        push!(eqs, poly - set.value)
    elseif set isa MOI.Interval
        if set.lower == set.upper
            push!(eqs, poly - set.lower)
        else
            push!(ineqs, poly - set.lower)
            push!(ineqs, set.upper - poly)
        end
    else
        error("Unsupported scalar set: $(typeof(set))")
    end

    return ineqs, eqs
end

function _nl_expr_to_poly(expr, polyvars, var_pos)
    if expr isa Real
        return Float64(expr)
    elseif expr isa MOI.VariableIndex
        return polyvars[var_pos[expr]]
    elseif expr isa Expr
        if expr.head == :ref
            length(expr.args) == 2 || error("Unsupported nonlinear variable reference: $expr")
            expr.args[1] == :x || error("Unsupported nonlinear variable container in expression: $expr")
            vidx = expr.args[2]
            vidx isa MOI.VariableIndex || error("Expected MOI.VariableIndex in nonlinear expression: $expr")
            return polyvars[var_pos[vidx]]
        elseif expr.head == :call
            op = expr.args[1]
            args = expr.args[2:end]
            if op === :+
                acc = _nl_expr_to_poly(args[1], polyvars, var_pos)
                for arg in args[2:end]
                    acc += _nl_expr_to_poly(arg, polyvars, var_pos)
                end
                return acc
            elseif op === :-
                if length(args) == 1
                    return -_nl_expr_to_poly(args[1], polyvars, var_pos)
                end
                acc = _nl_expr_to_poly(args[1], polyvars, var_pos)
                for arg in args[2:end]
                    acc -= _nl_expr_to_poly(arg, polyvars, var_pos)
                end
                return acc
            elseif op === :*
                acc = _nl_expr_to_poly(args[1], polyvars, var_pos)
                for arg in args[2:end]
                    acc *= _nl_expr_to_poly(arg, polyvars, var_pos)
                end
                return acc
            elseif op === :/
                length(args) == 2 || error("Unsupported nonlinear division expression: $expr")
                num = _nl_expr_to_poly(args[1], polyvars, var_pos)
                den = args[2]
                den isa Real || error("Only division by constants is supported in nonlinear expressions: $expr")
                return num / Float64(den)
            elseif op === :^
                length(args) == 2 || error("Unsupported nonlinear power expression: $expr")
                base = _nl_expr_to_poly(args[1], polyvars, var_pos)
                exponent = args[2]
                exponent isa Integer || (exponent isa Real && isinteger(exponent)) ||
                    error("Only integer exponents are supported in nonlinear expressions: $expr")
                power = Int(round(Float64(exponent)))
                power >= 0 || error("Negative exponents are not polynomial: $expr")
                return base^power
            elseif op in (:(==), :<=, :>=)
                error("Comparison operators should be handled at the constraint level: $expr")
            else
                error("Unsupported nonlinear operator $(repr(op)) in expression: $expr")
            end
        else
            error("Unsupported nonlinear Expr head $(expr.head) in expression: $expr")
        end
    end
    error("Unsupported nonlinear expression node $(typeof(expr)): $expr")
end

function _nl_constraint_polys(expr, bounds::MOI.NLPBoundsPair, polyvars, var_pos)
    ineqs = Any[]
    eqs = Any[]

    if expr isa Expr && expr.head == :call && !isempty(expr.args) && expr.args[1] in (:(==), :<=, :>=)
        op = expr.args[1]
        lhs = _nl_expr_to_poly(expr.args[2], polyvars, var_pos)
        rhs = _nl_expr_to_poly(expr.args[3], polyvars, var_pos)
        if op === :(==)
            push!(eqs, lhs - rhs)
        elseif op === :<=
            push!(ineqs, rhs - lhs)
        else
            push!(ineqs, lhs - rhs)
        end
        return ineqs, eqs
    end

    poly = _nl_expr_to_poly(expr, polyvars, var_pos)
    lower = Float64(bounds.lower)
    upper = Float64(bounds.upper)
    if isfinite(lower) && isfinite(upper)
        if lower == upper
            push!(eqs, poly - lower)
        else
            push!(ineqs, poly - lower)
            push!(ineqs, upper - poly)
        end
    elseif isfinite(lower)
        push!(ineqs, poly - lower)
    elseif isfinite(upper)
        push!(ineqs, upper - poly)
    else
        error("Encountered a nonlinear constraint with no finite bounds.")
    end

    return ineqs, eqs
end

function _affine_interval_without_var(func::MOI.ScalarAffineFunction,
                                      skip_var::MOI.VariableIndex,
                                      lb::Vector{Float64},
                                      ub::Vector{Float64},
                                      var_pos::Dict{MOI.VariableIndex, Int})
    lo = Float64(func.constant)
    hi = Float64(func.constant)
    for term in func.terms
        term.variable == skip_var && continue
        pos = var_pos[term.variable]
        li = lb[pos]
        ui = ub[pos]
        if !isfinite(li) || !isfinite(ui)
            return (-Inf, Inf)
        end
        coeff = Float64(term.coefficient)
        if coeff >= 0.0
            lo += coeff * li
            hi += coeff * ui
        else
            lo += coeff * ui
            hi += coeff * li
        end
    end
    return lo, hi
end

function _isolated_var_upper(func::MOI.ScalarAffineFunction,
                             rhs::Real,
                             target::MOI.VariableIndex,
                             lb::Vector{Float64},
                             ub::Vector{Float64},
                             var_pos::Dict{MOI.VariableIndex, Int})
    coeff = 0.0
    for term in func.terms
        if term.variable == target
            coeff += Float64(term.coefficient)
        end
    end
    iszero(coeff) && return nothing

    rest_lo, rest_hi = _affine_interval_without_var(func, target, lb, ub, var_pos)
    if !isfinite(rest_lo) || !isfinite(rest_hi)
        return nothing
    end

    val1 = (Float64(rhs) - rest_hi) / coeff
    val2 = (Float64(rhs) - rest_lo) / coeff
    return max(val1, val2)
end

function _derive_epigraph_ub(epigraph_var::MOI.VariableIndex,
                             scalar_constraints,
                             lb::Vector{Float64},
                             ub::Vector{Float64},
                             var_pos::Dict{MOI.VariableIndex, Int};
                             pad::Float64 = 1e-4)
    candidates = Float64[]

    for (func, set) in scalar_constraints
        func isa MOI.ScalarAffineFunction || continue
        if !any(term.variable == epigraph_var for term in func.terms)
            continue
        end

        if set isa MOI.LessThan
            bound = _isolated_var_upper(func, set.upper, epigraph_var, lb, ub, var_pos)
            bound !== nothing && push!(candidates, bound)
        elseif set isa MOI.GreaterThan
            bound = _isolated_var_upper(func, set.lower, epigraph_var, lb, ub, var_pos)
            bound !== nothing && push!(candidates, bound)
        elseif set isa MOI.EqualTo
            bound = _isolated_var_upper(func, set.value, epigraph_var, lb, ub, var_pos)
            bound !== nothing && push!(candidates, bound)
        elseif set isa MOI.Interval
            bound1 = _isolated_var_upper(func, set.lower, epigraph_var, lb, ub, var_pos)
            bound2 = _isolated_var_upper(func, set.upper, epigraph_var, lb, ub, var_pos)
            bound1 !== nothing && push!(candidates, bound1)
            bound2 !== nothing && push!(candidates, bound2)
        end
    end

    isempty(candidates) && return nothing
    raw = maximum(candidates)
    margin = max(pad, abs(raw) * pad)
    return raw + margin
end

function _objective_var_index(moi_model, var_pos)
    objective_type = MOI.get(moi_model, MOI.ObjectiveFunctionType())
    if objective_type == MOI.VariableIndex
        return MOI.get(moi_model, MOI.ObjectiveFunction{MOI.VariableIndex}())
    elseif objective_type <: MOI.ScalarAffineFunction
        func = MOI.get(moi_model, MOI.ObjectiveFunction{objective_type}())
        if iszero(func.constant) && length(func.terms) == 1
            term = func.terms[1]
            if term.coefficient == 1.0
                return term.variable
            end
        end
    end
    return nothing
end

function build_fem_instance(lp_path::AbstractString;
                            skip_simple_bounds::Bool = true,
                            branch_name_prefixes::Tuple{Vararg{String}} = ("x_", "y_"),
                            infer_epigraph_bound::Bool = true)
    endswith(lowercase(lp_path), ".nl") && return build_nl_instance(lp_path)
    model = load_lp_model(lp_path)
    moi_model, moi_vars, var_names, var_pos, name_to_pos = _collect_variables(model)
    polyvars = _polyvars(length(moi_vars))

    lb = fill(-Inf, length(moi_vars))
    ub = fill(Inf, length(moi_vars))
    scalar_constraints = Vector{Tuple{Any, Any}}()
    ineqs = Any[]
    eqs = Any[]

    constraint_types = MOI.get(moi_model, MOI.ListOfConstraintTypesPresent())
    for (F, S) in constraint_types
        for ci in MOI.get(moi_model, MOI.ListOfConstraintIndices{F, S}())
            func = MOI.get(moi_model, MOI.ConstraintFunction(), ci)
            set = MOI.get(moi_model, MOI.ConstraintSet(), ci)

            if func isa MOI.VariableIndex
                _capture_variable_bound!(lb, ub, var_pos, func, set)
                continue
            end

            push!(scalar_constraints, (func, set))
            if skip_simple_bounds && func isa MOI.ScalarAffineFunction
                maybe_bounds = _single_affine_bounds(func, set)
                if maybe_bounds !== nothing
                    for (vidx, sense, value) in maybe_bounds
                        _apply_bound!(lb, ub, var_pos[vidx], sense, value)
                    end
                    continue
                end
            end

            new_ineqs, new_eqs = _constraint_polys(func, set, polyvars, var_pos)
            append!(ineqs, new_ineqs)
            append!(eqs, new_eqs)
        end
    end

    epigraph_var = _objective_var_index(moi_model, var_pos)
    epigraph_idx = nothing
    if epigraph_var !== nothing
        epigraph_idx = var_pos[epigraph_var]
        if infer_epigraph_bound && !isfinite(ub[epigraph_idx])
            inferred = _derive_epigraph_ub(epigraph_var, scalar_constraints, lb, ub, var_pos)
            inferred !== nothing && (ub[epigraph_idx] = inferred)
        end
    end

    objective_type = MOI.get(moi_model, MOI.ObjectiveFunctionType())
    objective = _moi_function_to_poly(
        MOI.get(moi_model, MOI.ObjectiveFunction{objective_type}()),
        polyvars,
        var_pos,
    )

    objective_sense = MOI.get(moi_model, MOI.ObjectiveSense())
    if objective_sense == MOI.MAX_SENSE
        objective = -objective
    elseif objective_sense == MOI.FEASIBILITY_SENSE
        error("The LP file has no objective. A minimization objective is required.")
    end

    branch_idx = findall(eachindex(var_names)) do i
        any(startswith(var_names[i], prefix) for prefix in branch_name_prefixes)
    end
    isempty(branch_idx) && (branch_idx = collect(setdiff(1:length(var_names), [epigraph_idx])))

    for i in branch_idx
        if !isfinite(lb[i]) || !isfinite(ub[i])
            error("Branch variable $(var_names[i]) does not have finite bounds.")
        end
    end

    for i in eachindex(var_names)
        if !isfinite(lb[i]) || !isfinite(ub[i])
            error("Variable $(var_names[i]) still has an infinite bound. Add a finite bound before calling TSSOS.")
        end
    end

    return FEMPolyInstance(
        name = basename(lp_path),
        source_path = abspath(lp_path),
        vars = collect(polyvars),
        var_names = var_names,
        var_index = name_to_pos,
        objective = objective,
        inequalities = ineqs,
        equalities = eqs,
        lb = lb,
        ub = ub,
        branch_idx = branch_idx,
        epigraph_idx = epigraph_idx,
    )
end

function build_nl_instance(source::AbstractString;
                           fixed_atol::Float64 = 1e-10)
    nl_path = abspath(source)
    isfile(nl_path) || error("NL file not found: $nl_path")
    endswith(lowercase(nl_path), ".nl") || error("build_nl_instance expects a .nl file: $nl_path")

    model = load_lp_model(nl_path)
    moi_model, moi_vars, var_names, var_pos, name_to_pos = _collect_variables(model)
    polyvars = _polyvars(length(moi_vars))

    lb = fill(-Inf, length(moi_vars))
    ub = fill(Inf, length(moi_vars))
    scalar_constraints = Vector{Tuple{Any, Any}}()
    ineqs = Any[]
    eqs = Any[]

    constraint_types = MOI.get(moi_model, MOI.ListOfConstraintTypesPresent())
    for (F, S) in constraint_types
        for ci in MOI.get(moi_model, MOI.ListOfConstraintIndices{F, S}())
            func = MOI.get(moi_model, MOI.ConstraintFunction(), ci)
            set = MOI.get(moi_model, MOI.ConstraintSet(), ci)

            if func isa MOI.VariableIndex
                _capture_variable_bound!(lb, ub, var_pos, func, set)
                continue
            end

            push!(scalar_constraints, (func, set))
            new_ineqs, new_eqs = _constraint_polys(func, set, polyvars, var_pos)
            append!(ineqs, new_ineqs)
            append!(eqs, new_eqs)
        end
    end

    objective = 0.0
    if MOI.supports(moi_model, MOI.NLPBlock())
        nlp = MOI.get(moi_model, MOI.NLPBlock())
        evaluator = nlp.evaluator
        MOI.initialize(evaluator, [:ExprGraph])

        if nlp.has_objective
            objective = _nl_expr_to_poly(MOI.objective_expr(evaluator), polyvars, var_pos)
        end

        for i in eachindex(nlp.constraint_bounds)
            new_ineqs, new_eqs = _nl_constraint_polys(
                MOI.constraint_expr(evaluator, i),
                nlp.constraint_bounds[i],
                polyvars,
                var_pos,
            )
            append!(ineqs, new_ineqs)
            append!(eqs, new_eqs)
        end
    else
        objective_type = MOI.get(moi_model, MOI.ObjectiveFunctionType())
        objective = _moi_function_to_poly(
            MOI.get(moi_model, MOI.ObjectiveFunction{objective_type}()),
            polyvars,
            var_pos,
        )
    end

    objective_sense = MOI.get(moi_model, MOI.ObjectiveSense())
    if objective_sense == MOI.MAX_SENSE
        objective = -objective
    elseif objective_sense == MOI.FEASIBILITY_SENSE
        error("The NL file has no objective. A minimization objective is required.")
    end

    epigraph_var = _objective_var_index(moi_model, var_pos)
    epigraph_idx = isnothing(epigraph_var) ? nothing : var_pos[epigraph_var]
    branch_idx = _default_branch_idx(var_names, lb, ub, epigraph_idx; fixed_atol = fixed_atol)
    isempty(branch_idx) && error("No non-fixed branchable variables were found for $(basename(nl_path)).")

    for i in eachindex(var_names)
        if !isfinite(lb[i]) || !isfinite(ub[i])
            error("Variable $(var_names[i]) still has an infinite bound in $(basename(nl_path)).")
        end
    end

    return FEMPolyInstance(
        name = basename(nl_path),
        source_path = nl_path,
        vars = collect(polyvars),
        var_names = var_names,
        var_index = name_to_pos,
        objective = objective,
        inequalities = ineqs,
        equalities = eqs,
        lb = lb,
        ub = ub,
        branch_idx = branch_idx,
        epigraph_idx = epigraph_idx,
    )
end

function _default_branch_idx(var_names::AbstractVector{<:AbstractString},
                             lb::AbstractVector{<:Real},
                             ub::AbstractVector{<:Real},
                             epigraph_idx::Union{Nothing, Int};
                             fixed_atol::Float64 = 1e-10)
    idx = Int[]
    for i in eachindex(var_names)
        !isnothing(epigraph_idx) && i == epigraph_idx && continue
        if !isfinite(lb[i]) || !isfinite(ub[i])
            continue
        end
        _is_fixed_bound(lb[i], ub[i]; atol = fixed_atol) && continue
        push!(idx, i)
    end
    return idx
end

function build_minlplib_instance(source::AbstractString;
                                 download_if_missing::Bool = false,
                                 outdir::AbstractString = joinpath(pwd(), "minlplib_instances"),
                                 infer_epigraph_bound::Bool = true,
                                 fixed_atol::Float64 = 1e-10)
    lp_path = if isfile(source)
        abspath(source)
    else
        bare = replace(basename(source), r"\.lp$" => "")
        candidate = abspath(joinpath(outdir, bare * ".lp"))
        if isfile(candidate)
            candidate
        elseif download_if_missing
            download_minlplib_instance(bare; outdir = outdir)
        else
            error("MINLPLib LP file not found: $candidate. Set download_if_missing=true to fetch it.")
        end
    end

    base = build_fem_instance(
        lp_path;
        branch_name_prefixes = ("__minlplib_branch_placeholder__",),
        infer_epigraph_bound = infer_epigraph_bound,
    )

    branch_idx = _default_branch_idx(
        base.var_names,
        base.lb,
        base.ub,
        base.epigraph_idx;
        fixed_atol = fixed_atol,
    )

    isempty(branch_idx) && error("No non-fixed branchable variables were found for $(base.name).")

    return FEMPolyInstance(
        name = base.name,
        source_path = base.source_path,
        vars = base.vars,
        var_names = base.var_names,
        var_index = base.var_index,
        objective = base.objective,
        inequalities = base.inequalities,
        equalities = base.equalities,
        lb = base.lb,
        ub = base.ub,
        branch_idx = branch_idx,
        epigraph_idx = base.epigraph_idx,
    )
end

function build_minlplib_ts_instance(source::AbstractString;
                                    nl_root::AbstractString = joinpath(pwd(), "MINLPLib-TS", "nl"),
                                    fixed_atol::Float64 = 1e-10)
    nl_path = if isfile(source)
        src = abspath(source)
        if endswith(lowercase(src), ".nl")
            src
        elseif endswith(lowercase(src), ".mod")
            candidate = abspath(joinpath(dirname(dirname(src)), "nl", replace(basename(src), r"\.mod$" => ".nl")))
            isfile(candidate) || error("Matching .nl file not found for $src")
            candidate
        else
            error("Unsupported MINLPLib-TS source path: $src")
        end
    else
        bare = replace(basename(source), r"\.(mod|nl)$" => "")
        candidate = abspath(joinpath(nl_root, bare * ".nl"))
        isfile(candidate) || error("MINLPLib-TS .nl file not found: $candidate")
        candidate
    end

    return build_nl_instance(nl_path; fixed_atol = fixed_atol)
end

function scale_instance_to_unit_box(inst::FEMPolyInstance;
                                    fixed_atol::Float64 = 1e-10)
    n = length(inst.vars)
    length(inst.lb) == n || error("Lower bounds length does not match number of variables.")
    length(inst.ub) == n || error("Upper bounds length does not match number of variables.")

    shifts = Float64.(inst.lb)
    scales = Float64.(inst.ub) .- Float64.(inst.lb)
    substitutions = Pair[]
    scaled_lb = zeros(Float64, n)
    scaled_ub = ones(Float64, n)

    for i in 1:n
        if !isfinite(shifts[i]) || !isfinite(inst.ub[i])
            error("Variable $(inst.var_names[i]) must have finite bounds to scale to the unit box.")
        end

        if abs(scales[i]) <= fixed_atol
            push!(substitutions, inst.vars[i] => shifts[i])
            scaled_lb[i] = 0.0
            scaled_ub[i] = 0.0
            scales[i] = 0.0
        else
            push!(substitutions, inst.vars[i] => shifts[i] + scales[i] * inst.vars[i])
            scaled_lb[i] = 0.0
            scaled_ub[i] = 1.0
        end
    end

    scaled_objective = subs(inst.objective, substitutions...)
    scaled_ineqs = [subs(poly, substitutions...) for poly in inst.inequalities]
    scaled_eqs = [subs(poly, substitutions...) for poly in inst.equalities]

    scaled_inst = FEMPolyInstance(
        name = inst.name,
        source_path = inst.source_path,
        vars = inst.vars,
        var_names = inst.var_names,
        var_index = inst.var_index,
        objective = scaled_objective,
        inequalities = scaled_ineqs,
        equalities = scaled_eqs,
        lb = scaled_lb,
        ub = scaled_ub,
        branch_idx = inst.branch_idx,
        epigraph_idx = inst.epigraph_idx,
    )

    return scaled_inst, shifts, scales
end

function _is_fixed_bound(lb::Real, ub::Real; atol::Float64 = 1e-10)
    return isapprox(Float64(lb), Float64(ub); atol = atol, rtol = 0.0)
end

function box_bound_polynomials(vars,
                               lb::AbstractVector{<:Real},
                               ub::AbstractVector{<:Real};
                               encoding::Symbol = :quadratic)
    length(vars) == length(lb) == length(ub) || error("vars, lb, and ub must have the same length.")

    ineqs = Any[]
    eqs = Any[]
    for i in eachindex(vars)
        li = Float64(lb[i])
        ui = Float64(ub[i])
        !isfinite(li) && error("Lower bound for variable $i is not finite.")
        !isfinite(ui) && error("Upper bound for variable $i is not finite.")

        if _is_fixed_bound(li, ui)
            push!(eqs, vars[i] - li)
        elseif encoding == :linear
            push!(ineqs, vars[i] - li)
            push!(ineqs, ui - vars[i])
        elseif encoding == :quadratic
            push!(ineqs, (vars[i] - li) * (ui - vars[i]))
        else
            error("Unsupported bound encoding: $encoding. Use :linear or :quadratic.")
        end
    end

    return ineqs, eqs
end

function tssos_pop(inst::FEMPolyInstance;
                   include_box_bounds::Bool = true,
                   bound_encoding::Symbol = :quadratic)
    pop = Any[inst.objective]
    append!(pop, inst.inequalities)

    if include_box_bounds
        bound_ineqs, bound_eqs = box_bound_polynomials(
            inst.vars,
            inst.lb,
            inst.ub;
            encoding = bound_encoding,
        )
        append!(pop, bound_ineqs)
        append!(pop, bound_eqs)
    end

    append!(pop, inst.equalities)
    return pop, length(inst.equalities) + (include_box_bounds ? length(bound_eqs) : 0)
end

function instance_summary(inst::FEMPolyInstance)
    epigraph_name = isnothing(inst.epigraph_idx) ? nothing : inst.var_names[inst.epigraph_idx]
    return Dict(
        "name" => inst.name,
        "source_path" => inst.source_path,
        "n_vars" => length(inst.vars),
        "n_branch_vars" => length(inst.branch_idx),
        "n_ineqs" => length(inst.inequalities),
        "n_eqs" => length(inst.equalities),
        "epigraph_var" => epigraph_name,
        "branch_names" => inst.var_names[inst.branch_idx],
        "min_lb" => minimum(inst.lb),
        "max_ub" => maximum(inst.ub),
        "default_bound_encoding" => "quadratic",
    )
end

function validation_report(inst::FEMPolyInstance;
                           include_box_bounds::Bool = true,
                           bound_encoding::Symbol = :quadratic)
    x_count = count(name -> startswith(name, "x_"), inst.var_names)
    y_count = count(name -> startswith(name, "y_"), inst.var_names)
    epigraph_count = count(==("_"), inst.var_names)
    branch_name_count = count(name -> startswith(name, "x_") || startswith(name, "y_"), inst.var_names)
    bound_ineqs, bound_eqs = include_box_bounds ?
        box_bound_polynomials(inst.vars, inst.lb, inst.ub; encoding = bound_encoding) :
        (Any[], Any[])
    expected_pop = 1 + length(inst.inequalities) + length(inst.equalities) +
                   length(bound_ineqs) + length(bound_eqs)
    pop, numeq = tssos_pop(
        inst;
        include_box_bounds = include_box_bounds,
        bound_encoding = bound_encoding,
    )

    return Dict(
        "n_vars" => length(inst.vars),
        "x_count" => x_count,
        "y_count" => y_count,
        "epigraph_count" => epigraph_count,
        "branch_idx_count" => length(inst.branch_idx),
        "branch_name_count" => branch_name_count,
        "n_ineqs" => length(inst.inequalities),
        "n_eqs" => length(inst.equalities),
        "n_bound_ineqs" => length(bound_ineqs),
        "n_bound_eqs" => length(bound_eqs),
        "n_total_eqs_in_pop" => length(inst.equalities) + length(bound_eqs),
        "bound_encoding" => String(bound_encoding),
        "numeq" => numeq,
        "pop_length" => length(pop),
        "expected_pop_length" => expected_pop,
        "all_bounds_finite" => all(isfinite, inst.lb) && all(isfinite, inst.ub),
        "branch_names_match_idx" => length(inst.branch_idx) == branch_name_count,
        "numeq_matches_total_eqs" => numeq == length(inst.equalities) + length(bound_eqs),
        "pop_length_matches_formula" => length(pop) == expected_pop,
    )
end

function generic_instance_summary(inst::FEMPolyInstance)
    fixed_idx = findall(i -> _is_fixed_bound(inst.lb[i], inst.ub[i]), eachindex(inst.var_names))
    epigraph_name = isnothing(inst.epigraph_idx) ? nothing : inst.var_names[inst.epigraph_idx]
    return Dict(
        "name" => inst.name,
        "source_path" => inst.source_path,
        "n_vars" => length(inst.vars),
        "n_branch_vars" => length(inst.branch_idx),
        "n_fixed_vars" => length(fixed_idx),
        "n_ineqs" => length(inst.inequalities),
        "n_eqs" => length(inst.equalities),
        "epigraph_var" => epigraph_name,
        "branch_names" => inst.var_names[inst.branch_idx],
        "fixed_var_names" => inst.var_names[fixed_idx],
        "min_lb" => minimum(inst.lb),
        "max_ub" => maximum(inst.ub),
        "default_bound_encoding" => "quadratic",
    )
end

function generic_validation_report(inst::FEMPolyInstance;
                                   include_box_bounds::Bool = true,
                                   bound_encoding::Symbol = :quadratic)
    bound_ineqs, bound_eqs = include_box_bounds ?
        box_bound_polynomials(inst.vars, inst.lb, inst.ub; encoding = bound_encoding) :
        (Any[], Any[])
    expected_pop = 1 + length(inst.inequalities) + length(inst.equalities) +
                   length(bound_ineqs) + length(bound_eqs)
    pop, numeq = tssos_pop(
        inst;
        include_box_bounds = include_box_bounds,
        bound_encoding = bound_encoding,
    )
    fixed_idx = findall(i -> _is_fixed_bound(inst.lb[i], inst.ub[i]), eachindex(inst.var_names))
    expected_branch_idx = _default_branch_idx(inst.var_names, inst.lb, inst.ub, inst.epigraph_idx)

    return Dict(
        "n_vars" => length(inst.vars),
        "n_branch_vars" => length(inst.branch_idx),
        "n_fixed_vars" => length(fixed_idx),
        "n_ineqs" => length(inst.inequalities),
        "n_eqs" => length(inst.equalities),
        "n_bound_ineqs" => length(bound_ineqs),
        "n_bound_eqs" => length(bound_eqs),
        "n_total_eqs_in_pop" => length(inst.equalities) + length(bound_eqs),
        "bound_encoding" => String(bound_encoding),
        "numeq" => numeq,
        "pop_length" => length(pop),
        "expected_pop_length" => expected_pop,
        "all_bounds_finite" => all(isfinite, inst.lb) && all(isfinite, inst.ub),
        "branch_idx_matches_nonfixed" => inst.branch_idx == expected_branch_idx,
        "numeq_matches_total_eqs" => numeq == length(inst.equalities) + length(bound_eqs),
        "pop_length_matches_formula" => length(pop) == expected_pop,
    )
end

end
