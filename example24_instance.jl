using DynamicPolynomials

"""
    build_illustrative_pop()

Construct the illustrative POP instance used in this GitHub bundle.
"""
function build_illustrative_pop()
    @polyvar x1 x2

    f = -(x1 - 1)^2 - (x1 - x2)^2 - (x2 - 3)^2
    g1 = 1 - (x1 - 1)^2
    g2 = 1 - (x1 - x2)^2
    g3 = 1 - (x2 - 3)^2
    g4 = x1 - 0.3 * x2^2

    return POPInstance(
        name = "example_2_4",
        source_path = "manual",
        vars = [x1, x2],
        var_names = ["x1", "x2"],
        var_index = Dict("x1" => 1, "x2" => 2),
        objective = f,
        inequalities = Any[g1, g2, g3, g4],
        equalities = Any[],
        lb = [0.0, 2.0],
        ub = [2.0, 4.0],
        branch_idx = [1, 2],
        epigraph_idx = nothing,
    )
end

build_example24_instance() = build_illustrative_pop()
