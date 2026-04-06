include(joinpath(@__DIR__, "pop_bab_api.jl"))
include(joinpath(@__DIR__, "example24_instance.jl"))

function main()
    inst = build_illustrative_pop()

    println("CDK-guided Branch-and-Bound on an illustrative POP")
    println("===================================================")
    println("Instance            : $(inst.name)")
    println("Bounds              : x1 in [0, 2], x2 in [2, 4]")
    println("Expected optimizer  : (2, 2)")
    println("Expected optimum    : -2")
    println()

    res = branch_and_bound(
        inst;
        scale_to_unit_box = false,
        dir_strategy = :cdk,
        cut_strategy = :cdk,
        fix_CDK = true,
        d = 1,
        cdk_level = :marginal_dim,
        tol = 1.0e-4,
        max_iter = 10,
        seed = 100,
        verbose = true,
    )

    println(bab_summary_string(res; max_nodes = 12))
end

main()
