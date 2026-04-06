include(joinpath(@__DIR__, "src", "fem_bab_bridge.jl"))

const POPBaB = FEMBaBBridge
const POPInstance = FEMBaBBridge.FEMLPLoader.FEMPolyInstance
const branch_and_bound = FEMBaBBridge.branch_and_bound
const bab_summary_string = FEMBaBBridge.bab_summary_string
