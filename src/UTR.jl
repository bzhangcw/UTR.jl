__precompile__()
module UTR

using Printf

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T,Nothing}

# various utilities
include("utilities/autodiff.jl")
include("utilities/fbtools.jl")
include("utilities/iterationtools.jl")
include("utilities/displaytools.jl")
include("utilities/counter.jl")
include("utilities/interpolation.jl")
include("utilities/linesearches.jl")

# subproblem solvers
include("utilities/subp/atrs/ATRS.jl")
include("utilities/subp/ghm.jl")
include("utilities/subp/trs.jl")
include("utilities/subp/lanczos.jl")
include("utilities/subp/cg.jl")
include("utilities/subp/cubic.jl")

# algorithm interface + implementations needed for the UTR paper experiments
include("algorithms/interface.jl")
include("algorithms/utr.jl")
include("algorithms/atr.jl")
include("algorithms/atrms.jl")

# vanilla cubic regularization (baseline used in the experiments)
include("others/cubicreg_vanilla.jl")
# vanilla Monteiro-Svaiter accelerated method
include("others/ms.jl")

# Algorithm Aliases
# NOTE: the module itself is named `UTR`, so the universal trust-region
# constructor is exported under its full name `UniversalTrustRegion`
# (the `UTR` short alias would clash with the module binding).
CRM = CubicRegularizationVanilla
MS = AcceleratedMonteiroSvaiter
ATR = AcceleratedUniversalTrustRegion
ATRMS = AcceleratedUniversalTrustRegionMonteiroSvaiter

function __init__()
end

export Result
export UniversalTrustRegion, ATR, ATRMS
export CubicRegularizationVanilla, AcceleratedMonteiroSvaiter
export CRM, MS
end # module
