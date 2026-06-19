# UTR.jl: Universal Trust-Region Methods

A standalone Julia package for the **Universal Trust-Region (UTR)** method and its
accelerated enhancements, originally hosted at
[DRSOM.jl](https://github.com/bzhangcw/DRSOM.jl).

## Algorithms

| Exported name                                   | Method                              |
| ----------------------------------------------- | ----------------------------------- |
| `UniversalTrustRegion`                          | Universal Trust-Region (UTR)        |
| `ATR` (`= AcceleratedUniversalTrustRegion`)     | Accelerated universal trust-region  |
| `MS`  (`= AcceleratedMonteiroSvaiter`)          | Monteiro–Svaiter accelerated Newton |
| `CubicRegularizationVanilla`                    | Vanilla cubic regularization        |

> **Note.** The package/module is named `UTR`, so the universal trust-region
> constructor is exported under its full name `UniversalTrustRegion` (a `UTR`
> binding would clash with the module name).

## Usage

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using UTR, SparseArrays

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
g(x) = [-2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2), 200 * (x[2] - x[1]^2)]
H(x) = sparse([2-400*(x[2]-3x[1]^2) -400*x[1]; -400*x[1] 200.0])   # Hessian must be sparse

r = UniversalTrustRegion(name=:UTR)(;
    x0=[-1.2, 1.0], f=f, g=g, H=H,
    maxiter=200, tol=1e-7, subpstrategy=:direct, initializerule=:mishchenko,
)

r.state.x      # solution
r.trajectory   # per-iteration states (when bool_trace=true)
```

## Layout

```
src/
  UTR.jl                   # module entry point (include order + aliases)
  utilities/               # autodiff, counters, display, line searches, ...
    subp/                  # trust-region / regularized subproblem solvers
  algorithms/
    interface.jl           # IterativeAlgorithm / Result / oracle counting
    utr.jl                 # UniversalTrustRegion
    atr.jl                 # ATR
    ms.jl                  # MS (Monteiro–Svaiter)
  others/
    cubicreg_vanilla.jl    # CubicRegularizationVanilla baseline
test/
  setup.jl                 # sets up the (heavy) experiment environment
  Project.toml             # experiment-only deps
  Manifest.toml            # pinned to upstream DRSOM.jl's resolved environment
  lp.jl, tools.jl          # shared experiment helpers
  third-party/             # vendored AdaptiveRegularization.jl, LIBSVMFileIO.jl
  instances/               # LIBSVM datasets (a4a included)
  test_paper_utr/          # paper experiment scripts (UTR vs ARC / RegNewton)
  test_paper_utr_acc/      # accelerated-method scripts (ATR / MS)
```

## Running the experiments

The `test/` directory has its own environment that pulls in the heavy
experiment-only dependencies (Optim, Plots, NLPModels, AdaptiveRegularization, …),
plus two vendored packages under `test/third-party/`. Its `Manifest.toml` is the
same resolved environment as upstream DRSOM.jl, so the pinned versions are reused
as-is. Set it up with the setup script:

```bash
julia --project=test test/setup.jl
```

Then run a paper script from the repo root (so relative data paths resolve):

```bash
julia --project=test test/test_paper_utr/test_logistic.jl
julia --project=test test/test_paper_utr/test_soft_maximum.jl
julia --project=test test/test_paper_utr_acc/test_logistic_atr.jl   # ATR / MS comparison
```

## License

`UTR.jl` is licensed under the MIT License. See `LICENSE` for details.

## Developers

- Yuntian Jiang <yuntianjiang07@gmail.com>
- Chuwen Zhang <chuwzhang@gmail.com>


## Reference
Jiang, Y., He, C., Zhang, C. et al. Beyond Nonconvexity: A Universal Trust-Region Method with New Analyses. J Sci Comput 106, 28 (2026). https://doi.org/10.1007/s10915-025-03154-y