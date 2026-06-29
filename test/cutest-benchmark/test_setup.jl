include("../tools.jl")
include("./problem_cols.jl")


log_freq = 200
precision = tol_grad = 1e-5
max_iter = 20000
max_time = 200.0
test_before_start = true


# ------------------------------------------------------------
# filter test set
# ------------------------------------------------------------
filter_cutest_problem(nlp) = true
# small test
# filter_cutest_problem(nlp) = (4 <= nlp.meta.nvar <= 200)
# large_test
# filter_cutest_problem(nlp) = (1000 <= nlp.meta.nvar <= 5000)

# ------------------------------------------------------------
# filter methods
# @note, see ../tools.jl for more details
# ------------------------------------------------------------
# filter_optimization_method(k) = true
filter_optimization_method(k) = k ∈ [:iUTR]
# filter_optimization_method(k) = k == :ARC
# filter_optimization_method(k) = k == :TRST

# choose problem set
# PROBLEMS = UNC_PROBLEMS_221104
# PROBLEMS = TEST
PROBLEMS = intersect(UNC_PROBLEMS_4to200, UNC_PROBLEMS_GOOD)
# PROBLEMS = UNC_PROBLEMS_201to5000
# PROBLEMS = UNC_PROBLEMS_GOOD
# PROBLEMS = UNC_PROBLEMS_COMB[155:end]
# PROBLEMS = UNC_PROBLEMS_COMB
# PROBLEMS = UNC_PROBLEM_NO_PARAMS

if test_before_start
    ######################################################################
    # include a small test to make sure everything works
    @testset "TEST ALL DRSOM VARIANTS @ a CUTEst problem CHAINWOO" begin
        nlp = CUTEstModel("MSQRTALS", "-param", "P=7")
        println(nlp.meta)
        name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
        x0 = nlp.meta.x0
        loss(x) = NLPModels.obj(nlp, x)
        g(x) = NLPModels.grad(nlp, x)
        H(x) = NLPModels.hess(nlp, x)
        hvp(x, v, Hv) = NLPModels.hprod!(nlp, x, v, Hv)

        @testset "UTR" begin
            options = Dict(
                :maxiter => max_iter,
                :maxtime => max_time,
                :tol => 1e-5,
                :freq => log_freq
            )
            r = wrapper_utr(x0, loss, g, H, options)
            @test r.state.ϵ < 1e-4
        end
        finalize(nlp)
    end
end
