###############
# Second-order A-HPE: Accelerated Hybrid Proximal Extragradient (smooth case)
#
#   Implements Algorithm 2 ("Second-order A-HPE") of the companion note,
#   Section sec.ahpe.so, specialized to a smooth objective f = g
#   (the nonsmooth part h = 0 and the enlargement εₖ = 0).  This is the
#   Monteiro–Svaiter accelerated Newton-proximal-extragradient scheme.
#
#   Monteiro, R.D.C., Svaiter, B.F.: An Accelerated Hybrid Proximal
#   Extragradient Method for Convex Optimization and Its Implications
#   to Second-Order Methods. SIAM J. Optim. 23(2), 1092–1125 (2013).
#
# Notation follows the note (paper symbol [code field]):
#   yₖ    prox point / returned iterate              [state.y]
#   xₖ    estimate-sequence point                    [state.x]
#   x̃ₖ    extrapolation point                        [local y]
#   dₖ₊₁  Newton-prox step  (= yₖ₊₁ - x̃ₖ)            [local d]
#   vₖ₊₁  ∇g(yₖ₊₁)                                    [local gy]
#   Aₖ, aₖ₊₁   accumulated / current curve weights   [state.A, state.a]
#   λₖ₊₁  prox parameter (the regularization is 1/λ)  [state.λ]
#   M     Hessian-Lipschitz constant                 [iter.Mₕ]
#   σℓ, σu  large-step band on the HPE ratio          [iter.σl, iter.σu]
#
# Each outer iteration k, in the note's symbols:
#   1. pick λₖ₊₁ and the curve weight (eq. for aₖ₊₁ in Alg. 2)
#         aₖ₊₁ = (λₖ₊₁ + sqrt(λₖ₊₁² + 4 λₖ₊₁ Aₖ)) / 2 ;
#   2. extrapolate
#         x̃ₖ = (Aₖ/(Aₖ+aₖ₊₁)) yₖ + (aₖ₊₁/(Aₖ+aₖ₊₁)) xₖ ;
#   3. (smooth, exact) Newton-prox subproblem — solve for the step dₖ₊₁
#         (∇²g(x̃ₖ) + (1/λₖ₊₁) I) dₖ₊₁ = -∇g(x̃ₖ)      [Cholesky],
#      and set the prox point yₖ₊₁ = x̃ₖ + dₖ₊₁ .  The Cholesky solve is exact,
#      so the Newton subproblem is solved to zero residual ;
#   4. accept λₖ₊₁ by the large-step band (eq.ahpe.so.inner)
#         (2σℓ)/M ≤ λₖ₊₁ ‖dₖ₊₁‖ ≤ (2σu)/M ,
#      equivalently the large-step ratio  ls = (M λₖ₊₁ / 2) ‖dₖ₊₁‖ ∈ [σℓ, σu],
#      0 < σℓ < σu < 1 ;
#   5. set vₖ₊₁ = ∇g(yₖ₊₁) and update
#         Aₖ₊₁ = Aₖ + aₖ₊₁,   xₖ₊₁ = xₖ - aₖ₊₁ vₖ₊₁ .
###############
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using SparseArrays

const MS_LOG_SLOTS = @sprintf(
    "%5s | %5s | %11s | %9s | %8s | %7s | %8s | %6s | %6s \n",
    "k", "kₜ", "f", "|∇f|", "λ", "ls", "A", "t", "pts"
)

@doc raw"""
Iteration object for the second-order A-HPE (Monteiro–Svaiter) method,
Algorithm 2 of Section sec.ahpe.so, smooth case.

Required oracles: `f`, `g` (gradient), `H` (Hessian), and `Mₕ` — a function
returning an estimate of the Hessian-Lipschitz constant ``M`` at a point.
"""
Base.@kwdef mutable struct MSIteration{Tx,Tf,Tϕ,Tg,TH,Th}
    f::Tf             # f: smooth f
    ϕ::Tϕ = nothing   # ϕ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    hvp::Th = nothing # (unused here; kept for interface symmetry)
    H::TH = nothing   # hessian function
    Mₕ::Union{Function,Nothing} = nothing  # Hessian-Lipschitz estimate M_H(x)
    x0::Tx            # initial point
    t::Dates.DateTime = Dates.now()
    # ----------------------------------------------------------------
    # Monteiro–Svaiter parameters
    σl::Float64 = 0.2          # large-step band lower bound σℓ  (0 < σℓ < σu < 1)
    σu::Float64 = 0.8          # large-step band upper bound σu
    λ₀::Float64 = 1.0          # initial prox parameter λ₁
    λmin::Float64 = 1e-12      # bracketing floor for λ
    λmax::Float64 = 1e12       # bracketing ceiling for λ
    expand::Float64 = 4.0      # bracketing expansion factor
    itermax::Int64 = 30        # inner (λ search) iteration cap
    # ----------------------------------------------------------------
    direction = :warm
    linesearch = :none
    adaptive = :none
    verbose::Int64 = 1
    mainstrategy = :ms
    subpstrategy = :newton
    LOG_SLOTS::String = MS_LOG_SLOTS
    ALIAS::String = "MS"
    DESC::String = "Monteiro–Svaiter Accelerated Newton Proximal Extragradient"
    error::Union{Nothing,Exception} = nothing
end

Base.IteratorSize(::Type{<:MSIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct MSState{R,Tx}
    status::Bool = true # status
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at the iterate
    ∇fz::Tx           # gradient of f at z
    ∇fb::Tx           # buffer
    # ----------------------------------------------------------------
    x::Tx             # estimate-sequence point xₖ
    y::Tx             # prox point yₖ (returned iterate)
    v₀::Tx            # buffer
    z::Tx             # previous prox point yₖ₋₁
    d::Tx             # prox-point increment yₖ - yₖ₋₁ (display/stopping; ≠ Newton step dₖ₊₁)
    # ----------------------------------------------------------------
    a::R = 0.0        # current curve weight aₖ₊₁
    A::R = 0.0        # accumulated Aₖ
    λ::R = 1.0        # prox parameter λₖ₊₁ (regularization is 1/λ)
    ls::R = 0.0       # large-step ratio (M λₖ₊₁ / 2) ‖dₖ₊₁‖ ∈ [σℓ, σu]
    α::R = 1.0        # step size
    Δ::R = 0.0        # step norm proxy
    Δₙ::R = 0.0       # norm of the step
    dq::R = 0.0       # decrease of the quadratic model
    df::R = 0.0       # decrease of the real function value
    ρ::R = 0.0        # df / dq
    ϵ::R = 0.0        # ‖∇f‖ at the iterate
    r::R = 1.0        # alias of λ for display compatibility
    θ::R = 0.0        # alias of ls for display compatibility
    k::Int = 1        # outer iterations
    kᵥ::Int = 1       # (unused) krylov iterations
    kₜ::Int = 1       # inner (λ search) iterations of this step
    t::R = 0.0        # running time
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kgh::Int = 0      # gradient + hvp evaluations
    kH::Int = 0       # hessian evaluations
    kh::Int = 0       # hvp evaluations
    k₂::Int = 0       # cumulative inner iterations
    acc_style::Symbol = :_
end

# ---------------------------------------------------------------------------
# curve weight aₖ₊₁ (Alg. 2, step 1):
#   aₖ₊₁ = (λₖ₊₁ + sqrt(λₖ₊₁² + 4 λₖ₊₁ Aₖ)) / 2
# ---------------------------------------------------------------------------
_ms_a(λ, A) = (λ + sqrt(λ^2 + 4 * λ * A)) / 2
# extrapolation (Alg. 2, step 2):  x̃ₖ = (Aₖ/(Aₖ+aₖ₊₁)) yₖ + (aₖ₊₁/(Aₖ+aₖ₊₁)) xₖ
# here `y` is the prox point yₖ (state.y) and `x` is the estimate point xₖ (state.x)
function _ms_yeval(y, x, a, A)
    ratio = A / (A + a)
    return ratio .* y .+ (1 - ratio) .* x
end

@doc raw"""
Initialize the state; the 0-th iterate performs no optimization step.
"""
function Base.iterate(iter::MSIteration)
    iter.t = Dates.now()
    isnothing(iter.Mₕ) && throw(ErrorException(
        "MS requires a Hessian-Lipschitz estimate `Mₕ(x)`; none was provided."
    ))
    z = copy(iter.x0)
    fz = iter.f(z)
    ∇f = iter.g(z)
    gₙ = norm(∇f, 2)
    state = MSState(
        x=copy(z),        # estimate-sequence point x₀
        y=copy(z),        # prox point y₀
        v₀=zero(z),
        z=copy(z),
        d=zero(z),
        fx=fz,
        fz=fz,
        ∇f=∇f,
        ∇fz=copy(∇f),
        ∇fb=zero(∇f),
        ϵ=gₙ,
        Δ=gₙ * 1e1,
        a=0.0,
        A=0.0,
        λ=iter.λ₀,
    )
    return state, state
end

function Base.iterate(
    iter::MSIteration,
    state::MSState{R,Tx};
) where {R,Tx}
    state.z = z = state.y
    state.fz = fz = state.fx
    state.∇fz = state.∇f

    Mₕ = iter.Mₕ(state.y)
    σl, σu = iter.σl, iter.σu
    # ---- inner search (Alg. 2, step 4): pick λₖ₊₁ so the large-step ratio
    # ls = (M λₖ₊₁ / 2) ‖dₖ₊₁‖ lands in the band [σℓ, σu].
    # ls(λ) = (Mₕ λ / 2) ‖dₖ₊₁(λ)‖ is increasing in λ:
    #   λ too large  → ls > σu  → need smaller λ (lower the upper bound λ₊)
    #   λ too small  → ls < σl  → need larger λ  (raise the lower bound λ₋)
    # λ₋ / λ₊ start at 0 meaning "bound not yet known"; we expand geometrically
    # from the warm start until a bound is found, then plain-bisect once both
    # bounds exist (a valid bracket has been trapped).
    # ------------------------------------------------------------
    # @note: on λ:
    # 1. the most plain version, `cold-start` λ.
    #    λ = (iter.λmin + iter.λmax) / 2
    # 2. one can also clamp the λ.
    #    λ = clamp(state.λ, iter.λmin, iter.λmax)
    # 3. use the λ from the previous iteration.
    λ = state.λ
    λ₋, λ₊ = 0.0, 0.0
    k₂ = 0

    local d, gy, H, dₙ, ls, y
    while true
        a = _ms_a(λ, state.A)
        y = _ms_yeval(state.y, state.x, a, state.A)   # extrapolation y := x̃ₖ = (Aₖ/Aₖ₊₁) yₖ + (aₖ₊₁/Aₖ₊₁) xₖ
        gy = iter.g(y)                                # ∇g(x̃ₖ)
        H = iter.H(y)                                 # ∇²g(x̃ₖ)

        # exact Newton-prox step (Alg. 2, step 3): (∇²g(x̃ₖ) + (1/λ) I) dₖ₊₁ = -∇g(x̃ₖ)
        F = cholesky(Symmetric(Matrix(H) + (1 / λ) * I), check=false)
        pd = issuccess(F)
        if pd
            d = F \ (-gy)              # exact Newton solve (zero subproblem residual)
            dₙ = norm(d)
            ls = Mₕ * dₙ * λ / 2       # large-step ratio (M λₖ₊₁ / 2) ‖dₖ₊₁‖ ∈ [σℓ, σu]
        else
            dₙ = Inf
            ls = Inf                   # indefinite ⇒ ls > σu ⇒ shrink λ (more regularization)
        end

        state.a = a
        k₂ += 1
        # @printf(" |- λ: %.1e, λ₋: %.1e, λ₊: %.1e, ls: %.1e ∈ [%.1e, %.1e]\n", λ, λ₋, λ₊, ls, σl, σu)
        # accept when the large-step / HPE band is satisfied
        if pd && (σl <= ls <= σu)
            state.acc_style = :ls
            break
        end
        if k₂ >= iter.itermax
            # force-accept the last PD point; report a degraded style
            if !pd
                # last resort: shrink λ (more regularization) until PD
                λ = max(λ / iter.expand, iter.λmin)
                continue
            end
            state.acc_style = :force
            break
        end

        if ls > σu                     # λ too large → lower upper bound
            λ₊ = λ
            λ = λ₋ > 0 ? (λ₋ + λ₊) / 2 : max(λ / iter.expand, iter.λmin)
        else                           # ls < σl, λ too small → raise lower bound
            λ₋ = λ
            λ = λ₊ > 0 ? (λ₋ + λ₊) / 2 : min(λ * iter.expand, iter.λmax)
        end
    end

    # accept: prox point yₖ₊₁ = x̃ₖ + dₖ₊₁
    y₊ = y + d
    # vₖ₊₁ = ∇g(yₖ₊₁)  (HPE residual = gradient at the iterate)
    v = iter.g(y₊)
    # estimate-sequence update (Alg. 2, step 5): 
    #  xₖ₊₁ = xₖ - aₖ₊₁ vₖ₊₁,  Aₖ₊₁ = Aₖ + aₖ₊₁
    state.x .= state.x - state.a .* v
    state.A += state.a

    fx = iter.f(y₊)
    df = fz - fx
    dq = -d' * H * d / 2 - d' * gy     # model decrease at x̃ₖ (gy = ∇g(x̃ₖ) from the inner loop)

    state.y = y₊
    state.∇f = v                       # gradient at the returned iterate yₖ₊₁
    state.ϵ = norm(v)                  # ‖∇g(yₖ₊₁)‖ — residual at the returned point
    state.fx = fx
    state.df = df
    state.dq = dq
    state.ρ = dq != 0 ? df / dq : 0.0
    state.λ = λ
    state.r = state.λ
    state.ls = ls
    state.θ = ls
    state.d = y₊ - z
    state.Δ = dₙ
    state.Δₙ = dₙ
    state.kₜ = k₂
    state.k₂ += k₂
    state.t = (Dates.now() - iter.t).value / 1e3
    counting(iter, state)
    state.status = true
    state.k += 1
    checknan(state)
    return state, state
end

####################################################################################################
# Basic Tools
####################################################################################################
ms_stopping_criterion(tol, state::MSState) =
    (state.ϵ <= tol) || (state.Δ <= 1e-15)

function counting(iter::T, state::S) where {T<:MSIteration,S<:MSState}
    try
        state.kf = getfield(iter.f, :counter)
        state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
        state.kh = hasproperty(iter.hvp, :counter) ? getfield(iter.hvp, :counter) : 0
        state.kg = getfield(iter.g, :counter)
        state.kgh = state.kg + state.kh * 2
    catch
    end
end

function checknan(state::S) where {S<:MSState}
    if any(isnan, state.y)
        @warn(ErrorException("NaN detected in MS iterate, use debugging to fix"))
    end
end

function ms_display(k, state::MSState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", MS_LOG_SLOTS)
    end
    @printf(
        "%5d | %5d | %+.4e | %.3e | %.2e | %.1e | %+.1e | %6.1f | %s \n",
        k, state.kₜ, state.fx, state.ϵ,
        state.λ, state.ls, state.A, state.t, state.acc_style
    )
end

default_solution(::MSIteration, state::MSState) = state.y

AcceleratedMonteiroSvaiter(;
    name=:MS,
    stop=ms_stopping_criterion,
    display=ms_display
) = IterativeAlgorithm(MSIteration, MSState; name=name, stop=stop, display=display)

####################################################################################################
# call operator
####################################################################################################
function (alg::IterativeAlgorithm{T,S})(;
    maxiter=10000,
    maxtime=1e2,
    tol=1e-6,
    freq=10,
    verbose=1,
    direction=:cold,
    adaptive=:none,
    bool_trace=false,
    kwargs...
) where {T<:MSIteration,S<:MSState}

    arr = Vector{S}()
    kwds = Dict(kwargs...)

    for cf ∈ [:f :g :H :hvp]
        apply_counter(cf, kwds)
    end

    iter = T(; adaptive=adaptive, direction=direction, verbose=verbose, kwds...)
    for (_k, _) in kwds
        @debug _k getfield(iter, _k)
    end
    (verbose >= 1) && show(iter)
    for (k, state) in enumerate(iter)
        bool_trace && push!(arr, copy(state))
        if k >= maxiter || state.t >= maxtime || alg.stop(tol, state) || state.status == false
            (verbose >= 1) && alg.display(k, state)
            (verbose >= 1) && summarize(k, iter, state)
            return Result(name=alg.name, iter=iter, state=state, k=k, trajectory=arr)
        end
        (verbose >= 1) && (k == 1 || mod(k, freq) == 0) && alg.display(k, state)
    end
end

function Base.show(io::IO, t::T) where {T<:MSIteration}
    format_header(t.LOG_SLOTS)
    @printf io "  algorithm alias       := %s\n" t.ALIAS
    @printf io "  algorithm description := %s\n" t.DESC
    @printf io "  inner iteration limit := %s\n" t.itermax
    @printf io "  main       strategy   := %s\n" t.mainstrategy
    @printf io "  subproblem strategy   := %s (exact regularized Newton)\n" t.subpstrategy
    @printf io "  large-step band       := [%.2f, %.2f]\n" t.σl t.σu
    if t.H !== nothing
        @printf io "      second-order info := using provided Hessian matrix\n"
    else
        @printf io " unknown mode to compute Hessian info\n"
        throw(ErrorException("MS requires an explicit Hessian `H`\n"))
    end
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

function summarize(io::IO, k::Int, t::T, s::S) where {T<:MSIteration,S<:MSState}
    println(io, "-"^length(t.LOG_SLOTS))
    println(io, "summary:")
    @printf io " (main)          f       := %.2e\n" s.fx
    @printf io " (first-order)  |g|      := %.2e\n" s.ϵ
    println(io, "oracle calls:")
    @printf io " (main)          k       := %d  \n" s.k
    @printf io " (function)      f       := %d  \n" s.kf
    @printf io " (first-order)   g       := %d  \n" s.kg
    @printf io " (second-order)  H       := %d  \n" s.kH
    @printf io " (inner)         Σkₜ     := %d  \n" s.k₂
    @printf io " (running time)  t       := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:MSIteration,S<:MSState} =
    summarize(stdout, k, t, s)
