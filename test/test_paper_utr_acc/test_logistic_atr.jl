#############################################
# project: DRSOM
# created Date: Tu Mar 2022
# -----
# last Modified: Mon Apr 18 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# A script to test DRSOM on nonconvex logistic regression for 0-1 classification on LIBSVM
# @reference:
# 1. Zhu, X., Han, J., Jiang, B.: An Adaptive High Order Method for Finding Third-Order Critical Points of Nonconvex Optimization, http://arxiv.org/abs/2008.04191, (2020)
###############


include("../lp.jl")
include("../tools.jl")

using ArgParse
using Arpack
using UTR
using LineSearches
using Optim
using ProximalOperators
using ProximalAlgorithms
using Random
using Plots
using Printf
using KrylovKit
using LaTeXStrings
using LinearAlgebra
using Statistics
using LinearOperators
using Optim
using SparseArrays
using .LP
using LIBSVMFileIO

bool_q_preprocessed = true
bool_plot = true
bool_opt = true
# which method group to run:
#   0 = all, 1 = unaccelerated, 2 = accelerated, 3 = MS-type
mode_opt = 3
f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))
Lip2(Xv, N; dm=1.0) = begin
    λₘ = eigs(Xv' * Xv, nev=1, which=:LM, tol=1e-4)[1][]
    a = eachrow(Xv) .|> norm |> maximum
    return a * λₘ / N * dm
end

ε = 1e-9 # * max(g(x0) |> norm, 1)
λ = 1e-4
K = 2500
tol = 1e-9
if bool_q_preprocessed
    name = "a4a"
    # name = "a9a"
    # name = "w4a"
    # name = "w8a"
    # name = "covtype"
    # name = "news20"
    # name = "rcv1"

    X, y = libsvmread("test/instances/$name.libsvm"; dense=false)
    Is = vcat([x.nzind for (j, x) in enumerate(X)]...)
    Js = vcat([j * ones(Int, length(x.nzind)) for (j, x) in enumerate(X)]...)
    Vs = vcat([x.nzval for (j, x) in enumerate(X)]...)
    Xv = sparse(Is, Js, Vs)'
    Rc = 2.0 ./ f1(Xv)[:]
    Xv = (Rc |> Diagonal) * Xv
    X = Rc .* X

    if name in ["covtype"]
        y = convert(Vector{Float64}, (y .- 1.5) * 2)
    else
    end

    @info "data reading finished"

    # precompute Q Matrix
    Pv = y .^ 2 .* Xv

    n = Xv[1, :] |> length
    Random.seed!(1)
    N = y |> length

    x₀ = 1e1 * randn(Float64, n)

    # All oracles are written in terms of the margin m = y .* (Xv*w) so that the
    # exponentials saturate cleanly (→ 0 or → Inf) instead of giving Inf/Inf = NaN
    # when w (hence m) is large. See the σ(m), σ(m)(1-σ(m)) identities below.
    function loss(w)
        m = y .* (Xv * w)
        # log(1 + exp(-m)) via stable softplus: max(-m,0) + log1p(exp(-|m|))
        z = sum(@. max(-m, 0) + log1p(exp(-abs(m))))
        return z / N + 0.5 * λ * w'w
    end
    function g(w)
        m = y .* (Xv * w)
        fq = @. -1 / (1 + exp(m))          # = σ(m) - 1, stable (overflow → 0)
        return Xv' * (fq .* y) / N + λ * w
    end

    function H(w)
        m = y .* (Xv * w)
        fq = @. 1 / (exp(m) + 2 + exp(-m)) # = σ(m)(1-σ(m)), stable
        return ((fq .* Pv)' * Xv ./ N) + λ * I
    end

    function hvp(w, v, Hv)
        m = y .* (Xv * w)
        fq = @. 1 / (exp(m) + 2 + exp(-m)) # = σ(m)(1-σ(m)), stable
        copyto!(Hv, (fq .* Pv)' * (Xv * v) ./ N .+ λ .* v)
    end

    function hvpdiff(w, v, Hv; eps=1e-5)
        gn = g(w + eps * v)
        gf = g(w)
        copyto!(Hv, (gn - gf) / eps)
    end

    @info "data preparation finished"

    options = Optim.Options(
        g_tol=ε,
        iterations=10000,
        store_trace=true,
        show_trace=true,
        show_every=1,
        time_limit=500
    )
end


if bool_opt

    results = []
    # options for Optim.jl package
    options = Optim.Options(
        g_tol=ε,
        iterations=10000,
        store_trace=true,
        show_trace=true,
        show_every=1,
        time_limit=500
    )
    _Mconst = Lip2(Xv, N)
    Mₕ(x) = _Mconst

    # one named tuple per run: label, constructor, type (1=unaccel, 2=accel, 3=MS-type),
    # and method-specific kwargs. shared kwargs (x0, f, g, H, maxiter, bool_trace) are
    # supplied in the run loop below. optional `plot_k=true` adds an extra series vs the
    # outer-iteration count k (alongside the default :kH x-axis) when plotting.
    method_specs = [
        # (name="UTR (1)", method=ATR, type=1, kwargs=(; tol=tol / 2, freq=20, subpstrategy=:direct,
        #     initializerule=:given, Mₕ=Mₕ, adaptiverule=:constant, ratio_σ=2.0, ratio_Δ=15.0)),
        # (name="UTR (2)", method=ATR, type=1, kwargs=(; tol=tol / 2, freq=20, subpstrategy=:direct,
        #     initializerule=:given, Mₕ=Mₕ, adaptiverule=:constant, ratio_σ=5.0, ratio_Δ=15.0)),
        # (name="ATR", method=ATR, type=2, kwargs=(; tol=tol / 2, freq=20, subpstrategy=:nesterov,
        #     initializerule=:given, Mₕ=Mₕ, adaptiverule=:utr, ratio_σ=10.0, ratio_Δ=0.3, localthres=1e-5)),
        # (name="ATR (larger M)", method=ATR, type=2, kwargs=(; tol=tol / 2, freq=20, subpstrategy=:nesterov,
        #     initializerule=:given, Mₕ=(x) -> Mₕ(x) * 10, adaptiverule=:utr, ratio_σ=10.0, ratio_Δ=0.3, localthres=1e-5)),
        # # the accelerated UTR via the Monteiro–Svaiter inner solve
        # (name="ATR (MS)", method=ATRMS, type=3, kwargs=(; tol=tol / 2, freq=1, initializerule=:given,
        #     Mₕ=(x) -> Mₕ(x) / 10, adaptiverule=:constant, localthres=1e-5)),
        # the usual (large-step A-NPE) Monteiro–Svaiter accelerated method
        (name="MS", method=MS, type=3, plot_k=true, kwargs=(; tol=tol / 2, freq=10, Mₕ=(x) -> Mₕ(x), σl=0.2, σu=0.8, λstrategy=:cold)),
        # (name="MS (smaller M)", method=MS, type=3, plot_k=true, kwargs=(; tol=tol / 2, freq=10, Mₕ=(x) -> Mₕ(x) / 10, σl=0.2, σu=0.8, λstrategy=:cold)),
        (name="MS (warm-start)", method=MS, type=3, plot_k=true, kwargs=(; tol=tol / 2, freq=10, Mₕ=(x) -> Mₕ(x), σl=0.4, σu=0.6, λstrategy=:warm)),
        (name="CubicReg", method=CRM, type=1, kwargs=(; tol=tol, freq=20, subpstrategy=:direct, initializerule=:given, Mₕ=Mₕ)),
        (name="CubicReg-Acc", method=CRM, type=2, kwargs=(; tol=tol, freq=20, subpstrategy=:nesterov, initializerule=:given, Mₕ=Mₕ)),
    ]

    for spec in method_specs
        (mode_opt == 0 || mode_opt == spec.type) || continue
        rd = spec.method(name=Symbol(spec.name))(;
            x0=copy(x₀), f=loss, g=g, H=H, maxiter=K, bool_trace=true, spec.kwargs...)
        push!(results, (spec.name, rd, get(spec, :plot_k, false)))
    end

end


if bool_plot
    linestyles = [:dash, :dot, :dashdot, :dashdotdot]
    xaxis = :k
    metric = :ϵ
    @printf("plotting results\n")

    pgfplotsx()
    title = ""
    fig = plot(
        # xlabel=L"\textrm{Iterations}",
        xlabel=L"\texttt{#} of $\nabla^2f$ oracles",
        # ylabel=L"f(x) - f(x^*)",
        ylabel=L"\|\nabla f(x)\|",
        title=title,
        size=(600, 500),
        yticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-1, 1e1],
        # xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
        yscale=:log10,
        dpi=500,
        xtickfont=font(20),
        ytickfont=font(20),
        xlabelfontsize=20,
        ylabelfontsize=17,
        legend=:outertop,
        legendcolumns=2,
        legendfontsize=20,
        legend_background_color=RGBA(1.0, 1.0, 1.0, 0.7),
        legendfontfamily="sans-serif",
        legendfonthalign=:left,
        titlefontsize=22,
    )
    maxstep = K
    colors = palette(:Paired_8)[[1, 2, 3, 4, 5, 6, 7]]
    markers = [:rect, :rect, :rect, :rect, :circle, :circle, :diamond]
    for (k, (nm, rv, plot_k)) in enumerate(results)
        yfull = getresultfield(rv, metric)        # full series, kept for the optional :k plot
        xv = getresultfield(rv, :kH)
        maxlength = min(yfull |> length, xv[xv.<maxstep] |> length, maxstep)
        indices = 1:maxlength
        # yv .- f₊ .+ 1e-20,
        yv = yfull[indices]
        xv = xv[indices]
        @info "plotting $metric"
        @info "yv: $(yv[end])"
        plot!(fig,
            xv[end:-1:1],
            yv[end:-1:1],
            label=L"\texttt{%$nm}",
            linewidth=2.5,
            # linestyle=:dash,
            color=colors[k],
            # markershape=markers[k],
            # markersize=2.0,
            # markercolor=:match,
        )
        scatter!(fig,
            xv[end:-10:1],
            yv[end:-10:1],
            markershape=markers[k],
            markersize=4.0,
            markercolor=colors[k],
            label=nothing,
        )
        # optional extra series vs the outer-iteration count k (dashed, same color)
        if plot_k
            xk = getresultfield(rv, :k)
            mlk = min(yfull |> length, xk[xk.<maxstep] |> length, maxstep)
            idxk = 1:mlk
            yk = yfull[idxk]
            xk = xk[idxk]
            plot!(fig,
                xk[end:-1:1],
                yk[end:-1:1],
                label=L"\texttt{%$nm} ($k$)",
                linewidth=2.5,
                linestyle=:dash,
                color=colors[k],
            )
        end
    end
    savefig(fig, "/tmp/e-logistic-$name-$xaxis.tex")
    savefig(fig, "/tmp/e-logistic-$name-$xaxis.pdf")
    # savefig(fig, "/tmp/$metric-logistic-$name-$xaxis.png")
end
