using DelimitedFiles
using Bigsimr
using Distributions
using LinearAlgebra
using Plots
using CSV, Tables
using Random

ZeroInflated(dist, pzero) = MixtureModel([Dirac(0), dist], [pzero, 1 - pzero]);

function f(use_blocks, p, p_info)
    target_corrs = Dict()

    for corr in ["no_corr", "low_corr", "medium_corr", "high_corr"]
        if corr == "no_corr"
            coef = 0
            cst = 0.0
        else
            coef = 0.1
            if corr == "low_corr"
                cst = 0.2
            end
            if corr == "medium_corr"
                cst = 0.5
            end
            if corr == "high_corr"
                cst = 0.7
            end
        end
        base_corr = zeros((p, p))
        if use_blocks
            i = 1
            while i < p - 4
                base_corr[i:i+4, i:i+4] = coef * cor_randPD(5) .+ cst
                i += 5
            end
            target_corrs[corr] = base_corr
            target_corrs[corr][diagind(target_corrs[corr])] .= 1
        else
            base_corr[1:p_info, 1:p_info] = coef * cor_randPD(p_info) .+ cst
            target_corrs[corr] = base_corr
            target_corrs[corr][diagind(target_corrs[corr])] .= 1
        end

    end
    margins = Dict(
        "normal" => [Normal(0, 1) for i in 1:p],
        "NB" => [NegativeBinomial(2, 0.1) for i in 1:p],
    )
    return target_corrs, margins
end

p = parse(Int64, ARGS[1])
p_info = parse(Int64, ARGS[2])
marg = ARGS[3]
corr = ARGS[4]
use_blocks = (ARGS[5] == "yes")
i = 1

if marg == "normal" || marg == "ZI"
    marg = "normal"
end
if marg == "NB" || marg == "ZINB"
    marg = "NB"
end

if marg == "normal"
    margin = [Normal(0, 1) for i in 1:p]
else
    margin = [NegativeBinomial(2, 0.1) for i in 1:p]
end

if corr == "no_corr"
    coef = 0
    cst = 0.0
else
    coef = 0.1
    if corr == "low_corr"
        cst = 0.2
    end
    if corr == "medium_corr"
        cst = 0.5
    end
    if corr == "high_corr"
        cst = 0.7
    end
end
base_corr = zeros((p, p))
if use_blocks
    global i = 1
    while i < p - 4
        base_corr[i:i+4, i:i+4] = coef * cor_randPD(5) .+ cst
        global i += 5
    end
    base_corr[diagind(base_corr)] .= 1
else
    base_corr[1:p_info, 1:p_info] = coef * cor_randPD(p_info) .+ cst
end
base_corr[diagind(base_corr)] .= 1
target_corr = base_corr

dir = dirname(Base.source_path())

if use_blocks
    x = rvec(50000, target_corr, margin)
    CSV.write("$dir/Norta data $p_info (block)/$p feats $marg $corr.csv", Tables.table(x), writeheader=true)
    rng = MersenneTwister(42)
    rand_zero_one_mask = rand(rng, 1:10, size(x)) .>= 10
    x[rand_zero_one_mask] .= 0
    marg_zi = "ZI"
    if marg == "NB"
        marg_zi = "ZINB"
    end
    CSV.write("$dir/Norta data $p_info (block)/$p feats $marg_zi $corr.csv", Tables.table(x), writeheader=true)
else
    x = rvec(50000, target_corr, margin)
    CSV.write("$dir/Norta data $p_info/$p feats $marg $corr.csv", Tables.table(x), writeheader=true)
    rng = MersenneTwister(42)
    rand_zero_one_mask = rand(rng, 1:10, size(x)) .>= 10
    x[rand_zero_one_mask] .= 0
    marg_zi = "ZI"
    if marg == "NB"
        marg_zi = "ZINB"
    end
    CSV.write("$dir/Norta data $p_info/$p feats $marg_zi $corr.csv", Tables.table(x), writeheader=true)
end

#[100, 500, 1000, 2500, 5000, 7500, 10000]
#[10, 25, 50]
#["normal", "NB"]
#["no_corr", "low_corr", "medium_corr", "high_corr"]
#["yes", "no"]