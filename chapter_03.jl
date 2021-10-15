### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 3dc78b1b-58d3-40eb-a1c5-a2fa58f15abf
using Distributions, StatsPlots, PlutoUI, StatsBase, KernelDensity

# ╔═╡ 535b43f0-2780-11ec-1444-d98d954f2508
md""" # Using the posterior distribution

Usually samples from the posterior distribution are worked with rather than the disitrubtion itself, this for two reasons:
1. Using the whole posterior distribution involves integral calculus, whereas working with samples is just data summary. For  complex problems with many parameters, these integrals become very complex whereas the data summary remains much easier.
2. The most capable techniques for generating the posterior distribution, such as variants of the MCMC techniques, only produce samples. Using grid approximation will give you the whole posterior and will be very easy to work with but is impractical to work with for compelx problems.
"""

# ╔═╡ 9993e805-7265-4c1d-b8a9-30c73f91bc37
md""" ## Sampling from a grid approximate posterior

Using the globe tossing model
"""

# ╔═╡ 01914d0b-ba88-435f-8194-6acc5437c0e1
md""" Generate the grid approximated posterior distribution:"""

# ╔═╡ a7667076-b719-46e6-8ef7-690019c176b6
begin 
	N = 9
	S = 6
	n = 1001
	grid = range(0, 1, length=n)
	prior = @. pdf(Uniform(0,1), grid)
	likelihood = @. pdf(Binomial(N, grid), S)
	posterior = likelihood .* prior
	posterior /= sum(posterior)
end;

# ╔═╡ 2e20bfcb-9633-4808-9f29-d3ec2ed472fe
let
	plot(grid, prior, st=:line, label="prior", legend=:topright, xlims=(0,1.0), ylims=(0,2))
	plot!(grid, likelihood, st=:line, label="likelihood")
	plot!(grid, posterior, st=:line, label="posterior")
end

# ╔═╡ d63a4f5f-6274-40a6-90d9-b6933382e24d
md""" Sample 10,000 different parameter values ($P$) from the posterior:"""

# ╔═╡ 9356eb97-efae-47cb-a5a9-6676f00a5f1c
n_samples = 10_000

# ╔═╡ 63fdfebf-d08f-438c-a69b-9db2200a61a6
samples = wsample(grid, posterior, n_samples)

# ╔═╡ 0072d892-1ede-4432-afca-9132c5695586
md""" See how the sample distribution approaches the actual posterior as the number of samples increases:
"""

# ╔═╡ d5d3b50b-01b9-4201-b01c-19cbe0c87bcf
@bind _n Slider(10:10:2000, show_value=true)

# ╔═╡ cf44ec9a-7c16-4ab4-b547-3b57263da109
let
	post_true = Beta(1+S,(N-S)+1) # analytical density calculated from Beta dist.
	samples = rand(post_true, _n)
	plot(post_true, lw=3, label="true",title="posterior probability density", xticks=0:0.2:1.0, xaxis="proportion of water", legend=:topleft, xlims=(0.0,1.2), ylims=(0,4.0))
	density!(samples, lw=3, ls=:dash, label="sampled")
end

# ╔═╡ f88a09e1-d3dc-4228-856e-f82725df693d
md""" Not very useful as the sampling has only approximated the posterior which we already had!"""

# ╔═╡ 6784239e-5595-471a-b4e8-348491e29e5e
md""" ## Sampling to summarise
Exactly how the posterior is summarised depends on the purpose, but common questions can be divided into three main groups:
1. intervals of defined boundaries
2. intervals of defined probability mass
3. point estimates
"""

# ╔═╡ 41688bf0-3188-4d31-897c-f864de97558f
md""" ### Intervals of defined boundaries
For example, what is the posterior probability that the proportion of water is below 0.5?
"""

# ╔═╡ 9b4a897d-f0f2-42d9-a782-b47356065118
md""" For grid approximation it is simple:"""

# ╔═╡ d9e160f8-e203-4f23-ad03-2c7bd3410c5b
posterior[grid .< 0.5] |> sum

# ╔═╡ 3b2cca7a-3db1-4f08-af04-5d1df2955be4
md"""However, as has been said, grid approximation is not feasible for more complex problems whereas sampling from the posterior is:"""

# ╔═╡ cc34c6a3-ae53-41f6-9934-dc789508883f
sum(samples .< 0.5) / length(samples) 	# proportion of samples < 0.5

# ╔═╡ 0d0ed10d-78e3-4486-ab4e-bd0966be7cc5
md""" Similarly, how much of the distribution lies between two values, say 0.5 and 0.75:"""

# ╔═╡ bf7367bc-2375-4942-abeb-7a756f5a1dc4
sum(0.5 .< samples .< 0.75) / length(samples)

# ╔═╡ 1d838e46-53b6-45eb-ab56-c404773a6f94
md""" ### Intervals of defined mass
What is commonly referred to as *confidence* intervals in Bayesian statistics (but will be called *compatibility* intervals here, as the confidence of the interval will depend on the model's ability to replicate the "large" world).

These intervals report two parameter values, between which will hold the specified probability mass.
"""

# ╔═╡ afedb578-664f-41bb-96f5-1b9f5eebfcfd
md"""The lower 80% of the distribution:"""

# ╔═╡ 5f549033-b925-438d-b56b-f5f18fa50613
quantile(samples, 0.8)

# ╔═╡ 0da1daed-171f-4156-9247-21c2258d11c0
md"""Therefore 80% of the probability mass is contained between 0 and $(quantile(samples, 0.8)) """

# ╔═╡ 661b6777-0991-4274-8443-d0f6f1e94051
md"""The middle 80%:"""

# ╔═╡ f6c340ac-a923-4be5-b1ce-2a002d737cfc
quantile(samples, [0.1, 0.9])

# ╔═╡ a5beaffe-fa90-43a6-8efc-d9c0baa818e5
md"""These percentage intervals (PI) assume the distribution is symmetrical and give equal weight (and therefore chooses equal mass above and below the interval). Highest posterior density intervals (HPDI) provide the narrowest interval containing the specified amount of probability mass:"""

# ╔═╡ 8a037821-ec0f-4a4d-8292-21833f0c05f7
function hpdi(x::Vector{T}; alpha=0.11) where {T<:Real}
    n = length(x)
    m = max(1, ceil(Int, alpha * n))

    y = sort(x)
    a = y[1:m]
    b = y[(n - m + 1):n]
    _, i = findmin(b - a)

    return [a[i], b[i]]
end

# ╔═╡ 3785acb9-4e2c-4299-a925-c6006ce46890
a,b = hpdi(samples)

# ╔═╡ 68884922-52cc-4a7f-8a9c-b0d5cde80875
md"""### Point estimates
Bayesian analysis gives you the entire probability distribution of possible parameter values so point estimates often pointless.

Choice of estimate will depend on the question being answered.
"""

# ╔═╡ f594e927-1d66-4107-8fec-7f2b60617110
md"""The *maximum a posteriori* (MAP) estimate:"""

# ╔═╡ 7dfb3173-b7af-4546-bbcb-495cb7498a3b
Pmap = grid[argmax(posterior)]

# ╔═╡ 64635021-340b-4968-adc4-8ea3e9d7c5cf
md"""If you have samples from the posterior you can approximate this with the *mode* of the samples:"""

# ╔═╡ 82db866d-62af-412a-aeee-64940c64fa10
begin
	d = Dict()
	for x in samples
		p = round(x, digits=2)
		p in keys(d) ? d[p] += 1 : d[p] = 1
	end
	Pmap_est = findmax(d)
end

# ╔═╡ fa055e68-a5d5-4ecd-970b-de20766e4a0a
md"""You could also report the *mean* or *median*:"""

# ╔═╡ 57a9acfa-6665-4936-8864-a54269e79a6b
Pmean = mean(samples)

# ╔═╡ b5f28886-2239-4cb3-8248-00c4f32cfa91
Pmedian = median(samples)

# ╔═╡ 369febaa-8600-409e-9035-00e46483b70c
begin
	plot(grid, posterior, label=:none, legend=:topleft)
	vline!([Pmap], label="MAP")
	vline!([Pmap_est], label="MAP estimate")
	vline!([Pmean], label="mean")
	vline!([Pmedian], label="median")
end

# ╔═╡ ddf696bd-e6ac-48eb-b421-d5d778c2673b
md""" An alternative way would be to choose a loss function that tells you the loss associated with any particular estimate.

For example, using the distance from the true value:

``loss = \sum \lvert x - P \rvert``, where ``x`` is the estimate and ``P`` is the true value.

The true value will be unknown though, so weight the estimate by the posterior probability:
"""

# ╔═╡ b6b6fe45-e22f-4ca4-93f0-2ad1971d5339
begin
	estimate = 0.5 # p=0.5
	posterior .* abs.(estimate .- grid) |> sum
end

# ╔═╡ 455f6d2e-9abd-45ee-bd8d-83c3d92218ec
md"""Repeating over every possible ``P`` value and taking the minimum actually gives the median of the posterior:"""

# ╔═╡ 4ed753dc-44a8-48fc-82aa-80a95f40278b
begin
	loss = map(x -> sum(posterior.*abs.(x .- grid)), grid)
	plot(grid, loss, label="", title="loss", legend=:top)
	vline!([grid[argmin(loss)]], lw=2, label="minimum loss : $(round(grid[argmin(loss)], digits=3))")
	vline!([Pmedian], lw=2, ls=:dash, label="median: $(round(Pmedian, digits=3))")
	end

# ╔═╡ 02a8b42e-340e-4b2e-90f1-38db6f2aa03c
md"""Taking the square of the deviation (the quadratic loss) gives the mean.

The context in which you conduct your analysis may dictate a unique loss function, e.g. decision on whether to evacuate based on estimate of wind speed; the damage to property increases very steeply with wind speed, whereas ordering an evacuation when none needed will decrease slowly as wind speed falls below estimate - requiring a highly asymmetrical loss function.
"""

# ╔═╡ bdee2725-7fde-4548-afea-2b220cadf951
md"""## Sampling to simulate prediction

A common use for samples of the posterior is to simulate the model's implied observations. Simluating implied obervations is useful for:

- Model design: sampling from the prior to understand its implications and see what the model expects before the data.
- After the model is updated with data, simulating implied observations and investigate model behaviour.
- Software validation simulate observations under.
- Research design and evaluating whether it will be effecttive or now. In a narrow sense, doing power analysis.
- Forecasting and simulating new predictions.
"""

# ╔═╡ 4c4b553d-4ba5-4fc4-9119-86f3b5a1849d
md"""### Generating dummy data

Using the globe tossing model, you can simulate the outcomes of the globe tossing under a value for the parameter ``P``:
"""

# ╔═╡ 2b75ec16-9fd7-4d86-88e0-94240bdf9722
md"""Analytical likelihoods:"""

# ╔═╡ 288e009a-df1b-4c52-9bd6-1e7fc9040f43
@. pdf(Binomial(9, 0.7), 0:9)

# ╔═╡ 3b35eaa3-e704-4ca7-a82b-ffa4bea195c8
md"""Estimated likelihoods from simulation:"""

# ╔═╡ 219e4064-12a6-4686-b181-8e99cc24cb38
rand(Binomial(9, 0.7), 10_000) |> x -> counts(x) / 10_000

# ╔═╡ 2acc5752-5b3e-47d3-8e4c-f18d24bfa586
begin
	analytical = @. pdf(Binomial(9, 0.7), 0:9)
	estimated = rand(Binomial(9, 0.7), 10_000) |> x -> counts(x, 0:9) / 10_000
	groupedbar([analytical estimated], labels=["true" "estimate"], legend=:topleft, xaxis="number of water", yaxis="proportion")
end

# ╔═╡ 5a2fa84f-457e-4839-81f2-633e92ee77a0
md"""### Posterior predicitive distribution

This is the distribution of possible predicitions from your model. It incorporates the uncertainty in the observation (if you know the true value of ``P`` you can't predict what will come next, unless ``P=0`` or ``P=1``), and uncertainty in the value of ``P``, i.e. the posterior distribution. The uncertainty in ``P`` is carried forward into the implied predictions by taking a weighted average of the predictions by the posterior distribution.
"""

# ╔═╡ a58923d4-090a-4c44-948d-90e163dc5a00
md"""Generating a prediction:"""

# ╔═╡ d56fa818-a031-4fd5-a05c-f6fb1912be34
@bind sampled_posterior Button("Sample a random value from the posterior")

# ╔═╡ b5d3fc06-1e64-4e47-9d46-a38a9c569a46
let
	sampled_posterior
	sampled_p = wsample(grid, posterior)
	sample = rand(Binomial(N, sampled_p), 10_000)
	proportions = counts(sample, 0:9) ./ 10_000
	p1 = plot(grid, posterior, label="", title="posterior distribution", xaxis="P", legend=:none)
	vline!([sampled_p], ls=:dash, label="sampled P = $(round(sampled_p, digits=3))")
	p2 = bar([0:9], proportions, title="sampling distribution for\n P = $(round(sampled_p, digits=3))", legend=:none, xticks=0:9, ylims=(0,0.5), xaxis="number of water" )
	
	plot(p1,p2)
end

# ╔═╡ d91ad027-e086-4675-93fc-eec8f151191f
md"""Repeating over the whole posterior:"""

# ╔═╡ 7da185e4-9868-46ce-bac2-4e2ceda8f17b
simulated_samples = @. rand(Binomial(N,samples), n_samples)

# ╔═╡ 4e336b94-e9c2-47e3-8088-07eff6f02ecb
md"""**Taking values of P from `samples` means they will be in proportion to the posterior distribution, incorporating its uncertainty**"""

# ╔═╡ c428d7c0-2073-4275-a50c-4a9d411b82f6
simulated_counts = map(x -> counts(x, 0:9), simulated_samples)

# ╔═╡ 35d9ec1c-f212-4d87-a70e-7d14f0956fed
simulated_proportions = sum(simulated_counts) ./ (n_samples*n_samples)

# ╔═╡ 2dc72b11-bdaf-4e03-869c-15abc6c5e048
bar(0:9, simulated_proportions, title="posterior predictive distribution", xticks=0:9, legend=:none, xaxis="number of water tosses", yaxis="proportion of throws")

# ╔═╡ e50dd983-ac8a-4f6e-8b3f-e97b42b488e9
md"""## Practice problems

### Hard
#### Introduction

The data below indicates the gender (male=1, female=0) of reported first and second born children in 100 two-child families.
"""

# ╔═╡ 995f788a-850b-4025-be00-02af762957de
birth1 = [1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,0,1,1,1,1]

# ╔═╡ b249b896-7165-45e4-b172-0e2ecc6b2b69
birth2 = [0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0]

# ╔═╡ 84ff7ecc-97df-4532-b054-84520e8340f8
md"""##### 3H1
Using grid approximation, computer the posterior dsitribution of a birth being a boy. Assume uniform prior.

Which parameter value maximises the posterior probability?

- The parameter of interest is the proportion of boys born ``P``.
- Assume births are independent

"""

# ╔═╡ 283d84a7-d8d8-419e-91f4-086e0e8a4b1d
let
	N = 200 	# number of births, assumed independent
	Nboys = sum(birth1) + sum(birth2) # number of boys
	n = 1001 	# grid granularity
	grid = range(0, 1, length=n)
	prior = @. pdf(Uniform(0,1), grid)
	likelihood = @. pdf(Binomial(N, grid), Nboys)
	posterior = likelihood .* prior
	posterior /= sum(posterior)
	plot(grid, posterior, label="", title="posterior distribution")
	map = grid[argmax(posterior)]
	vline!([map], label="MAP = $(round(map, digits=3))")
end

# ╔═╡ 9b155b0f-c605-46d0-9bd2-b69be3e21042
md"""##### 3H2
Draw 10,000 random parameter values from the posterior dsitribution and use these to estimate the 50%, 89% and 97% HPDI.
"""

# ╔═╡ 3c0ee869-a119-4f4c-89e9-9e57b2090505
let
	# the model
	N = 200 	# number of births, assumed independent
	Nboys = sum(birth1) + sum(birth2) # number of boys
	n = 101 	# grid granularity
	grid = range(0, 1, length=n)
	prior = @. pdf(Uniform(0,1), grid)
	likelihood = @. pdf(Binomial(N, grid), Nboys)
	posterior = likelihood .* prior
	posterior /= sum(posterior)
	n_samples = 10_000
	samples = wsample(grid, posterior, n_samples)
	de = kde(samples)
	plt = plot(de.x, de.density, label="", lw=2, palette=palette([:gray], 10))
	alpha = [0.5, 0.11, 0.03]
	as = Vector{Float64}(undef,3)
	bs = Vector{Float64}(undef, 3)
	for i in 1:3
		a, b = hpdi(samples, alpha=alpha[i])
		as[i], bs[i] = a,b
		start = argmin(abs.(a .- de.x))
		stop = argmin(abs.(b .- de.x))
		_x = de.x[start:stop]
		_y = de.density[start:stop]
		plot!(_x, zeros(length(_x)), fillrange=_y, fillalpha=0.3, label="",linecolor=:white)
		plot!([a b; a b], [0 0; de.density[start] de.density[stop]], lw=3, lc=:black, label="")
	end
	for i in 1:3
		a, b = hpdi(samples, alpha=alpha[i])
		as[i], bs[i] = a,b
		start = argmin(abs.(a .- de.x))
		stop = argmin(abs.(b .- de.x))
		plot!([a b; a b], [0 0; de.density[start] de.density[stop]], lw=3, lc=:black, label=["HPDI = $(1-alpha[i])" ""])
	end
	plt
end

# ╔═╡ 79cfd95f-0e98-4396-b5a3-35fabf5a6fd1
md"""##### 3H3

Simulate 10,000, 200 births, giving the predictive psoterior distribution of the proportion of boys ``P``. Does the model fit the data well?
"""

# ╔═╡ 382f2550-32b0-4ca8-92c5-3d04c29c5b83
let
	# the model
	N = 200 	# number of births, assumed independent
	Nboys = sum(birth1) + sum(birth2) # number of boys
	n = 101 	# grid granularity
	grid = range(0, 1, length=n)
	prior = @. pdf(Uniform(0,1), grid)
	likelihood = @. pdf(Binomial(N, grid), Nboys)
	posterior = likelihood .* prior
	posterior /= sum(posterior)
	
	# sampled from the posterior
	n_samples = 1_000
	samples = wsample(grid, posterior, n_samples)

	# simulated observations
	n_simulations = 10_000
	simulated_samples = @. rand(Binomial(N, samples), n_simulations)
	simulated_boys = sum(map(x -> counts(x, 0:N), simulated_samples)) ./ (n_simulations*n_samples)
	bar(0:N, simulated_boys, xticks=0:20:N, label="predicted distribution of P")
	vline!([Nboys], lw=2, label="actual number of boys")
end

# ╔═╡ b68063cd-d0d3-42a8-93e4-8eed50465d34
md"""##### 3H4

Now compare 10,000 counts of boys from 100 simulated first borns only to the number of boys in the first births, `birth1`. How does the model compare now?
"""

# ╔═╡ 88254a2d-af98-4c50-8d74-ca1a64245479
let
	# the model
	N = 200 	# number of births, assumed independent
	Nboys = sum(birth1) + sum(birth2)
	n = 101 	# grid granularity
	grid = range(0, 1, length=n)
	prior = @. pdf(Uniform(0,1), grid)
	likelihood = @. pdf(Binomial(N, grid), Nboys)
	posterior = likelihood .* prior
	posterior /= sum(posterior) 
	
	# sampled from posterior
	n_samples = 1_000
	samples = wsample(grid, posterior, n_samples)

	# simulated observations
	n_simulations = 10_000
	simulated_samples = @. rand(Binomial(100, samples), n_simulations)
	simulated_boys = sum(map(x -> counts(x, 0:100), simulated_samples)) ./ (n_simulations*n_samples)
	bar(0:100, simulated_boys, xticks=0:20:100, label="predicted distribution of P")
	vline!([sum(birth1)], lw=2, label="actual number of boys")
end

# ╔═╡ bd502d36-7bf6-43a7-847a-8ce2986dd35e
md"""##### 3H5

The model assumes that sex of first and second borns are independent, check this assumption by focusing on second births that followed from female first borns. 

Compare 10,000 simulated counts of boys to only those second births that followed girls.


"""


# ╔═╡ dca7b0ac-6c94-469b-a438-645eb5d44e05
let
	# the model
	N = 200 	# number of births, assumed independent
	Nboys = sum(birth1) + sum(birth2) 
	n = 101 	# grid granularity
	grid = range(0, 1, length=n)
	prior = @. pdf(Uniform(0,1), grid)
	likelihood = @. pdf(Binomial(N, grid), Nboys)
	posterior = likelihood .* prior
	posterior /= sum(posterior) 
	
	# sampled from posterior
	n_samples = 1_000
	samples = wsample(grid, posterior, n_samples)

	# simulated observations
	n_simulations = 10_000
	n_girls1 = sum(birth1 .== 0) # number of first born girls
	simulated_samples = @. rand(Binomial(n_girls1, samples), n_simulations)
	simulated_boys2 = sum(map(x -> counts(x, 0:n_girls1), simulated_samples)) ./ (n_simulations*n_samples)
	bar(0:n_girls1, simulated_boys2, label="simulated boys")
	
	# actual observations
	n_boys2 = sum(birth2[birth1 .== 0])
	vline!([n_boys2], lw=2, label="observed boys")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.18"
KernelDensity = "~0.6.3"
PlutoUI = "~0.7.15"
StatsBase = "~0.33.10"
StatsPlots = "~0.14.28"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a325370b9dd0e6bf5656a6f1a7ae80755f8ccc46"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.2"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "ff7890c74e2eaffbc0b3741811e3816e64b6343d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.18"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "29890dfbc427afa59598b8cfcc10034719bd7744"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4c8c0719591e108a83fb933ac39e32731c7850ff"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.60.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "f6532909bf3d40b308a0f360b6a0e626c0e263a8"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.1"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "a8709b968a1ea6abc2dc1967cb1db6ac9a00dfb6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.5"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "6841db754bd01a91d281370d9a0f8787e220ae08"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.4"

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "633f8a37c47982bff23461db0076a33787b17ecd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.15"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "793793f1df98e3d7d554b65a107e9c9a6399a6ed"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.7.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "eb007bb78d8a46ab98cd14188e3cec139a4476cf"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.28"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "019acfd5a4a6c5f0f38de69f2ff7ed527f1881da"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.1.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─535b43f0-2780-11ec-1444-d98d954f2508
# ╟─9993e805-7265-4c1d-b8a9-30c73f91bc37
# ╠═3dc78b1b-58d3-40eb-a1c5-a2fa58f15abf
# ╟─01914d0b-ba88-435f-8194-6acc5437c0e1
# ╠═a7667076-b719-46e6-8ef7-690019c176b6
# ╟─2e20bfcb-9633-4808-9f29-d3ec2ed472fe
# ╟─d63a4f5f-6274-40a6-90d9-b6933382e24d
# ╟─9356eb97-efae-47cb-a5a9-6676f00a5f1c
# ╠═63fdfebf-d08f-438c-a69b-9db2200a61a6
# ╟─0072d892-1ede-4432-afca-9132c5695586
# ╟─d5d3b50b-01b9-4201-b01c-19cbe0c87bcf
# ╟─cf44ec9a-7c16-4ab4-b547-3b57263da109
# ╟─f88a09e1-d3dc-4228-856e-f82725df693d
# ╟─6784239e-5595-471a-b4e8-348491e29e5e
# ╟─41688bf0-3188-4d31-897c-f864de97558f
# ╟─9b4a897d-f0f2-42d9-a782-b47356065118
# ╠═d9e160f8-e203-4f23-ad03-2c7bd3410c5b
# ╟─3b2cca7a-3db1-4f08-af04-5d1df2955be4
# ╠═cc34c6a3-ae53-41f6-9934-dc789508883f
# ╟─0d0ed10d-78e3-4486-ab4e-bd0966be7cc5
# ╠═bf7367bc-2375-4942-abeb-7a756f5a1dc4
# ╟─1d838e46-53b6-45eb-ab56-c404773a6f94
# ╟─afedb578-664f-41bb-96f5-1b9f5eebfcfd
# ╠═5f549033-b925-438d-b56b-f5f18fa50613
# ╟─0da1daed-171f-4156-9247-21c2258d11c0
# ╟─661b6777-0991-4274-8443-d0f6f1e94051
# ╠═f6c340ac-a923-4be5-b1ce-2a002d737cfc
# ╟─a5beaffe-fa90-43a6-8efc-d9c0baa818e5
# ╠═8a037821-ec0f-4a4d-8292-21833f0c05f7
# ╠═3785acb9-4e2c-4299-a925-c6006ce46890
# ╟─68884922-52cc-4a7f-8a9c-b0d5cde80875
# ╟─f594e927-1d66-4107-8fec-7f2b60617110
# ╟─7dfb3173-b7af-4546-bbcb-495cb7498a3b
# ╟─64635021-340b-4968-adc4-8ea3e9d7c5cf
# ╠═82db866d-62af-412a-aeee-64940c64fa10
# ╟─fa055e68-a5d5-4ecd-970b-de20766e4a0a
# ╟─57a9acfa-6665-4936-8864-a54269e79a6b
# ╟─b5f28886-2239-4cb3-8248-00c4f32cfa91
# ╟─369febaa-8600-409e-9035-00e46483b70c
# ╟─ddf696bd-e6ac-48eb-b421-d5d778c2673b
# ╠═b6b6fe45-e22f-4ca4-93f0-2ad1971d5339
# ╟─455f6d2e-9abd-45ee-bd8d-83c3d92218ec
# ╟─4ed753dc-44a8-48fc-82aa-80a95f40278b
# ╟─02a8b42e-340e-4b2e-90f1-38db6f2aa03c
# ╟─bdee2725-7fde-4548-afea-2b220cadf951
# ╟─4c4b553d-4ba5-4fc4-9119-86f3b5a1849d
# ╟─2b75ec16-9fd7-4d86-88e0-94240bdf9722
# ╠═288e009a-df1b-4c52-9bd6-1e7fc9040f43
# ╟─3b35eaa3-e704-4ca7-a82b-ffa4bea195c8
# ╠═219e4064-12a6-4686-b181-8e99cc24cb38
# ╟─2acc5752-5b3e-47d3-8e4c-f18d24bfa586
# ╟─5a2fa84f-457e-4839-81f2-633e92ee77a0
# ╟─a58923d4-090a-4c44-948d-90e163dc5a00
# ╟─d56fa818-a031-4fd5-a05c-f6fb1912be34
# ╟─b5d3fc06-1e64-4e47-9d46-a38a9c569a46
# ╟─d91ad027-e086-4675-93fc-eec8f151191f
# ╠═7da185e4-9868-46ce-bac2-4e2ceda8f17b
# ╟─4e336b94-e9c2-47e3-8088-07eff6f02ecb
# ╠═c428d7c0-2073-4275-a50c-4a9d411b82f6
# ╠═35d9ec1c-f212-4d87-a70e-7d14f0956fed
# ╟─2dc72b11-bdaf-4e03-869c-15abc6c5e048
# ╟─e50dd983-ac8a-4f6e-8b3f-e97b42b488e9
# ╟─995f788a-850b-4025-be00-02af762957de
# ╟─b249b896-7165-45e4-b172-0e2ecc6b2b69
# ╟─84ff7ecc-97df-4532-b054-84520e8340f8
# ╠═283d84a7-d8d8-419e-91f4-086e0e8a4b1d
# ╟─9b155b0f-c605-46d0-9bd2-b69be3e21042
# ╠═3c0ee869-a119-4f4c-89e9-9e57b2090505
# ╟─79cfd95f-0e98-4396-b5a3-35fabf5a6fd1
# ╠═382f2550-32b0-4ca8-92c5-3d04c29c5b83
# ╟─b68063cd-d0d3-42a8-93e4-8eed50465d34
# ╠═88254a2d-af98-4c50-8d74-ca1a64245479
# ╟─bd502d36-7bf6-43a7-847a-8ce2986dd35e
# ╠═dca7b0ac-6c94-469b-a438-645eb5d44e05
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
