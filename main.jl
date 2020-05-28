using Flux
include("base.jl")
include("individual.jl")
include("population.jl")
include("lemonade.jl")


ind = Individual()
ind.genotype = [MConvBlock((40, 40, 4), (40, 40, 16)), MDepthwiseConvBlock((40, 40, 16), (40, 40, 32)), MSkipConnection((40, 40, 16), (40, 40, 48)), MConv((40, 40, 48), (40, 40, 128))]
println(ind.genotype)
construct_model!(ind, 4, LAYER_OUTPUT)
println(ind.model)
X = randn(Float32, 40, 40, 4, 1)
println(ind.model(X))

mutate!(ind)
println(ind.genotype)