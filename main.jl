using Flux
include("lemonade.jl")
include("base.jl")

ind = Individual()
ind.genotype = [MConvBlock(), MDepthwiseConvBlock(), MBatchNorm(), MReLU()]

construct_model!(ind, 4, LAYER_OUTPUT)

X = randn(Float32, 40, 40, 4, 1)
println(ind.model(X))
