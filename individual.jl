include("base.jl")

########################################
########################################
#########      INDIVIDUAL      #########
########################################
########################################
mutable struct Individual
    genotype::Array{MGene}
    # fitness::function
    model::Chain
    Individual() = new()

end

function construct_model!(ind::Individual, input_channels::Int64, output_layer::Dense)
    layers = []
    # last_output_channels = input_channels
    last_gene = MConv((40, 40, 4), (40, 40, 4))
    last_pheno = Nothing

    for g in ind.genotype
        if typeof(g) <: MBlockGene

            ph = get_phenotype(last_gene, g)
            last_pheno = Chain(ph...)
            last_gene = g

            push!(layers, last_pheno)
        else
            if typeof(g) <: MSkipConnection
                ph = get_phenotype(last_gene, g, last_pheno)
                pop!(layers)
                push!(layers, ph)
            else
                ph = get_phenotype(last_gene, g)
                push!(layers, ph)
            end
            last_pheno = ph
            last_gene = g
            # println(ph)
        end
    end

    ### TODO: placeholder
    ex_data = randn(Float32, 40, 40, 4, 1)
    dense_in_size = size(Chain(layers..., x -> reshape(x, :, size(x, 4)))(ex_data))[1]
    ###

    ind.model = Chain(layers..., x -> reshape(x, :, size(x, 4)), Dense(dense_in_size, 1, relu; initb = ones))
end
