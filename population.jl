include("individual.jl")

########################################
########################################
#########      POPULATION      #########
########################################
########################################
mutable struct Population
    pop::Array{Individual}
    Population() = new()
end
