let
    include("H_dmrg.jl") 
    include("H_obs.jl")
    include("H_processing.jl")
end

Nx = 32; Ny = 6; t = 1; tp = 0.2; J = 0.4; U = 4(t^2)/J
α=0.0/((Nx-1)*Ny); doping = 1/16; max_linkdim = 3501; yperiodic = false; reupload = false; psi0 = nothing
prev_alpha = 1.0/((Nx-1)*Ny)
input_par = Dict(:Nx => Nx, :Ny => Ny, :t => t, :tp => tp, :J => J, :U=>U, :α => α, :doping => doping, 
            :max_linkdim => max_linkdim, :yperiodic => yperiodic, :reupload => reupload, :prev_alpha => prev_alpha, :psi0 => nothing)

main_dmrg(; input_par...)
main_obs(; input_par...)
#main_processing(; input_par...)
