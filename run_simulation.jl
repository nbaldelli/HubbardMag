let
    #include("H_dmrg.jl") 
    #include("H_obs.jl")
    include("H_processing.jl")
    #include("H_parallel_dmrg.jl")
    include("parhub.jl")
end

Nx = 4; Ny = 3; t = 1.; tp = 0.2; J = 0.4; U = 4(t^2)/J
α=2.0/((Nx-1)*Ny); doping = 2/9; max_linkdim = 20; yperiodic = true; reupload = false; psi0 = nothing
prev_alpha = 1.0/((Nx-1)*Ny); 
type = 1

input_par = Dict(:Nx => Nx, :Ny => Ny, :t => t, :tp => tp, :J => J, :U=>U, :α => α, :doping => doping, :type => type,
            :max_linkdim => max_linkdim, :yperiodic => yperiodic, :reupload => reupload, :prev_alpha => prev_alpha, :psi0 => nothing)

#main_dmrg(; input_par...)
#main_space(Nx = Nx, Ny = Ny, U = U, t = t, one_site = true, doping = doping, max_linkdim = max_linkdim)
main_kspace(; input_par...)
#main_ky(Nx = Nx, Ny = Ny, U = U, t = t, tp = tp, α=α, doping = doping, max_linkdim = max_linkdim)
#main_obs(; input_par...)
#main_processing(; input_par...)
#