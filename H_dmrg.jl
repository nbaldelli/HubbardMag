using LinearAlgebra, ITensors, ITensors.HDF5, Random

function square_lattice_nn(Nx::Int, Ny::Int; kwargs...)::Lattice
    yperiodic = get(kwargs, :yperiodic, false)
    yperiodic = yperiodic && (Ny > 2)
    N = Nx * Ny
    Nbond = 2N - Ny + (yperiodic ? 0 : -Nx)
    Nbond_nn = 2(Nx-1)*(Ny - (yperiodic ? 0 : 1))
    latt = Lattice(undef, Nbond + Nbond_nn)
    b = 0
    for n in 1:N
      x = div(n - 1, Ny) + 1
      y = mod(n - 1, Ny) + 1

      #nearest neighbour
      if x < Nx
        latt[b += 1] = LatticeBond(n, n + Ny, x, y, x + 1, y, "1")
      end
      if Ny > 1
        if y < Ny
          latt[b += 1] = LatticeBond(n, n + 1, x, y, x, y + 1, "1")
        end
        if yperiodic && y == 1
          latt[b += 1] = LatticeBond(n, n + Ny - 1, x, y, x, y + Ny - 1, "1")
        end
      end

      #next nearest neighbour
      if x < Nx && Ny > 1
        if y < Ny
            latt[b += 1] = LatticeBond(n, n + Ny + 1, x, y, x + 1, y + 1, "2")
        end
        if y > 1
            latt[b += 1] = LatticeBond(n, n + Ny - 1, x, y, x + 1, y - 1, "2")
        end
        if yperiodic && y == Ny
            latt[b += 1] = LatticeBond(n, n + 1, x, Ny, x + 1, 1, "2")
        end
        if yperiodic && y == 1
            latt[b += 1] = LatticeBond(n, n + 2Ny - 1, x, 1, x + 1, Ny, "2")
        end
      end
    end
    return latt
end

function dmrg_run_tj(Nx, Ny, t, tp, J, lattice; doping = 1/16, sparse_multithreading = true, maxlinkdim = 100)
    
    if sparse_multithreading
        ITensors.Strided.set_num_threads(1)
        BLAS.set_num_threads(1)
        ITensors.enable_threaded_blocksparse()
    else
        ITensors.Strided.set_num_threads(16)
        BLAS.set_num_threads(16)
        ITensors.disable_threaded_blocksparse()
    end

    N = Nx * Ny
    #Nϕ = yperiodic ? α*(Nx-1)*(Ny) : α*(Nx-1)*(Ny-1) #number of fluxes threading the system

    sites = siteinds("tJ", N, conserve_qns = true) #number of fermions is conserved, magnetization is NOT conserved
    #lattice = square_lattice_nn(Nx, Ny; yperiodic = yperiodic)

    #=set up of DMRG schedule=#
    sweeps = Sweeps(20)
    setmaxdim!(sweeps, max_linkdim)
    setcutoff!(sweeps, 1e-9)
    setnoise!(sweeps, 1e-8, 1e-10, 0) #how much??
    en_obs = DMRGObserver(energy_tol = 1e-7)

    holes = floor(Int,N*doping)
    num_f = N - holes
    in_state = ["Up" for _ in 1:floor(num_f/2)]
    append!(in_state, ["Dn" for _ in 1:(num_f - floor(num_f/2))])
    append!(in_state, ["0" for _ in 1:holes ])
    shuffle!(in_state) #DANGER: changes at every run

    psi0 = randomMPS(sites, in_state, linkdims = 50)

    #= hamiltonian definition tj=#
    ampo = OpSum()  
    for bond in lattice
        if bond.type == "1"
            ampo += -t, "Cdagup", bond.s1, "Cup", bond.s2 #nearest-neighbour hopping
            ampo += -t, "Cdagdn", bond.s1, "Cdn", bond.s2
            ampo += -t, "Cdagup", bond.s2, "Cup", bond.s1
            ampo += -t, "Cdagdn", bond.s2, "Cdn", bond.s1

            ampo += J/2, "S+", bond.s1, "S-", bond.s2 #heisenberg interaction
            ampo += J/2, "S-", bond.s1, "S+", bond.s2
            ampo += J, "Sz", bond.s1, "Sz", bond.s2
            
            ampo += -J/4, "Ntot", bond.s1, "Ntot", bond.s2 #density-density interaction
        end
        if bond.type == "2"
            ampo += -tp, "Cdagup", bond.s1, "Cup", bond.s2 #next-nearest-neighbour hopping
            ampo += -tp, "Cdagdn", bond.s1, "Cdn", bond.s2
            ampo += -tp, "Cdagup", bond.s2, "Cup", bond.s1
            ampo += -tp, "Cdagdn", bond.s2, "Cdn", bond.s1
        end
    end

    H = MPO(ampo, sites)

    if sparse_multithreading
        H = splitblocks(linkinds,H)
    end

    ####################################
    #= DMRG =#
    energy, psi = @time dmrg(H, psi0, sweeps, observer = en_obs, verbose=false);
    ####################################

    #= write mps to disk =#
    #=
    f = h5open("trial.h5","cw")
    write(f,"Nx=($Nx)_Ny=($Ny)_2",psi)
    close(f)
    =#
    #=
    H2 = inner(H, psi, H, psi)
    variance = H2-energy^2
    @show real(variance) #variance to check for convergence
    =#

    #= observables computation =#
    return psi
end

function dmrg_run_hubbard(Nx, Ny, t, tp, U, lattice; psi0=nothing, α=0, doping = 1/16, sparse_multithreading = true, max_linkdim = 100)
    
    if sparse_multithreading
        ITensors.Strided.set_num_threads(1)
        BLAS.set_num_threads(1)
        ITensors.enable_threaded_blocksparse()
    else
        ITensors.Strided.set_num_threads(16)
        BLAS.set_num_threads(16)
        ITensors.disable_threaded_blocksparse()
    end

    N = Nx * Ny

    #lattice = square_lattice_nn(Nx, Ny; yperiodic = yperiodic)

    holes = floor(Int,N*doping)
    num_f = N - holes

    #Random.seed!(1234)
    
    if psi0 === nothing
        sites = siteinds("Electron", N, conserve_qns = true) #number of fermions is conserved, magnetization is zero
        in_state = ["Up" for _ in 1:(floor(num_f/2)-1)]
        append!(in_state, ["Dn" for _ in 1:(num_f - floor(num_f/2)) -1])
        append!(in_state, ["UpDn"])
        append!(in_state, ["0" for _ in 1:(holes+1) ])
        shuffle!(in_state) #DANGER: changes at every run
        println(in_state)
        psi0 = randomMPS(sites, in_state, linkdims=10)
    end
    
    #=
    if psi0 === nothing
        sites = siteinds("Electron", N, conserve_qns = true) #number of fermions is conserved, magnetization is zero
        in_state = []
        for i in 1:Int(Nx/2)
            push!(in_state, [isodd(n) ? "Up" : "Dn" for n in 1:Ny]...)
            push!(in_state, [isodd(n) ? "Dn" : "Up" for n in 1:Ny]...)
        end
        distr_pol = round.([(Nx/((holes/2)+1))*i for i in 1:Int(holes/2)])
        distr_holes = [Int.((Ny .* distr_pol) .- (Ny/2))]
        distr_holes2 = [Int.((Ny .* distr_pol) .- (Ny/2) .+ 1)]
        in_state[distr_holes...] .= "0"; in_state[(distr_holes2)...] .= "0"; 
        println(in_state)
        psi0 = productMPS(sites, in_state)
    end
    =#
    sites = siteinds(psi0)

    #=set up of DMRG schedule=#
    sweeps = Sweeps(60)
    setmaxdim!(sweeps, 500, 1000, 1500, 2000, max_linkdim)
    setcutoff!(sweeps, 1e-9)
    setnoise!(sweeps, 1e-8, 1e-10, 0)
    en_obs = DMRGObserver(energy_tol = 1e-7)
     
    #= hamiltonian definition hubbard=#
    ampo = OpSum()  
    for bond in lattice
        if bond.type == "1"
            pf = bond.y2 - bond.y1
            if abs(pf) > 1.1; pf = -sign(pf) end
            X = bond.x1 - (Nx)/2
            ampo += -t*exp(1im*2pi*α*X*pf), "Cdagup", bond.s1, "Cup", bond.s2 #nearest-neighbour hopping
            ampo += -t*exp(1im*2pi*α*X*pf), "Cdagdn", bond.s1, "Cdn", bond.s2
            ampo += -t*exp(-1im*2pi*α*X*pf), "Cdagup", bond.s2, "Cup", bond.s1
            ampo += -t*exp(-1im*2pi*α*X*pf), "Cdagdn", bond.s2, "Cdn", bond.s1
        end
        if bond.type == "2"
            pf = bond.y2-bond.y1
            if abs(pf)>1.1; pf = -sign(pf) end
            X = bond.x1 - (Nx)/2
            ampo += -tp*exp(1im*2pi*α*(X+0.5)*pf), "Cdagup", bond.s1, "Cup", bond.s2 #next-nearest-neighbour hopping
            ampo += -tp*exp(1im*2pi*α*(X+0.5)*pf), "Cdagdn", bond.s1, "Cdn", bond.s2
            ampo += -tp*exp(-1im*2pi*α*(X+0.5)*pf), "Cdagup", bond.s2, "Cup", bond.s1
            ampo += -tp*exp(-1im*2pi*α*(X+0.5)*pf), "Cdagdn", bond.s2, "Cdn", bond.s1
        end
    end
    for n in 1:N
        ampo += U, "Nup", n, "Ndn", n #density-density interaction
    end

    H = MPO(ampo, sites)

    if sparse_multithreading
        H = splitblocks(linkinds,H)
    end

    ####################################
    #= DMRG =#
    energy, psi = @time dmrg(H, psi0, sweeps, observer = en_obs, verbose=false);
    ####################################
    #= observables computation =#
    return psi
end

function main_dmrg(; Nx = 6, Ny = 4, t = 1, tp = 0.2, J = 0.4, U=10., α=1/60, doping = 1/16, 
                    yperiodic = true, type = nothing,
                    max_linkdim = 450, reupload = true, prev_alpha = 1/60, psi0 = nothing)

    if reupload;
        f = h5open("ceph/MPS.h5","r")
        psi0 = read(f,"psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($prev_alpha)_mlink($max_linkdim)",MPS)
        close(f)
    end

    println("DMRG run: #threads=$(Threads.nthreads()), tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
    
    lattice = square_lattice_nn(Nx, Ny, yperiodic = yperiodic)
    #psi = dmrg_run_tj(Nx, Ny, t, tp, J, doping = doping, maxlinkdim = maxlinkdim)
    psi = dmrg_run_hubbard(Nx, Ny, t, tp, U, lattice, psi0=psi0, α=α, doping = doping, max_linkdim = max_linkdim)

    #=
    h5open("ceph/MPS.h5","cw") do f
        if haskey(f, "psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
            delete_object(f, "psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        end
        write(f,"psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)",psi)
    end
    =#
end

#main_dmrg()

