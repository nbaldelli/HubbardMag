#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, ITensors, ITensors.HDF5, MKL, DelimitedFiles, ITensorGLMakie, Random, JLD

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
          latt[b += 1] = LatticeBond(n, n + Ny - 1, x, y, x, y + Ny, "1")
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

function SC_rho(psi, Nx, Ny)
    sites = siteinds(psi)  
    lat = square_lattice(Nx, Ny, yperiodic = true) #generate all bonds   
    C = zeros(length(lat), length(lat))
    println(length(lat)^2)
    Threads.@threads for i in 1:length(lat) #multithreading on 16 threads
        b1 = lat[i]
        for j in i:length(lat)
            b2 = lat[j]
            if intersect([b1.s1 b1.s2], [b2.s1 b2.s2]) == [] #exclude bonds sharing a common site
                println("($i,$j)")
                
                os = OpSum()            
                os += 1/2,"Cdagup", b1.s1, "Cdagdn", b1.s2, "Cdn", b2.s1, "Cup", b2.s2
                os += 1/2,"Cdagdn", b1.s1, "Cdagup", b1.s2, "Cup", b2.s1, "Cdn", b2.s2
                os += -1/2,"Cdagup", b1.s1, "Cdagdn", b1.s2, "Cup", b2.s1, "Cdn", b2.s2
                os += -1/2,"Cdagdn", b1.s1, "Cdagup", b1.s2, "Cdn", b2.s1, "Cup", b2.s2
                C[j, i] = inner(psi', MPO(os, sites), psi) #SC correlation function
            end
        end
    end
    return C
end

function ITensors.op(::OpName"SCdag", ::SiteType"tJ", s1::Index, s2::Index)    
    sc = (1/sqrt(2))*(op("Cdagup", s1)*op("Cdagdn", s2) - op("Cdagdn", s1)*op("Cdagup", s2)) 
    return sc
end

function ITensors.op(::OpName"SC", ::SiteType"tJ", s1::Index, s2::Index)    
    sc = (1/sqrt(2))*(op("Cdn", s1)*op("Cup", s2) - op("Cup", s1)*op("Cdn", s2)) 
    return sc
end

function ITensors.op!(Op::ITensor, ::OpName"Sy", ::SiteType"tJ", s::Index)
    Op[s' => 2, s => 3] = -0.5im
    return Op[s' => 3, s => 2] = 0.5im
end

function dmrg_run_tj(Nx, Ny, t, tp, J; doping = 1/16, sparse_multithreading = true, yperiodic = true, maxlinkdim = 100)
    
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
    lattice = square_lattice_nn(Nx, Ny; yperiodic = true)

    #=set up of DMRG schedule=#
    sweeps = Sweeps(20)
    setmaxdim!(sweeps, maxlinkdim)
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
            
            ampo += -J/4, "Ntot", bond.s1, "Ntot", bond.s1 #density-density interaction
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
    
    f = h5open("trial.h5","cw")
    write(f,"Nx=($Nx)_Ny=($Ny)_2",psi)
    close(f)
    
    #=
    H2 = inner(H, psi, H, psi)
    variance = H2-energy^2
    @show real(variance) #variance to check for convergence
    =#

    #= observables computation =#
    return psi
end


t = 1; tp = 0.2; J = 0.4; doping = 1/16; maxlinkdim = 800
Nx = 16; Ny = 4

f = h5open("corr.h5","r")
SC = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_mlink($maxlinkdim)")
corrup = read(f,"densup_H_tp($tp)_Nx($Nx)_Ny($Ny)_mlink($maxlinkdim)")
corrdn = read(f,"densdn_H_tp($tp)_Nx($Nx)_Ny($Ny)_mlink($maxlinkdim)")
close(f)
corr_dens = corrup + corrdn

f = h5open("MPS.h5", "r")
psi = read(f,"psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_mlink($maxlinkdim)", MPS)
close(f)


let
    dens = diag(corr_dens)
    println(sum(dens))
    mean_dens = zeros(Nx)
    for i in 1:Nx
        mean_dens[i] = 1-sum(dens[(1+(i-1)*Ny):(i*Ny)])/Ny
    end

    plt.figure(1)
    plt.plot(1:Nx,mean_dens)
    plt.scatter(1:Nx,mean_dens)
    plt.title("Nx=$Nx, Ny=$Ny, maxlinkdim=$maxlinkdim")
    plt.grid(true); plt.xlabel("x"); plt.ylabel("Mean hole density")

    plt.savefig("dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_mlink($maxlinkdim).pdf")
end

let 
    Cd = SC+SC'
    vals, vecs = eigen(C)

    plt.scatter(1:length(vals), reverse(vals))
    plt.title("Nx=$Nx, Ny=$Ny, maxlinkdim=$maxlinkdim")
    plt.ylabel("Mac. eigenvalue")
    plt.xlabel("eig. number")
    plt.grid(true)
    plt.savefig("spectrum_H_$Nx.pdf")
    

    dens = 1 .- diag(corr_dens)

    couples = Any[]
    lat = square_lattice(Nx, Ny, yperiodic = true) #generate all bonds   
    for b in lat
        push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
    end

    #= plot of current and density =#
    num = length(lat)-1; neig = length(lat)-num+1
    eig_v = real(vecs[:,num])
    fig, ax = plt.subplots(1, dpi = 150)
    ax.set_ylim(0.5,Ny+0.5)
    line_segments = plt.matplotlib.collections.LineCollection(couples,array=eig_v, 
                                                            norm=plt.matplotlib.colors.Normalize(vmin=minimum(eig_v), vmax=maximum(eig_v)),
                                                            linewidths=5, cmap=plt.get_cmap("RdBu_r"), rasterized=true, zorder = 0)
    pl_curr = ax.add_collection(line_segments)
    pl_dens = ax.scatter(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx), c=dens, s=100, marker="s", zorder = 1, cmap=plt.get_cmap("PuBu"), edgecolors="black")
    plt.gca().set_aspect("equal")
    plt.title("tp=$tp, Nx=$Nx, Ny=$Ny, mlink=$maxlinkdim, #eig.=$neig")
    plt.colorbar(pl_dens, ax=ax, location="bottom", label=" hole density", shrink=0.7, pad=0.03, aspect=50)
    plt.colorbar(pl_curr, ax=ax, location="bottom", label="pairing", shrink=0.7, pad=0.07, aspect=50)
    plt.tight_layout()
    plt.savefig("fulldens_H_($Nx)_($neig).pdf")
end


av_pair = zeros(Nx); num = length(lat)-4
for (i, b) in enumerate(lat)
    av_pair[Int(b.x1)] += (-1)^(b.x2-b.x1)*vecs[i,num]
end
plt.plot(av_pair)

for b in lat 
    println(b.x2-b.x1)
end