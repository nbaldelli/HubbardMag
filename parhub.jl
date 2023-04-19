#using MPI
#using MKL
#using Distributed
#using ClusterManagers
#using SlurmClusterManager
using ITensors, ITensors.HDF5
using LinearAlgebra
using Random
using ITensorParallel

#rmprocs(setdiff(procs(), 1))
#addprocs(SlurmManager(4))
#@show nprocs()

#@everywhere using ITensors
#@everywhere using ITensorParallel

include(joinpath(pkgdir(ITensors), "examples", "src", "electronk.jl"))
include(joinpath(pkgdir(ITensors), "examples", "src", "hubbard.jl"))

using LinearAlgebra, ITensors, ITensors.HDF5, Random
using DelimitedFiles

function hubbard_tt_2d_ky(; Nx::Int, Ny::Int, t=1.0, U=0.0, tp=0.0, α=0.0)
  ampo = OpSum()
  for x in 0:(Nx - 1) #along ky
    for ky in 0:(Ny - 1)
      s = x * Ny + ky + 1
      disp = -2 * t * cos(((2 * π / Ny) * ky + 2pi * α * (x - (Nx)/2)))
      if abs(disp) > 1e-12
        ampo .+= disp, "Nup", s
        ampo .+= disp, "Ndn", s
      end
    end
  end
  for x in 0:(Nx - 2) #along x hopping
    for ky in 0:(Ny - 1)
      s1 = x * Ny + ky + 1
      s2 = (x + 1) * Ny + ky + 1
      pref = -t - 2 * tp * cos((2 * π / Ny) * ky + 2pi * α * (x - (Nx)/2 + 0.5))
      ampo .+= pref, "Cdagup", s1, "Cup", s2
      ampo .+= pref, "Cdagup", s2, "Cup", s1
      ampo .+= pref, "Cdagdn", s1, "Cdn", s2
      ampo .+= pref, "Cdagdn", s2, "Cdn", s1
    end
  end
  
  if U ≠ 0
    for x in 0:(Nx - 1)
      for ky in 0:(Ny - 1)
        for py in 0:(Ny - 1)
          for qy in 0:(Ny - 1)
            s1 = x * Ny + (ky + qy + Ny) % Ny + 1
            s2 = x * Ny + (py - qy + Ny) % Ny + 1
            s3 = x * Ny + py + 1
            s4 = x * Ny + ky + 1
            ampo .+= (U / Ny), "Cdagdn", s1, "Cdagup", s2, "Cup", s3, "Cdn", s4
          end
        end
      end
    end
  end
  return ampo
end

function number_ky(C, Nx, Ny)
  n_xy = zeros(Nx, Ny)
  for x in 0:(Nx - 1) #along ky
    for y in 0:(Ny - 1) #along q
      temp = 0 
      for ky in 0:(Ny - 1)
        s1 = x * Ny + ky + 1
        for qy in 0:(Ny - 1)
          s2 = x * Ny + qy + 1
          temp += C[s1, s2] * exp(-im * (2 * pi * (ky - qy) / Ny) * y)
        end
      end
      n_xy[x + 1, y + 1] = temp / Ny
    end
  end
  return n_xy
end

function main_parallel_dist(;
  parallel = "None",
  blocksparse = true,
  Nx::Int=4,
  Ny::Int=4,
  U::Float64=10.0,
  t::Float64=1.0,
  doping = 2/16,
  tp=0.0,
  α=0.0,
  max_linkdim::Int=3000,
  )
  
  #MPI.Init() #every output inside this function is printed by all the processes
  ITensors.BLAS.set_num_threads(Threads.nthreads())
  ITensors.disable_threaded_blocksparse()
  @show Threads.nthreads()

  if blocksparse == true
    ITensors.BLAS.set_num_threads(1)
    ITensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse()
  end
  @show ITensors.using_threaded_blocksparse()

  seed = 1234
  Random.seed!(seed)

  N = Nx * Ny

  holes = floor(Int,N*doping)
  num_f = N - holes
  
  sites = siteinds("ElecK", N; conserve_qns = true, conserve_ky = true, modulus_ky = Ny)
  if parallel == "MPI"
     sites = MPI.bcast(sites, 0, MPI.COMM_WORLD)
  end
  
  in_state = []
  
  for i in 1:Int(Nx/2)
      push!(in_state, [isodd(n) ? "Up" : "Dn" for n in 1:Ny]...)
      push!(in_state, [isodd(n) ? "Dn" : "Up" for n in 1:Ny]...)
  end

  distr_pol = round.([((Nx+1)/((holes/2)+1))*i for i in 1:Int(holes/2)])
  distr_holes = [Int.((Ny .* distr_pol) .- Ny .+ 2)]
  distr_holes2 = [Int.((Ny .* distr_pol))]
  distr_holes3 = [Int.((Ny .* distr_pol) .- 1)]
  in_state[distr_holes3...] .= in_state[distr_holes2...];
  in_state[distr_holes...] .= "0"; in_state[(distr_holes2)...] .= "0"; 
  display(reshape(in_state,Ny,Nx))
  
  #in_state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  #psi0 = productMPS(sites, in_state)
  #if parallel == "MPI"
  #   psi0 = MPI.bcast(psi0, 0, MPI.COMM_WORLD)
  #end =#
    
  #in_state = ["Dn", "Up", "Dn", "Up", "Up", "Dn", "Up", "Dn", "Dn", "Up", "Dn", "Up", "Up", "Dn", "Up", "Dn", "Dn", "0", "Up", "0","Up", "Dn", "Up", "Dn", "Dn", "Up", "Dn", "Up", "Up", "Dn", "Up", "Dn", "Dn", "Up", "Dn", "Up"]
  psi0 = productMPS(sites, in_state)
  if parallel == "MPI"
     psi0 = MPI.bcast(psi0, 0, MPI.COMM_WORLD)
  end

  @show flux(psi0)
  
  os = hubbard_tt_2d_ky(Nx = Nx, Ny = Ny, t = t, U = U, tp = tp, α = α/((Nx-1)*Ny))

  if parallel == "MPI" 
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    @show nprocs
    Hs = partition(os, nprocs; in_partition=ITensorParallel.default_in_partition)
    which_proc = MPI.Comm_rank(MPI.COMM_WORLD) + 1
    PH = MPISumTerm(MPO(Hs[which_proc], sites), MPI.COMM_WORLD)
  elseif parallel == "Threaded"
    nthreads = Threads.nthreads()
    Hs = partition(os, nthreads; in_partition=ITensorParallel.default_in_partition)
    H = Vector{MPO}(undef, nthreads)
    Threads.@threads for n in 1:length(Hs)
      Hn = MPO(Hs[n], sites)
      Hn = splitblocks(linkinds, Hn)
      H[n] = Hn
    end
    @show maximum(maxlinkdim.(H))
    PH = ThreadedProjMPOSum(H)
  elseif parallel == "None"
    PH = MPO(os, sites)
    if blocksparse == true
      PH = splitblocks(linkinds, PH)
    end
    @show maxlinkdim(PH)
  elseif parallel == "Distributed"
    npartitions = 4
    ℋs = partition(os, npartitions; in_partition=ITensorParallel.default_in_partition)
    Hs = [splitblocks(linkinds,MPO(ℋ, sites)) for ℋ in ℋs]
    @show maxlinkdim.(Hs)
    PH = DistributedSum(Hs)
  end 

  en_obs = DMRGObserver(energy_tol = 1e-5)
  nsweeps = 15
  max_maxdim = max_linkdim
  maxdim = min.([100, 200, 500, max_maxdim], max_maxdim)
  mindim = min.([100, 200, 500, max_maxdim], max_maxdim)
  cutoff = 1e-9
  noise = [1e-6, 1e-7, 1e-8, 0.0]

  @show nsweeps
  @show maxdim
  @show cutoff
  @show noise
  #@show en_obs


  energy, psi = @time dmrg(
    PH, psi0; nsweeps, mindim, maxdim, cutoff, noise, observer = en_obs,
  )

  C = correlation_matrix(psi, "Cdagup", "Cup")
  Cd = correlation_matrix(psi, "Cdagdn", "Cdn")
  #display(diag(C.+Cd))

  n_xy = number_ky(C, Nx, Ny) + number_ky(Cd, Nx, Ny)
  display(n_xy)

  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  
  if parallel == "MPI" 
    #MPI.Finalize()
  end
  #return energy, H, psi
  return psi
end

#MPI.Init()
#Nx = 10; Ny = 6; t = 1.; U = 10.; max_linkdim = 3000; tp = 0.2; doping = 2/60
#main_parallel_dist(; Nx = Nx, Ny = Ny, t = t, tp = tp, α = 2.0, doping = doping,  U = U,  parallel = "None", blocksparse = true, max_linkdim = max_linkdim)

function main_kspace(; Nx = 4, Ny = 4, t = 1, tp = 0.0, J = 0.4, U=10., α=0, doping = 0, 
                    type = nothing, yperiodic = nothing,
                    max_linkdim = 900, reupload = false, prev_alpha = 0/60, psi0 = nothing)

    if reupload;
        f = h5open("merda.h5","r")
        psi0 = read(f,"psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($prev_alpha)_mlink($max_linkdim)_kspace",MPS)
        close(f)
    end

    println("DMRG run (kspace): #threads=$(Threads.nthreads()), tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)")
    
    #psi = dmrg_run_tj(Nx, Ny, t, tp, J, doping = doping, maxlinkdim = maxlinkdim)
    psi = main_parallel_dist(; Nx = Nx, Ny = Ny, t = t, tp = tp, α = α, doping = doping,  U = U,  parallel = "None", blocksparse = true, max_linkdim = max_linkdim)

    h5open("merda.h5","cw") do f
        if haskey(f, "psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_kspace")
            delete_object(f, "psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_kspace")
        end
        write(f,"psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_kspace",psi)
    end
end
