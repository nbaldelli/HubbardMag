using LinearAlgebra, ITensors, ITensors.HDF5, Random
using ITensorParallel
using MPI
MPI.Init()

include(joinpath(pkgdir(ITensors), "examples", "src", "electronk.jl"))
include(joinpath(pkgdir(ITensors), "examples", "src", "hubbard.jl"))
include(joinpath(pkgdir(ITensorParallel), "examples", "hubbard_conserve_momentum", "hubbard_ky.jl"))

function hubbard_2d_ky(; Nx::Int, Ny::Int, t=1.0, U=0.0)
    ampo = OpSum()
    for x in 0:(Nx - 1)
      for ky in 0:(Ny - 1)
        s = x * Ny + ky + 1
        disp = -2 * t * cos(((2 * π / Ny) * ky))
        if abs(disp) > 1e-12
          ampo .+= disp, "Nup", s
          ampo .+= disp, "Ndn", s
        end
      end
    end
    for x in 0:(Nx - 2)
      for ky in 0:(Ny - 1)
        s1 = x * Ny + ky + 1
        s2 = (x + 1) * Ny + ky + 1
        ampo .+= -t, "Cdagup", s1, "Cup", s2
        ampo .+= -t, "Cdagup", s2, "Cup", s1
        ampo .+= -t, "Cdagdn", s1, "Cdn", s2
        ampo .+= -t, "Cdagdn", s2, "Cdn", s1
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
      ampo .+= -pref, "Cdagup", s1, "Cup", s2
      ampo .+= -pref, "Cdagup", s2, "Cup", s1
      ampo .+= -pref, "Cdagdn", s1, "Cdn", s2
      ampo .+= -pref, "Cdagdn", s2, "Cdn", s1
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

function hubbard_2d(; Nx::Int, Ny::Int, t=1.0, U=0.0, yperiodic::Bool=true)
  N = Nx * Ny
  lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)
  ampo = OpSum()
  for b in lattice
    ampo .+= -t, "Cdagup", b.s1, "Cup", b.s2
    ampo .+= -t, "Cdagup", b.s2, "Cup", b.s1
    ampo .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
    ampo .+= -t, "Cdagdn", b.s2, "Cdn", b.s1
  end
  if U ≠ 0
    for n in 1:N
      ampo .+= U, "Nupdn", n
    end
  end
  return ampo
end

function hubbard(; Nx::Int, Ny::Int=1, t=1.0, U=0.0, yperiodic::Bool=true, ky::Bool=false)
  if Ny == 1
    ampo = hubbard_1d(; N=Nx, t=t, U=U)
  elseif ky
    ampo = hubbard_2d_ky(; Nx=Nx, Ny=Ny, t=t, U=U)
  else
    ampo = hubbard_2d(; Nx=Nx, Ny=Ny, yperiodic=yperiodic, t=t, U=U)
  end
  return ampo
end

function main_parallel_dist(;
    mpi = false,
    Nx::Int=6,
    Ny::Int=3,
    U::Float64=4.0,
    t::Float64=1.0,
    tp=0.0,
    α=0.0,
    maxdim::Int=3000,
    doping = 1/16,
    max_linkdim::Int=3000,
    psi0 = nothing,
    conserve_ky=true,
    use_splitblocks=true,
    in_partition=ITensorParallel.default_in_partition,
    )
    ITensors.BLAS.set_num_threads(1)
    ITensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse()
   
    @show Threads.nthreads()
    @show ITensors.using_threaded_blocksparse()
  
    N = Nx * Ny
  
    holes = floor(Int,N*doping)
    num_f = N - holes

    sweeps = Sweeps(60)
    setmaxdim!(sweeps, 500, 1000, 1500, 2000, max_linkdim)
    setcutoff!(sweeps, 1e-9)
    setnoise!(sweeps, 1e-8, 1e-10, 0)
    en_obs = DMRGObserver(energy_tol = 1e-7)
  
    sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky=conserve_ky, modulus_ky=Ny)
  
    os = hubbard_tt_2d_ky(; Nx=Nx, Ny=Ny, t=t, U=U, tp = tp, α=α)
  
    if mpi
      nprocs = MPI.Comm_size(MPI.COMM_WORLD)
      Hs = partition(os, nprocs; in_partition=in_partition)  
      n = MPI.Comm_rank(MPI.COMM_WORLD) + 1
      PH = MPISum(ProjMPO(MPO(Hs[n], sites)))
      println("MPO distributed Hamiltonian")
    else
      PH = MPO(os, sites)
      @show maxlinkdim(PH)
    end

    # Number of structural nonzero elements in a bulk
    # Hamiltonian MPO tensor
   
    #Random.seed!(1234)
    # Create start state
    if psi0 === nothing
        in_state = ["Up" for _ in 1:(floor(num_f/2)-1)]
        append!(in_state, ["Dn" for _ in 1:(num_f - floor(num_f/2)) -1])
        append!(in_state, ["Up"])
        append!(in_state, ["Dn" for _ in 1:(holes+1) ])
        shuffle!(in_state) #DANGER: changes at every run
        println(in_state)
        psi0 = randomMPS(sites, in_state, linkdims=10)
    end
    
    
    energy, psi = @time dmrg(
      PH, psi0, sweeps; svd_alg="divide_and_conquer", observer = en_obs
    )
    @show Nx, Ny
    @show t, U
    @show flux(psi)
    @show maxlinkdim(psi)
    @show energy
    #return energy, H, psi
    
end

function main_ky(;
    Nx::Int=6,
    Ny::Int=3,
    U::Float64=4.0,
    t::Float64=1.0,
    tp=0.0,
    α=0.0,
    maxdim::Int=3000,
    doping = 1/16,
    max_linkdim::Int=3000,
    psi0 = nothing,
    conserve_ky=true,
    use_splitblocks=true,
    )

    Random.seed!(1234)
    ITensors.Strided.set_num_threads(1)
    BLAS.set_num_threads(1)
    ITensors.disable_threaded_blocksparse()

    @show Threads.nthreads()
    @show ITensors.using_threaded_blocksparse()
  
    N = Nx * Ny
  
    holes = floor(Int,N*doping)
    num_f = N - holes

    sweeps = Sweeps(20)
    setmaxdim!(sweeps, 500, 1000, 1500, 2000, max_linkdim)
    setcutoff!(sweeps, 1e-9)
    setnoise!(sweeps, 1e-8, 1e-10, 0)
    en_obs = DMRGObserver(energy_tol = 1e-7)

    sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky=conserve_ky, modulus_ky=Ny)
  
    os = hubbard_tt_2d_ky(; Nx=Nx, Ny=Ny, t=t, U=U, tp = tp, α=α)
    H = MPO(os, sites; splitblocks=use_splitblocks)
    @show maxlinkdim(H)
  
    # Number of structural nonzero elements in a bulk
    # Hamiltonian MPO tensor
    @show nnz(H[end ÷ 2])
    @show nnzblocks(H[end ÷ 2])
  
    # Create start seed
    #Random.seed!(1234)
    # Create start state
    if psi0 === nothing
        in_state = ["Up" for _ in 1:(floor(num_f/2)-1)]
        append!(in_state, ["Dn" for _ in 1:(num_f - floor(num_f/2)) -1])
        append!(in_state, ["Up"])
        append!(in_state, ["Dn" for _ in 1:(holes+1) ])
        shuffle!(in_state) #DANGER: changes at every run
        println(in_state)
        psi0 = randomMPS(sites, in_state, linkdims=10)
    end
  
    energy, psi = @time dmrg(
      H, psi0, sweeps, observer = en_obs, verbose = false
    )
    @show Nx, Ny
    @show t, U
    @show flux(psi)
    @show maxlinkdim(psi)
    @show energy
    #return energy, H, psi
end

function main_space(;
  Nx::Int=6,
  Ny::Int=3,
  U::Float64=10.0,
  t::Float64=1.0,
  doping = 1/16,
  max_linkdim::Int=3000,
  psi0 = nothing,
  splitblocks= true,
  one_site = false
  )

  ITensors.Strided.set_num_threads(1)
  BLAS.set_num_threads(1)
  ITensors.enable_threaded_blocksparse()

  @show Threads.nthreads()
  @show ITensors.using_threaded_blocksparse()

  N = Nx * Ny

  holes = floor(Int,N*doping)
  num_f = N - holes

  sweeps = Sweeps(60)
  setmaxdim!(sweeps, max_linkdim)
  setcutoff!(sweeps, 1e-9)
  setnoise!(sweeps, 1e-8, 1e-10, 0)
  en_obs = DMRGObserver(energy_tol = 1e-7)

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor

  # Create start state
  #Random.seed!(1234)
  # Create start state
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

  os = hubbard(; Nx=Nx, Ny=Ny, t=t, U=U, ky=false)
  H = MPO(os, sites; splitblocks=splitblocks)

  @show maxlinkdim(H)

  @show nnz(H[end ÷ 2])
  @show nnzblocks(H[end ÷ 2])

  if one_site == false
    energy, psi = @time dmrg(
      H, psi0, sweeps, observer = en_obs, verbose = false
    )
  end

  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  #return energy, H, psi
end


# shouldnt we construct the MPO carefully for the ky formulation to work?
#  i.e. for width 4 real sp. mpo dim 18 for mom. sp. it is 36
# ky and real space converge to slightly different energies at low BD
# I get weird ups and downs of the energy of dmrg for real space 32x6
# I get not the same energy if I use a not bipartite lattice, reason? ky formulation gets stuck
# where is the autofermion?
# can we do correlators with MPOs efficiently instead of applying single gate?

