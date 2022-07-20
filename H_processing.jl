#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("Qt5Agg"); ENV["MPLBACKEND"] = "Qt5Agg"; plt.pygui(true); plt.ion()

using LinearAlgebra, ITensors, ITensors.HDF5, Random

function currents_from_correlation_V3(t, tp, lattice::Vector{LatticeBond}, C, α, Nx, Ny)
    curr_plot = zeros(ComplexF64,length(lattice))
    couples = Any[]
    for (ind,b) in enumerate(lattice)
        push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        curr_plot[ind] += 1im*t*(b.x2-b.x1)*C[b.s2, b.s1]-1im*t*(b.x2-b.x1)*C[b.s1, b.s2]
        curr_plot[ind] += -1im*t*(b.y2-b.y1)*exp(2pi*1im*α*(b.x1-(Nx+1)/2)*(b.y2-b.y1))*C[b.s1,b.s2]
        curr_plot[ind] += 1im*t*(b.y2-b.y1)*exp(-2pi*1im*α*(b.x1-(Nx+1)/2)*(b.y2-b.y1))*C[b.s2,b.s1]
    end
    return couples, real(curr_plot)
end

function main_processing(; Nx = 6, Ny = 4, t = 1, tp = 0.2, J = 0.4, U = 10, α=1/60, doping = 1/16, max_linkdim = 450,
                            yperiodic = true, kwargs...)

    h5open("data/corr.h5","r") do f
        SC = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        corr_dens = read(f,"dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        dens = real(diag(corr_dens))

        mean_dens = zeros(Nx)
        for i in 1:Nx
            mean_dens[i] = 1-sum(dens[(1+(i-1)*Ny):(i*Ny)])/Ny
        end
        
        plt.figure(1)
        plt.plot(1:Nx,mean_dens)
        plt.scatter(1:Nx,mean_dens, label="$α")
        plt.title("Nx=$Nx, Ny=$Ny, α=$α, maxlinkdim=$max_linkdim")
        plt.grid(true); plt.xlabel("x"); plt.ylabel("Mean hole density")
        plt.legend(title="α")

        #plt.savefig("dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_mlink($maxlinkdim).pdf")
        
        Cd = SC+SC'
        vals, vecs = eigen(Cd)

        plt.figure(2)
        plt.scatter(1:length(vals), reverse(vals))
        plt.title("Nx=$Nx, Ny=$Ny, α=$α, maxlinkdim=$max_linkdim")
        plt.ylabel("Mac. eigenvalue")
        plt.xlabel("eig. number")
        plt.grid(true)
        #plt.savefig("spectrum_H_$Nx.pdf")
        
        dens = 1 .- diag(corr_dens)

        couples = Any[]

        lattice = square_lattice(Nx, Ny, yperiodic = true)
        for b in lattice
            push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        end

        #= plot of current and density =#
        num = length(lattice); #choose which eigenvector
        neig = length(lattice)-num+1
        eig_v = real.(vecs[:,num])
        println(imag(vecs[1:end,num]))


        av_pair = zeros(ComplexF64,Nx);
        for (i, b) in enumerate(lattice)
            av_pair[Int(b.x1)] += (-1)^(b.x2-b.x1)*vecs[i,num]
        end

        plt.figure(3)
        plt.plot(1:Nx,real.(av_pair))
        plt.scatter(1:Nx,real.(av_pair), label="$α, real")
        plt.plot(1:Nx,imag.(av_pair))
        plt.scatter(1:Nx,imag.(av_pair), label="$α, imag")
        plt.title("Nx=$Nx, Ny=$Ny maxlinkdim=$max_linkdim")
        plt.grid(true); plt.xlabel("x"); plt.ylabel("Mean pairing")
        plt.legend(title="α")

        fig, ax = plt.subplots(1, dpi = 150)
        ax.set_ylim(0.5,Ny+0.5)
        line_segments = plt.matplotlib.collections.LineCollection(couples,array=eig_v, 
                                                                norm=plt.matplotlib.colors.Normalize(vmin=minimum(eig_v), vmax=maximum(eig_v)),
                                                                linewidths=5, cmap=plt.get_cmap("RdBu_r"), rasterized=true, zorder = 0)
        pl_curr = ax.add_collection(line_segments)
        pl_dens = ax.scatter(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx), c=dens, s=100, marker="s", zorder = 1, cmap=plt.get_cmap("PuBu"), edgecolors="black")
        plt.gca().set_aspect("equal")
        plt.title("tp=$tp, Nx=$Nx, Ny=$Ny, α=$α, mlink=$max_linkdim, #eig.=$neig")
        plt.colorbar(pl_dens, ax=ax, location="bottom", label=" hole density", shrink=0.7, pad=0.03, aspect=50)
        plt.colorbar(pl_curr, ax=ax, location="bottom", label="pairing", shrink=0.7, pad=0.07, aspect=50)
        plt.tight_layout()
        #display(fig)
        #plt.close()    

        #plt.savefig("fulldens_H_($Nx)_($neig).pdf")
        
        lattice = square_lattice(Nx, Ny, yperiodic = yperiodic)
        couples, curr_plot = currents_from_correlation_V3(t, tp, lattice, corr_dens, α, Nx, Ny) #current
        println(sum(curr_plot))
        
        fig, ax = plt.subplots(1, dpi = 150)
        ax.set_ylim(0.5,Ny+0.5)

        line_segments = plt.matplotlib.collections.LineCollection(couples,array=curr_plot, 
                                                                norm=plt.matplotlib.colors.Normalize(vmin=minimum(curr_plot), vmax=maximum(curr_plot)),
                                                                linewidths=5, cmap=plt.get_cmap("RdBu_r"), rasterized=true, zorder = 0)
        pl_curr = ax.add_collection(line_segments)
        pl_dens = ax.scatter(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx), c=dens, s=100, marker="s", zorder = 1, cmap=plt.get_cmap("PuBu"), edgecolors="black")
        plt.gca().set_aspect("equal")
        plt.colorbar(pl_dens, ax=ax, location="bottom", label="density", shrink=0.7, pad=0.03, aspect=50)
        plt.colorbar(pl_curr, ax=ax, location="bottom", label="current", shrink=0.7, pad=0.07, aspect=50)
        plt.title("Parameters: α=$α, Nx=$Nx, Ny=$Ny, U=$U, BD=$max_linkdim")
        plt.tight_layout()
        #display(fig)
        #plt.close()    
        #plt.savefig("fulldens_H_($Nx)_($neig).pdf")
    end
end

#main_processing()