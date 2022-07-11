#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, ITensors, ITensors.HDF5, MKL, DelimitedFiles, ITensorGLMakie, Random, JLD

function currents_from_correlation_V2(t, tp, lattice, C; α=0)
    curr_plot = Any[]
    couples = Any[]
    for (ind,b) in enumerate(lattice)
        #=
        if ((b.y2-b.y1)>1)
            pf=0.0
            push!(couples, [(b.x1, b.y1), (b.x2, pf)])
        elseif ((b.y2-b.y1)<-1)
            pf=b.y1+1
            push!(couples, [(b.x1, b.y1), (b.x2, pf)])
        else            
            pf=b.y2
            push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        end
        =#
        if (b.type)=="1" #grid bonds
            
            if (b.x1 != b.x2) #horizontal bonds
                push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
                push!(curr_plot, 1im*t*(C[b.s1, b.s2]-C[b.s2, b.s1]))
                #curr_plot[ind] += 1im*t*(C[b.s1, b.s2]-C[b.s2, b.s1])
            end
            
            if (b.y1 != b.y2) #vertical bonds
                push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
                push!(curr_plot, 1im*t*exp(-1im*2pi*α*b.x1)*C[b.s1,b.s2]-1im*t*exp(1im*2pi*α*b.x1)*C[b.s2,b.s1] )
                #curr_plot[ind] += 1im*t*exp(-1im*2pi*α*b.x1)*C[b.s1,b.s2]
                #curr_plot[ind] += -1im*t*exp(1im*2pi*α*b.x1)*C[b.s2,b.s1]
            end
        end
        if (b.type)=="2" #diagonal
            #=
            pf = b.y2-b.y1
            if abs(pf)>1.1; pf = -sign(pf) end
            curr_plot[ind] += 1im*tp*exp(-1im*2pi*α*(b.x1+0.5)*pf)*C[b.s1,b.s2]
            curr_plot[ind] += -1im*tp*exp(1im*2pi*α*(b.x1+0.5)*pf)*C[b.s2,b.s1]
            =#
        end
    end
    return couples, real(curr_plot)
end

function currents_from_correlation_V3(t, tp, lattice::Vector{LatticeBond}, C, α)
    curr_plot = zeros(ComplexF64,length(lattice))
    couples = Any[]
    for (ind,b) in enumerate(lattice)
        push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        curr_plot[ind] += 1im*t*(b.x2-b.x1)*C[b.s2, b.s1]-1im*t*(b.x2-b.x1)*C[b.s1, b.s2]
        curr_plot[ind] += -1im*t*(b.y2-b.y1)*exp(2pi*1im*α*b.x1*(b.y2-b.y1))*C[b.s1,b.s2]
        curr_plot[ind] += 1im*t*(b.y2-b.y1)*exp(-2pi*1im*α*b.x1*(b.y2-b.y1))*C[b.s2,b.s1]
        println(curr_plot[ind])
    end
    return couples, real(curr_plot)
end

let
    t = 1; tp = 0.0; J = 0.4; U=(4*t^2)/J; α=1/60; U=0; doping = 1/16; max_linkdim = 500
    Nx = 5; Ny = 4

    h5open("data/corr.h5","r") do f
        SC = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        #corr_dens = read(f,"densup_H_tp($tp)_Nx($Nx)_Ny($Ny)_mlink($max_linkdim)") + read(f,"densdn_H_tp($tp)_Nx($Nx)_Ny($Ny)_mlink($max_linkdim)")
        corr_dens_up = read(f,"dens_up_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        corr_dens_dn = read(f,"dens_dn_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")

        dens = real(diag(corr_dens_up)+diag(corr_dens_dn))
        println(sum(dens))
        mean_dens = zeros(Nx)
        for i in 1:Nx
            mean_dens[i] = 1-sum(dens[(1+(i-1)*Ny):(i*Ny)])/Ny
        end
        #=
        plt.figure(1)
        plt.plot(1:Nx,mean_dens)
        plt.scatter(1:Nx,mean_dens, label="$α")
        plt.title("Nx=$Nx, Ny=$Ny, α=$α, maxlinkdim=$max_linkdim")
        plt.grid(true); plt.xlabel("x"); plt.ylabel("Mean hole density")
        plt.legend()

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
        lat = square_lattice(Nx, Ny, yperiodic = true) #generate all bonds   
        for b in lat
            push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        end

        #= plot of current and density =#
        num = length(lat); neig = length(lat)-num+1
        eig_v = real(vecs[:,num])

        av_pair = zeros(ComplexF64,Nx);
        for (i, b) in enumerate(lat)
            av_pair[Int(b.x1)] += (-1)^(b.x2-b.x1)*vecs[i,num]
        end

        plt.figure(3)
        plt.plot(1:Nx,abs.(av_pair))
        plt.scatter(1:Nx,abs.(av_pair))
        plt.title("Nx=$Nx, Ny=$Ny, α=$α, maxlinkdim=$max_linkdim")
        plt.grid(true); plt.xlabel("x"); plt.ylabel("Mean pairing")

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
        #plt.savefig("fulldens_H_($Nx)_($neig).pdf")
        =#
        lat=square_lattice(Nx,Ny,yperiodic=false)
        couples, curr_plot_up = currents_from_correlation_V3(t, tp, lat, corr_dens_up, α) #current
        couples, curr_plot_dn = currents_from_correlation_V3(t, tp, lat, corr_dens_up, α) #current
        curr_plot = curr_plot_up + curr_plot_up
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
        plt.title("Parameters: α=$α, Nx=$Nx, Ny=$Ny U=$U")
        plt.tight_layout()
        display(fig)
        plt.close()    
        #plt.savefig("fulldens_H_($Nx)_($neig).pdf")
    end
end

av_pair = zeros(Nx); num = length(lat)-4
for (i, b) in enumerate(lat)
    av_pair[Int(b.x1)] += (-1)^(b.x2-b.x1)*vecs[i,num]
end
plt.plot(av_pair)

for b in lat 
    println(b.x2-b.x1)
end