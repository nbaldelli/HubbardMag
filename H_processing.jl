#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, ITensors, ITensors.HDF5, Random
using DelimitedFiles

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
    
function supercurrent(chi, Nx, Ny, alpha)
    ph = angle.(chi)
    mod = abs.(chi) .^2
    println(ph[end-1])
    println(ph[end])
    curr_x = []; curr_y = []; abs_value = []
    for j in (2:2:(length(ph)-Ny*3))
        if ((j-2)%(2*Ny) == 0) 
            push!(curr_x, (ph[j + 2*Ny - 1] - ph[j-1]) * (mod[j + 2*Ny - 1] + mod[j-1])/2)
        else
            push!(curr_x, (ph[j + 2*Ny] - ph[j]) * (mod[j + 2*Ny] + mod[j])/2)
        end
    end
    curr_x_2 = [fill(0, Ny)..., curr_x..., fill(0, Ny)...]

    for j in 3:2:(Nx*(Ny*2-1)+Ny*2-1)
        if ((j+1)%(Ny*2) == 0)
            push!(curr_y, 0) 
            push!(abs_value, 0)
        elseif ((j-1)%(Ny*2) == 0)
            push!(curr_y, 0)    
            push!(abs_value, 0)
        elseif ((j-3)%(2*Ny)==0) 
            push!(curr_y, (ph[j + 2] - ph[j-1]) * (mod[j + 2] + mod[j-1])/2)
            push!(abs_value, (mod[j + 2] + mod[j-1])/2)
        else 
            push!(curr_y, (ph[j + 2] - ph[j]) * (mod[j + 2] + mod[j])/2 )
            push!(abs_value, (mod[j + 2] + mod[j])/2)
        end 
    end
    push!(curr_y, 0)
    push!(abs_value, 0)
    for j in (Nx*(Ny*2-1)+Ny*2+2):(length(ph)-1)
        if (j == Nx*(Ny*2-1)+Ny*2+2)
            push!(curr_y, (ph[j + 2] - ph[j]) * (mod[j + 2] + mod[j])/2)
            push!(abs_value, (mod[j + 2] + mod[j])/2)
        else
            push!(curr_y, (ph[j + 1] - ph[j]) * (mod[j + 1] + mod[j])/2)
            push!(abs_value, (mod[j + 1] + mod[j])/2)
        end
    end

    curr_y_2 = [0, curr_y..., 0] .- 2*2pi*α* [0,abs_value...,0] .*  repeat(collect(1:Nx) .- Nx/2, inner=6)
    return curr_x_2, curr_y_2
end

function main_processing(; Nx = 6, Ny = 4, t = 1, tp = 0.2, J = 0.4, U = 10, α=1/60, doping = 1/16, max_linkdim = 450,
                            yperiodic = true, kwargs...)

    h5open("corr.h5","r") do f
        #α = α /((Nx-1)*Ny)
        SC_1 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_1")        
        SC_2 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_2")        
        SC_3 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_3")        
        SC_4 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_4")        
        #SC = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        SC = SC_1 .+ SC_2 .+ SC_3 .+ SC_4

        corr_dens = read(f,"dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)")
        dens = real(diag(corr_dens))
        println(sum(dens))

        plot_alpha = α * ((Nx-1)*Ny)
        
        mean_dens = zeros(Nx)
        for i in 1:Nx
            mean_dens[i] = sum(dens[(1+(i-1)*Ny):(i*Ny)])/Ny
        end
        dens = 1 .- diag(corr_dens)
        open("dens.txt", "a") do io
            writedlm(io,real.(dens))
         end

        Cd = SC+SC'
        vals, vecs = eigen(Cd)

        couples = Any[]
        lattice = square_lattice(Nx, Ny, yperiodic = true)
        for b in lattice
            push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        end

        #= plot of current and density =#
        num = length(lattice); #choose which eigenvector
        neig = length(lattice)-num+1
        eig_v = (vecs[:,num])

        #==
        println("flag")
        a, b = supercurrent(eig_v, Nx, Ny, α)
        fig, ax = plt.subplots(1, dpi = 200)
        #C = abs.(eig_v) .^2
        pl_quiver = ax.quiver(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx),a,b, scale_units="inches", scale=0.4, headlength=5)
        plt.gca().set_aspect("equal")
        println(a)
        println(b)
        ==#
        vortex_pattern = [22,31,32,40,48,55,54,61,52,50,41,33,25,26,20,21]
        vortex =  [(angle(eig_v[i])+2pi)%2pi for i in vortex_pattern]
        winding = 0
        for i in 1:(length(vortex)-1)
            winding += vortex[i+1] - vortex[i]
            println(winding)
        end
        winding += vortex[1] - vortex[length(vortex)]
        println(winding)
        println("winding = $winding")

        av_pair = zeros(ComplexF64, Nx, Ny);
        for (i, b) in enumerate(lattice), j in 1:Ny
            av_pair[Int(b.x1),j] += (1)^(b.x2-b.x1)*abs.(vecs[i,num-j+1])
        end

        fig, host = plt.subplots(figsize=(8,6), dpi = 250)
        par1 = host.twinx()

        host.plot(1:Nx,mean_dens, "-o", color="black", label="Hole dens.")
        par1.plot(1:Nx,real.(av_pair[:,1] ./sqrt(sum(av_pair[:,1].^2))), "-s", label="1")
        par1.plot(1:Nx,real.(av_pair[:,2] ./sqrt(sum(av_pair[:,2].^2))), "-^", label="2")
        #par1.plot(1:Nx,real.(av_pair[:,3] ./sqrt(sum(av_pair[:,3].^2))), "-*", label="3")
        #par1.plot(1:Nx,real.(av_pair[:,5]), "-x", label="5")
        #plt.title("Nx=$Nx, Ny=$Ny, α=$α, maxlinkdim=$max_linkdim")
        #plt.grid(true); 
        host.set_xlabel("x"); host.set_ylabel("Averaged Hole Density")
        par1.set_ylabel("Averaged Pairing")
        plt.legend(title="Mac. eigenv.")
        plt.savefig("Plots/average_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim).png")
        #plt.close()    

        fig, ax = plt.subplots(1, dpi = 250, figsize = (4,4))
        ax.plot(1:length(vals), reverse(vals), ".", markersize = 5, label="$plot_alpha")
        #ax.set_title("Nx=$Nx, Ny=$Ny, α=$α, maxlinkdim=$max_linkdim")
        ax.set_ylabel("Mac. eigenvalue")
        ax.set_xlabel("eig. number")
        axins = ax.inset_axes([0.3, 0.65, 0.2, 0.2])
        axins.plot(1:length(vals), reverse(vals), ".", markersize = 10)
        println("vals=", sum(vals))

        x1, x2, y1, y2 = -1, 8, 0.18, 0.245
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        ax.indicate_inset_zoom(axins, edgecolor="black")

        plt.tight_layout()
        #plt.grid(true)
        #plt.legend(title="α")
        plt.savefig("Plots/spectrum_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim).png")
        #plt.close()    

        fig, ax = plt.subplots(1, dpi = 200)
        ax.set_ylim(0.5,Ny+0.5)
        #line_segments = plt.matplotlib.collections.LineCollection(couples,array=eig_v, 
        #                                                        norm=plt.matplotlib.colors.Normalize(vmin=minimum(eig_v), vmax=maximum(eig_v)),
        #                                                        linewidths=5, cmap=plt.get_cmap("RdBu_r"), rasterized=true, zorder = 0)
        #pl_curr = ax.add_collection(line_segments)
        X = zeros(length(couples)); Y = zeros(length(couples))
        U = zeros(length(couples)); V = zeros(length(couples))
        C = zeros(length(couples))

        for i in 1:length(couples)
            X[i] =(couples[i][1][1]+couples[i][2][1])/2
            Y[i] =(couples[i][1][2]+couples[i][2][2])/2
            if (couples[i][2][2]==Ny+1) Y[i] = Ny+0.5 end
        end

        U .= real.(eig_v) #./abs.(eig_v)
        V .= imag.(eig_v) #./abs.(eig_v) 
        C = abs.(eig_v)
        println(maximum(C))

        pl_quiver = ax.quiver(X,Y,U,V,C, scale_units="inches", scale=0.9, headlength=5)
        pl_dens = ax.scatter(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx), c=dens, s=25, marker="s", zorder = 1, cmap=plt.get_cmap("Greys"), edgecolors="black")
        plt.gca().set_aspect("equal")
        #plt.title("tp=$tp, Nx=$Nx, Ny=$Ny, α=$plot_alpha, mlink=$max_linkdim, #eig.=$neig")
        plt.colorbar(pl_dens, ax=ax, location="bottom", label=" hole density", shrink=0.7, pad=0.05, aspect=50)
        plt.colorbar(pl_quiver, ax=ax, location="bottom", label="pairing", shrink=0.7, pad=0.08, aspect=50)
        plt.tight_layout()
        #display(fig)
        #plt.close()    

        corre = []
        for i in 1:length(couples)
            #plt.text(X[i],Y[i],string(i), fontsize=5)
            if (Y[i]) == 3
                push!(corre, i+1)
            end
        end

        plt.savefig("Plots/fulldens_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_$num.png")
        #plt.close()    

        lattice = square_lattice(Nx, Ny, yperiodic = false)
        couples, curr_plot = currents_from_correlation_V3(t, tp, lattice, corr_dens, α, Nx, Ny) #current
        
        fig, ax = plt.subplots(1, dpi = 150)
        ax.set_ylim(0.5,Ny+0.5)

        line_segments = plt.matplotlib.collections.LineCollection(couples,array=curr_plot, 
                                                                norm=plt.matplotlib.colors.Normalize(vmin=minimum(curr_plot), vmax=maximum(curr_plot)),
                                                                linewidths=5, cmap=plt.get_cmap("PuOr_r"), rasterized=true, zorder = 0)
                                                            
        pl_curr = ax.add_collection(line_segments)
        pl_dens = ax.scatter(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx), c=dens, s=50, marker="s", zorder = 1, cmap=plt.get_cmap("Greys"), edgecolors="black")
        plt.gca().set_aspect("equal")
        plt.colorbar(pl_dens, ax=ax, location="bottom", label="density", shrink=0.7, pad=0.03, aspect=50)
        plt.colorbar(pl_curr, ax=ax, location="bottom", label="current", shrink=0.7, pad=0.07, aspect=50)
        #plt.title("Parameters: α=$plot_alpha, Nx=$Nx, Ny=$Ny, U=$U, BD=$max_linkdim")
        plt.tight_layout()
        #display(fig)
        plt.savefig("Plots/current_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim).png")
        #plt.close()   
    end

    #=
    h5open("EE.h5","r") do f
        EE = read(f,"EE_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        plot_alpha = α * ((Nx-1)*Ny)
        #=
        plt.figure(8)
        plt.plot(4:(Nx*Ny-4),EE[4:end-4])
        plt.scatter(4:(Nx*Ny-4),EE[4:end-4], label="$plot_alpha")
        plt.title("Nx=$Nx, Ny=$Ny, maxlinkdim=$max_linkdim")
        plt.grid(true); plt.xlabel("Cut position"); plt.ylabel("Entanglement entropy")
        plt.legend(title="α")
        =#
    end
    =#
end

function hdf5_to_txt(; Nx = 24, Ny = 6, t = 1, tp = 0.2, J = 0.4, U = 10, α=6.0, doping = 1/36,  max_linkdim = 4001, nn = 1)
    h5open("corr.h5","r") do f
        α = α/((Nx-1)*Ny)
        SC_1 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_1")        
        SC_2 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_2")        
        SC_3 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_3")        
        SC_4 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_4")        

        #SC = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        SC = SC_1 .+ SC_2 .+ SC_3 .+ SC_4

        corr_dens = read(f,"dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)")
        dens = real(diag(corr_dens))

        plot_alpha = α * ((Nx-1)*Ny)
        nholes = Int(floor(doping*Nx*Ny))
        BC = "O"
        
        open("dens_tp($tp)_Nx($Nx)_Ny($Ny)_nholes($nholes)_alpha_($plot_alpha)_mlink($max_linkdim)_$BC.txt", "w") do io
            writedlm(io,real.(dens))
            end

        Cd = SC+SC'
        vals, vecs = eigen(Cd)

        couples = Any[]
        lattice = square_lattice(Nx, Ny, yperiodic = true)
        for b in lattice
            push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        end

        #= plot of current and density =#
        num = length(lattice)-nn+1; #choose which eigenvector
        neig = length(lattice)-num+1
        eig_v = (vecs[:,num])

        #pl_curr = ax.add_collection(line_segments)
        X = zeros(length(couples)); Y = zeros(length(couples))
        U = zeros(length(couples)); V = zeros(length(couples))
        C = zeros(length(couples))

        for i in 1:length(couples)
            X[i] =(couples[i][1][1]+couples[i][2][1])/2
            Y[i] =(couples[i][1][2]+couples[i][2][2])/2
            if (couples[i][2][2]==Ny+1) Y[i] = Ny+0.5 end
        end

        U .= real.(eig_v) #./abs.(eig_v)
        V .= imag.(eig_v) #./abs.(eig_v) 
        C = abs.(eig_v)
        SC_data = [X Y U V C]
        open("SC_tp($tp)_Nx($Nx)_Ny($Ny)_nholes($nholes)_alpha_($plot_alpha)_mlink($max_linkdim)_neig($neig)_$BC.txt.txt", "w") do io
            writedlm(io,SC_data)
        end
    end
end

#main_processing(; Nx = 32, Ny = 4, t = 1, tp = 0.2, J = 0.4, U = 10, α=5.0, doping = 1/16,  max_linkdim = 3000,)
hdf5_to_txt()