#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, ITensors, ITensors.HDF5, Random

function currents_from_correlation_V3(t, tp, lattice::Vector{LatticeBond}, C, α, Nx, Ny)
    curr_plot = zeros(ComplexF64,length(lattice))
    couples = Any[]
    for (ind,b) in enumerate(lattice)
        push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        curr_plot[ind] += 1im*t*(b.x2-b.x1)*C[b.s2, b.s1]-1im*t*(b.x2-b.x1)*C[b.s1, b.s2]
        curr_plot[ind] += -1im*t*(b.y2-b.y1)*exp(2pi*1im*α*(b.x1-(Nx)/2)*(b.y2-b.y1))*C[b.s1,b.s2]
        curr_plot[ind] += 1im*t*(b.y2-b.y1)*exp(-2pi*1im*α*(b.x1-(Nx)/2)*(b.y2-b.y1))*C[b.s2,b.s1]
    end
    return couples, real(curr_plot)
end

function main_processing(; Nx = 6, Ny = 4, t = 1, tp = 0.2, J = 0.4, U = 10, α=1/60, doping = 1/16, max_linkdim = 450,
                            yperiodic = true, kwargs...)

    h5open("corr.h5","r") do f
        SC_1 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_1")        
        SC_2 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_2")        
        SC_3 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_3")        
        SC_4 = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)_4")        

        SC = SC_1 .+ SC_2 .+ SC_3 .+ SC_4

        corr_dens = read(f,"dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_d($doping)_alpha_($α)_mlink($max_linkdim)")
        dens = real(diag(corr_dens))
        println(sum(dens))

        plot_alpha = α * ((Nx-1)*Ny)
        
        mean_dens = zeros(Nx)
        for i in 1:Nx
            mean_dens[i] = 1-sum(dens[(1+(i-1)*Ny):(i*Ny)])/Ny
        end
        dens = 1 .- diag(corr_dens)
        
        Cd = SC+SC'
        vals, vecs = eigen(Cd)

        couples = Any[]
        lattice = square_lattice(Nx, Ny, yperiodic = true)
        for b in lattice
            push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        end

        #= plot of current and density =#
        num = length(lattice)-1; #choose which eigenvector
        neig = length(lattice)-num+1
        eig_v = (vecs[:,num])

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

        fig, host = plt.subplots(figsize=(8,5))
        par1 = host.twinx()
        host.plot(1:Nx,mean_dens, "-o", color="black", label="Hole dens.")
        par1.plot(1:Nx,real.(av_pair[:,1]), "-s", label="1")
        par1.plot(1:Nx,real.(av_pair[:,2]), "-^", label="2")
        par1.plot(1:Nx,real.(av_pair[:,3]), "-*", label="3")
        #par1.plot(1:Nx,real.(av_pair[:,4]), "-x", label="4")
        plt.title("Nx=$Nx, Ny=$Ny, α=$α, maxlinkdim=$max_linkdim")
        #plt.grid(true); 
        host.set_xlabel("x"); host.set_ylabel("Averaged Hole Density")
        par1.set_ylabel("Averaged Pairing")
        plt.legend(title="Mac. eigenv.")
        #plt.savefig("Plots/average_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim).png")
        #plt.close()    

        plt.figure(5, dpi=200, figsize=(5,5))
        plt.scatter(1:length(vals), reverse(vals), label="$plot_alpha")
        plt.title("Nx=$Nx, Ny=$Ny, α=$α, maxlinkdim=$max_linkdim")
        plt.ylabel("Mac. eigenvalue")
        plt.xlabel("eig. number")
        plt.tight_layout()
        #plt.grid(true)
        #plt.legend(title="α")
        #plt.savefig("Plots/spectrum_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim).png")
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

        pl_quiver = ax.quiver(X,Y,U,V,C, scale_units="inches", scale=0.9, headlength=5)
        pl_dens = ax.scatter(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx), c=dens, s=25, marker="s", zorder = 1, cmap=plt.get_cmap("Greys"), edgecolors="black")
        plt.gca().set_aspect("equal")
        plt.title("tp=$tp, Nx=$Nx, Ny=$Ny, α=$plot_alpha, mlink=$max_linkdim, #eig.=$neig")
        plt.colorbar(pl_dens, ax=ax, location="bottom", label=" hole density", shrink=0.7, pad=0.03, aspect=50)
        plt.colorbar(pl_quiver, ax=ax, location="bottom", label="pairing", shrink=0.7, pad=0.07, aspect=50)
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

        plt.figure(91)
        plt.plot(real.(SC[55,corre]))
        plt.plot(imag.(SC[55,corre]))
        println(SC[55,corre])
        plt.xlim(6,32)
        plt.xscale("log")
        plt.yscale("log")
        

        #plt.savefig("Plots/fulldens_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim).png")
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
        #plt.savefig("Plots/current_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim).png")
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


function slider_plot(Nx)
    Nx = 32; Ny = 4; t = 1; tp = 0.2; J = 0.4; U = 4(t^2)/J
    doping = 1/16; max_linkdim = 2500; yperiodic = false; reupload = true; psi0 = nothing
    
    alphas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] ./(Ny*(Nx-1))

    lattice = square_lattice(Nx, Ny, yperiodic = true)
    couples = Any[]
    
    for b in lattice
        push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
    end
    X = zeros(length(couples)); Y = zeros(length(couples))
    U = zeros(length(couples)); V = zeros(length(couples))
    C = zeros(length(couples))

    for i in 1:length(couples)
        X[i] =(couples[i][1][1]+couples[i][2][1])/2
        Y[i] =(couples[i][1][2]+couples[i][2][2])/2
        if (couples[i][2][2]==Ny+1) Y[i] = Ny+0.5 end
    end

    SCS = zeros(ComplexF64, length(lattice), length(alphas))
    denses = zeros(Nx*Ny, length(alphas))

    f=h5open("corr.h5","r")
    for (i,α) in enumerate(alphas)
        SC = read(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")        
        corr_dens = read(f,"dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        dens = 1 .- real(diag(corr_dens))

        Cd = SC+SC'
        vals, vecs = eigen(Cd)

        num = length(lattice); #choose which eigenvector
        neig = length(lattice)-num+1
        eig_v = (vecs[:,num])

        SCS[:,i] .= eig_v
        denses[:,i] .= dens

        fig, ax = plt.subplots(1, dpi = 200, figsize = (10,6))
        ax.set_ylim(0.5,Ny+0.5)
        plt.gca().set_aspect("equal")
        U .= real.(SCS[:,i]) ./abs.(eig_v)
        V .= imag.(SCS[:,i]) ./abs.(eig_v) 
        C = abs.(SCS[:,i])

        pl_quiver = ax.quiver(X,Y,U,V,C, scale_units="inches", scale=0.9, headlength=5)
        pl_dens = ax.scatter(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx), c=denses[:,i], s=25, marker="s", zorder = 1, cmap=plt.get_cmap("Greys"), edgecolors="black")
        
        plot_alpha = α * (Ny*(Nx-1))
        plt.title("tp=$tp, Nx=$Nx, Ny=$Ny, α=$plot_alpha, mlink=$max_linkdim, #eig.=1")
        plt.colorbar(pl_dens, ax=ax, location="bottom", label=" hole density", shrink=0.7, pad=0.05, aspect=50)
        plt.colorbar(pl_quiver, ax=ax, location="bottom", label="pairing", shrink=0.7, pad=0.09, aspect=50)
    
        display(fig)
        plt.close()    
    end

    #=
    axslid = plt.axes([0.1, 0.25, 0.0225, 0.63])
    slider = plt.matplotlib.widgets.Slider(
    ax=axslid,
    label="α",
    valmin=0.0,
    valmax=10.0,
    valinit=0.0,
    valstep=alphas,
    orientation="vertical")

    plt.tight_layout()

    function update(val)
        U .= real.(SCS[:,slider.val]) #./abs.(eig_v)
        V .= imag.(SCS[:,slider.val]) #./abs.(eig_v) 
        C = abs.(SCS[:,slider.val])
        pl_quiver.set_UVC(U,V,C)
        pl_dens.set_color(dens[:,slider.val])
        fig.canvas.draw_idle()
    end

    slider.on_changed(update)
    plt.show()
    #line_segments = plt.matplotlib.collections.LineCollection(couples,array=eig_v, 
    #                                                        norm=plt.matplotlib.colors.Normalize(vmin=minimum(eig_v), vmax=maximum(eig_v)),
    #                                                        linewidths=5, cmap=plt.get_cmap("RdBu_r"), rasterized=true, zorder = 0)
    #pl_curr = ax.add_collection(line_segments)
    =#
end

#slider_plot(32)