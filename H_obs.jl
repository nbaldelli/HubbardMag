using LinearAlgebra, ITensors, ITensors.HDF5, MKL, Random 

function correlation_matrix_bond(psi0, Nx, Ny, o1, o2, o3, o4)

    N=Nx*Ny
    lat = square_lattice(Nx, Ny, yperiodic = true)
    indx = Dict([(b.s1,b.s2)=>i for (i,b) in enumerate(lat)])
    C = zeros(ComplexF64, length(lat), length(lat))

    #easy term: second bond after the first (bulk of computation), both bonds either H or V
    println("Easy term")
    
    #set F operators on sites where creation/destruction operators act (required for spinful fermions)
    if (o1 == "Adagup" && o3 == "Aup") j_1 = "F"; j_2 = "F"; j_3 = "F"; j_4 = "F"; pf = 1
    elseif (o1 == "Adagdn" && o3 == "Aup") j_1 = "Id"; j_2 = "Id"; j_3 = "F"; j_4 = "F"; pf = -1
    elseif (o1 == "Adagup" && o3 == "Adn") j_1 = "F"; j_2 = "F"; j_3 = "Id"; j_4 = "Id"; pf = -1
    else j_1 = "Id"; j_2 = "Id"; j_3 = "Id"; j_4 = "Id"; pf = 1 end

    psi = copy(psi0)    
    orthogonalize!(psi,1)
    psi_dag = prime(linkinds, dag(psi))

    for i in 1:(N-2) #choose first site (o1)
        orthogonalize!(psi, i) #after orthogonalize weird things happen to the indices, have to do it before taking indices
        s = siteinds(psi) #this can be done before orth.
        l = linkinds(psi) #this CAN'T
        psi_dag = prime(linkinds, dag(psi)) #opposite MPS to contract
        op_psi_i = apply(op(o1, s[i]), apply(op(j_1,s[i]),psi[i])) #why apply instead of multiplication? bc it changes the site to be primed
        L = (i>1 ? delta(dag(l[i-1])',l[i-1]) : 1.) * op_psi_i * psi_dag[i]  #left system to conserve between contractions
        
        #choice of bonds
        neigh = [1, Ny] #vertical or horizontal bond
        i%Ny==1 ? neigh = [1, Ny-1, Ny] : neigh=neigh #extra bond for first row (cylinder)
        i%Ny==0 ? neigh = [Ny] : neigh=neigh #one bond less for last row (cylinder), this way of writing is terrible
        #(probably better to find a way to add the bond going around the cylinder on the last row and remove a line)
        
        for M in neigh #choose second site (o2)
            if (M==Ny && i>(N-Ny)) continue end #last column: no H bond

            #println("(o1,o2):($i,$(i+M))")
            L_t = copy(L) #copy unnecessary
            for str in 1:(M-1) #insert JW string between o1 and o2
                L_t = L_t* apply(op("F", s[i + str]), psi[i + str]) * psi_dag[i + str] 
            end
            op_psi_i1 =  apply(op(o2, s[i+M]), apply(op(j_2,s[i+M]),psi[i+M])) #apply o2
            L_t = L_t * op_psi_i1 * psi_dag[i+M]  #generate left tensor to store
            
            for j in (i+M+1):N #choose third site (o3)
                op_psi_j = apply(op(j_3, s[j]), apply(op(o3,s[j]),psi[j])) #apply o3

                neigh_j = [1, Ny] #vertical or horizontal bond
                j%Ny==1 ? neigh_j = [1, Ny-1, Ny] : neigh_j=neigh_j #extra bond for first row (cilinder), same considerations as before
                for O in neigh_j #vertical-horizontal second bond, choose fourth site
                    if (O==1 && j%Ny==0) continue end #new column: no V bond
                    if (O==Ny && j>(N-Ny)) continue end #last column: no H bond
                    #println("(o3,o4):($j,$(j+O))")

                    op_psi_j1 =  apply(op(j_4, s[j+O]), apply(op(o4,s[j+O]),psi[j+O])) #apply o4

                    R = ((j+O)<length(psi) ? delta(dag(l[j+O]),l[j+O]') : 1.) * op_psi_j1 * psi_dag[j+O] #create right system
                    for str in (O-1):-1:1 #insert JW string between o3 and o4
                        R = R * psi_dag[j + str] * apply(op("F", s[j + str]), psi[j + str]) #ASK MATT HOW THE ORDER IN THIS MULTIPLICATION IMPACTS COMPUTATION TIME
                    end        
                    R = R * op_psi_j * psi_dag[j] 

                    C[get(indx,(i,i+M),0),get(indx,(j,j+O),0)] = pf*inner(dag(L_t),R) #get matrix element
                end
                L_t = L_t * psi[j] * psi_dag[j]  #update left tensor (EFFICIENT PART)
            end
        end
    end
    
    #annoying term 1: second bond in the middle: (o1,o3,o4,o2) (first bond always H, second always V)
    println("Annoying term 1")

    psi = copy(psi0)
    orthogonalize!(psi,1)
    psi_dag = prime(linkinds, dag(psi))

    #set F operators on sites where creation/destruction operators act (required for spinful fermions)
    if (o1 == "Adagup" && o3 == "Aup") j_1 = "F"; j_3 = "Id"; j_4 = "Id"; j_2 = "F"; pf = 1
    elseif (o1 == "Adagdn" && o3 == "Aup") j_1 = "Id"; j_3 = "Id"; j_4 = "Id"; j_2 = "Id"; pf = -1
    elseif (o1 == "Adagup" && o3 == "Adn") j_1 = "F"; j_3 = "F"; j_4 = "F"; j_2 = "F"; pf = -1 
    else j_1 = "Id"; j_3 = "F"; j_4 = "F"; j_2 = "Id"; pf = 1 end

    for i in 1:(N-Ny) #choose first site
        orthogonalize!(psi, i)
        s = siteinds(psi)
        l = linkinds(psi)
        psi_dag = prime(linkinds, dag(psi)) #opposite MPS to contract

        op_psi_i = apply(op(o1, s[i]), apply(op(j_1,s[i]),psi[i])) #apply o1
        op_psi_i1 = apply(op(o2,s[i+Ny]), apply(op(j_2,s[i+Ny]),psi[i+Ny])) #apply o2 (always H bond)

        L = (i>1 ? delta(dag(l[i-1])',l[i-1]) : 1.) * op_psi_i * psi_dag[i] #left system 
        R = (i+Ny < (length(psi)) ? delta(dag(l[i+Ny]),l[i+Ny]') : 1.) * op_psi_i1 * psi_dag[i+Ny] #right system
        #println("(o1,o2):($i,$(i+Ny))")

        for j in (i+1):(i+Ny-2) #choose third site
            if (j%Ny==0) #skip bond if it would go to another column
                L = L * apply(op("F", s[j]), psi[j]) * psi_dag[j] 
            continue end 
            #println("(o3,o4):($j,$(j+1))")
            L_t = copy(L)
            op_psi_j = apply(op(j_3, s[j]),apply(op(o3,s[j]), psi[j])) #apply o3
            op_psi_j1 = apply(op(j_4, s[j+1]),apply(op(o4,s[j+1]), psi[j+1])) #apply o4 (always V bond)
            L_t = L_t * op_psi_j * psi_dag[j] * op_psi_j1 * psi_dag[j+1]
            for k in (j+2):(i+Ny-1) #add empty sites between o4 and o2
                L_t = L_t * psi_dag[k] * apply(op("F", s[k]), psi[k])
            end
            C[get(indx,(i,i+Ny),0),get(indx,(j,j+1),0)] = pf * inner(dag(L_t),R)
            L = L * apply(op("F", s[j]), psi[j]) * psi_dag[j] #add JW term to account for space between o1 and o3
        end
    end
    
    #annoying term 2: overlapping bonds (o1,o3,o2,o4) (first bond always H, second always H) DOESNT WORK
    println("Annoying term 2")

    psi = copy(psi0)
    orthogonalize!(psi,1)
    psi_dag = prime(linkinds, dag(psi))

    #set F operators on sites where creation/destruction operators act (required for spinful fermions)
    if (o1 == "Adagup" && o3 == "Aup") j_1 = "F"; j_3 = "Id"; j_2 = "Id"; j_4 = "F"; pf = 1
    elseif (o1 == "Adagdn" && o3 == "Aup") j_1 = "Id"; j_3 = "Id"; j_2 = "F"; j_4 = "F"; pf = -1
    elseif (o1 == "Adagup" && o3 == "Adn") j_1 = "F"; j_3 = "F"; j_2 = "Id"; j_4 = "Id"; pf = -1
    else j_1 = "Id"; j_3 = "F"; j_2 = "F"; j_4 = "Id"; pf = 1 end
    
    for i in 1:(N-Ny) #choose first site
        orthogonalize!(psi, i) 
        s = siteinds(psi)
        l = linkinds(psi)
        psi_dag = prime(linkinds, dag(psi)) #opposite MPS to contract

        op_psi_i = apply(op(o1, s[i]), apply(op(j_1,s[i]),psi[i])) #apply o1
        op_psi_i1 = apply(op(o2,s[i+Ny]), apply(op(j_2,s[i+Ny]),psi[i+Ny])) #second site (always H bond)

        L = (i>1 ? delta(dag(l[i-1])',l[i-1]) : 1.) * op_psi_i * psi_dag[i] #left system
        #println("(o1,o2):($i,$(i+Ny))")

        for j in (i+1):(i+Ny-1)
            if j%Ny==1 && i%Ny!=0
                L_t = copy(L)
                op_psi_j = apply(op(j_3, s[j]),apply(op(o3,s[j]), psi[j]))
                L_t = L_t * op_psi_j * psi_dag[j] 
                #println("(o3,o4):($j,$(j+Ny-1))")
    
                op_psi_j1 = apply(op(j_4, s[j+Ny-1]),apply(op(o4,s[j+Ny-1]), psi[j+Ny-1]))
                R = (j+Ny-1 < (length(psi)) ? delta(dag(l[j+Ny-1]),l[j+Ny-1]') : 1.) * op_psi_j1 * psi_dag[j+Ny-1] #apply rightmost operator (second bond)
    
                for k in (j+1):(i+Ny-1) #add empty sites between the third op. and the second
                    L_t = L_t * psi_dag[k] * psi[k] #should add JW? NO
                end
                L_t = L_t * op_psi_i1 * psi_dag[i+Ny]
                for k in (i+Ny+1):(j+Ny-2) #add empty sites between the second op. and the fourth
                    L_t = L_t * psi_dag[k] * apply(op("F", s[k]), psi[k]) #should add JW? YES
                end            
    
                C[get(indx,(i,i+Ny),0),get(indx,(j,j+Ny-1),0)] = pf*inner(dag(L_t),R)
        
            end
            
            if j>(N-Ny) continue end 
            L_t = copy(L)
            op_psi_j = apply(op(j_3, s[j]),apply(op(o3,s[j]), psi[j]))
            L_t = L_t * op_psi_j * psi_dag[j]
            #println("(o3,o4):($j,$(j+Ny))")

            op_psi_j1 = apply(op(j_4, s[j+Ny]),apply(op(o4,s[j+Ny]), psi[j+Ny]))
            R = (j+Ny < (length(psi)) ? delta(dag(l[j+Ny]),l[j+Ny]') : 1.) * op_psi_j1 * psi_dag[j+Ny] #apply rightmost operator (second bond)

            for k in (j+1):(i+Ny-1) #add empty sites between the third op. and the second
                L_t = L_t * psi_dag[k] * psi[k] #should add JW? NO
            end
            L_t = L_t * op_psi_i1 * psi_dag[i+Ny]
            for k in (i+Ny+1):(j+Ny-1) #add empty sites between the second op. and the fourth
                L_t = L_t * psi_dag[k] * apply(op("F", s[k]), psi[k]) #should add JW? YES
            end            

            C[get(indx,(i,i+Ny),0),get(indx,(j,j+Ny),0)] = pf*inner(dag(L_t),R)

            L = L * apply(op("F", s[j]), psi[j]) * psi_dag[j] #should add JW?
        end
    end
    
    #extremely annoying term: bonds around the cylinder (o1,o3,o4,o2) (first bond always V, second V), this can be inserted in the two previous terms probably
    psi = copy(psi0)
    orthogonalize!(psi,1)
    psi_dag = prime(linkinds, dag(psi))

    println("term ext1")
   
    if (o1 == "Adagup" && o3 == "Aup") j_1 = "F"; j_3 = "Id"; j_4 = "Id"; j_2 = "F"; pf = 1
    elseif (o1 == "Adagdn" && o3 == "Aup") j_1 = "Id"; j_3 = "Id"; j_4 = "Id"; j_2 = "Id"; pf = -1
    elseif (o1 == "Adagup" && o3 == "Adn") j_1 = "F"; j_3 = "F"; j_4 = "F"; j_2 = "F"; pf = -1 
    else j_1 = "Id"; j_3 = "F"; j_4 = "F"; j_2 = "Id"; pf = 1 end
    
    for i in 1:Ny:N #choose first site
        orthogonalize!(psi, i) #after orthogonalize weird things happen to the indices, have to do it before taking indices
        s = siteinds(psi)
        l = linkinds(psi)
        psi_dag = prime(linkinds, dag(psi)) #opposite MPS to contract

        op_psi_i = apply(op(o1, s[i]), apply(op(j_1,s[i]),psi[i])) #why apply instead of multiplication?
        L = (i>1 ? delta(dag(l[i-1])',l[i-1]) : 1.) * op_psi_i * psi_dag[i] #apply leftmost operator not sure 
        #println("(o1,o2):($i,$(i+Ny-1))")

        op_psi_i1 = apply(op(o2, s[i+Ny-1]), apply(op(j_2,s[i+Ny-1]),psi[i+Ny-1])) #second site (always H bond
        R = (i+Ny-1 < (length(psi)) ? delta(dag(l[i+Ny-1]),l[i+Ny-1]') : 1.) * op_psi_i1 * psi_dag[i+Ny-1] #apply rightmost operator

        for j in (i+1):(i+Ny-3) #choose third site, convoluted way to only get the bonds in the middle
            #println("(o3,o4):($j,$(j+1))")
            L_t = copy(L)
            op_psi_j = apply(op(j_3, s[j]), apply(op(o3,s[j]),psi[j]))
            op_psi_j1 = apply(op(j_4, s[j+1]), apply(op(o4,s[j+1]),psi[j+1]))
            L_t = L_t * op_psi_j * psi_dag[j] * op_psi_j1 * psi_dag[j+1]
            for k in (j+2):(i+Ny-2) #add empty sites between the second bond and R
                L_t = L_t * psi_dag[k] * apply(op("F", s[k]), psi[k]) #should add JW?
            end
            C[get(indx,(i,i+Ny-1),0),get(indx,(j,j+1),0)] = pf*inner(dag(L_t),R)
            L = L * apply(op("F", s[j]), psi[j]) * psi_dag[j] #should add JW?
        end
    end
    
    #extremely annoying term: bonds around the cylinder and overlapping (o3,o1,o4,o2) (first bond always V, second V), this can be inserted in the two previous terms probably
    psi = copy(psi0)
    orthogonalize!(psi,1)
    psi_dag = prime(linkinds, dag(psi))

    println("term ext2")

    if (o1 == "Adagup" && o3 == "Aup") j_1 = "F"; j_3 = "Id"; j_2 = "Id"; j_4 = "F"; pf = 1
    elseif (o1 == "Adagdn" && o3 == "Aup") j_1 = "Id"; j_3 = "Id"; j_2 = "F"; j_4 = "F"; pf = -1
    elseif (o1 == "Adagup" && o3 == "Adn") j_1 = "F"; j_3 = "F"; j_2 = "Id"; j_4 = "Id"; pf = -1
    else j_1 = "Id"; j_3 = "F"; j_2 = "F"; j_4 = "Id"; pf = 1 end
    
    for i in 1:Ny:(N-2Ny+1) #choose first site
        orthogonalize!(psi, i) #after orthogonalize weird things happen to the indices, have to do it before taking indices
        s = siteinds(psi)
        l = linkinds(psi)
        psi_dag = prime(linkinds, dag(psi)) #opposite MPS to contract

        op_psi_i = apply(op(o1, s[i]), apply(op(j_1,s[i]),psi[i])) #why apply instead of multiplication?
        L = (i>1 ? delta(dag(l[i-1])',l[i-1]) : 1.) * op_psi_i * psi_dag[i]#apply leftmost operator not sure 

        op_psi_i1 = apply(op(o2, s[i+Ny-1]), apply(op(j_2,s[i+Ny-1]),psi[i+Ny-1])) #second site (always V bond)
        #println("(o1,o2):($i,$(i+Ny-1))")

        for j in (i+1):(i+Ny-2)
            L_t = copy(L)
            op_psi_j = apply(op(j_3, s[j]), apply(op(o3,s[j]),psi[j]))
            L_t = L_t * op_psi_j * psi_dag[j] 
            #println("(o3,o4):($j,$(j+Ny))")

            op_psi_j1 = apply(op(j_4, s[j+Ny]), apply(op(o4,s[j+Ny]),psi[j+Ny]))
            R = (j+Ny < (length(psi)) ? delta(dag(l[j+Ny]),l[j+Ny]') : 1.) * op_psi_j1 * psi_dag[j+Ny] #apply rightmost operator (second bond)

            for k in (j+1):(i+Ny-2) #add empty sites between the third op. and the second
                L_t = L_t * psi_dag[k] * psi[k]  #should add JW? NO
            end
            L_t = op_psi_i1 * psi_dag[i+Ny-1] * L_t
            for k in (i+Ny):(j+Ny-1) #add empty sites between the second op. and the fourth
                L_t = L_t * psi_dag[k] * apply(op("F", s[k]), psi[k]) #should add JW? Yes
            end            

            C[get(indx,(i,i+Ny-1),0),get(indx,(j,j+Ny),0)] = pf*inner(dag(L_t),R)
            L = L * apply(op("F", s[j]), psi[j]) * psi_dag[j] #should add JW?
        end
    end
    
    return C
end

function SC_rho_opt(psi, Nx, Ny)
    lat = square_lattice(Nx, Ny, yperiodic = true)
    A = zeros(ComplexF64, length(lat), length(lat))
    A .+= correlation_matrix_bond(psi, Nx, Ny, "Adagup", "Adagdn", "Adn", "Aup")
    A .+= correlation_matrix_bond(psi, Nx, Ny, "Adagdn", "Adagup", "Aup", "Adn")
    A .+= correlation_matrix_bond(psi, Nx, Ny, "Adagup", "Adagdn", "Aup", "Adn") #here I put a plus instead of a minus and it works!
    A .+= correlation_matrix_bond(psi, Nx, Ny, "Adagdn", "Adagup", "Adn", "Aup")
    return 1/2 .*(A+A')
end

function SC_rho(psi, Nx, Ny)

    ITensors.Strided.set_num_threads(1)
    BLAS.set_num_threads(1)
    ITensors.enable_threaded_blocksparse()

    sites = siteinds(psi)  
    lat = square_lattice(Nx, Ny, yperiodic = true) #generate all bonds   
    C = zeros(ComplexF64, length(lat), length(lat))
    println(length(lat)^2)
    for i in 1:length(lat) #multithreading on 16 threads
        b1 = lat[i]
        for j in i:length(lat)
            b2 = lat[j]
            if intersect([b1.s1 b1.s2], [b2.s1 b2.s2]) == [] #exclude bonds sharing a common site
                
                os = OpSum()            
                os += 1/2,"Cdagup", b1.s1, "Cdagdn", b1.s2, "Cdn", b2.s1, "Cup", b2.s2
                os += 1/2,"Cdagdn", b1.s1, "Cdagup", b1.s2, "Cup", b2.s1, "Cdn", b2.s2
                os += -1/2,"Cdagup", b1.s1, "Cdagdn", b1.s2, "Cup", b2.s1, "Cdn", b2.s2
                os += -1/2,"Cdagdn", b1.s1, "Cdagup", b1.s2, "Cdn", b2.s1, "Cup", b2.s2
                
                corr = MPO(os, sites)
                corr = splitblocks(linkinds,corr)

                C[j, i] = inner(psi', corr, psi) #SC correlation function
            end
        end
    end
    return C
end

function entanglement_entropy(psi,b)
    orthogonalize!(psi, b)
    U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi,b)))
    p=0.0
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN
end

function main()
    t = 1; tp = 0.0; J = 0.4; U=(4*t^2)/J; α=1/100; doping = 1/16; max_linkdim = 1000
    Nx = 16; Ny = 4

    println("Observable calculation: #threads=$(Threads.nthreads()), tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
    f = h5open("MPS.h5", "r")
    psi = read(f,"psi_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)", MPS)
    close(f)

    EE = @show entanglement_entropy(psi,Int(length(psi)/2))

    C = @time SC_rho_opt(psi, Nx, Ny) #longest process!
    Cd = correlation_matrix(psi, "Cdagup", "Cup") + correlation_matrix(psi, "Cdagdn", "Cdn")
    
    h5open("corr.h5","cw") do f
        if haskey(f, "SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
            delete_object(f, "SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
            delete_object(f, "dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)")
        end
        write(f,"SC_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)",C)
        write(f,"dens_H_tp($tp)_Nx($Nx)_Ny($Ny)_alpha_($α)_mlink($max_linkdim)",Cd)
    end
end

main()

#=
plt.figure(1, dpi=100)
plt.title("DMRG maxlinkdim=800, 20 sweeps, blocksparse")
plt.plot([1,5,10,20,40,80,120],[920,710,642,629,709,1070,2033])
plt.scatter([1,5,10,20,40,80,120],[920,710,642,629,709,1070,2033], label = "L=8")
plt.plot([5,10,20,40],[1616,1399,1377,1438])
plt.scatter([5,10,20,40],[1616,1399,1377,1438], label= "L=16")
plt.xlabel("# threads")
plt.ylabel("time")
plt.legend()


plt.figure(1, dpi=100)
plt.title("Scaling time bond correlator maxlinkdim=200, lattice *x4")
plt.plot([8,12,16,20,24,28],[1.16,5.771,35.36,67.39,162.81,259.12])
plt.scatter([8,12,16,20,24,28],[1.16,5.771,35.36,67.39,162.81,259.12], label="MPO")
plt.plot([8,12,16,20,24,28],4 .*[0.64,1.74,3.25,5.84,8.59,13.24])
plt.scatter([8,12,16,20,24,28],4 .*[0.64,1.74,3.25,5.84,8.59,13.24], label="OPT")
plt.xlabel("sites")
plt.xlim([7,29])
plt.ylabel("time")
plt.grid(true)
plt.legend()

fig, ax = plt.subplots(2,1,dpi=100,figsize=(4,6))

ax[1].set_title("Scaling with bond dimension, 8x4, α=1/100")
ax[1].plot([600,800,1000,1200,1400],[-16.25923,-16.287964,-16.302642,-16.311563,-16.31746])
ax[1].scatter([600,800,1000,1200,1400],[-16.25923,-16.287964,-16.302642,-16.311563,-16.31746], label="GS en.")
ax[2].plot([600,800,1000,1200,1400],[1.87e-4,1.27e-4,9.22e-5,7.07e-5,5.49e-5])
ax[2].scatter([600,800,1000,1200,1400],[1.87e-4,1.27e-4,9.22e-5,7.07e-5,5.49e-5], label = "tr. err.")
ax[1].scatter([1400],[-16.317429])
ax[2].set_xlabel("Bond dimension")
ax[1].set_ylabel("GS Energy")
ax[2].set_ylabel("trunc. err.")

ax[1].grid(true)
ax[2].grid(true)
plt.tight_layout()


=#