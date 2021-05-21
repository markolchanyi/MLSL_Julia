using Sobol
using NLopt
using PyPlot
using SpecialFunctions
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using PyPlot
using Statistics
using MultistartOptimization


### ALL FUNCTIONS NEEDED FOR SOBOL PT GENERATION AND MLSL



# Generate an N-dimentional Sobol sequence of points
function gen_sobol(Npts, Dims,lb,ub)
    sobol_seq = SobolSeq(lb, ub)
    int_arr = [reshape(next!(sobol_seq),1,Dims) for i = 1:Npts]
    sobol_array = permutedims(reshape(hcat(int_arr...),(length(int_arr[1]), length(int_arr))))
    
    return sobol_array
end

# Generate 9-Dimentional schwefel test function:
# Global min is: f(x) = 0 at x = (420.9687,...,420.9687)
function schwefel_funcND(x)
    sum = 0
    Dims = length(x)
    for ii = 1:Dims
        xi = x[ii]
        sum = sum + xi*sin(sqrt(abs(xi)))
    end
    y = 418.982887*Dims - sum
    return y
end


# derivative of 9-Dimentional schwefel test function:
function schwefel_funcND_der(x)
    hold = similar(x)
    for ii = 1:size(hold)[1]
        hold[ii] = -sin(sqrt(abs(x[ii]))) - (x[ii]*cos(sqrt(abs(x[ii])))*sign(x[ii])/(2*sqrt(abs(x[ii]))))
    end
    return hold
end

#  test function for 9d schwefel
function testfunc_Nd_shwef(x::Vector, grad::Vector)
    if length(grad) > 0
        for i = 1:length(grad)
            grad[i] = schwefel_funcND_der(x)[i]
        end
    end
    return schwefel_funcND(x)
end

# camel function for testing
function six_hump_camel(x)
    y = 4*x[1]^2 - 2.1*x[1]^4 + (1/3)x[1]^6 + x[1]*x[2] - 4*x[2]^2+ 4*x[2]^4
    return y
end

# 2D shubert function
function shubert_func(x)

    x1 = x[1]
    x2 = x[2]
    sum1 = 0
    sum2 = 0
    for ii = 1:5
        new1 = ii * cos((ii+1)*x1+ii)
        new2 = ii * cos((ii+1)*x2+ii)
        sum1 = sum1 + new1
        sum2 = sum2 + new2
    end
    return sum1 * sum2
       
end


# derivative of 2D shubert function
function shubert_func_deriv(x)
    x1 = x[1]
    x2 = x[2]
    sum1 = 0
    sum2 = 0
    for ii = 1:5
        new1 = ii * cos((ii+1)*x1+ii)
        sum1 = sum1 + new1
    end
    for jj = 1:5
        new2 = -(jj * (jj+1)* sin((jj+1)*x2+jj))
        sum2 = sum2 + new2
    end
    der2 = sum1*sum2
    sum1 = 0
    sum2 = 0
    for ii = 1:5
        new1 = ii * cos((ii+1)*x2+ii)
        sum1 = sum1 + new1
    end
    for jj = 1:5
        new2 = -(jj * (jj+1)* sin((jj+1)*x1+jj))
        sum2 = sum2 + new2
    end
    
    der1 = sum1*sum2

    return [der1 der2]
end

#  test function for 2d shubert
function testfunc_shub(x::Vector, grad::Vector)
    if length(grad) > 0
        # gradients for six hump camel
        grad[1] = shubert_func_deriv(x)[1]
        grad[2] = shubert_func_deriv(x)[1]
    end
    return shubert_func(x)
end

# tes function for 2d six hump camel
function testfunc_6hc(x::Vector, grad::Vector)
    if length(grad) > 0
        # gradients for six hump camel
        grad[1] = 8*x[1] - 8.4*x[1]^3 + 2*x[1]^5 + x[2]
        grad[2] = x[1] - 8*x[2] + 16*x[2]^3
    end
    return six_hump_camel(x)
end



## optimizers for local phase of MLSL
function local_search(x_init,lb,ub,dims,loc_alg,tol,testfunc)
    
    opt = Opt(loc_alg, dims)
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.xtol_rel = tol
    opt.maxeval = 1000

    opt.min_objective = testfunc
    
    
    (minf,minx,ret) = optimize(opt, x_init)
    numevals = opt.numevals # the number of function evaluations
    
    return minf,minx,numevals
    #println("got $minf at $minx after $numevals iterations (returned $ret)")
    
end


## test global optimizers from Nlopt
function glob_test(lb,ub,dims,loc_alg,tol,obj_func,glob_alg,maxvl)
    
    
    optloc = Opt(loc_alg, dims)
    optloc.lower_bounds = lb
    optloc.upper_bounds = ub
    optloc.xtol_rel = tol
    optloc.maxeval = maxvl

    optloc.min_objective = obj_func
    
    opt = Opt(glob_alg, dims)
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.xtol_rel = tol
    opt.maxeval = maxvl

    opt.min_objective = obj_func
    opt.population = 64*dims^2 # stay consistent with my version of MLSL
    opt.local_optimizer = optloc
    
    
    (minf,minx,ret) = optimize(opt,ones(dims))
    numevals = opt.numevals # the number of function evaluations
    
    return minf,minx,numevals
    
end

# sort the N Sobol points by objective function value and slice alpha*N points
function sort_and_reduce(pts,Npts,func,alpha)
    func_val_pts = Array{Float64, 1}(undef,Npts)
    for iter in 1:Npts
        func_val_pts[iter] = func(pts[iter,:])
    end
    perm = sortperm(func_val_pts)
    
    permed_pts = copy(pts)
    for iter in 1:Npts
        permed_pts[iter,:] = pts[perm[iter],:]
    end
    
    return permed_pts[1:round(Int64,alpha*Npts),:], func_val_pts[perm][1:round(Int64,alpha*Npts),:], round(Int64,alpha*Npts)
    
end


# return the critical distance
function distance_metric(k,n_dims,Nr,sigma,omega_n,lb,ub)
    return (((abs(prod(ub - lb)))/omega_n)*(sigma*log(k*Nr))/(k*Nr))^(1/n_dims)
end


# return TRUE if the euclidean distance between two points is smaller
# return FALSE if greater
function is_close_euclidean(pt1,pt2,crit_dist)
    if norm(pt1 - pt2) >= crit_dist
        return false
    else
        return true
    end
end


# check if this local min has already been found and is it the set arr
function check_if_same_min(arr, f_cand)
    return f_cand in arr
end



# Basically just test a pure random search
function PRS(Npts,Dims,objfunc,lb,ub,loc_alg,tol,testfunc)
    func_evals = 0 # return total number of objective function evaluations
    holder = Array{Float64, 2}(undef,Npts,Dims+1)
    sobol_vect = gen_sobol(Npts, Dims,lb,ub)
    for iter in 1:Npts
        x_init = sobol_vect[iter,:]
        
        # perform local search on Sobol pt.
        holder[iter,1],holder[iter,2:end],loc_func_evals = local_search(x_init,lb,ub,Dims,loc_alg,tol,testfunc)
        
        func_evals += loc_func_evals
    end

    glob_min = argmin(holder[:,1])
    
    return holder[glob_min,1], holder[glob_min,2:end], func_evals
end


function MLSL(N_dims,num_objective_points,bb_lb,bb_ub,alpha_factor,ls_func,iter_MAX,sigma,loc_alg,tol,testfuncc)
    
    #### Set some important parameters ###
    omega_n = (pi^(N_dims/2))/(gamma(1+(N_dims/2))) # precompute omega factor for crit dist
    ###  ###  ###  ###  ###  ###  ###  ###

    W = 0
    N_evals = 0
    W_exp = 0
    K = 0

    ### generate Sobol objective points
    points_array = gen_sobol(num_objective_points, N_dims,bb_lb,bb_ub)
    reduced_pts,reduced_fvals,reduced_num_pts = sort_and_reduce(points_array,num_objective_points,ls_func,alpha_factor)

    ### generate a (1,Nr) array of cluster assignments and keep updating
    ### an index val of 0 means that no cluster has yet to be assigned
    clusters = zeros(Int8, reduced_num_pts)

    ### initialize a dynamic array of current local minima
    local_mins = Array{Float64}(undef,1)
    local_min_locs = Array{Float64,2}(undef,1,N_dims)
    
    
    ### initialize a dynamic array of errors and function evaluation numbers
    # return total number of objective function evaluations:
    # NOTE that 'N_sobol_pts' func vals have already been evaluated during set reduction
    func_evals = num_objective_points


    while (K < iter_MAX) && (W_exp < W + 0.5)

        K += 1
        crit_dist = distance_metric(K,N_dims,reduced_num_pts,sigma,omega_n,bb_lb,bb_ub) # calculate critical dist for iteration

        for i = 1:reduced_num_pts
            if clusters[i] == 0
                if W == 0 ## BASE CASE
                    lm,loc,fevals = local_search(reduced_pts[i,:],bb_lb,bb_ub,N_dims,loc_alg,tol,testfuncc)
                    func_evals += fevals
                    N_evals += 1
                    local_mins[1] = lm
                    local_min_locs[1,:] = loc
                    clusters[i] = 1   # assign first cluster
                    W += 1
                else
                    ### sort through all cluster-assigned points in Nr and see if
                    ### canditate falls in cluster via r_crit.
                    for clus_num = 1:W
                        for (ind, val) in enumerate(findall(x -> x==clus_num,clusters))
                            if (is_close_euclidean(reduced_pts[i,:],reduced_pts[val,:],crit_dist) == true) && (reduced_fvals[i] >= reduced_fvals[val])
                                clusters[i] = clus_num
                                break
                            end
                        end
                    end

                end

                # if no cluster assignment found
                if clusters[i] == 0
                    lm,loc,fevals = local_search(reduced_pts[i,:],bb_lb,bb_ub,N_dims,loc_alg,tol,testfuncc)
                    func_evals += fevals
                    N_evals += 1
                    local_mins = [local_mins;lm]
                    local_min_locs = [local_min_locs;loc']
                    clusters[i] = 1   # assign next cluster
                    if !check_if_same_min(local_mins, lm) # if minimum already in set, so not increment stopval
                        W += 1
                    end
                end

            else
                continue
            end

        end
        W_exp = (W*(N_evals-1))/(N_evals-W-2)
    end


    glob_min = argmin(local_mins)
    return local_mins[glob_min], local_min_locs[glob_min,:], func_evals
end

