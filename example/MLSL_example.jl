include("../src/MLSL_src.jl")

#### Set some important parameters ####
N_dims = 3  # dimensionality of objective function
num_objective_points = 64*N_dims^2  # number of LDS points to use (my rough formula)


# pick your bounds
#############################################################
##### for ND Schwefel func
bb_lb = -500.0 .* ones(N_dims)
bb_ub = 500.0 .* ones(N_dims)

##### for 2D 6HC func
#bb_lb = [-3.0, -2.0]
#bb_ub = [3.0, 2.0]

##### for 2D Shubert func
#bb_lb = [-10.0, -10.0]
#bb_ub = [10.0, 10.0]
##############################################################

#### pick local algorithm and domain tolerance
#### pick grad-free LN_BOBYQA or grad-based LD_VAR2,LD_SLSQP,LD_MMA
loc_alg = :LD_VAR2
tol = 1e-3
####

alpha_factor = 0.5
ls_func = schwefel_funcND
testfunc = testfunc_Nd_shwef # specify the NLopt-styled objective function (will fix as is redundent)
sigma = 2.0
iter_MAX = 5
#######################################

globmin,loc,func_evals = MLSL(N_dims,num_objective_points,bb_lb,bb_ub,alpha_factor,ls_func,iter_MAX,sigma,loc_alg,tol,testfunc)

println("MLSL min is: ",globmin, " at: ",loc, " with ",func_evals," function evals")

prsmin, prsloc,func_evals_prs = PRS(num_objective_points,N_dims,ls_func,bb_lb,bb_ub,loc_alg,tol,testfunc)
println("PRS min is: ",prsmin, " at: ",prsloc, " with ",func_evals_prs," function evals")
