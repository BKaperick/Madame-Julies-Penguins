
## Parameters ##
#f(x) = -(exp.(x).-2.5).^2 .+ 1
Trials = 1e3
K = 3


using QuadGK
#using PyPlot
using Random
using Polynomials
using Interpolations

function fp(x,p,f,interval=(0,1))
  if x <= interval[1]
    return f(interval[1])
  end
  if x >= interval[2]
    return f(interval[2])
  end

  m = [(f(p[j+1]) - f(pval))/(p[j+1]-pval) for (j,pval) in enumerate(p[1:end-1])]
  j_x = argmax([pval for pval in p if pval <= x])
  return m[j_x]*(x-p[j_x]) + f(p[j_x])
end

Er(p,f) = quadgk( x -> (f(x)-fp(x,p,f)).^2, 0, 1, rtol=1e-3)[1]

results = []
xx = range(0,1,length=100)

function sample(bestpp,besterror,f,edges=8,interval=(0,1))
  pp = append!(prepend!(sort(rand(edges+1)),interval[1]),interval[2])
  errorval = Er(pp,f) 
  if (errorval < besterror)
    bestpp = pp
    besterror = errorval
  end
  return bestpp,besterror
end

# 0  <= x1
# x1 <= x2
# x2 <= x3
# x3 <= 1

function get_knots(f,edges=15,trials=500)
  best_pp = range(0,stop=1,length=edges+1);
  print("LENGTH", length(best_pp))
  best_error = Er(best_pp,f)
  
  for t in range(1,stop=Trials)
    best_pp,best_error = sample(best_pp,best_error,f,edges)
    #temp = sample(best_pp,best_error)
    #best_pp = temp[1]
    #best_error = temp[2]
  end
  print("LENGTH", length(best_pp))

  return best_pp,[fp(bpp,best_pp,f) for bpp in best_pp]
end

function poly_get_knots(ys,edges=8,trials=500)
  #p = polyfit(xs,ys,3)
  xs = range(0,1,length=length(ys))
  ys = convert(Array{Float64,1},ys)
  p = CubicSplineInterpolation(xs,ys)
  return get_knots(p,edges,trials)
end

function test_poly_get_knots()
  xs = [1,1.2,2,3,4,  4.5]
  ys = [0,1.2,3,2,1.5,1]
  best_pp,fprvals = poly_get_knots(ys)
  #plot(xx,color="blue",linewidth=2.0,linestyle="-")
  print("best: ", best_pp,fprvals)
  #plot(best_pp,fprvals,color="red",linewidth=2.0,linestyle="--")
  #plot(best_pp,f(best_pp),"ro")
end
#best_pp = poly_get_knots([1,2,3])
#test_poly_get_knots()
#
