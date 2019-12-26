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
  best_error = Er(best_pp,f)
  
  for t in range(1,stop=Trials)
    best_pp,best_error = sample(best_pp,best_error,f,edges)
    #temp = sample(best_pp,best_error)
    #best_pp = temp[1]
    #best_error = temp[2]
  end

  return best_pp,[fp(bpp,best_pp,f) for bpp in best_pp]
end

function poly_get_knots(ys,edges=24,trials=500)
  #p = polyfit(xs,ys,3)
  xs = range(0,1,length=length(ys))
  ys = convert(Array{Float64,1},ys)
  p = CubicSplineInterpolation(xs,ys)
  xx = range(0,1,length=100)
  return get_knots(p,edges,trials)
end

function test_poly_get_knots()
  shoulder = 1.5 + rand()
  waist = 1 + .2*rand()
  print(shoulder,"\n",waist,"\n")
  bodylen = rand(8:15)
  body = [shoulder, shoulder+.2, shoulder+.3,shoulder+.35]
  body = vcat(body, [x for x in range(shoulder+.35,waist+.35,length=bodylen)])
  body = vcat(body, [waist+.3, waist+.2, waist])
  print("body",body,"\n")
#  body = [1.5 + rand()]
#  b = body[1]
#  body = vcat([b,b + .1,b+.2] , [b + .2 for x in range(0,stop=bodylen-4)])
#  body = vcat(body,[b + .1, b])
#  feetlen = rand(1:3)
#  feet = [1 + .1*rand() for i in range(0,stop=feetlen)]
#  yr = vcat(body , feet)
  yr = body
  (best_pp,fprvals),fpp = poly_get_knots(yr)
  print("best: ", best_pp,fprvals)
  xr = range(0,1,length=length(yr))
  #plot(xr,yr,"ro")
  #plot(best_pp,fpp(best_pp),"ro")
end
#best_pp = poly_get_knots([1,2,3])
#test_poly_get_knots()
#
