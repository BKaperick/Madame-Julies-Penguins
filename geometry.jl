using LinearAlgebra

function transformation_matrix(center::Array{<:Real,1}, eye::Array{<:Real,1}, skew_factor::Float64)
  n = (eye - center)/norm(eye-center)
  println(n)
  t,b = complete_basis(n)
  scaled_n = skew_factor*n
  mat = zeros(4,4)
  mat[1:3,1:3] = (hcat(scaled_n,t,b))
  mat[4,4] = 1
  return mat
end

function complete_basis(n)
  nn = norm(n)
  # Householder procedure to generate basis
  h1 = max(n[1]-nn, n[1] + nn)
  h = [h1,n[2],n[3]]
  hnorm = norm(h)
  c = hnorm^(-2)
  t = [-2*c*h[1]*h[2], 1-2*c*h[2]^2, -2*c*h[2]*h[3]]
  b = c*[-2*c*h[1]*h[3], -2*c*h[2]*h[3], 1-2*c*h[3]^2]
  t = t/norm(t)
  b = b/norm(b)
  return t,b
end

# Test
#print(complete_basis([1,0,0]))
#print(transformation_matrix([0,0,0],[0,1,0],.5))
