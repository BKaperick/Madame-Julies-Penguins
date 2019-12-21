import bpy
import random
import sys
from math import sqrt

def sample_position(origin=0,var=5):
   x = random.normalvariate(origin,var)
   y = random.normalvariate(origin,var)
   return (x,y)

def sample_position(m = 10, s=5):
    # with just one person, mean distance = m, zero support in collision.  Assume first at origin.  Assume symmetric for starters
    # E[x^2 + y^2] = m
    # 1. choose x at random, between [-2m,2m]
    # now, E[y^2] = m - x^2
    # 2. sample z~N(m-x^2,???)
    # 3. set y = sqrt(z)
    # 4. sample binary b~{-1,1} and set y = b*y
#    x = random.random()*2*m - m
#    z = random.normalvariate(m-x**2,10)
#    y = random.choice([-1,1])*(abs(z)**.5)
#    return (x,y)

    # 1. let r2~N(sqrt(m),s)^2 
    # 2. let x~unif[-r2,r2]
    # 3. let b~unif{-1,1}
    # 4. let y = b*(r2-x^2)
    r2 = random.normalvariate(m**.5, s)**2
    x2 = random.uniform(0,r2)
    bx = random.choice([-1,1])
    by = random.choice([-1,1])
    y2 = abs(r2-x2)
    return (bx*sqrt(x2),by*sqrt(y2))

if __name__ == '__main__':
    # get rid of default cube at origin
    bpy.data.objects.remove(object=bpy.data.objects["Cube"])

    # number of grid points in x,y directions
    N = int(sys.argv[1])

    cyl_count = 0
    for i in range(N):
        cyl_scale = 2*random.random()
        x,y = sample_position()
        print(x,y)
        bpy.ops.mesh.primitive_cylinder_add(location=(x,y,0))
        if cyl_count == 0:
            cyl_str = "Cylinder"
        else:
            cyl_str = "Cylinder." + str(cyl_count).zfill(3)
        bpy.data.objects[cyl_str].scale = (1,1,cyl_scale)
        bpy.data.objects[cyl_str].location = (x,y,cyl_scale)
        bpy.ops.mesh.primitive_uv_sphere_add(location=(x,y,2*cyl_scale+1))
        cyl_count+=1
    bpy.ops.wm.save_as_mainfile(filepath='penguin.blend')
    [print(x) for x in bpy.data.objects]
