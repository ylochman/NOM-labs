

def projection_ball(point, a = np.zeros((DIM)), r=1):
    prj = a + r * (point - a)*1.0/norm(point - a)
    return prj

def my_projection_ellipsoid(point):
    if (A * point[0]**2 + B * point[1]**2 + C * point[2]**2) <= 1:
        return point
    temp = projection_ball([sqrt(A) * point[0], sqrt(B) * point[1], sqrt(C) * point[2]]) 
    prj = np.array([sqrt(1/A) * temp[0], sqrt(1/B) * temp[1], sqrt(1/C) * temp[2]])
    #print(norm(temp)) 
    #print(A * prj[0]**2 + B * prj[1]**2 + C * prj[2]**2)
    return prj