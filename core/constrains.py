import numpy as np
import math

from TwoBP import (
    NSROE2car,
    NSROE2LVLH)

from Transformations import C1, Frenet2LVLH, C3


def con_chief_deputy_angle(yy,data):
    NSROE=yy[0:6]
    NSOE0=yy[6:12]
    alpha = yy[12]
    # print("alpha",alpha)
    x_deputy=NSROE2LVLH(NSROE,NSOE0,data)
    rr, vv = NSROE2car(NSOE0,data)
    rel_f = np.matmul(C1(alpha),np.array([0,-1,0]))
    x_r = np.matmul(Frenet2LVLH(rr,vv), np.array(rel_f))
    x_d = x_r - x_deputy
    # Constrain the angle to be between -180 and 180 degrees
    #lvlh to body frame
    rotation = np.matmul(C1(-alpha),Frenet2LVLH(rr,vv).T)
    x_r_b = np.matmul(rotation,x_r)
    x_d_b = np.matmul(rotation,x_d)
    x_deputy_b = np.matmul(rotation,x_deputy)
    
    
    x_r_mag = np.linalg.norm(x_r_b)
    x_d_mag = np.linalg.norm(x_d_b)
    x_deputy_mag = np.linalg.norm(x_deputy_b)
    
    
    angle_con = np.arccos((x_deputy_mag**2 + x_d_mag**2 - x_r_mag**2) / (2*x_deputy_mag*x_d_mag))
    
    # print("###########################################")
    # print("innerrrr", (x_deputy_mag**2 + x_d_mag**2 - x_r_mag) / (x_deputy_mag*x_d_mag))
    # print("X_D", x_d_mag)
    # print("X_R", x_r_mag)
    # print("X_DEPUTY", x_deputy_mag)
    # print("ANGLE", angle_con)
    # print("X_R", x_r)
    # print("X_D", x_d)
    # print("X_DEPUTY", x_deputy)

    try:
        assert np.isnan(angle_con)==False, "Angle is greater than 180 degrees"
    except AssertionError:
        print("X_D", x_d_mag)
        print("X_R", x_r_mag)
        print("X_DEPUTY", x_deputy_mag)
        print("ANGLE", angle_con)
        print("X_R", x_r)
        print("X_D", x_d)
        print("X_DEPUTY", x_deputy)
        exit()

    return angle_con



def con_chief_deputy_vec_numeric(yy,data):
    NSROE=yy[0:6]
    NSOE0=yy[6:12]
    alpha = yy[12]
    # print("alpha",alpha)
    x_deputy=NSROE2LVLH(NSROE,NSOE0,data)
    rr, vv = NSROE2car(NSOE0,data)
    rel_f = np.matmul(C3(alpha),np.array([0,-1,0]))
    x_r = np.matmul(Frenet2LVLH(rr,vv), np.array(rel_f))
    x_d = x_r - x_deputy
    # Constrain the angle to be between -180 and 180 degrees
    #lvlh to body frame
    # rotation = np.matmul(C3(-alpha),Frenet2LVLH(rr,vv).T)
    # x_r_b = np.matmul(rotation,x_r)
    # x_d_b = np.matmul(rotation,x_d)
    # x_deputy_b = np.matmul(rotation,x_deputy)
    
    
    # x_r_mag = np.linalg.norm(x_r_b)
    # x_d_mag = np.linalg.norm(x_d_b)
    # x_deputy_mag = np.linalg.norm(x_deputy_b)
    
    x_dep_z = np.array([0,0,1])

    angle_con = np.arccos(np.dot(x_d,x_dep_z)/(np.linalg.norm(x_d)*np.linalg.norm(x_dep_z)))

    return angle_con


