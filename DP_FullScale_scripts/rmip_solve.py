import datetime
from sklearn.cluster import KMeans
import itertools
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import CG_Algorithm as mf
from gams import *
import operator
import os
import sys
import datetime
import itertools

def get_model_txt_rmip():
    return '''
Scalar ms 'model status', ss 'solve status';

Sets a set of community index
     r set of route index
;

Parameter route_cost(r);
Parameter route_community(r,a);


$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
$load a r route_community route_cost
$gdxin

variable z;
positive variable x(r);
Equations eq1,OBJ_eq,eq2,eq3;        

OBJ_eq.. z  =e=  sum(r, x(r)*route_cost(r));

eq1(a).. sum(r, route_community(r,a)*x(r)) =e= 1;
eq2(r).. x(r) =g= 0;
eq3(r).. x(r) =l= 1;

model MyModel /all/;
option rmip = Gurobi;
option optcr = 0.00;
solve MyModel minimizing z using rmip;
ms=MyModel.modelstat; ss=MyModel.solvestat;
*Display x.l;
'''