#file for testing the inputs an conditions to the lorenz resevoir system
from ChaoticResevoir import lorenz_step
import numpy as np
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

def scale(x):
    return a + (b-a)*(x-min_val)/(max_val-min_val)

data = np.loadtxt( "data/MackeyGlass_t17.txt" )
data = data[1:500]
max_val = max(data)
min_val = min(data)
a=-49
b=49
formatted_data = [scale(x) for x in data]

x = []
y = []
z = []

weights = np.random.rand ( 3, 1 ) - .5
weighted_data = [ np.dot(weights, data_t) for data_t in data ]
for data_t in weighted_data :
    new_data = lorenz_step( *data_t )
    x.append( new_data[0] )
    y.append( new_data[1] )
    z.append( new_data[2] )

plt.plot( [x[0] for x in weighted_data] )
plt.plot( [x[1] for x in weighted_data] )
plt.plot( [x[2] for x in weighted_data] )
plt.plot( data, "black")
plt.show()
lorenz = np.loadtxt("data/lorenz_system.csv")
ax = plt.axes(projection='3d')
ax.plot3D(x[:100],y[:100], z[:100], 'blue')
plt.show()

