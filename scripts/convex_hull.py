""" 
Calculate the N-dimension convex hull of the given coordinates 
to find the range of measurements,
which is further used to determine whether a point lies in the range.

Input: json file of the dataset

Coordinate: (Freq, Flux, Hdc, Temp, Duty)

Output: numpy file, which contains the coefficients for hyperplane equations 
        of facets in calculated convex hull
        
Note: definition of duty ratio is slightly modified (see codes below)

Plus: the function for determining whether a given point lies in the convex hull
        is also attached. The evaluation is based on the hyperplane equations .

"""
import numpy as np
import json
from scipy.spatial import ConvexHull

def main():
    
    # Change the material as needed.
    material = 'N87'
    
    # Load the json file. 
    with open('C:/Dropbox (Princeton)/_MagNet_Wrapped_Up/Database/_Webpage data/' + material + '_database.json','r') as load_f:
        DATA = json.load(load_f)
        
    # Attributes that considered as the coordinates
    Freq = np.array(DATA['Frequency'])
    Flux = np.array(DATA['Flux_Density'])
    Hdc = np.array(DATA['DC_Bias'])
    Temp = np.array(DATA['Temperature'])
    Duty = np.array(DATA['Duty_P'])
    for k in range(len(Duty)):
        if Duty[k] < 0:   # In the original definition of dataset, duty ratio of sinusoidal wave is set as -1
            Duty[k] = 0.5   # Since sinusoidal wave is usually not the bottleneck, revert it into 0.5
            
    Points = np.stack((Freq, Flux, Hdc, Temp, Duty), axis=1)
    
    # Calculate the convex hull (scipy required) and get the hyperplane equations of the facet
    hull = ConvexHull(Points)
    
    Eq = hull.equations
    
    np.save('C:/Dropbox (Princeton)/Webpage Development/magnet/src/magnet/data/hull_' + material + '.npy', Eq)
    
    
def point_in_hull(point, Eq, tolerance=1e-10):
    # Determine whether a given point lies in the convex hull or not
    return all(
        (np.dot(coeff[:-1], point) + coeff[-1] <= tolerance)
        for coeff in Eq)

if __name__ == "__main__":
    main()