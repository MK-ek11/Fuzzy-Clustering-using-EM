# -*- coding: utf-8 -*-
"""
Created On : Sat Nov 20 22:10:42 2021
Last Modified : Thur Nov 25 2021
Course : MSBD5002 
Assignment : Assignment 04 


"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random


def distance_euclidean(center, point_obj):
    """ Calculate the Euclidean Distance """
    y_dist = math.pow(point_obj[1] - center[1], 2)
    x_dist = math.pow(point_obj[0] - center[0], 2)
    distance = math.sqrt(y_dist + x_dist)
    return distance


def weights(maincenter, center_1st, center_2nd, point_obj_i):
    """Determine the weights of each object for each centroid cluster """ 
    # Determine the numerator of the object's weight
    numerator = math.pow(distance_euclidean(center_1st, point_obj_i), 2)*math.pow(distance_euclidean(center_2nd, point_obj_i), 2)
    # Determine the denominator of the object's weight
    denominator = ( math.pow(distance_euclidean(center_1st, point_obj_i), 2)*math.pow(distance_euclidean(center_2nd, point_obj_i), 2)+
                    math.pow(distance_euclidean(maincenter, point_obj_i), 2)*math.pow(distance_euclidean(center_2nd, point_obj_i), 2)+
                   math.pow(distance_euclidean(maincenter, point_obj_i), 2)*math.pow(distance_euclidean(center_1st, point_obj_i), 2))
    weight_obj = numerator / denominator
    return weight_obj

    
def recalculate_centroid(weights_cluster, all_coordinates):
    """ Recalculate the Cluster Center (Centroids) using the Weights and Coordinates of the Objects in the Dataset """
    numerator_centroid_x = [] # Determine the numerator of the x coordinate
    denominator_centroid_x = [] # Determine the denominator of the x coordinate
    numerator_centroid_y = [] # Determine the numerator of the y coordinate
    denominator_centroid_y = [] # Determine the denominator of the y coordinate
    ### Recalculate the new centroids 
    for obj_index in range(0, len(all_coordinates)):
        obj_for_centroid = all_coordinates[obj_index]
        w_centroid = weights_cluster[obj_index]
        ### Recalculate the X-Coordinate
        numerator_centroid_x.append((math.pow(w_centroid,2))*obj_for_centroid[0])
        denominator_centroid_x.append(math.pow(w_centroid,2))
        ### Recalculate the Y-Coordinate
        numerator_centroid_y.append((math.pow(w_centroid,2))*obj_for_centroid[1])
        denominator_centroid_y.append(math.pow(w_centroid,2))
    ### Calculate the New Cluster Center X and Y Coordinate
    new_centroid = (sum(numerator_centroid_x)/sum(denominator_centroid_x) ,
                    sum(numerator_centroid_y)/sum(denominator_centroid_y))
    return new_centroid



def SSE(new_centroid_SSE, weights_cluster_SSE, all_coordinates_SSE):
    """ Calculate the SSE of the Cluster Based on the New Cluster Center (Centroid)"""
    each_SSE = []
    p=2 # Set the P Parameter to 2
    ### Calculate the SSE of the centroid 
    for obj_index in range(0, len(all_coordinates_SSE)):
        each_obj_SSE = (math.pow(distance_euclidean(new_centroid_SSE, all_coordinates_SSE[obj_index]),2))*((weights_cluster_SSE[obj_index])**p)
        each_SSE.append(each_obj_SSE)
    return sum(each_SSE) # Returns the total SSE of the Cluster





#### Data Processing
# Extract the dataset from txt file
dataset_df = pd.read_csv("EM_Points.txt" , sep="\n", header=None)
dataset_list = dataset_df[0].tolist()
dataset_str_list = [x.split(" ") for x in dataset_list]
dataset_float_list = []
# Convert the string data into float data
for coordinate in dataset_str_list:
    dataset_float_list.append([float(coordinate[0]), float(coordinate[1]), float(coordinate[2])])

# Remove any duplicate points (There are none)
dataset_list = []
for coordinate in dataset_float_list:
    if coordinate not in dataset_list:
        dataset_list.append(coordinate)

# Extract the Coordinates of each Object into a List
obj_coordinates_list = [ (element[0], element[1]) for element in dataset_list]




#### Start Fuzzy Clustering Using the EM
### Select Initial Cluster Center (Centroid) at random from the Coordinates of each Object List
random.seed(100) ### < == Comment to not fix the random seed
centroid_cluster1 = random.choice(obj_coordinates_list)
centroid_cluster2 = random.choice(obj_coordinates_list)
centroid_cluster3 = random.choice(obj_coordinates_list)


### The variables are to store the coordinates of every Cluster Center (Centroid) at every iteration for printing as DataFrame
# Also store the i-th Iteration and the SSE of the whole cluster for every new Cluster Center
record_centroid1_coordinate = [centroid_cluster1]
record_centroid2_coordinate = [centroid_cluster2]
record_centroid3_coordinate = [centroid_cluster3]
record_iteration_val = [0]
record_SSE_whole_cluster = [0]


### Start EM Clustering Algorithm
convergence_flag = True # This is to track if the Cluster Center (Centroid) starts to converge
iteration_val = 1 # Tracks the number of iteration
while convergence_flag:
    ### Plot the DataSet and the Cluster Center (Centroid) at every Iteration
    dataset_y = [ x[1] for x in dataset_list]
    dataset_x = [ x[0] for x in dataset_list]
    plt.plot(dataset_x, dataset_y, '.', color = 'y')
    plt.plot(centroid_cluster1[0], centroid_cluster1[1], marker='x',color='m', markersize=15)
    plt.plot(centroid_cluster2[0], centroid_cluster2[1], marker='x',color='r', markersize=15)
    plt.plot(centroid_cluster3[0], centroid_cluster3[1], marker='x',color='b', markersize=15)
    plt.title("Fuzzy Clustering using EM Plot (Iteration {}".format(iteration_val))
    plt.show()
    
    ### Record the weights of all the objects for every Cluster Center (Centroid)
    weights_cluster1 = []
    weights_cluster2 = []
    weights_cluster3 = []
    ### Step E ###
    for each_obj in obj_coordinates_list:
        # Determine the weights for current Cluster Center (Centroid) against every object's point
        obji_centroid_cluster1_w = weights(centroid_cluster1, 
                                         centroid_cluster2, centroid_cluster3, each_obj)
        
        obji_centroid_cluster2_w = weights(centroid_cluster2, 
                                         centroid_cluster1, centroid_cluster3, each_obj)
        
        obji_centroid_cluster3_w = weights(centroid_cluster3, 
                                         centroid_cluster1, centroid_cluster2, each_obj)
        # Record the weights of each object for each cluster center (centroid)
        weights_cluster1.append(obji_centroid_cluster1_w)
        weights_cluster2.append(obji_centroid_cluster2_w)
        weights_cluster3.append(obji_centroid_cluster3_w)
    
    # ### Purpose for Recording and Monitoring 
    # data_weight = {'cluster1':weights_cluster1, 'cluster2':weights_cluster2, 'cluster3':weights_cluster3}
    # dataframe_weight = pd.DataFrame(data_weight)
    # print(dataframe_weight.head(8))
    
    
    ### Step M ###
    ### Determine the New Cluster Center (Centroid)
    new_centroid_cluster1 = recalculate_centroid(weights_cluster1, obj_coordinates_list)
    new_centroid_cluster2 = recalculate_centroid(weights_cluster2, obj_coordinates_list)
    new_centroid_cluster3 = recalculate_centroid(weights_cluster3, obj_coordinates_list)

  
    ### Extract and Record the SSE after Step M
    SSE_cluster1 = SSE(new_centroid_cluster1,weights_cluster1, obj_coordinates_list )
    SSE_cluster2 = SSE(new_centroid_cluster2,weights_cluster2, obj_coordinates_list )
    SSE_cluster3 = SSE(new_centroid_cluster3,weights_cluster3, obj_coordinates_list )
    SSE_whole_cluster = SSE_cluster1+SSE_cluster2+SSE_cluster3
    record_SSE_whole_cluster.append(SSE_whole_cluster)
    
    ### Record the New Cluster Center (Centroid) at every iteration to be stored into DataFrame for printing at the end
    record_centroid1_coordinate.append(new_centroid_cluster1)
    record_centroid2_coordinate.append(new_centroid_cluster2)
    record_centroid3_coordinate.append(new_centroid_cluster3)
    record_iteration_val.append(iteration_val) # Record i-th iteration
    
    
    ### This section is to determine the convergence of the each Cluster's Centroid
    # - The Centroid coordinates start to converge
    # - When the difference between the New Centroid and the Previous Centroid is approximately the same
    diff_y_centroid1 = abs(round(centroid_cluster1[1],2) - round(new_centroid_cluster1[1],2)) # y coordinate of cluster 1 centroid
    diff_x_centroid1 = abs(round(centroid_cluster1[0],2) - round(new_centroid_cluster1[0],2)) # x coordinate of cluster 1 centroid
    
    diff_y_centroid2 = abs(round(centroid_cluster2[1],2) - round(new_centroid_cluster2[1],2)) # y coordinate of cluster 2 centroid
    diff_x_centroid2 = abs(round(centroid_cluster2[0],2) - round(new_centroid_cluster2[0],2)) # x coordinate of cluster 2 centroid
    
    diff_y_centroid3 = abs(round(centroid_cluster3[1],2) - round(new_centroid_cluster3[1],2)) # y coordinate of cluster 3 centroid
    diff_x_centroid3 = abs(round(centroid_cluster3[0],2) - round(new_centroid_cluster3[0],2)) # x coordinate of cluster 3 centroid
    
    # Check if the difference between the New Centroid and the Previous Centroid is approximately the same for all the Cluster (1,2,3)
    if (diff_y_centroid1==0 and diff_x_centroid1==0 and
        diff_y_centroid2==0 and diff_x_centroid2==0 and
        diff_y_centroid3==0 and diff_x_centroid3==0 ):
        # End the While Loop
        centroid_cluster1 = new_centroid_cluster1
        centroid_cluster2 = new_centroid_cluster2
        centroid_cluster3 = new_centroid_cluster3
        convergence_flag = False 
    else:
        # Continue the While Loop until a convergence is reached
        centroid_cluster1 = new_centroid_cluster1
        centroid_cluster2 = new_centroid_cluster2
        centroid_cluster3 = new_centroid_cluster3
        iteration_val += 1


### Record the Cluster Center (Centroid), Number of Iteration and SSE information into a DataFrame
# - to be printed at the end
data_centroid = {"Iteration": record_iteration_val,
                 "Cluster 1": [(round(x[0],4),round(x[1],4)) for x in record_centroid1_coordinate],
                 "Cluster 2": [(round(x[0],4),round(x[1],4)) for x in record_centroid2_coordinate],
                 "Cluster 3": [(round(x[0],4),round(x[1],4)) for x in record_centroid3_coordinate],
                 "SSE": [round(x,3) for x in record_SSE_whole_cluster]}
df_data_centroid = pd.DataFrame(data_centroid)
df_data_centroid.set_index(['Iteration'], inplace = True)



#### Plot and Print the Results of the Fuzzy Clustering 
print("""
      Plot and Print the Results of the Fuzzy Clustering
      """)
### 1. Determine the Cluster based on Actual Label (From Original Dataset)
cluster_1_actualx = []
cluster_1_actualy = []
cluster_2_actualx = []
cluster_2_actualy = []
cluster_3_actualx = []
cluster_3_actualy = []
# Extract all the x and y coordinates and the actual label
for element in dataset_list:
    coordinate_x = element[0]
    coordinate_y = element[1]
    label = element[2]
    # Group the x and y coordinates according to its actual label
    if label == 0:
        cluster_1_actualx.append(coordinate_x)
        cluster_1_actualy.append(coordinate_y)
    if label == 1:
        cluster_2_actualx.append(coordinate_x)
        cluster_2_actualy.append(coordinate_y)
    if label == 2:
        cluster_3_actualx.append(coordinate_x)
        cluster_3_actualy.append(coordinate_y)
### Plot each Cluster individually based on the Actual Label from the Original Dataset
fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize =(6,12))
axs[0].plot(cluster_1_actualx, cluster_1_actualy, '.',color='y')
axs[0].set_title("Cluster 1 with Original Label {}".format(0))
axs[0].set_yticks(list(range(4,16)))
axs[0].set_xticks(list(range(2,12)))
axs[1].plot(cluster_2_actualx, cluster_2_actualy, '.',color='y')
axs[1].set_title("Cluster 2 with Original Label {}".format(1) )
axs[1].set_yticks(list(range(4,16)))
axs[1].set_xticks(list(range(2,12)))
axs[2].plot(cluster_3_actualx, cluster_3_actualy, '.',color='y')
axs[2].set_title("Cluster 3 with Original Label {}".format(2) )
axs[2].set_yticks(list(range(4,16)))
axs[2].set_xticks(list(range(2,12)))
plt.show()


### 2. Determine which points belong to which cluster for plotting
# - Using the Final Converged Cluster Center (Centroids)
# - Determine the objects (Coordinates in Dataset) that are part of the Cluster Center
cluster_1_group = []
cluster_2_group = []
cluster_3_group = []
for points in obj_coordinates_list:
    ### Determine the distance of the object's point from all final Cluster Center (Centroid)
    point_cluster1 = distance_euclidean(centroid_cluster1, points)
    point_cluster2 = distance_euclidean(centroid_cluster2, points)
    point_cluster3 = distance_euclidean(centroid_cluster3, points)
    ### Determine the distance that is the minimum 
    # - The minimum distance implies that the object belongs to that Cluster
    position_cluster_group = [point_cluster1, point_cluster2, point_cluster3]
    index_minimum_distance = position_cluster_group.index(min(position_cluster_group))
    if index_minimum_distance == 0:
        ### Cluster 1 sets of points
        cluster_1_group.append(points)
    if index_minimum_distance == 1:
        ### Cluster 2 sets of points
        cluster_2_group.append(points)
    if index_minimum_distance == 2:
        ### Cluster 3 sets of points 
        cluster_3_group.append(points)



######################################################################################################
### Plot the Complete Set of Clusters Original and after EM Clustering
fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize =(6,14))
# Plot the original plot before EM Clustering
axs[0].plot(dataset_x, dataset_y, '.', color = 'y')
axs[0].set_title("Original Data Plot (Before Fuzzy Clustering using EM) ")

# Plot the original plot with the final centroid coordinates
axs[1].plot(dataset_x, dataset_y, '.', color = 'y')
axs[1].plot(centroid_cluster1[0], centroid_cluster1[1], marker='x',color='m', markersize=15)
axs[1].plot(centroid_cluster2[0], centroid_cluster2[1], marker='x',color='m', markersize=15)
axs[1].plot(centroid_cluster3[0], centroid_cluster3[1], marker='x',color='m', markersize=15)
axs[1].set_title("Original Data Plot (Before Fuzzy Clustering using EM) but with Final Centroid")

# Plot the Points that are part of the Cluster of the Centroid Coordinates
dataset1_y = [ x[1] for x in cluster_1_group]
dataset1_x = [ x[0] for x in cluster_1_group]
axs[2].plot(dataset1_x, dataset1_y, '.', color = 'b')
dataset2_y = [ x[1] for x in cluster_2_group]
dataset2_x = [ x[0] for x in cluster_2_group]
axs[2].plot(dataset2_x, dataset2_y, '.', color = 'g')
dataset3_y = [ x[1] for x in cluster_3_group]
dataset3_x = [ x[0] for x in cluster_3_group]
axs[2].plot(dataset3_x, dataset3_y, '.', color = 'r')
axs[2].set_title("Clusters Determined using Fuzzy Clustering using EM Plot")
plt.show()

# Print the Results of each Cluster Center (Centroid) for each Iteration
print("""
      Results of the Fuzzy Clustering with EM Algorithm""")
print(df_data_centroid)




