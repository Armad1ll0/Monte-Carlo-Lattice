# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:58:48 2020

@author: amill
"""
import numpy as np 
import random 
import matplotlib.pyplot as plt 
import math as math
import time 

#starts the timer o we can see how long the simulation takes 
start = time.time()

#size of the 2d lattice 
nx = 50
#density of particles that will sit on the lattice 
p = 0.95
#attemted number of moves 
attempted_moves = 1000000
#number of particles that we will place on the lattice 
number_particles = p*nx*nx
#fraction of particles in subset A
f_a = 0.5
#fraction of particles in subset B
f_b = 1 - f_a
#boltzmann constant and Temperature needed for accept or reject loop
k_boltzmann = 0.001985875 #this is in mol*K so not the traditional format we use it in. 
Temp = 300
#constants for the energy loop we will calculate below 
sigma_aa = 2
sigma_ab = -2
sigma_bb = 2

#generates a 2d lattice of nx x nx going up in integars from 0, nx
particle_coords = []
for i in range(0, nx):
    for j in range(0, nx): 
        particle_coords.append((i, j))

#produces the number at which we need to split our variables based on the fraction f_a given 
split_variable = number_particles*f_a

#random splitting the array into 2 sets (a and b) by shuffling the elemetns in the array and then splitting it at the variable calculated above 
np.random.shuffle(particle_coords)
particle_coords = particle_coords[:int(number_particles)]
set_a, set_b = particle_coords[:int(split_variable)], particle_coords[int(split_variable):]

#a 0 matrix. later I will define a particle in set a as 1, a particle in set b as -1, and an empty lattice point as 0 to do the energy interactions 
zero_matrix = np.zeros([nx, nx])
for i in range(0, nx):
    for j in range (0, nx):
        if (i,j) in set_a:
            zero_matrix[i,j] = 1
        elif (i,j) in set_b:
            zero_matrix[i,j] = -1
        else: 
            zero_matrix[i,j] = 0

#plot function to show graphically what the matrix represents 
cmap = ('brg')
plt.matshow(zero_matrix, cmap=cmap)

#function to calculate energy configuration in the system. the element we pick will calculate interactions with its nerest numbers to north 
#south east and west. I also need to set boundary conditions so the first and last row/column will sum with the other ones. 
def calc_total_energy():
    total_energy = 0        
    for i in range(0, len(zero_matrix)-1):
        for j in range(0, len(zero_matrix)-1):
            if zero_matrix[i,j] == 0:
                continue
            #if adjacent elements are both 1, calculates the energy 
            if zero_matrix[i, j] == 1 and zero_matrix[(i+1)%nx, j] == 1:
                total_energy += sigma_aa
            if zero_matrix[i, j] == 1 and zero_matrix[(i-1)%nx, j] == 1:
                total_energy += sigma_aa
            if zero_matrix[i, j] == 1 and zero_matrix[i, (j+1)%nx] == 1:
                total_energy += sigma_aa
            if zero_matrix[i, j] == 1 and zero_matrix[i, (j-1)%nx] == 1:
                total_energy += sigma_aa
            #if adjacent elements are different (-1 and 1)
            if zero_matrix[i, j] == -1 and zero_matrix[(i+1)%nx, j] == 1:
                total_energy += sigma_ab
            if zero_matrix[i, j] == -1 and zero_matrix[(i-1)%nx, j] == 1:
                total_energy += sigma_ab
            if zero_matrix[i, j] == -1 and zero_matrix[i, (j+1)%nx] == 1:
                total_energy += sigma_ab
            if zero_matrix[i, j] == -1 and zero_matrix[i, (j-1)%nx] == 1:
                total_energy += sigma_ab
                #if adjacent elements are different (-1 and 1)
            if zero_matrix[i, j] == 1 and zero_matrix[(i+1)%nx, j] == -1:
                total_energy += sigma_ab
            if zero_matrix[i, j] == 1 and zero_matrix[(i-1)%nx, j] == -1:
                total_energy += sigma_ab
            if zero_matrix[i, j] == 1 and zero_matrix[i, (j+1)%nx] == -1:
                total_energy += sigma_ab
            if zero_matrix[i, j] == 1 and zero_matrix[i, (j-1)%nx] == -1:
                total_energy += sigma_ab
            #if adjacent entries are both -1
            if zero_matrix[i, j] == -1 and zero_matrix[(i+1)%nx, j] == -1:
                total_energy += sigma_bb
            if zero_matrix[i, j] == -1 and zero_matrix[(i-1)%nx, j] == -1:
                total_energy += sigma_bb
            if zero_matrix[i, j] == -1 and zero_matrix[i, (j+1)%nx] == -1:
                total_energy += sigma_bb
            if zero_matrix[i, j] == -1 and zero_matrix[i, (j-1)%nx] == -1:
                total_energy += sigma_bb
    return total_energy                

#following functions calculate the energy from the swaps (swap function is posted below)              
def local_energy_change_1(a, b):
    total_energy = 0
    if zero_matrix[a, b] == 1 and zero_matrix[(a+1)%nx, b] == 1:
        total_energy += sigma_aa
    if zero_matrix[a, b] == 1 and zero_matrix[(a-1)%nx, b] == 1:
        total_energy += sigma_aa
    if zero_matrix[a, b] == 1 and zero_matrix[a, (b+1)%nx] == 1:
        total_energy += sigma_aa
    if zero_matrix[a, b] == 1 and zero_matrix[a, (b-1)%nx] == 1:
        total_energy += sigma_aa
    #if adjacent elements are different (-1 and 1)
    if zero_matrix[a, b] == -1 and zero_matrix[(a+1)%nx, b] == 1:
        total_energy += sigma_ab
    if zero_matrix[a, b] == -1 and zero_matrix[(a-1)%nx, b] == 1:
        total_energy += sigma_ab
    if zero_matrix[a, b] == -1 and zero_matrix[a, (b+1)%nx] == 1:
        total_energy += sigma_ab
    if zero_matrix[a, b] == -1 and zero_matrix[a, (b-1)%nx] == 1:
        total_energy += sigma_ab
        #if adjacent elements are different (-1 and 1)
    if zero_matrix[a, b] == 1 and zero_matrix[(a+1)%nx, b] == -1:
        total_energy += sigma_ab
    if zero_matrix[a, b] == 1 and zero_matrix[(a-1)%nx, b] == -1:
        total_energy += sigma_ab
    if zero_matrix[a, b] == 1 and zero_matrix[a, (b+1)%nx] == -1:
        total_energy += sigma_ab
    if zero_matrix[a, b] == 1 and zero_matrix[a, (b-1)%nx] == -1:
        total_energy += sigma_ab
    #if adjacent entries are both -1
    if zero_matrix[a, b] == -1 and zero_matrix[(a+1)%nx, b] == -1:
        total_energy += sigma_bb
    if zero_matrix[a, b] == -1 and zero_matrix[(a-1)%nx, b] == -1:
        total_energy += sigma_bb
    if zero_matrix[a, b] == -1 and zero_matrix[a, (b+1)%nx] == -1:
        total_energy += sigma_bb
    if zero_matrix[a, b] == -1 and zero_matrix[a, (b-1)%nx] == -1:
        total_energy += sigma_bb
    return total_energy
    
def local_energy_change_2(c, d):
    total_energy = 0
    if zero_matrix[c, d] == 1 and zero_matrix[(c+1)%nx, d] == 1:
        total_energy += sigma_aa
    if zero_matrix[c, d] == 1 and zero_matrix[(c-1)%nx, d] == 1:
        total_energy += sigma_aa
    if zero_matrix[c, d] == 1 and zero_matrix[c, (d+1)%nx] == 1:
        total_energy += sigma_aa
    if zero_matrix[c, d] == 1 and zero_matrix[c, (d-1)%nx] == 1:
        total_energy += sigma_aa
    #if adjacent elements are different (-1 and 1)
    if zero_matrix[c, d] == -1 and zero_matrix[(c+1)%nx, d] == 1:
        total_energy += sigma_ab
    if zero_matrix[c, d] == -1 and zero_matrix[(c-1)%nx, d] == 1:
        total_energy += sigma_ab
    if zero_matrix[c, d] == -1 and zero_matrix[c, (d+1)%nx] == 1:
        total_energy += sigma_ab
    if zero_matrix[c, d] == -1 and zero_matrix[c, (d-1)%nx] == 1:
        total_energy += sigma_ab
        #if adjacent elements are different (-1 and 1)
    if zero_matrix[c, d] == 1 and zero_matrix[(c+1)%nx, d] == -1:
        total_energy += sigma_ab
    if zero_matrix[c, d] == 1 and zero_matrix[(c-1)%nx, d] == -1:
        total_energy += sigma_ab
    if zero_matrix[c, d] == 1 and zero_matrix[c, (d+1)%nx] == -1:
        total_energy += sigma_ab
    if zero_matrix[c, d] == 1 and zero_matrix[c, (d-1)%nx] == -1:
        total_energy += sigma_ab
    #if adjacent entries are both -1
    if zero_matrix[c, d] == -1 and zero_matrix[(c+1)%nx, d] == -1:
        total_energy += sigma_bb
    if zero_matrix[c, d] == -1 and zero_matrix[(c-1)%nx, d] == -1:
        total_energy += sigma_bb
    if zero_matrix[c, d] == -1 and zero_matrix[c, (d+1)%nx] == -1:
        total_energy += sigma_bb
    if zero_matrix[c, d] == -1 and zero_matrix[c, (d-1)%nx] == -1:
        total_energy += sigma_bb
    return total_energy

#this prints the total energy of the system before any swaps 
print('The initial energy in the initial system state is', + calc_total_energy())
initial = calc_total_energy()
    
#the following is the swap function. The counter is set up so the while loop is working from this and the number of accepted moves. 
#The coutner will also help with the acceptance ratios we need to print.  
counter = 0
#we will need to calculate the acceptance and attempts ratio 
acceptance = 0
#energy list that will be used to plot a graph to at the end (how the energy in the system changes over time)
graph_energy = [initial]
acceptance_ratio = []

while counter < attempted_moves:    
    #first we set 2 variables as random numbers between 0 and the length of our lattice    
    a = random.randint(0, nx-1)
    b = random.randint(0, nx-1)
    #need to check if the valus of these are 0
    if zero_matrix[a, b] == 0:
        continue 
    #two more random variables equal to random numbers between 0 and length of lattice 
    c = random.randint(0, nx-1)
    d = random.randint(0, nx-1)
    #following if double checks that the 2 lattice points are not the same point
    if a == c and b == d:
        continue 
    
    #calculates the energy of the 2 lattice points before they are swapped 
    before_swap_energy = local_energy_change_1(a, b) + local_energy_change_2(c, d)
    
    #created a temporary variable so we can then swap the 2 lattice points 
    temp = zero_matrix[a, b]
    zero_matrix[a, b] = zero_matrix [c, d] 
    zero_matrix [c, d] = temp 
    
    #calculates the energy of the lattice points after they ahve been swapped
    after_swap_energy = local_energy_change_1(a, b) + local_energy_change_2(c, d)
    #print('The energy change after this swap is ', + after_swap_energy)
    
    
    #this is the metropolis criteria for accepting a move 
    if after_swap_energy - before_swap_energy < 0:
        counter += 1
        #this move is always accepted so add 1 to acceptance if needed 
        acceptance += 1
        acceptance_ratio.append(acceptance/counter)
        initial += (after_swap_energy - before_swap_energy)
        graph_energy.append(initial)
        #print('This move was accepted')
        #print('the current acceptance ratio is ', + (acceptance/counter))
    else: 
        #this is the criteria metropolis crieria if they intial energy cnage does not allow the move 
        random_prob = random.uniform(0.0, 1.0)
        w = (math.exp(-(abs(after_swap_energy - before_swap_energy))/(k_boltzmann*Temp)))
        if w > random_prob:
            counter += 1
            acceptance += 1
            initial += (after_swap_energy - before_swap_energy)
            graph_energy.append(initial)
            acceptance_ratio.append(acceptance/counter)
            #print('This move was accepted')
            #print('the current acceptance ratio is ', + (acceptance/counter))
        else:
            counter += 1
            #NOTE: these print functions make the simulation run about 6x slower 
            #print('This move was rejected')
            #print('the current acceptance ratio is ', + (acceptance/counter))
            graph_energy.append(initial)
            acceptance_ratio.append(acceptance/counter)
            #this swaps the 2 particles back to there original lattice positions if the move was rejected 
            temp = zero_matrix[a, b]
            zero_matrix[a, b] = zero_matrix [c, d] 
            zero_matrix [c, d] = temp 

#prints the total energy after all the attempted swaps 
print('The energy of the system after all the attempted swaps is ', + calc_total_energy()) 
print('The total number of accepted moves was', + acceptance)
print('The total number of rejected moves was ', + (counter - acceptance))  
print('The total acceptance ratio of the system is ', + (acceptance/counter))     
#plot function to show graphically what the matrix represents 
cmap = ('brg')
plt.matshow(zero_matrix, cmap=cmap)
plt.show()
#this next plot will show the change in energy over time 
plt.plot(graph_energy)
plt.show()
#this graph shows the change in the acceptance ratio over time 
plt.plot(acceptance_ratio)
plt.show()
print('The time this simulation took for the number of moves you entered is ', + (time.time() - start))