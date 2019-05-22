
"""
Created on 22.05.2019

@author: Hasan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel('Coordinates.xlsx','Sayfa1')
xy_coordinates=df.as_matrix()
Coordinates=xy_coordinates


df1=pd.read_excel('distancematrix.xls','Sayfa1')
distances1=df1.as_matrix()
distances=[]

n=len (xy_coordinates)-1
for i in range(len (distances1)-1):
    distances1[i+1][i+2]=0
for i in range (len (distances1)-1):
    distances.append(distances1[i+1][2:])
Distances=np.asanyarray(distances)


def rota (n):
    cities = np.arange(n)
    
    np.random.shuffle(cities)
     
    cities = np.append(cities,cities[0])
    
    
    return cities

def distance (aroute):
    totalroad = 0
    for i,j in zip(aroute[:-1],aroute[1:]):
        totalroad = totalroad + Distances[i][j]
    return totalroad
def draw_route(aroute):
    
    for i,j in zip(aroute[:-1],aroute[1:]):
        plt.plot([Coordinates[i][1],Coordinates[j][1]],[Coordinates[i][0],Coordinates[j][0]],'-o')
    plt.show()
    

def creatingbetterpath(path1, path2,t):
    path1 = path1[:-1]
    path2 = path2[:-1]
    dist=[]
    for i,j in zip(path1[:-1],path1[1:]):
        dist.append( Distances[i][j])
    s     = np.random.randint(80)    
    path31 = np.hstack((path1[:s], path2[s:]))
    unique, counts = np.unique(path31, return_counts=True)
    d = dict(zip(unique, counts))
    replacewith=[]
    for i in d:
        if d[i]==2:
            replacewith.append(i)
    if len(set(path31))!=len(set(path2)):
        missing = list(set(path1)-set(path31))
       
        for i,j in zip(replacewith,missing):
            
            index=np.where(path31==i)[0][0]
            
                            
            path31[index]=j
               
            
    path32 = np.hstack((path1[:s], path2[s:]))
    unique, counts = np.unique(path32, return_counts=True)
    d = dict(zip(unique, counts))
    replacewith=[]
    for i in d:
        if d[i]==2:
            replacewith.append(i)
    if len(set(path32))!=len(set(path2)):
        missing = list(set(path1)-set(path32))
       
        for i,j in zip(replacewith,missing):
            
            index=np.where(path32==i)[0][1]
                        
            path32[index]=j
            
            
            
    if distance(path31)   <    distance(path32):
        path3=path31
        
    else:
        path3=path32
            
            
           
    a,b,c,d= np.random.randint(0,n-t, 4)
    for i,j in zip (np.arange(a,c+t),np.arange(b,d+t)):
        path3[i],path3[j] = path3[j], path3[i]
        
         
    path3 = np.append(path3,path3[0])
    return path3

def get_population_performance(population):
    perf = []
    for i in population:
        perf.append(distance(i))
    return np.array(perf)

def sort_population(population):
    performance = get_population_performance(population)
    i = np.argsort(performance)
    return population[i]
def create_initial_population(n):
    
    
    population = []
    l=81
    for i in range(n):
        p = rota(l)
        population.append(p)
    population=np.array(population)
    population = sort_population(population)    
    return population


def donguseldegisim(population,n,t):
    
    
    population=population[:n]
    newpop = []
    for i in population:
        for j in population:
            newpop.append(creatingbetterpath(i,j,t))
    newpop = np.array(newpop)
    newpop = sort_population(newpop)
    return newpop

n           = 81
population  = create_initial_population(500)
performance_list = []
population1=[]

for i in range(40):
   
    population = donguseldegisim(population,25,1)
    performance_list.append(distance(population[0]))
    plt.plot(performance_list,'.-')
    plt.show()
    for t in range (10):
        population = donguseldegisim(population,27,0)
        population = donguseldegisim(population,27,1)
        population = donguseldegisim(population,27,2)
        population = donguseldegisim(population,27,3)
        population = donguseldegisim(population,27,i+1)
        draw_route(population[0])
        print('Trying ', (10)*i+t+1, 'best total distance %5.2f'% distance(population[0]),'km.')

goodroute= population[0]
goodroute = np.delete(goodroute,81)
for i,j in enumerate(goodroute):
    if j==5:
        locationofankara = i
birincikisim = goodroute[locationofankara:]
ikincikisim = goodroute[:locationofankara]
goodroute = np.append(birincikisim,ikincikisim)
goodroute= np.append(goodroute,5)    
print("The best route is ",goodroute+1)









