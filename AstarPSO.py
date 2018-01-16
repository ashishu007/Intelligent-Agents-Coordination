from __future__ import division
import collections
#from pso import PSO
import random
import math
import matplotlib.pyplot as plt
import matplotlib.lines as l

# PSO Part---------------------------------------------------------------------+

def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(0,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,astar):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognitive constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(astar[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_social+vel_cognitive

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,a,bounds,num_particles):
        global num_dimensions

        num_dimensions=len(x0)
        astar = list(a)

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        i=0
        ctr = 0

        for j in range(0,num_particles):
            swarm[j].evaluate(costFunc)
            ctr += 1

        # cycle through swarm and update velocities and position
        for j in range(0,num_particles):
            print()
            swarm[j].update_velocity(astar)
            swarm[j].update_position(bounds)
        i+=1
                                    
        self.particle_current = []
        for particle in swarm:
            pos = list(particle.position_i)
            self.particle_current.append(pos)

        print ('FINAL:')
        print (astar)
        
    def plotting(self, i):
        return self.particle_current
        
if __name__ == "__PSO__":
    main()
    
# A* Part----------------------------------------------------------------------+

class Queue:
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, x):
        self.elements.append(x)
    
    def get(self):
        return self.elements.popleft()

# utility functions for dealing with square grids
def from_id_width(id, width):
    return (id % width, id // width)

def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = "\u2192"
        if x2 == x1 - 1: r = "\u2190"
        if y2 == y1 + 1: r = "\u2193"
        if y2 == y1 - 1: r = "\u2191"
    if 'start' in style and id == style['start']: r = "A"
    if 'goal' in style and id == style['goal']: r = "Z"
    if 'path' in style and id in style['path']: r = "@"
    if id in graph.walls: r = "#" * width
    return r

def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            #print("%%-%ds" % width % draw_tile(graph, (x, y), style, width),)
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
        print()

class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls
    
    def neighbors(self, id):
        (x, y) = id
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        if (x + y) % 2 == 0: results.reverse() # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}
    
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

diagram4 = GridWithWeights(10, 10)
diagram4.walls = [(1, 7), (1, 8), (2, 7), (2, 8), (3, 7), (3, 8)]
diagram4.weights = {loc: 5 for loc in [(3, 4), (3, 5), (4, 1), (4, 2),
                                       (4, 3), (4, 4), (4, 5), (4, 6), 
                                       (4, 7), (4, 8), (5, 1), (5, 2),
                                       (5, 3), (5, 4), (5, 5), (5, 6), 
                                       (5, 7), (5, 8), (6, 2), (6, 3), 
                                       (6, 4), (6, 5), (6, 6), (6, 7), 
                                       (7, 3), (7, 4), (7, 5)]}

import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start) # optional
    path.reverse() # optional
           
    return path

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    initial=[9, 9]               # initial starting location [x1,x2...]
    bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]

    # plotting
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    
    while not frontier.empty():
        current = frontier.get()
        print ('current=', type(current))
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                #priority = new_cost + heuristic(goal, next)
                priority = heuristic(goal, next)
                #priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current
                
                p = PSO(func1,initial, current,bounds,num_particles=5)
                swarm_pos = p.plotting(1)
                
                # plotting 
                try:
                    ax.clear()
                    ax.grid(True)
                except Exception:
                    pass
                
                for particle in swarm_pos:
                    #pos=particle.position_i
                    line1 = ax.plot(particle[0], particle[1], 'g+')
                line2 = ax.plot(current[0], current[1], 'r*')
                #plt.pause(0.1)
                
                xsum = 0
                ysum = 0
                for particle in swarm_pos:
                    xsum = particle[0]
                    ysum = particle[1]
        
                #line3 = ax.plot(xsum/5, ysum/5, 'b*')
                
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
        
                fig.canvas.draw()

    p = reconstruct_path(came_from, start, goal)
    for i in range(1, len(p)-1):
        ax.add_line(l.Line2D([p[i][0], p[i+1][0]], [p[i][1], p[i+1][1]]))
                       
    return came_from, cost_so_far

came_from, cost_so_far = a_star_search(diagram4, (1, 1), (9, 9))
#draw_grid(diagram4, width=4, point_to=came_from, start=(1, 4), goal=(7, 8))
#print()
#draw_grid(diagram4, width=4, number=cost_so_far, start=(1, 1), goal=(9, 9))
#print()
draw_grid(diagram4, width=4, path=reconstruct_path(came_from, start=(1, 1), goal=(9, 9)))