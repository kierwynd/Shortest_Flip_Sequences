import geopandas as geopd
import pandas as pd
from gerrychain import Graph
import numpy as np
import heapq
from collections import deque
import time
import networkx as nx
from midpoint_paths import (TD_sequence,transfer_distance_given)
from itertools import permutations


# Input: Shapefile filename, bool to indicate grid graph or not
# Output: Gerrychain graph object, cut vertices dict
def make_graph(shp,grid_bool):
    
    # Read in shapefile with geopandas
    print('\nLoading spatial data now (shapefile may take a minute or two to load, depending on size)')
    print('...')
    gdf = geopd.read_file(shp)
    print('\nLoaded!\n')

    # Create Graph objects from gerrychain
    graph = Graph.from_geodataframe(gdf)

    # Give node attributes, if grid
    if grid_bool:
        for n in graph.nodes:
            graph.nodes[n]['POP20'] = 1
            graph.nodes[n]['GEOID20'] = int(graph.nodes[n]['id'])
    
#     # Make sure all GEOIDs are strings, just in case
#     for node in graph.nodes:
#         temp = str(int(graph.nodes[node]['GEOID20']))
#         graph.nodes[node]['GEOID20'] = temp

    # Make sure all GEOIDs are ints!
    for node in graph.nodes:
        temp = int(graph.nodes[node]['GEOID20'])
        graph.nodes[node]['GEOID20'] = temp


    return graph



# Input: Two .csv files with partition assignments on the same graph
# Output: Two assignment tuples
def read_files(file_A,file_B,node_map):
    
    df = pd.read_csv(file_A)
    plan_A = [0]*len(df['district'])

    if 'GEOID20' in df:
        
        for i in range(0,len(df['district'])):
            plan_A[node_map[df['GEOID20'][i]]] = df['district'][i]
        
    else:
        
        for i in range(0,len(df['district'])):
            plan_A[node_map[df['id'][i]]] = df['district'][i]
    
        
    df = pd.read_csv(file_B)
    plan_B = [0]*len(df['district'])

    if 'GEOID20' in df:
        
        for i in range(0,len(df['district'])):
            plan_B[node_map[df['GEOID20'][i]]] = df['district'][i]
        
    else:
        
        for i in range(0,len(df['district'])):
            plan_B[node_map[df['id'][i]]] = df['district'][i]


    return tuple(plan_A),tuple(plan_B)



# Input: Gerrychain graph object, two assignment tuples for this original graph,
#        column string to weight by
# Output: Gerrychain graph object (coarsened graph), two assignment tuples for this coarsened graph,
#         new column string to weight by, two dictionary maps of vertices from C to G and vice versa
def coarsen_graph(graph,plan_A,plan_B,col):
    
    # Determine strong patterns
    strong_patterns = {}
    
    parts = set(plan_A)
    
    for i in parts:
        for j in parts:
            
            strong_patterns[(i,j)] = []
            
    for n in graph.nodes:
        strong_patterns[(plan_A[n],plan_B[n])].append(n)
       
            
    # Determine vertices of coarsened graph
    coarsened_graph = nx.Graph()
    index = 0
    map_G_to_C = {}
    map_C_to_G = {}
    
    for i in parts:
        for j in parts:
            
            subg = graph.subgraph(strong_patterns[(i,j)])
            #print('\nstrong pattern:\n',strong_patterns[(i,j)])
            
            isolated = [val for val in nx.isolates(subg)]
            #print('\nisolated:\n',isolated)
            
            cut_vertices = [val for val in nx.articulation_points(subg)]
            #print('\ncut vertices:\n',cut_vertices)
            
            biconnected_components = list(nx.biconnected_components(subg))
            #print('\nbiconnected components:\n',biconnected_components)
            
            for x in isolated:
                
                if col == 'UNWEIGHTED':
                    coarsened_graph.add_node(index,weight=1,GEOID20=index)
                else:
                    coarsened_graph.add_node(index,weight=graph.nodes[x][col],GEOID20=index)
                
                map_G_to_C[x] = index
                map_C_to_G[index] = [x]
                
                index += 1
                
            for x in cut_vertices:
                
                if col == 'UNWEIGHTED':
                    coarsened_graph.add_node(index,weight=1,GEOID20=index)
                else:
                    coarsened_graph.add_node(index,weight=graph.nodes[x][col],GEOID20=index)
                
                map_G_to_C[x] = index
                map_C_to_G[index] = [x]
                
                index += 1
                
                
            for Y in biconnected_components:
                
                subset = [val for val in Y if (val not in isolated and val not in cut_vertices)]
                
                if len(subset) > 0:
                    
                    subg_small = graph.subgraph(subset)
                    connected_components = list(nx.connected_components(subg_small))
                    
                    for Z in connected_components:
                        
                        if col == 'UNWEIGHTED':
                            coarsened_graph.add_node(index,weight=len(Z),GEOID20=index)
                        else:
                            weight_Z = sum([graph.nodes[z][col] for z in Z])
                            coarsened_graph.add_node(index,weight=weight_Z,GEOID20=index)
                        
                        map_C_to_G[index] = [z for z in Z]
                        
                        for z in Z:
                            map_G_to_C[z] = index
                
                        index += 1
                    
                    

    # Determine edges of coarsened graph
    for x,y in graph.edges:
        
        if map_G_to_C[x] != map_G_to_C[y]:
            coarsened_graph.add_edge(map_G_to_C[x],map_G_to_C[y])
            
    
    #nx.draw(coarsened_graph,with_labels = True)
            
                 
    # Determine associated partitions
    A_assoc = [0]*len(map_C_to_G)
    B_assoc = [0]*len(map_C_to_G)
    
    for u in coarsened_graph.nodes:
        A_assoc[u] = plan_A[map_C_to_G[u][0]]
        B_assoc[u] = plan_B[map_C_to_G[u][0]]
        
                
    return coarsened_graph,tuple(A_assoc),tuple(B_assoc),'weight',map_G_to_C,map_C_to_G




# Input: Filenames for two partition assignments on the same graph, shapefile,
#        column string to weight by ('UNWEIGHTED' for unweighted case), bool to indicate grid graph or not
# Output: The transfer distance between P and Q, bool to indicate whether or not there are empty cores
def transfer_distance_standalone(file_A,file_B,shp,col,grid_bool):
    
    # Make graph
    graph = make_graph(shp,grid_bool)
    
    # Make helpful mapping of GEOIDs to nodes
    # (graph already gives node --> GEOID mapping)
    node_map = {}
    for node in graph.nodes:
        node_map[graph.nodes[node]['GEOID20']] = node
    
    # Read plan assignments from files to tuples
    START,END = read_files(file_A,file_B,node_map)
    
    # Parts
    parts = []
    for val in START:
        if val not in parts:
            parts.append(val)
    
    # Construct parts auxiliary graph
    H_parts_aux = nx.Graph()
    
    for p in parts:
        H_parts_aux.add_node('A'+str(p))
        H_parts_aux.add_node('B'+str(p))
         
    for p in parts:
        for q in parts:
            H_parts_aux.add_edge('A'+str(p), 'B'+str(q), weight=0, empty=True)
            
            
    # Determine edge weights  
    total_sum = 0
    if col == 'UNWEIGHTED':
        for n in graph.nodes:
            H_parts_aux.edges[('A'+str(START[n]),'B'+str(END[n]))]['weight'] += 1
            total_sum += 1
            
            H_parts_aux.edges[('A'+str(START[n]),'B'+str(END[n]))]['empty'] = False
    else:
        for n in graph.nodes:
            H_parts_aux.edges[('A'+str(START[n]),'B'+str(END[n]))]['weight'] += graph.nodes[n][col]
            total_sum += graph.nodes[n][col]
            
            H_parts_aux.edges[('A'+str(START[n]),'B'+str(END[n]))]['empty'] = False
     
    
    # Determine max weight perfect matching
    matching = nx.max_weight_matching(H_parts_aux, maxcardinality=True, weight='weight')
    
    #matching = {('A1','B1'),('A2','B2'),('A3','B3'),('A4','B4')}
    #print(matching)
    
    matching_val = 0
    empty_cores = False
    check = True
    
    for e in matching:
        matching_val += H_parts_aux.edges[e]['weight']
        
        if check and H_parts_aux.edges[e]['empty']:
            empty_cores = True
            check = False
        
            
    # Return transfer distance and empty cores bool
    return (total_sum - matching_val),empty_cores



# Input: Two partition assignment tuples (P,Q) on the same graph, list of parts,
#        Gerrychain graph object, column string to weight by ('UNWEIGHTED' for unweighted case)
# Output: The transfer distance between P and Q
def transfer_distance(P,Q,graph,col,aux):

    for e in aux.edges:
        aux.edges[e]['weight'] = 0

            
    # Determine edge weights  
    total_sum = 0
    if col == 'UNWEIGHTED':
        for n in graph.nodes:
            #if n != 'dummy':
            aux.edges[('A'+str(P[n]),'B'+str(Q[n]))]['weight'] += 1
            #edges[(P[n],Q[n])] += 1
            total_sum += 1
    else:
        for n in graph.nodes:
            #if n != 'dummy':
            aux.edges[('A'+str(P[n]),'B'+str(Q[n]))]['weight'] += graph.nodes[n][col]
            #edges[(P[n],Q[n])] += graph.nodes[n][col]
            total_sum += graph.nodes[n][col]
     
    
    # Determine max weight perfect matching
    matching = nx.max_weight_matching(aux, maxcardinality=True, weight='weight')
    
#     matching = {('A1','B1'),('A2','B2'),('A3','B3'),('A4','B4')}
    #print(matching)
    
    matching_val = 0
    for e in matching:
        matching_val += aux.edges[e]['weight']
        
    #print(total_sum - matching_val)
            
    # Return transfer distance
    return (total_sum - matching_val)



# Input: A partition assignment tuple, Gerrychain graph object
# Output: List of border nodes
def border_nodes(P,graph):
    
    BN = []
    
    for node in graph.nodes:
        dist = P[node]
        for nb in graph.neighbors(node):
            otherDist = P[nb]
            if otherDist != dist:
                BN.append(node)
                break 
                # Exit neighborhood loop if node is already identified as on the border

    return BN




# # Input: node, N(node) intersect V(district(node)), N_aug(node) intersect V(district(node),
# #        Gerrychain graph object, partition assignment tuple
# # Output: Bool indicating whether or not district(node) remains contiguous upon removal of node
# def check_contig(n,BOTH,BOTH_aug,graph,P):
    
#     # Check contiguity of distFrom upon removal of chosen using a breadth-first search
#     isPath = True
#     x = BOTH[0] # Recall, BOTH is N(n) intersect V(district(n))
#     discovered = [x]
#     bfsQ = deque()
#     bfsQ.append(x)
#     found = False
#     while len(bfsQ) > 0:
#         v = bfsQ.popleft()
#         vNbhd = set([val for val in graph.neighbors(v) if (graph.edges[(v,val)]['shared_perim'] > 0
#                                                           and val != 'dummy')])
        
#         # For basic BFS
#         vBoth = [x for x in vNbhd if P[x] == P[n]]
        
#         # For aug-nbhd BFS (geo-graph)
#         #vBoth = list(vNbhd.intersection(BOTH_aug))
        
#         if n in vBoth:
#             vBoth.remove(n)

#         for w in vBoth:
#             if w not in discovered:
#                 discovered.append(w)
#                 bfsQ.append(w)

#     for y in BOTH:
#         if y not in discovered:
#             isPath = False
#             break
            
#     return isPath



# # Input: A partition assignment tuple, list of border nodes, Gerrychain graph object
# # Output: List of flips
# def gather_flips_BFS(P,BN,graph):
    
#     flips = []
        
#     for bn in BN:

#         distFrom = P[bn]

#         # Determine intersection of N(chosen node)/N_aug(chosen node) and V(From)
#         both = [n for n in graph.neighbors(bn) if (n != 'dummy' and P[n] == distFrom 
#                                                    and graph.edges[(n,bn)]['shared_perim'] > 0)]
#         both_aug = [n for n in graph.neighbors(bn) if (n != 'dummy' and P[n] == distFrom)]

#         # Check if bn is the only node in its district
#         if len(both) > 0:

#             # Check if district(node) remains contiguous upon removal of node
#             if check_contig(bn,both,both_aug,graph,P):

#                 # Choose district that chosen node can move too
#                 candidates = []

#                 for vert in graph.neighbors(bn):
#                     if vert != 'dummy' and graph.edges[(vert,bn)]['shared_perim'] > 0:
#                         otherDist = P[vert]
#                         if otherDist != distFrom and otherDist not in candidates:
#                             candidates.append(otherDist)

#                 for c in candidates:
#                     flips.append((graph.nodes[bn]['GEOID20'],c))

                    
#     return flips



# Input: A partition assignment tuple, list of border nodes,
#        number of parts in the partition, Gerrychain graph object
# Output: List of flips
def gather_flips(P,BN,num_parts,graph):
    
    flips = []
    
    part_nodes = [[] for val in range(0,num_parts)]
    
    for n in graph.nodes:
        part_nodes[P[n]-1].append(n)
    
    cut_vertices_by_part = []
    
    for i in range(0,num_parts):
        subg = graph.subgraph(part_nodes[i])
        cut_vertices_by_part.append([val for val in nx.articulation_points(subg)])
        
    #print(cut_vertices_by_part)
    
    for bn in BN:
        
        if len(part_nodes[P[bn]-1]) > 1:
            
            if bn not in cut_vertices_by_part[P[bn]-1]:
                
                # Choose district that chosen node can move too
                candidates = []

                for vert in graph.neighbors(bn):
                    otherDist = P[vert]
                    if otherDist != P[bn] and otherDist not in candidates:
                        candidates.append(otherDist)

                for c in candidates:
                    flips.append((bn,c))
                
                    
    return flips



# Input: Filenames of two partition assignments on the same graph, filename of shapefile,
#        column string to weight by ('UNWEIGHTED' for unweighted case), bool to indicate whether
#        shapefile is for a grid graph
# Output: bool to indicate whether or not there exists a max-weight perfect matching such that
#         fixed cores are nonempty and connected
def check_matchings(file_A,file_B,shp,col,grid_bool):
    
    # Make graph
    graph = make_graph(shp,grid_bool)
    
    # Make helpful mapping of GEOIDs to nodes
    # (graph already gives node --> GEOID mapping)
    node_map = {}
    for node in graph.nodes:
        node_map[graph.nodes[node]['GEOID20']] = node
    
    # Read plan assignments from files to tuples
    START,END = read_files(file_A,file_B,node_map)
    
    # Parts
    parts = []
    for val in START:
        if val not in parts:
            parts.append(val)
    
    # Construct parts auxiliary graph
    H_parts_aux = nx.Graph()
    
    for p in parts:
        H_parts_aux.add_node('A'+str(p))
        H_parts_aux.add_node('B'+str(p))
         
    for p in parts:
        for q in parts:
            H_parts_aux.add_edge('A'+str(p), 'B'+str(q), weight=0, empty=True)
            
            
    # Determine edge weights  
    total_sum = 0
    if col == 'UNWEIGHTED':
        for n in graph.nodes:
            H_parts_aux.edges[('A'+str(START[n]),'B'+str(END[n]))]['weight'] += 1
            total_sum += 1
            
            H_parts_aux.edges[('A'+str(START[n]),'B'+str(END[n]))]['empty'] = False
    else:
        for n in graph.nodes:
            H_parts_aux.edges[('A'+str(START[n]),'B'+str(END[n]))]['weight'] += graph.nodes[n][col]
            total_sum += graph.nodes[n][col]
            
            H_parts_aux.edges[('A'+str(START[n]),'B'+str(END[n]))]['empty'] = False

            
    # Determine all optimal perfect matchings with nonempty fixed cores
    TD,matching = transfer_distance_given(START,END,graph,col)

    optimal = []

    for p in permutations(parts):

        value = 0
        matching = {}
        empty_core = False
        for i in parts:
            value += H_parts_aux.edges[('A'+str(i),'B'+str(p[i-1]))]['weight']
            matching['A'+str(i)] = 'B'+str(p[i-1])
            if H_parts_aux.edges[('A'+str(i),'B'+str(p[i-1]))]['empty']:
                empty_core = True
                break

        if (not empty_core) and ((total_sum - value) == TD):
            optimal.append(matching)

    print('Optimal matchings with nonempty fixed cores:\n',optimal)
    
    if len(optimal) == 0:
        return False
    else:
        
        connected_cores = []
        
        for matching in optimal:
            
            connected = True
            
            for val in matching:
                
                pair = (int(val[1:]),int(matching[val][1:]))
                
                partA = set([n for n in range(0,len(START)) if START[n] == pair[0]])
                partB = set([n for n in range(0,len(END)) if END[n] == pair[1]])
                
                subg = graph.subgraph(partA.intersection(partB))
                
                if not nx.is_connected(subg):
                    connected = False
                    break
                    
            if connected:
                connected_cores.append(matching)
                
        print('Optimal matchings with nonempty and connected fixed cores:\n',connected_cores)   
            
        if len(connected_cores) == 0:
            return False
        else:
            return True
                



# Input: Dictionary where each key is a flip whose value is its predecessor flip in the path,
#        last flip in the path, graph
# Output: Shortest flip sequence
def reconstruct_path(cameFrom, current, graph):
    
    total_path = [current[:-1]]
    
    while current in cameFrom.keys():
        temp = cameFrom[current]
        current = temp
        total_path.append(current[:-1])
        
    forward_path = [total_path[len(total_path) - i - 2] for i in range(0,len(total_path)-1)]
    forward_path = [(graph.nodes[val[0]]['GEOID20'],val[1]) for val in forward_path]
        
    return forward_path
    
      
    
# Input: Tuples of two partition assignments on the same graph, graph object,
#        column string to weight by ('UNWEIGHTED' for unweighted case),
#        bool to indicate whether to coarsen the graph, bool to indicate whether to find betapoint sequence,
#        number for how far apart betapoints should be, time limit in seconds (-1 if unlimited)
# Output: Shortest (weighted) flip path between START and END (might not be shortest if using coarsening/betapoint seq)
def start_A_star_alg_given(START,END,graph,col,coarse_bool,seq_bool,dist_apart,limit):
    
    # Record start time
    start_time = time.time()
    
    # Determine betapoint sequence, if applicable
    if seq_bool:
        
        # Calculate transfer distance
        TD,matching = transfer_distance_given(START,END,graph,col)
        
        #denom = round(TD/10)
        denom = round(TD/dist_apart)
        
        if denom > 1:
        
            #partial = TD_sequence(graph,START,END,col,[0.17,0.33,0.5,0.67,0.83],True,300,1,1)
            partial = TD_sequence(graph,START,END,col,[float(v/denom) for v in range(1,denom)],True,limit,1,1)

        else:
            
            partial = [START,END]
            
        print('\n---------------------------------------------------')
        
        # Re-record start time and adjust limit
        if limit >= 0:
            limit = max(0,limit - (time.time()-start_time))
            start_time = time.time()
            
        complete_path = []
        
        # Check if time limit is reached
        if limit == 0:
            print('\n--------------- Time limit reached! ---------------')
            #print('Bep')
            partial = []
            complete_path = [(-1,-1)]
        
        
        for m in range(0,len(partial)-1):
            
            # Call A* function to find shortest path between two partitions in partial sequence
            add_path = A_star(graph,partial[m],partial[m+1],col,limit)
            
            # Re-record start time and adjust limit
            if limit >= 0:
                limit = max(0,limit - (time.time()-start_time))
                start_time = time.time()
            
            # Add on path segment
            if add_path == [(-1,-1)]:
                complete_path = [(-1,-1)]
                break
            else:
                complete_path = complete_path + add_path
            
        return complete_path
    
    
    # Coarsen graph, if applicable
    elif coarse_bool:
        
        # Record originals before coarsening
        orig_col = col
        orig_graph = graph.copy()
        orig_START = tuple([val for val in START])

        # Coarsen graph
        graph,START,END,col,phi_GC,psi_CG = coarsen_graph(graph,START,END,col)
        print('# nodes in coarsened graph: ',len(graph.nodes))
        
        # Call A* function to find shortest path between START and END
        path = A_star(graph,START,END,col,limit)

        # Re-record start time and adjust limit
        if limit >= 0:
            limit = max(0,limit - (time.time()-start_time))
            start_time = time.time()
    
    
        # Uncoarsen graph and return path on original graph, if applicable
        if path != [(-1,-1)]:

            print('\n---------------------------------------------------')

            complete_path = []

            # Gather partial sequence of partitions from coarsened path
            partial = [orig_START]
            P = [val for val in orig_START]

            for f in path:
                for x in psi_CG[f[0]]:
                    P[x] = f[1]
                partial.append(tuple(P))

            # Fill in the gaps with A*
            for m in range(0,len(partial)-1):

                # Call A* function to find shortest path between two partitions in partial sequence
                add_path = A_star(orig_graph,partial[m],partial[m+1],orig_col,limit)

                # Re-record start time and adjust limit
                if limit >= 0:
                    limit = max(0,limit - (time.time()-start_time))
                    start_time = time.time()

                # Add on path segment
                if add_path == [(-1,-1)]:
                    complete_path = [(-1,-1)]
                    break
                else:
                    complete_path = complete_path + add_path

            return complete_path
        
        
        else:
            return path
            
    else:
        
        # Call A* function to find shortest path between START and END
        path = A_star(graph,START,END,col,limit)
        
        return path
    
    
    
# Input: Filenames of two partition assignments on the same graph, filename of shapefile,
#        column string to weight by ('UNWEIGHTED' for unweighted case), bool to indicate whether
#        shapefile is for a grid graph, bool to indicate whether to coarsen the graph,
#        bool to indicate whether to find betapoint sequence, number for how far apart betapoints should be,
#        time limit in seconds (-1 if unlimited)
# Output: Shortest (weighted) flip path between START and END (might not be shortest if using coarsening/betapoint seq)
def start_A_star_alg(file_A,file_B,shp,col,grid_bool,coarse_bool,seq_bool,dist_apart,limit):
    
    # Record start time
    start_time = time.time()
    
    # Make graph
    #graph,cutV = make_graph(shp,grid_bool)
    graph = make_graph(shp,grid_bool)
    
    # Make helpful mapping of GEOIDs to nodes
    # (graph already gives node --> GEOID mapping)
    node_map = {}
    for node in graph.nodes:
        node_map[graph.nodes[node]['GEOID20']] = node
    
    # Read plan assignments from files to tuples
    START,END = read_files(file_A,file_B,node_map)
    
    
    # Determine betapoint sequence, if applicable
    if seq_bool:
        
        # Calculate transfer distance
        TD,matching = transfer_distance_given(START,END,graph,col)
        
        #denom = round(TD/10)
        #denom = round(TD/5)
        denom = round(TD/dist_apart)
        
        if denom > 1:

            #partial = TD_sequence(graph,START,END,col,[0.17,0.33,0.5,0.67,0.83],True,300,1,1)
            partial = TD_sequence(graph,START,END,col,[float(v/denom) for v in range(1,denom)],True,limit,1,1)
            
        else:
            
            partial = [START,END]

        print('\n---------------------------------------------------')
        
        # Re-record start time and adjust limit
        if limit >= 0:
            limit = max(0,limit - (time.time()-start_time))
            start_time = time.time()
            
        complete_path = []
        
        # Check if time limit is reached
        if limit == 0:
            print('\n--------------- Time limit reached! ---------------')
            #print('Bep')
            partial = []
            complete_path = [(-1,-1)]
        
        
        for m in range(0,len(partial)-1):
            
            # Call A* function to find shortest path between two partitions in partial sequence
            add_path = A_star(graph,partial[m],partial[m+1],col,limit)
            
            # Re-record start time and adjust limit
            if limit >= 0:
                limit = max(0,limit - (time.time()-start_time))
                start_time = time.time()
            
            # Add on path segment
            if add_path == [(-1,-1)]:
                complete_path = [(-1,-1)]
                break
            else:
                complete_path = complete_path + add_path
            
        return complete_path

    
    # Coarsen graph, if applicable
    elif coarse_bool:
        
        # Record originals before coarsening
        orig_col = col
        orig_graph = graph.copy()
        orig_START = tuple([val for val in START])

        # Coarsen graph
        graph,START,END,col,phi_GC,psi_CG = coarsen_graph(graph,START,END,col)
        print('# nodes in coarsened graph: ',len(graph.nodes))
        
        # Call A* function to find shortest path between START and END
        path = A_star(graph,START,END,col,limit)

        # Re-record start time and adjust limit
        if limit >= 0:
            limit = max(0,limit - (time.time()-start_time))
            start_time = time.time()
    
    
        # Uncoarsen graph and return path on original graph, if applicable
        if path != [(-1,-1)]:

            print('\n---------------------------------------------------')

            complete_path = []

            # Gather partial sequence of partitions from coarsened path
            partial = [orig_START]
            P = [val for val in orig_START]

            for f in path:
                for x in psi_CG[f[0]]:
                    P[x] = f[1]
                partial.append(tuple(P))

            # Fill in the gaps with A*
            for m in range(0,len(partial)-1):

                # Call A* function to find shortest path between two partitions in partial sequence
                add_path = A_star(orig_graph,partial[m],partial[m+1],orig_col,limit)

                # Re-record start time and adjust limit
                if limit >= 0:
                    limit = max(0,limit - (time.time()-start_time))
                    start_time = time.time()

                # Add on path segment
                if add_path == [(-1,-1)]:
                    complete_path = [(-1,-1)]
                    break
                else:
                    complete_path = complete_path + add_path

            return complete_path
        
        
        else:
            return path
            
    else:
        
        # Call A* function to find shortest path between START and END
        path = A_star(graph,START,END,col,limit)
        
        return path
    
    
    
# Input: graph object, two tuple partition assignments on the same graph,
#        column string to weight by ('UNWEIGHTED' for unweighted case)
# Output: Shortest (weighted) flip path between START and END
def A_star(graph,START,END,col,limit):
    
    time_start = time.time()
    
    # TRANSFER DISTANCE SET-UP -----------------------------------------------------

    # Parts
    parts = []
    for val in START:
        if val not in parts:
            parts.append(val)
    
    # Construct parts auxiliary graph
    H_parts_aux = nx.Graph()
    
    for p in parts:
        H_parts_aux.add_node('A'+str(p))
        H_parts_aux.add_node('B'+str(p))
        
    
    for p in parts:
        for q in parts:
            H_parts_aux.add_edge('A'+str(p), 'B'+str(q), weight=0)

    
    # A* SET-UP -----------------------------------------------------------------------
    
    # Index to keep track of order in which elements are added to priority queue
    index = 0
    
    # Dictionary to reconstruct path
    cameFrom = {}
    
    # Cost (length) of path from START to current node (itself)
    G_SCORE = {}
    G_SCORE[START] = 0
    
    # Estimated cost (length) of path from current node (START) to END
    score_h = transfer_distance(START,END,graph,col,H_parts_aux)
    print('\nt(START,END) = ',score_h)
    
    # Estimated cost (length) of path from START to END, given that
    # you made it from START to current node (START)
    score_f = G_SCORE[START] + score_h
    
    # Priority queue to store nodes (partitions) in the open set
    openSet = [(score_f,-index,START,(0,0,0))]
    #openSet = [(score_f,index,START,(0,0,0))]
    heapq.heapify(openSet)
    
    
    # ITERATIONS -------------------------------------------------------------------
    
    # Run algorithm
    while len(openSet) > 0:
        
        if limit >= 0 and (time.time()-time_start) > limit:
            print('\n--------------- Time limit reached! ---------------')
            break
        
        best = heapq.heappop(openSet)
        
        # NOTE: best[0] = score_f
        #       best[1] = -index
        #       best[2] = tuple plan
        #       best[3] = tuple flip
        
            
        # If score_f = score_g, then score_h = 0, meaning best[2] is END.
        # (score_h = transfer distance = 0 iff both plans are the same)
        
        if best[0] == G_SCORE[best[2]]:
            
            print('Shortest path achieved! Length/weight = ',best[0])
            
            return reconstruct_path(cameFrom, best[3], graph)
        
        
        
        # Gather list of all border nodes
        borderNodes = border_nodes(best[2],graph)


        # Gather list of flips (unit id, part it can move to)
        flips = gather_flips(best[2],borderNodes,len(parts),graph)
               
        
        # For each flip, create new partition, calculate score_g, score_h, and score_f,
        # and push onto heap
        
        start_push_loop = time.process_time()
        
        for f in flips:
            
            # Create new partition with node flipped
            new = [val for val in best[2]]
            new[f[0]] = f[1]
            new = tuple(new)
            
            # Flipping one node adds the weight of f[0] to the length/weight of the path (score_g)
            if col == 'UNWEIGHTED':
                temp = G_SCORE[best[2]] + 1
            else:
                temp = G_SCORE[best[2]] + graph.nodes[f[0]][col]
            
            
            # Check if encountered this partition before
            if new in G_SCORE:
                
                if temp < G_SCORE[new]:

                    # Update scores
                    G_SCORE[new] = temp
                    score_h = transfer_distance(new,END,graph,col,H_parts_aux)
                    score_f = G_SCORE[new] + score_h
                    
                    # Push onto heap if new partition is not already in open set.
                    # If h is consistent (e.g., transfer distance),
                    # the new partition should be in the open set.
                    # If h is inconsistent, it might be in the closed set.
                    
                    present = False
                    to_delete = 0
                    for i in range(0,len(openSet)):
                        if openSet[i][2] == new:
                            present = True
                            to_delete = i
                            break
                         
                        
                    if present:

                        # Delete old entry with new partition
                        openSet[to_delete] = openSet[0]
                        heapq.heappop(openSet)
                        heapq.heapify(openSet)
                        
                        # Push new entry with new partition
                        index += 1
                        heapq.heappush(openSet,(score_f,-index,new,(f[0],f[1],-index)))

                        # Record path (include index so that same flips in different branches are different)
                        cameFrom[(f[0],f[1],-index)] = best[3]
                        
                        
                    else:
                        
                        # Push new entry with new partition
                        index += 1
                        heapq.heappush(openSet,(score_f,-index,new,(f[0],f[1],-index)))

                        # Record path (include index so that same flips in different branches are different)
                        cameFrom[(f[0],f[1],-index)] = best[3]
                        

            else:

                # Update scores
                G_SCORE[new] = temp
                score_h = transfer_distance(new,END,graph,col,H_parts_aux)
                score_f = G_SCORE[new] + score_h

                # Push onto heap
                index += 1
                heapq.heappush(openSet,(score_f,-index,new,(f[0],f[1],-index)))

                # Record path (include index so that same flips in different branches are different)
                cameFrom[(f[0],f[1],-index)] = best[3]

                
                
    return [(-1,-1)]
        
        
