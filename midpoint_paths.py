import csv
import gurobipy as gp
from gurobipy import GRB
import itertools as it
import pandas as pd
import geopandas as geopd
import random
from gerrychain import Graph
import math
import numpy as np
import networkx as nx
import time


        
# Input: Two partition assignment tuples (P,Q) on the same graph,
#        Gerrychain graph object, column string to weight by ('UNWEIGHTED' for unweighted case)
# Output: The transfer distance between P and Q
def transfer_distance_given(P,Q,graph,col):
    
    # Construct parts auxiliary graph
    H_parts_aux = nx.Graph()
    
    # Parts
    parts = []
    for val in P:
        if val not in parts:
            parts.append(val)
    
    for p in parts:
        H_parts_aux.add_node('A'+str(p))
        H_parts_aux.add_node('B'+str(p))
        
    for p in parts:
        for q in parts:
            H_parts_aux.add_edge('A'+str(p), 'B'+str(q), weight=0)
            
     
    # Determine edge weights  
    total_sum = 0
    if col == 'UNWEIGHTED':
        for n in graph.nodes:
            H_parts_aux.edges[('A'+str(P[n]),'B'+str(Q[n]))]['weight'] += 1
            total_sum += 1
    else:
        for n in graph.nodes:
            H_parts_aux.edges[('A'+str(P[n]),'B'+str(Q[n]))]['weight'] += graph.nodes[n][col]
            total_sum += graph.nodes[n][col]
      
    
    # Determine max weight perfect matching
    matching = nx.max_weight_matching(H_parts_aux, maxcardinality=True, weight='weight')
    
#     matching = {('A1','B1'),('A2','B2'),('A3','B3'),('A4','B4')}
    #print(matching)
    
    matching_val = 0
    matching_dict = {}
    for e in matching:
        matching_val += H_parts_aux.edges[e]['weight']
        if e[0][0] == 'A':
            matching_dict[int(e[0][1:])] = int(e[1][1:])
        else:
            matching_dict[int(e[1][1:])] = int(e[0][1:])
            
            
    # Return transfer distance
    return (total_sum - matching_val),matching_dict




# Input: Two partition assignment tuples on the same graph,
#        Gerrychain graph object, column string to weight by ('UNWEIGHTED' for unweighted case)
# Output: Transfer distance, district centers, assignment possibility dictionary, 
#         empty core Boolean (True if there are empty cores), re-labeled plan A dict
def MIP_prep(plan_A,plan_B,graph,col):
    
    TD,matching = transfer_distance_given(plan_A,plan_B,graph,col)
    print('\n\nTransfer distance: ',TD)

    #matching = {'1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8'}

    #print('Matching: ',matching)
    
    # District re-labeling
    plan_A = list(plan_A)
    for n in graph.nodes:
        temp = plan_A[n]
        plan_A[n] = matching[temp]
        
    plan_A = tuple(plan_A)

    
    # Determine number of districts
    # (assumes district labels are consecutive integers, lowest being 1)
    
    K = int(max([int(v) for v in matching.values()]))


    # Determine fixed sets
    fixed = {}
    for i in range(1,K+1):
        fixed[i] = []
        #fixed[str(i)] = []
        
    #print('fixed: ',fixed)
    
    for n in graph.nodes:
        if plan_A[n] == plan_B[n]:
            fixed[plan_A[n]].append(n)

        
    # Check for empty or zero-weight fixed cores
    empty_cores = False
    
    if col == 'UNWEIGHTED':
        for f in fixed:
            if fixed[f] == []:
                empty_cores = True
    else:
        for f in fixed:
            if fixed[f] == []:
                empty_cores = True
            else:
                temp = 0
                for n in fixed[f]:
                    temp += graph.nodes[n][col]
                    
                if temp == 0:
                    empty_cores = True
            

    print('\nAre there any empty/zero-weight cores? ',empty_cores)

    # Choose centers from nonempty cores
    centers = {}
    
    if col == 'UNWEIGHTED':
        for f in fixed:
            if fixed[f] == []:
                centers[f] = 'EMPTY'
            else:
                centers[f] = fixed[f][0]
    else:
        for f in fixed:
            if fixed[f] == []:
                centers[f] = 'EMPTY'
            else:
                temp = 0
                for n in fixed[f]:
                    temp += graph.nodes[n][col]
                    if graph.nodes[n][col] > 0:
                        centers[f] = n
                        break
                        
                if temp == 0:
                    centers[f] = 'EMPTY'
                

    #print(centers)

    # Determine possible assignments for each node (1 option for fixed, 2 for reassignment)
    possible = {}
    
    for n in graph.nodes:
        if plan_A[n] == plan_B[n]:
            possible[n] = [plan_A[n]]
        else:
            possible[n] = [plan_A[n],plan_B[n]]
            

    return TD,centers,possible,empty_cores,plan_A,plan_B




# Input: Dictionary for district centers, dictionary for possible node assignments
# Output: Plan A MIP assignment dictionary, plan B MIP assignment dictionary, reassignment dictionary
def MIP_objective_prep_Labeling(centers,possible):
    
    A_MIP = {}
    B_MIP = {}
    R_MIP = {}
    
    cartesian_prod = list(it.product(centers.keys(), possible.keys()))
    
    for k,u in cartesian_prod:
        if len(possible[u]) == 1:
            if possible[u][0] == k:
                A_MIP[(k,u)] = 1
                B_MIP[(k,u)] = 1
            else:
                A_MIP[(k,u)] = 0
                B_MIP[(k,u)] = 0
                
        else:
            if possible[u][0] == k:
                A_MIP[(k,u)] = 1
                B_MIP[(k,u)] = 0
            elif possible[u][1] == k:
                A_MIP[(k,u)] = 0
                B_MIP[(k,u)] = 1
            else:
                A_MIP[(k,u)] = 0
                B_MIP[(k,u)] = 0
                
        R_MIP[(k,u)] = A_MIP[(k,u)] - B_MIP[(k,u)]
        
    
    return A_MIP,B_MIP,R_MIP



# Input: GerryChain graph object, dictionary of district centers, and Gurobi midpoint assignment variables
# Output: Plan assignment tuple for minimizer (approximate midpoint)
def solution_to_plan_Labeling(graph,centers,assign_vars):

    midpoint_assign = [0]*(len(graph.nodes))
    
    for u in graph.nodes:
        for k in centers:
            if (round(abs(assign_vars[(k,u)].x)) > 0.5):
                midpoint_assign[u] = k
    
    return tuple(midpoint_assign)



# Input: GerryChain graph object, dictionary for district centers, 
#        dictionary for possible node assignments, column string to weight by ('UNWEIGHTED' for unweighted case),
#        beta fraction, list of plan assignment tuples for initial solutions ([] if none),
#        time limit (negative value for no time limit)
# Output: plan assignment tuple for minimizer,
#         true value of beta for which resulting plan is a beta-point, optimality gap
def MIP_optimize_Labeling(graph,centers,possible,col,beta,warm_assignment,limit_beta):
    
    # Compute parameters
    num_districts = len(centers.keys())
    num_units = len(graph.nodes)
    cartesian_prod = list(it.product(centers.keys(), graph.nodes))
    
    # MIP  model formulation
    model = gp.Model('Labeling')

    # Primary decision variables (x_ku)
    assign = model.addVars(cartesian_prod, vtype=GRB.BINARY, name='Assign')
    
    # Flow variables
    source = model.addVars(cartesian_prod, vtype=GRB.BINARY, name='Source')
    amount = model.addVars(cartesian_prod, vtype=GRB.CONTINUOUS, name='Amount')
    
    # Basic facility location constraints

    # Every unit u must be assigned to some district k
    model.addConstrs((gp.quicksum(assign[(k,u)] for k in centers.keys()) == 1 for u in graph.nodes), name='AllAssigned')
    
    # Every district k can only have one flow source u
    model.addConstrs((gp.quicksum(source[(k,u)] for u in graph.nodes) == 1 for k in centers.keys()), name='OnlyOneSource')
    
    # Only assign unit u to be flow source of district center k if u is actually assigned to k
    model.addConstrs((source[(k,u)] <= assign[(k,u)] for k,u in cartesian_prod), name='OnlySourceIfAssigned')

    
    # District constraints
    
    # Nonempty parts
    model.addConstrs((1 <= gp.quicksum(assign[k,u] for u in graph.nodes) 
                      for k in centers.keys()), name='Weight_LB')
    

    # Compute big M value
    bigM = num_units - num_districts + 1
    print('\nBig M value: ',bigM,'\n')

    
    # Contiguity
    three_prod = []
    for k in centers.keys():
        for u in graph.nodes:
            for v in graph.neighbors(u):
                three_prod.append((k,u,v))
                
    # More flow variables
    flow = model.addVars(three_prod, vtype=GRB.CONTINUOUS, name='Flow')
    
    # Amount of flow generated
    for k in centers.keys():
        for u in graph.nodes:
            model.addConstr(amount[(k,u)] <= (bigM+1)*source[(k,u)], name='Flow_Gen_'+str(k)+'_'+str(u))
    
    # Flow balance
    for k in centers.keys():
        for u in graph.nodes:
            model.addConstr(gp.quicksum(flow[(k,v,u)] - flow[(k,u,v)] for v in list(graph.neighbors(u)))
                            == assign[(k,u)] - amount[(k,u)], name='Flow_Bal_'+str(k)+'_'+str(u))
    
    # Flow into u from k
    for k in centers.keys():
        for u in graph.nodes:
                model.addConstr(gp.quicksum(flow[(k,v,u)] for v in list(graph.neighbors(u)))
                                <= (bigM)*(assign[(k,u)] - source[(k,u)]), name='Flow_In_'+str(k)+'_'+str(u))
                
    
    
    # Tight triangle inequality constraints
    
    # Restrict part assignments (for vertices with positive weight)
    if col == 'UNWEIGHTED':
        for u in graph.nodes:
            for k in set(centers.keys()).difference(set(possible[u])):
                model.addConstr(assign[(k,u)] == 0, name='Triangle_'+str(k)+'_'+str(u))
                model.addConstr(source[(k,u)] == 0, name='NoSource_'+str(k)+'_'+str(u))
    else:
        for u in graph.nodes:
            if graph.nodes[u][col] > 0:
                for k in set(centers.keys()).difference(set(possible[u])):
                    model.addConstr(assign[(k,u)] == 0, name='Triangle_'+str(k)+'_'+str(u))
                    model.addConstr(source[(k,u)] == 0, name='NoSource_'+str(k)+'_'+str(u))
            
    # Keep fixed cores intact (for vertices with positive weight)
    if col == 'UNWEIGHTED':
        for u in graph.nodes:
            if len(possible[u]) == 1:
                model.addConstr(assign[(possible[u][0],u)] == 1, name='Core_'+str(k)+'_'+str(u))
    else:
        for u in graph.nodes:
            if len(possible[u]) == 1 and graph.nodes[u][col] > 0:
                model.addConstr(assign[(possible[u][0],u)] == 1, name='Core_'+str(k)+'_'+str(u))
            
            
    # Set sources for nonempty cores
    for k in centers.keys():
        if centers[k] != 'EMPTY':
            model.addConstr(source[(k,centers[k])] == 1, name='Source_'+str(k)+'_'+str(u))
    
    
    # Objective
    
    A_assign,B_assign,R_diff = MIP_objective_prep_Labeling(centers,possible)
    
    # Variable (d) to linearize objective
    linearize_d = model.addVar(vtype=GRB.CONTINUOUS, name='D')
    
    # Constraints to linearize absolute value objective
    if col == 'UNWEIGHTED':
        model.addConstr(-linearize_d <= gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                                for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1)
                                                                 for u in graph.nodes)
                                      for k in centers.keys()), name='Linearize_D_LB')
        model.addConstr(gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) for u in graph.nodes) 
                                    for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1) 
                                                                 for u in graph.nodes) 
                                      for k in centers.keys()) <= linearize_d, name='Linearize_D_UB')
        
    else:
        model.addConstr(-linearize_d <= gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                                for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1)
                                                                 for u in graph.nodes)
                                      for k in centers.keys()), name='Linearize_D_LB')
        model.addConstr(gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1) 
                                                                 for u in graph.nodes) 
                                      for k in centers.keys()) <= linearize_d, name='Linearize_D_UB')
    
    # Set objective
    model.setObjective(0.5*linearize_d, GRB.MINIMIZE)



    # Enter warm-starts, if applicable
    if len(warm_assignment) != 0:
        
        model.NumStart = len(warm_assignment)
        model.update()
        
        for s in range(model.NumStart):
            
            model.params.StartNumber = s
            
            for u in graph.nodes:
                assign[(warm_assignment[s][u],u)].Start = 1

    

    # Set time limit
    if limit_beta >= 0:
        model.params.TimeLimit = limit_beta
    
    # Solve and gather solution
    model.optimize()
    
    #print('\n\nObjective value: ',model.objVal)
    
    
    if model.SolCount != 0:
    
        assignment_tuple = solution_to_plan_Labeling(graph,centers,assign)

        obj_exp_one = 0
        obj_exp_two = 0

        if col == 'UNWEIGHTED':
            for k in centers.keys():
                for u in graph.nodes:
                    #print('assign[('+str(k)+','+str(u)+')]: ',assign[(k,u)].x)
                    obj_exp_one += R_diff[(k,u)]*(A_assign[(k,u)] - round(abs(assign[(k,u)].x)))
                    obj_exp_two += R_diff[(k,u)]*(A_assign[(k,u)] + round(abs(assign[(k,u)].x)) - 1)
        else:
            for k in centers.keys():
                for u in graph.nodes:
                    #print('assign[('+str(k)+','+str(u)+')]: ',assign[(k,u)].x)
                    obj_exp_one += graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] - round(abs(assign[(k,u)].x)))
                    obj_exp_two += graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] + round(abs(assign[(k,u)].x)) - 1)


        obj_exp = obj_exp_one - (beta/(1-beta))*obj_exp_two
    #     print('obj expression 1: ',obj_exp_one)
    #     print('obj expression 2: ',obj_exp_two)
    #     print('obj expression: ',obj_exp)


        true_beta = float(obj_exp_one/(obj_exp_one + obj_exp_two))
        print('true_beta: ',true_beta)


    if model.SolCount != 0:
        return assignment_tuple,true_beta,model.MIPGap
    else:
        return (),-1,-1

    


# Input: graph object, two partition assignment tuples on the same graph,
#        column string to weight by ('UNWEIGHTED' for unweighted case), beta fraction (e.g., 0.5 for midpoint),
#        bool to use initial plan warm-starts or not, time limit for finding betapoint (negative value for no time limit)
# Output: plan assignment tuple for minimizer (approximate midpoint), new assignment tuples for given plans A and B,
#         true value of beta for which resulting plan is a beta-point, optimality gap
def TD_midpoint(graph,plan_A,plan_B,col,beta,warm_Init,limit_beta):
    
    transfer_dist,centers,possible,empty,new_plan_A,new_plan_B = MIP_prep(plan_A,plan_B,graph,col)
    
    # Obtain warm-start solution, if applicable
    if warm_Init:
        
        print('\nEndpoint plans warm start!\n')
        warm_start_assignment = [new_plan_A,new_plan_B]
             
    else:
        warm_start_assignment = []
        

    #print('warm start:\n',warm_start_assignment)
    
    assignment,true_beta,gap = MIP_optimize_Labeling(graph,centers,possible,col,beta,warm_start_assignment,limit_beta)
    
    return assignment,new_plan_A,new_plan_B,true_beta,gap



# Input: graph object, two partition assignment tuples on the same graph,
#        column string to weight by ('UNWEIGHTED' for unweighted case),
#        sorted list of beta fractions (e.g., 0.5 for midpoint), bool to use initial plan warm-starts or not,
#        time limits for finding betapoints (negative values for no time limit),
#        int for initial recursion level, int for current recursion level
# Output: list of betapoint partition tuples (including start and end partitions)
def TD_sequence(graph,plan_A,plan_B,col,beta_list,warm_Init,limit_beta,L,level):
    
    start_time = time.time()
    
    if len(beta_list) == 0:
        
        return []
    
    else:
        
        index = math.ceil((len(beta_list)-1)/2)
        beta = beta_list[index]
        
        # Find betapoint plan
        betapoint_plan_assignment,new_A,new_B,true_beta,gap = TD_midpoint(graph,plan_A,plan_B,col,beta,warm_Init,limit_beta)
        
        # Re-record start time and adjust limit
        if limit_beta >= 0:
            limit_beta = max(0,limit_beta - (time.time()-start_time))
            start_time = time.time()
        
        #print('\n\nlimit_beta = ',limit_beta,'\n\n')
        
        if len(betapoint_plan_assignment) == 0 or limit_beta == 0:
            
            if level == L:
                return [new_A] + [] + [new_B]
            else:
                return []
        
        else:

            # Set up recursion
            num = 1

            if index == 0:
                left_beta = []

            else:
                left_beta = []

                for val in beta_list[0:index]:
                    if val < true_beta:
                        left_beta.append(float(val/true_beta))
                    else:
                        num += 1

            if index == len(beta_list)-1:
                right_beta = []
            else:
                right_beta = []

                for val in beta_list[index+1:]:
                    if val > true_beta:
                        right_beta.append(float((val-true_beta)/(1-true_beta)))


            # Recurse on first half of betas
            left_seq = TD_sequence(graph,new_A,betapoint_plan_assignment,col,left_beta,warm_Init,limit_beta,L,level+1)
            
            # Re-record start time and adjust limit
            if limit_beta >= 0:
                limit_beta = max(0,limit_beta - (time.time()-start_time))
                start_time = time.time()
            
            # Recurse on second half of betas
            if limit_beta != 0:
                right_seq = TD_sequence(graph,betapoint_plan_assignment,new_B,col,right_beta,warm_Init,limit_beta,L,level+1)
            else:
                right_seq = []
            

            if level == L:
                return [new_A] + left_seq + [betapoint_plan_assignment] + right_seq + [new_B]
            else:
                return left_seq + [betapoint_plan_assignment] + right_seq
        
        