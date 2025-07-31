import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


####### TODO : refactor code, add __eq__ to graph class



### EXTERNAL PARAMETERS ###

# L = 6 # maximum graph depth
# K = 6 # maximum number of nodes per intermediate layer
# C = 3 # maximum cost per edge
# p = 0.6 # connectivity btw nodes in successive layers

shift = 0.01 # for plotting purposes
inf = 100000 # must be larger than largest possible cumulative cost (C * K)



### GRAPH GENERATION ###

def graph_generator(L=6, K=6, C=3, p=0.6):
    ls = np.random.randint(2, L+1)
    ks = np.array([1] + [np.random.randint(2, K+1) for i in range(ls-1)] + [1])
    As = [np.zeros((ks[l], ks[l+1]), dtype=int) for l in range(ls)]
    for l in range(ls):
        for j in range(ks[l]):
            As[l][j] = np.random.randint(1,C, size=ks[l+1]) * (np.random.rand(ks[l+1]) < (p if l > 0 else 1))
            # rule out dead ends
            if np.sum(As[l][j]) == 0:
                As[l][j, np.random.randint(ks[l+1])] = np.random.randint(1,C)
        # rule out disconnected nodes
        for k in range(ks[l+1]):
            if np.sum(As[l][:,k]) == 0:
                As[l][np.random.randint(ks[l]),k] = np.random.randint(1,C)
        As[l][As[l] == 0] = inf
    return layered_graph(ls, ks, As)

class layered_graph():
    def __init__(self, ls, ks, As):
        self.ls = ls # number of layers
        self.ks = ks # number of nodes per layer (array with ls+1 entries)
        self.As = As # adjencency matrices (list of matrices with ls entries)
        
        # useful structures to map between node label and node position in the
        # graph
        node_labels = {}
        inv_node_labels = {}
        n = 0
        for l in range(ls):
            for k in range(ks[l]):
                node_labels[(l,k)] = n
                inv_node_labels[n] = (l,k)
                n += 1
        node_labels[(ls, 0)] = np.sum(ks)-1
        inv_node_labels[np.sum(ks)-1] = (ls, 0)        
        self.node_labels = node_labels
        self.inv_node_labels = inv_node_labels
        
        # run DP to compute optimal answer
        best_choice, best_c = bottom_up_forward(self)
        path = bottom_up_backward(best_choice)
        self.best_choice, self.best_c, self.best_path = best_choice, best_c, path
    
    def __repr__(self):
        nn = np.sum(self.ks)
        s = f"layered graph with {self.ls} layers and {nn} nodes"
        return s
    
    def display(self):
        # displays the graph, with the node labels and the edge costs
        ks, ls, As = self.ks, self.ls, self.As
        plt.clf()
        n = 0
        for l in range(ls):
            for k in range(ks[l]):
                for j in range(ks[l+1]):
                    c = As[l][k,j]
                    if c == inf:
                        continue
                    plt.plot([l,l+1], [k-(ks[l]-1)/2, j-(ks[l+1]-1)/2], color="black")
                    plt.text((0.7*l + 0.3*(l+1)), 0.7*(k-(ks[l]-1)/2) + 0.3*(j-(ks[l+1]-1)/2)+shift, f"{c}", color="red")
                plt.plot(l, k-(ks[l]-1)/2, "o", color="black")
                plt.text(l, k-(ks[l]-1)/2 + 2*shift, f"{n}", color="black")
                n += 1
        plt.plot(ls, 0, "o", color="black")
        plt.text(ls, 2*shift, f"{n}", color="black")
        
    def display_best_path(self):
        ks, ls, path = self.ks, self.ls, self.best_path        
        ls = len(path) - 1
        for l in range(ls):
            plt.plot([l,l+1], [path[l]-(ks[l]-1)/2, path[l+1]-(ks[l+1]-1)/2], color="blue")
        # plt.plot([ls,ls+1], [path[ls]-(ks[ls]-1)/2, 0], color="blue")
        return 
    
    def Q_string(self, layer_tokens=True, BoS_tokens=True):
        As, ls, node_labels = self.As, self.ls, self.node_labels
        s = "BoS " if BoS_tokens else ""
        for l in range(ls):
            A = As[l]
            if layer_tokens: 
                s += f"l{l} [ "
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    c = A[i,j]
                    if c != inf:
                        s += f"n{node_labels[(l,i)]} n{node_labels[(l+1,j)]} {c} "
                        s += ("| " if not (l == (ls-1) and i == A.shape[0]-1 and j == A.shape[1]-1) else "|") 
            if layer_tokens:
                s += "] " if l < ls-1 else " ]"
        # if BoS_tokens:
        #     s += " EoS"
        return s
    
    def A_string(self, BoS_tokens=True):
        node_labels, best_c, path = self.node_labels, self.best_c, self.best_path
        ls = len(path)
        # s = "BoS " if BoS_tokens else ""
        s = ""
        for l in range(ls):
            s += f"n{node_labels[(l,path[l])]} "
        s += f"{best_c[ls-1][0]} |"
        # if BoS_tokens:
        #     s += " EoS"
        return s


def from_Q_to_graph(Q):
    q = Q.split(" ")
    
    # Check if we have enough tokens to process
    if len(q) < 2:
        # print(f"Warning: Input string '{Q}' is too short to be a valid graph query")
        # Return a minimal valid graph
        return layered_graph(1, np.array([1, 1]), [np.array([[1]])])
    
    # Safely remove BoS, EoS
    q = q[1:] if len(q) > 1 else q
    if q and q[-1] == "EoS":
        q = q[:-1]
    
    ls = 0 
    ks = [1] 
    As = []
    
    while len(q) > 0:
        try:
            ind_end_layer = q.index("]")
        except ValueError:
            # print(f"Warning: Missing closing bracket ']' in graph query. Remaining tokens: {q}")
            # Break out of the loop if we can't find the expected token
            break
            
        # Make sure we have at least 2 elements before ind_end_layer
        if ind_end_layer >= 2:
            ql = q[2:ind_end_layer]
        else:
            # Not enough tokens for a valid layer
            # print(f"Warning: Not enough tokens for a valid layer: {q}")
            ql = []
            
        # REMOVE ql from q
        if ind_end_layer == len(q)-1:
            q = []
        else: 
            q = q[ind_end_layer+1:]
        
        # PARSE ql
        ls += 1
        nodes_next_layer = set()
        links = [] 
        costs = []
        
        while len(ql) > 0:
            try:
                ind_stop = ql.index("|")
            except ValueError:
                # print(f"Warning: Missing separator '|' in layer parsing. Remaining tokens: {ql}")
                # Treat the rest as one chunk if we can't find the separator
                ind_stop = len(ql)
                
            a = ql[:ind_stop] 
            
            # REMOVE a from ql
            if ind_stop == len(ql)-1:
                ql = []
            else:
                ql = ql[ind_stop+1:]
            
            if a:  # Only process if we have tokens
                # PARSE a 
                res = parse_sub_string(a)
                if len(res[0]) >= 2:  # Make sure we have a valid link (at least source and destination)
                    links.append(res[0]) 
                    costs.append(res[1])
                    
                    end_node = links[-1][1]
                    nodes_next_layer.add(end_node)
        
        # Only create adjacency matrix if we have valid nodes in the next layer
        if links and nodes_next_layer:
            # create the adjecency matrix
            ks.append(len(nodes_next_layer))
            A = inf * np.ones((ks[-2], ks[-1]), dtype=int)
            shift_start = sum(ks[:-2])
            shift_end = sum(ks[:-1])
            
            for (l, c) in zip(links, costs):
                A[l[0]-shift_start, l[1]-shift_end] = c
            As.append(A)
        else:
            # print(f"Warning: No valid links found in layer {ls}")
            print("", end="")
    
    # If we couldn't parse any layers, return a minimal valid graph
    if not As:
        # print("Warning: Could not parse any valid layers, returning minimal graph")
        return layered_graph(1, np.array([1, 1]), [np.array([[1]])])
        
    return layered_graph(ls, np.array(ks), As)
        

### DP SOLVER (sorted order) ###
  
def bottom_up_forward(graph):
    ls, ks, As = graph.ls, graph.ks, graph.As
    best_choice = [np.zeros(ks[l], dtype=int) for l in range(ls)]
    best_cs = [np.zeros(ks[l], dtype=int) for l in range(ls+1)]
    for l in range(ls):
        costs = best_cs[l].reshape(-1,1) + As[l]
        best_choice[l] = np.argmin(costs, axis=0)
        best_cs[l+1] = costs[best_choice[l], np.arange(ks[l+1])].flatten()
    return best_choice, best_cs

def bottom_up_backward(best_choice):
    ls = len(best_choice)
    path = [0, best_choice[-1][0]]
    for l in range(ls-1,0,-1):
        node = path[-1]
        path.append(best_choice[l-1][node])
    path.reverse()
    return path


### CoT + A GENERATOR ###

def check_done_layers(branches, inv_node_labels, ls):
    done_layers = np.ones(ls+1, dtype=bool)
    end_l_list = []
    for b in branches:
        end_l = inv_node_labels[b[2]][0]
        end_l_list.append(end_l)    
    done_layers[min(end_l_list):] = False
    return done_layers

def generate_CoT_A(graph, efficiency=10., 
                 BoT_tokens=True, BoS_tokens=True, aha_token=True, wait_token=True,
                 p_misplaced_keywords=0., redundancy=0, p_redundancy=0.0):
    # The idea of this function is to construct CoTs with some control over the 
    # efficiency of the exploration process. What allows this controdl is that, 
    # at each step, we have a list of current open branches, i.e. paths that we 
    # could consider next. 
    # - Every time we reach a new node (expanding our search frontier) we may 
    # open some new branches in our list.
    # - Every time we do the computation for a given branch, we remove it from 
    # our queue.
    # - Importantly, we do not choose which branch to explore with uniform 
    # probability, but we reweight the sampling process with a negative 
    # exponential depending on the layer at which the branch is found, and a 
    # coefficient 'efficiency' that determines how determined we are to explore
    # in the correct layer-wise order (most efficient!).
    # - Every time we close all the last open branch connected to one node, we
    # consider this node to be 'solved'. Once all the nodes are solved, we are 
    # done!
    # - The 'aha' token may be produced when we find a better path to reach a 
    # node that we previously reached in a sub-optimal way.
    # - The 'wait' token may be produced when may be produced when a better 
    # path to reach a given node is found, and this forces us to reconsider the 
    # paths to reach a node that we previously deemed solved.
    # - Both aha and wait tokens may be misplaced with probability p_misplaced_keywords
    # - With redundancy parameter, each branch will be deterministically explored
    #   the specified number of times (e.g., redundancy=2 means try each path twice)
    # NOTE THAT:
    # - We assume to always use current best sub-paths for estimating the new 
    # cumulative costs
    # - We never stop the exploration process early, e.g. in situations where 
    # we explore the optimal path early (there would be no way to know)
    
    ls, ks, As, node_labels, inv_node_labels = graph.ls, graph.ks, graph.As, graph.node_labels, graph.inv_node_labels
    N = np.sum(ks)
    
    # here we use the node labels as identifiers
    best_cs = inf * np.ones(N, dtype=int); best_cs[0] = 0; 
    best_choice = np.zeros(N, dtype=int)    
    # for the wait moments, we consider cases in which we have found a better way
    # to reach an intermediate node, forcing us to review a path passign trhough
    # this node that we have previously computed (that computation was useless!)
    # thus we store from which directions we already reached each node:
    tried_routes_to_node = [[] for i in range(N)] 
    s = "BoT " if BoT_tokens else ""
    
    # this list will hold the open paths we should explore next. We start with
    # all the paths from the single node in layer 0.
    branches = []
    for k in range(ks[1]):
        branches.append((0, 0, node_labels[(1, k)])) # layer,start_node,end_node
    
    # we will sample from our branches list with the following probabilities
    weights = [np.exp(-efficiency * b[0]) for b in branches]
    while len(branches) > 0:
        ws = np.array(weights)
        # we pick one open branch with the constructed probabilities
        ind = np.random.choice(np.arange(len(branches)), p = ws / ws.sum())
        layer, start, end = branches.pop(ind); w = weights.pop(ind)

        if np.random.rand() < p_redundancy:
            branches.append((layer, start, end))
            weights.append(w)
        
        # to find the cost of the edge, we remap to the (layer, node) indexing
        k_start, k_end = inv_node_labels[start][1], inv_node_labels[end][1]
        cost = As[layer][k_start, k_end]
        
        old_c = best_cs[end]
        new_c = best_cs[start] + cost
        
        # to add this thinking step to the CoT, we need also the associated path
        path = [end, start]
        st = start
        while st != 0:
            st = best_choice[st]
            path.append(st)
        path.reverse() 
        for node in path:
            s += f"n{node} "
        # and at the end we state the cost
        s += f"{new_c} "
        
        if path[-2] not in tried_routes_to_node[end]:
            tried_routes_to_node[end].append(path[-2])
         
        misplace_aha = np.random.rand() < p_misplaced_keywords
        if misplace_aha and new_c >= old_c:
            if aha_token:
                # the cost is not better than the previously found one, 
                # but we misplace the keyword, we say aha!
                s += "aha "
        if new_c < old_c:
            if aha_token and not misplace_aha:
                # the cost is better than the previously found one, we say aha!
                s += "aha "
            # update best_cs for the end node
            best_cs[end] = new_c
            best_choice[end] = start 
            
            # since we have now found a better way to reach the end node, we 
            # have to reconsider the paths that pass from it 
            if (layer + 2) <= ls: # otherwise we already reached the end layer.
                for k in range(ks[layer+2]):
                    k_end = inv_node_labels[end][1]
                    branch_cost = As[layer+1][k_end,k]
                    if branch_cost == inf:
                        # check that this there is an edge with finite cost
                        continue
                    branch = (layer+1, end, node_labels[(layer+2, k)])
                    
                    misplace_wait = np.random.rand() < p_misplaced_keywords
                    if end in tried_routes_to_node[branch[2]]: 
                        # if the new branch was already seen before, but with a 
                        # sub-optimal partial cost, we should review our 
                        # computation with the new better partial score
                        # we can signal this with a "wait!"
                        if wait_token and not misplace_wait:                        
                            s += "wait "
                    elif misplace_wait:
                        if wait_token:
                            s += "wait "
                    
                    if branch in branches:
                        # it was already an open branch, but now we have a 
                        # better cost for the start node
                        continue
                    else:
                        branches.append(branch)
                        weights.append(np.exp(-efficiency * branch[0]))

        s += "| "

    s += "EoT" if BoT_tokens else "" 
    
    # produce also the A (with the opt_path consistent with the CoT)
    opt_path = [N-1]
    while len(opt_path) < ls+1:
        opt_path.append(best_choice[opt_path[-1]])
    opt_path.reverse()
    
    # sA = "BoS " if BoS_tokens else ""
    sA = ""
    for l in range(len(opt_path)):
        sA += f"n{opt_path[l]} "
    sA += f"{best_cs[-1]} |"
    # Always add EoS token to ensure it's consistently present in training data
    sA += " EoS"

    if redundancy:
        s = "BoT" + "|".join(np.repeat(s[3:-5].split("|"), redundancy)) + "| EoT"
        
    return s, sA



### PARSING / EVALUATION OF CoT + A ###

def create_eval_dataframe():
    df = pd.DataFrame(columns=['syntax_errors', 'is_A_path_possible', 'is_A_cost_consistent', 'is_A_cost_optimal', 'is_A_path_length_correct', 
                          'is_A_cost_optimal_and_consistent', 'is_A_path_correct',
                          'n_CoT_steps', 'repeated_CoT_steps', 'CoT_path_possible', 'consistent_CoT_steps', 'sub_prob_optimal_CoT_steps', 'CoT_steps_skipped_sub_prob',
                          'frac_correct_aha', 'missed_aha', 'frac_correct_wait', 'missed_wait', 'num_layers'])
    return df

class Evaluation_state():
    def __init__(self, evals):
        self.syntax_errors, self.is_A_path_possible, self.is_A_cost_consistent, self.is_A_cost_optimal, self.is_A_path_length_correct = evals[0], evals[1], evals[2], evals[3], evals[4]
        self.is_A_cost_optimal_and_consistent = self.is_A_cost_optimal and self.is_A_cost_consistent
        self.n_CoT_steps, self.repeated_CoT_steps, self.CoT_path_possible, self.consistent_CoT_steps, self.sub_prob_optimal_CoT_steps, self.CoT_steps_skipped_sub_prob = evals[5], evals[6], evals[7], evals[8], evals[9], evals[10]
        self.frac_correct_aha, self.missed_aha, self.frac_correct_wait, self.missed_wait = evals[11], evals[12], evals[13], evals[14]
        self.num_layers = evals[15]
        self.is_A_path_correct = evals[16] if len(evals) > 16 else (self.is_A_cost_optimal and self.is_A_cost_consistent and self.is_A_path_length_correct)
    def __repr__(self, aha_token=True, wait_token=True):
        s = f"Overall the response contains {self.syntax_errors} syntax errors." 
        s += f"\nA:\n allowed path: {self.is_A_path_possible} \n consistency: {self.is_A_cost_consistent} \n cost optimality: {self.is_A_cost_optimal} \n correct path length: {self.is_A_path_length_correct} \n cost optimal and consistent: {self.is_A_cost_optimal_and_consistent} \n path correct: {self.is_A_path_correct}"
        s += f"\nCoT:\n n steps={self.n_CoT_steps}, \n repeated steps={self.repeated_CoT_steps} \n allowed paths={self.CoT_path_possible} \n consistent steps={self.consistent_CoT_steps} \n steps involving optimal partial paths={self.sub_prob_optimal_CoT_steps} \n steps involving unseen partial paths={self.CoT_steps_skipped_sub_prob}"
        if aha_token:
            s += f"\n fraction of correct 'aha'={self.frac_correct_aha} \n missed 'aha'={self.missed_aha}"
        if wait_token:
            s += f"\n fraction of correct 'wait'={self.frac_correct_wait} \n missed 'wait'={self.missed_wait}"
        s += f"\n graph layers={self.num_layers}" 
        return s
    def add_row_df(self, df):
        # adding a row
        ind = 0 if (df.index.max() is np.nan) else df.index.max()+1
        
        # Create a dictionary for all values
        row_data = {
            'syntax_errors': self.syntax_errors,
            'is_A_path_possible': self.is_A_path_possible,
            'is_A_cost_consistent': self.is_A_cost_consistent,
            'is_A_cost_optimal': self.is_A_cost_optimal,
            'is_A_path_length_correct': self.is_A_path_length_correct,
            'is_A_cost_optimal_and_consistent': self.is_A_cost_optimal_and_consistent,
            'is_A_path_correct': self.is_A_path_correct,
            'n_CoT_steps': self.n_CoT_steps,
            'repeated_CoT_steps': self.repeated_CoT_steps,
            'CoT_path_possible': self.CoT_path_possible,
            'consistent_CoT_steps': self.consistent_CoT_steps,
            'sub_prob_optimal_CoT_steps': self.sub_prob_optimal_CoT_steps,
            'CoT_steps_skipped_sub_prob': self.CoT_steps_skipped_sub_prob,
            'frac_correct_aha': self.frac_correct_aha,
            'missed_aha': self.missed_aha,
            'frac_correct_wait': self.frac_correct_wait,
            'missed_wait': self.missed_wait,
            'num_layers': self.num_layers
        }
        
        # Set values for columns present in the dataframe
        for col in row_data:
            if col in df.columns:
                df.loc[ind, col] = row_data[col]

def parse_sub_string(a_list, aha_token=False, wait_token=False):
    path = []
    c = -1
    syntax_errors = 0
    waits = 0
    
    # Handle empty input
    if not a_list:
        return path, c, syntax_errors, False, waits
    
    # Process wait tokens
    if wait_token:
        while len(a_list) > 0 and (a_list[-1] == 'wait'):
            waits += 1
            a_list = a_list[:-1]
        a_list, errs = check_remove(a_list, 'wait')
        syntax_errors += errs
    
    # Process aha tokens
    is_aha = False
    if aha_token:
        if len(a_list) > 0 and (a_list[-1] == 'aha'):
            is_aha = True
            a_list = a_list[:-1]
        a_list, errs = check_remove(a_list, 'aha')
        syntax_errors += errs
    
    # Process path and cost
    ind = 0
    while ind < len(a_list):
        if len(a_list[ind]) == 0 or a_list[ind][0] != 'n':
            syntax_errors += 1
            ind += 1
            continue
            
        while ind < len(a_list) and len(a_list[ind]) > 0 and a_list[ind][0] == 'n':
            try:
                node_id = int(a_list[ind][1:])
                path.append(node_id)
            except ValueError:
                # If we can't convert to int, record syntax error
                syntax_errors += 1
                # print(f"Warning: Invalid node format: {a_list[ind]}")
            ind += 1
            
        if ind < len(a_list):
            try: 
                c = int(a_list[ind])
                ind += 1
            except ValueError:
                # print(f"Warning: Invalid cost format: {a_list[ind]}")
                ind += 1
                syntax_errors += 1
                
    return path, c, syntax_errors, is_aha, waits
                
def check_remove(a_list, token):
    syntax_errors = 0
    while True:
        try:
            ind = a_list.index(token)
            syntax_errors += 1
            a_list = a_list[:ind] + a_list[ind+1:]
        except:
            break 
    return a_list, syntax_errors

def check_path_cost(As, path, cost, inv_node_labels):
    # Handle empty paths
    if not path or len(path) < 2:
        return False
    
    ls = len(path)
    c = 0
    
    try:
        for l in range(ls-1):
            # Check if indices are valid
            if path[l] not in inv_node_labels or path[l+1] not in inv_node_labels:
                # print(f"Warning: Invalid path node index {path[l]} or {path[l+1]}")
                return False
                
            node_l = inv_node_labels[path[l]]
            node_l_plus_1 = inv_node_labels[path[l+1]]
            
            # Make sure layer is valid for As
            if l >= len(As):
                # print(f"Warning: Layer index {l} out of bounds for As with length {len(As)}")
                return False
                
            # Check dimensions for the adjacency matrix
            if (node_l[1] >= As[l].shape[0] or node_l_plus_1[1] >= As[l].shape[1]):
                # print(f"Warning: Node indices {node_l[1]},{node_l_plus_1[1]} out of bounds for layer {l} with shape {As[l].shape}")
                return False
                
            cc = As[l][node_l[1], node_l_plus_1[1]]
            c += cc
            
        return c == cost
    except Exception as e:
        # print(f"Warning: Error checking path cost: {e}")
        return False

def check_nodes_and_correct_layer_order(path, inv_node_labels):
    """
    Check if a path consists of valid nodes in the correct layer order.
    
    Args:
        path: Path through the graph
        inv_node_labels: Map from flat node indices to (layer, node) tuples
        
    Returns:
        True if the path is valid, False otherwise
    """
    # Check if path is empty
    if not path:
        # print("Warning: Empty path provided for checking")
        return False
        
    ls = len(path)
    ok_nodes = ls > 0
    ok_path = ls > 0
    
    for l in range(ls):
        try:
            # Check if the node index is in the mapping
            if path[l] not in inv_node_labels:
                # print(f"Warning: Path node {path[l]} not found in node labels")
                ok_nodes = False
                ok_path = False
                continue
                
            (ll, k) = inv_node_labels[path[l]]
            ok_path &= (l == ll)
            
            if l != ll:
                # print(f"Warning: Path node {path[l]} at position {l} has incorrect layer {ll}")
                print("", end="")
        except Exception as e:
            # print(f"Warning: Error checking node order: {e}")
            ok_nodes = False
            ok_path = False
            
    return (ok_nodes and ok_path)

def evaluate_A(graph, A, 
               BoS_tokens=True, BoT_tokens=True, aha_token=True, wait_token=True,
               correct_costs=None, sum_table=None, correct_sum_table=None):
    
    ls, ks, As = graph.ls, graph.ks, graph.As
    best_c, best_path = graph.best_c, graph.best_path
    node_labels, inv_node_labels = graph.node_labels, graph.inv_node_labels

    if correct_costs is not None:
        correct_costs_list = [eval(c) for c in correct_costs]
        assert sum_table is not None
        assert correct_sum_table is not None

    # Handle empty input
    if not A or not A.strip():
        # print("Warning: Empty or whitespace-only input")
        return Evaluation_state([1, False, False, False, False, 0, 0, 0, 0, 0, 0, None, None, None, None])

    a_list = A.split(); cot_list = [];
     
    syntax_errors = 0   
    
    if BoT_tokens and a_list:
        # check the correct positioning of BoT/EoT and remove them
        if not a_list or a_list[0] != 'BoT':
            syntax_errors += 1
            # print("Warning: Missing BoT token at beginning")
        else: 
            a_list = a_list[1:]
     
        a_list, errs = check_remove(a_list, 'BoT')
        syntax_errors += errs
        try:
            ind = a_list.index('EoT')
            cot_list = a_list[:ind]
            a_list = a_list[ind+1:]
        except ValueError:
            syntax_errors += 1
            # print("Warning: Missing EoT token")
            # In case of missing EoT, assume everything is part of cot_list
            cot_list = a_list.copy()
            a_list = []
            
        cot_list, errs = check_remove(cot_list, 'BoT')
        syntax_errors += errs
        a_list, errs = check_remove(a_list, 'EoT')
        syntax_errors += errs
            
    if BoS_tokens and a_list:
        # check the presence of and remove the BoS_tokens from A
            
        a_list, errs = check_remove(a_list, 'BoS')
        syntax_errors += errs    
        
        if not a_list or a_list[-1] != 'EoS':
            syntax_errors += 1
            # print("Warning: Missing EoS token at end")
        else: 
            a_list = a_list[:-1] 
            
        a_list, errs = check_remove(a_list, 'EoS')
        syntax_errors += errs
      
    
    ### CHECKING A (a_list) ### (OK)
    if not a_list or a_list[-1] != '|':
        syntax_errors += 1
        # print("Warning: Missing '|' token at end of answer")
    else: 
        a_list = a_list[:-1] 
        
    a_list, errs = check_remove(a_list, '|')
    syntax_errors += errs    
    
    A_path, A_c, errs, _, _ = parse_sub_string(a_list, aha_token=False, wait_token=False)
    syntax_errors += errs
    
    is_A_path_possible = check_nodes_and_correct_layer_order(A_path, inv_node_labels)
    is_A_path_length_correct = (len(A_path) == len(best_path))
    is_A_cost_consistent = check_path_cost(As, A_path, A_c, inv_node_labels) if is_A_path_possible else False
    is_A_cost_optimal = (A_c == best_c[-1][0]) if is_A_path_possible else False 
    
    n_CoT_steps = 0 if BoT_tokens else None
    repeated_CoT_steps = 0 if BoT_tokens else None
    CoT_path_possible = 0 if BoT_tokens else None
    consistent_CoT_steps = 0 if BoT_tokens else None
    sub_prob_optimal_CoT_steps = 0 if BoT_tokens else None 
    CoT_steps_skipped_sub_prob = 0 if BoT_tokens else None 
    correct_aha, missed_aha, tot_aha = 0, 0, 0
    correct_wait, missed_wait, tot_wait = 0, 0, 0
    
    if BoT_tokens and cot_list:
        ### CHECKING CoT (cot_list) ###
        N = np.sum(ks)
        
        best_cs = inf * np.ones(N, dtype=int); best_cs[0] = 0; 
        tried_routes_to_node = [[] for i in range(N)]
        
        seen_paths = []
        
        # using the separators '|', split in sub-strings the CoT
        while cot_list:
            try: 
                # break the list up to the last separator
                try:
                    ind = cot_list.index('|')
                    a_sub = cot_list[:ind]
                except ValueError:
                    # print("Warning: Missing '|' separator in CoT")
                    # Treat the rest as one chunk
                    ind = len(cot_list) - 1
                    a_sub = cot_list
                    syntax_errors += 1
                
                n_CoT_steps += 1
                go_next = False
                if ind != len(cot_list)-1:
                    go_next = True
                    cot_list = cot_list[ind+1:]
                
                # parse and evaluate the identified substring (reasoning step)
                sub_path, sub_c, errs, is_aha, waits = parse_sub_string(a_sub, aha_token=aha_token, wait_token=wait_token)
                tot_aha += is_aha; tot_wait += (waits > 0); 
                syntax_errors += errs

                if correct_costs is not None:
                    correct_sum_table[best_cs[sub_path[-2]], sub_c-best_cs[sub_path[-2]]-1] += correct_costs_list[n_CoT_steps-1]
                    sum_table[best_cs[sub_path[-2]], sub_c-best_cs[sub_path[-2]]-1] += 1
                
                # Check if we have a valid path
                if not sub_path:
                    # print("Warning: Empty path in CoT step")
                    if go_next:
                        continue
                    else:
                        break
                
                end = sub_path[-1] if len(sub_path) > 0 else None
                
                # Make sure end is a valid index
                if end is None or end >= N:
                    # print(f"Warning: Invalid end node index {end}")
                    if go_next:
                        continue
                    else:
                        break
                
                is_path_possible = check_nodes_and_correct_layer_order(sub_path, inv_node_labels)
                if is_path_possible:  
                    CoT_path_possible += 1
                    
                    # Make sure sub_path has at least 2 elements before accessing [-2]
                    if len(sub_path) >= 2:
                        if sub_path[-2] not in tried_routes_to_node[end]:
                            tried_routes_to_node[end].append(sub_path[-2])
                    
                    if sub_path in seen_paths:
                        repeated_CoT_steps += 1
                    else:
                        seen_paths.append(sub_path)
                    
                    better_cost = sub_c < best_cs[end]
                    # update current estimate of best cost to this end node
                    if better_cost:
                        best_cs[end] = sub_c
                        if wait_token:
                            try:
                                l_end, k_end = inv_node_labels[end]
                                if l_end != ls and l_end < len(As): 
                                    correct_waits = 0
                                    for k in range(As[l_end].shape[1]):
                                        if As[l_end][k_end, k] == inf: 
                                            continue 
                                        try:
                                            dest = node_labels.get((l_end+1, k))
                                            if dest is not None and end in tried_routes_to_node[dest]:
                                                correct_waits += 1
                                        except (KeyError, IndexError) as e:
                                            # print(f"Warning: Node label lookup error: {e}")
                                            print("", end="")
                                    correct_wait += ((waits > 0) & (correct_waits == waits))
                                    missed_wait += (waits == 0) * correct_waits
                            except (KeyError, IndexError) as e:
                                # print(f"Warning: Invalid node index in inv_node_labels: {e}")
                                print("", end="")
                            
                    if aha_token:
                        # check if aha was correct
                        correct_aha += (is_aha & better_cost)
                        missed_aha += (~is_aha & better_cost)
        
                    # check if the cost computation makes sense
                    consistent_CoT_steps += check_path_cost(As, sub_path, sub_c, inv_node_labels)
                    
                    # check if the current optimal paths to the intermediate nodes are 
                    # considered (otherwise this step is avoidable!)
                    are_sub_paths_current_optimal = True
                    for l in range(1, len(sub_path)-1):
                        try:
                            sub_end = sub_path[l]
                            if sub_end >= N:
                                are_sub_paths_current_optimal = False
                                break
                                
                            current_opt_c = best_cs[sub_end]
                            if best_cs[sub_end] == inf:
                                CoT_steps_skipped_sub_prob += 1
                                are_sub_paths_current_optimal = False
                                break
                            else:
                                are_sub_paths_current_optimal &= check_path_cost(As, sub_path[:l+1], current_opt_c, inv_node_labels)
                        except (IndexError, KeyError) as e:
                            # print(f"Warning: Error checking sub-path optimality: {e}")
                            are_sub_paths_current_optimal = False
                            break
                                
                    sub_prob_optimal_CoT_steps += are_sub_paths_current_optimal
                
                if go_next:
                    # continue if there is more stuff after the separator
                    continue
                else:
                    # stop if this is the last sub-string
                    break
            except Exception as e: 
                # print(f"Warning: Error processing CoT step: {e}")
                syntax_errors += 1
                break
    
    # Calculate metrics only with positive denominators
    frac_correct_aha = (correct_aha / tot_aha) if (aha_token and (tot_aha > 0)) else None
    frac_correct_wait = (correct_wait / tot_wait) if (wait_token and (tot_wait > 0)) else None
    missed_aha = missed_aha if aha_token else None
    missed_wait = missed_wait if wait_token else None
    
    # Calculate the new is_A_path_correct metric
    is_A_path_correct = is_A_cost_optimal and is_A_cost_consistent and is_A_path_length_correct
    
    ev = Evaluation_state([syntax_errors, is_A_path_possible, is_A_cost_consistent, is_A_cost_optimal, is_A_path_length_correct, 
                          n_CoT_steps, repeated_CoT_steps, CoT_path_possible, consistent_CoT_steps, sub_prob_optimal_CoT_steps, CoT_steps_skipped_sub_prob,
                          frac_correct_aha, missed_aha, frac_correct_wait, missed_wait, ls, is_A_path_correct])
    
    return ev