import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import math as math
import numpy as np

from graph import *

#test_cold_item_item_ids = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\test_cold_item_item_ids.csv")
#print(test_cold_item_ids) # 49975 rows
#print(len(test_cold_item_ids["ids"].unique().tolist())) # 49975 target items

#test_cold_user_item_ids = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\test_cold_user_item_ids.csv")
#print(user_ids) # 42153 rows
#print(len(user_ids["ids"].unique().tolist())) # 42153 users

#test_cold_item = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\test_cold_item.csv")
#print(test_cold_item) # 199028 entries
#print(len(test_cold_item["user"].unique().tolist())) # 85343 users
#print(len(test_cold_item["item"].unique().tolist())) # 49975 items
#print(test_cold_item["interaction"].unique().tolist()) # interactions = {1, 2, 3, 5}

#test_cold_user = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\test_cold_user.csv")
#print(test_cold_user) # 169479 entries
#print(len(test_cold_user["user"].unique().tolist())) # 47755 users
#print(len(test_cold_user["item"].unique().tolist())) # 42153 items
#print(test_cold_user["interaction"].unique().tolist()) # interactions = {1, 2, 3, 5}


#test_warm_item_ids = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\test_warm_item_ids.csv")
#print(test_warm_item_ids) # 62443 rows

train_set = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\train.csv")
#print(train_set) # 19M
#print(len(train_set["user"].unique().tolist())) # 1M users
#print(len(train_set["item"].unique().tolist())) # 0.5M items
#print(train_set["interaction"].unique().tolist()) # interactions = {0, 1, 2, 3, 5}

interactions = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\interactions.csv")
interactions_not5 = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\interactions_not5.csv")
interactions1235 = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\data\interactions1235.csv")
interactions123 = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\data\interactions123.csv")



def get_duplicates():
    duplicates = train_set[train_set.duplicated(subset=['user', 'item'], keep=False)]
    print("All duplicate rows:")
    print(duplicates)

    only_duplicates = train_set[train_set.duplicated(subset=['user', 'item'], keep='first')]
    print("\nDuplicate rows (excluding first occurrences):")
    print(only_duplicates)


def plot_applications_user(train_set):

    users_apply = train_set

    # Count the occurrences of each number
    number_counts = Counter(users_apply["user"])


    # Count how many times each count appears
    occurrence_counts = Counter(number_counts.values())

    print("Nº of applications: ",sum([a for a in number_counts.values()]))
    print("Nº of users: ", len(number_counts.keys()))
    # Plotting the data
    plt.figure(figsize=(10, 5))
    plt.scatter(occurrence_counts.keys(), occurrence_counts.values(), color='green')

    
    
    max_val = np.max(list(occurrence_counts.values()))
    min_val = np.min(list(occurrence_counts.values()))

    max_interaction = np.max(list(number_counts.values()))
    print("The user with more interaction: ",[(k,v) for k,v in 
        number_counts.items() if v == max_interaction]) 
    
    print("The average of interactions per users is:",np.mean(list(number_counts.values())))
    print("Nº of interactions from users: ", train_set.shape)

    plt.axhline(max_val, color='blue', linestyle='--', linewidth=1, label=f'Max: {max_val}')
    plt.axhline(min_val, color='purple', linestyle='--', linewidth=1, label=f'Min: {min_val}')

    plt.text(140, 17660, f'Max: {max_val}', color='blue', va='center')
    plt.text(140, 12620, f'Min: {min_val}', color='purple', va='center')


    # Adding title and labels
    plt.title('Interactions per user')
    plt.xlabel('Nº of interactions')
    plt.ylabel('Nº of users')

    # Display the plot
    plt.show()

def plot_users_companies():

    companies_click = train_set[(train_set['interaction'] == 5)]
    
    # Count the occurrences of each number
    number_counts = Counter(companies_click["item"])

    

    # Count how many times each count appears
    occurrence_counts = Counter(number_counts.values())
    
    print("Nº of clicks from companies: ", companies_click.shape)

    max_interest = np.max(list(number_counts.values()))
    print("The company with interested in more users: ",[(k,v) for k,v in 
        number_counts.items() if v == max_interest]) 
    print("The average of interest for users is:",np.mean(list(number_counts.values())))
    
    

    # Plotting the data
    plt.figure(figsize=(10, 5))
    bars = plt.bar(occurrence_counts.keys(), occurrence_counts.values(), color='skyblue')

    

    # Extract the positions (x-coordinates) and heights (y-coordinates) of the bars
    coords = sorted([(bar.get_x() + bar.get_width() / 2, bar.get_height())for bar in bars], key = lambda x: x[1])
    x_coords = [a for a,_ in coords]
    y_coords = [b for _,b in coords]

    
    median_val = np.median(list(occurrence_counts.values()))
    max_val = np.max(list(occurrence_counts.values()))
    min_val = np.min(list(occurrence_counts.values()))

    

    # Plot mean, median, max, and min lines
    plt.axhline(max_val, color='blue', linestyle='--', linewidth=1, label=f'Max: {max_val}')
    plt.axhline(min_val, color='purple', linestyle='--', linewidth=1, label=f'Min: {min_val}')


    
    plt.text(len(occurrence_counts.keys())-0.5, 660, f'Max: {max_val}', color='blue', va='center')
    plt.text(len(occurrence_counts.keys())-0.5, 620, f'Min: {min_val}', color='orange', va='center')
    # Plot the points
    plt.scatter(x_coords, y_coords, color='red', zorder=5)  # zorder to bring points on top
    plt.plot(x_coords, y_coords, color='red', linestyle='-', marker='o', linewidth=2)
# Plot the lines connecting the points

    # Adding title and labels
    plt.title('Companies interested in n users')
    plt.xlabel('Nº of users')
    plt.ylabel('Nº of companies')

    
    # Display the plot
    plt.show()

def plot_match():

    companies_click = train_set[(train_set['interaction'] == 5)]
    users_applied = train_set[(train_set['interaction'] == 3)]
    couples = []
    result = []
    print("Size of companies_click: ", len(companies_click))
    print("Size of users_applied: ", len(users_applied))
    for index, row in companies_click.iterrows():
        couples.append((row['user'], row['item']))
    
    print("Size of couples: ", len(couples))
    for index, row in users_applied.iterrows():
        if (row["user"], row['item']) in couples:
            result.append((row["user"], row['item']))
    print("Size of result: ", len(result))
    print(result)


def get_interactions(train_set):
    companies_click = train_set[(train_set['interaction'] == 5)]
    no_interaction = train_set[(train_set['interaction'] == 0)]

    couples = []
    result = defaultdict(set)
    print("Size of companies_click: ", len(companies_click))
    print("Size of no interactions: ", len(no_interaction))
    
    interested_items = set()
    interesting_users = set()
    for index, row in companies_click.iterrows():
        couples.append((row['user'], row['item']))
        interested_items.add(row["item"])
        interesting_users.add(row["user"])

    users_applied = train_set[((train_set['interaction'] == 3) | 
                              (train_set["interaction"] == 2) | 
                              (train_set["interaction"] == 1))
                              & (train_set["user"].isin(interesting_users))
                              ]
    
    print("Number of interactions from users who are interesting:", len(users_applied))
    count, count2 = 0,0
    users_in_match = set()
    interactions_not5 = users_applied
    for i, row in users_applied.iterrows():
        if (row["user"], row["item"]) not in couples: # A
            count+=1
            users_applied = users_applied.drop(index=i)
        else:
            if row["user"] in interesting_users:
                count2+=1
                users_in_match.add(row["user"])
                # We don't want the match rows
                interactions_not5 = interactions_not5.drop(index = i)
    
    print("Size of not5 then:", interactions_not5.shape)
    for i, row in interactions_not5.iterrows():
         if row["user"] not in users_in_match:
             interactions_not5 = interactions_not5.drop(index = i)
    print("Size of not5 now:", interactions_not5.shape)

    print("Count: ", count)
    print("Count 2:", count2)
    print("Users in match: ", len(users_in_match))
    print("Size of couples: ", len(couples))
    print("Size of interested_items: ", len(interested_items))
    print("Size of interesting users: ", len(interesting_users))
    print("Size of matches: ", len(users_applied))
    users_applied.to_csv('interactions.csv', index=False)
    interactions_not5.to_csv('interactions_not5.csv', index=False)

def get_interactions1235(train_set):
    interactions1235 = train_set[(train_set['interaction'] != 0)]
    interactions1235.to_csv("warm/data/interactions1235.csv", index = False)

def get_interactions123(interactions1235):
    interactions123 = interactions1235[(interactions1235["interaction"]!=5)]
    interactions123.to_csv("warm/data/interactions123.csv", index = False)

    

def do_graph(interactions, interactions_not5, n):
    g = Graph()
    vertices = 0
    users = 0
    for index, row in interactions.iterrows():
        user = Vertex(row["user"], "user")
        item = Vertex(row["item"], "item")
        if g.has_vertex(user):
            if g.has_vertex(item):
                if g.are_connected(user,item):
                    pass
                else:
                   g.add_edge(Edge(user, item,0))
                   g.add_edge(Edge(item, user, 0)) 
            else:
                g.add_vertex(item)
                vertices +=1
                g.add_edge(Edge(user, item,0))
                g.add_edge(Edge(item, user, 0))
        else:
            g.add_vertex(user)
            vertices +=1
            users +=1
            if g.has_vertex(item):
                g.add_edge(Edge(user, item,0))
                g.add_edge(Edge(item, user, 0))
            else:
                g.add_vertex(item)
                vertices+=1
                g.add_edge(Edge(user, item,0))
                g.add_edge(Edge(item, user, 0))

    print("Number of users here: ", users)
    for index, row in interactions_not5.iterrows():
        if vertices >= n:
            break
        else:
            user = Vertex(row["user"], "user")
            item = Vertex(row["item"], "item")
            if g.has_vertex(user):
                if g.has_vertex(item):
                    if g.are_connected(user,item):
                        pass
                    else:
                        g.add_edge(Edge(user, item,0))
                else:
                    g.add_vertex(item)
                    vertices+=1
                    g.add_edge(Edge(user, item,0))
            else:
                print("hola")
                pass

    return g

def do_graph1235(interactions1235):
    g = Graph()
    vertices = 0
    users = 0
    interactions5 = 0
    for index, row in interactions1235.iterrows():
        user = Vertex(row["user"], "user")
        item = Vertex(row["item"], "item")
        if row["interaction"] == 5:
            g = connect_vertices(g, item, user)
            interactions5 +=1
        else:
            g = connect_vertices(g, user, item)

    print("Number of vertices", g.total_vertices())
    print("Number of users", g.num_vertices("user"))
    print("Number of items", g.num_vertices("item"))
    print("Number of 5", interactions5)


    print("Testing number of users", 
          g.num_vertices("user") == len(interactions1235["user"].unique().tolist()))
    print("Testing number of items", 
          g.num_vertices("item") == len(interactions1235["item"].unique().tolist()))

    return g

def connect_vertices(g, v1, v2):
    if g.has_vertex(v1):
            if g.has_vertex(v2):
                if g.are_connected(v1,v2):
                    print("Hola")
                else:
                   g.add_edge(Edge(v1, v2,0)) 
            else:
                g.add_vertex(v2)
                g.add_edge(Edge(v1, v2,0))
    else:
        g.add_vertex(v1)
        if g.has_vertex(v2):
            g.add_edge(Edge(v1, v2,0))
        else:
            g.add_vertex(v2)
            g.add_edge(Edge(v1, v2,0))
    return g

def compute_similarity(list1, list2):
    # Convert lists to sets to remove duplicates
    set1 = set(list1)
    set2 = set(list2)
    
    # Compute the intersection of the sets (common elements)
    common_elements = set1.intersection(set2)
    if (list1 == None) | (list2 == None):
        print("Empty list")
    # Return the size of the intersection
    return len(common_elements)

def do_users_graph(g):
    g2 = Graph()
    for vertex, adjacent_vertices in g.graph.items():
        if vertex.type != "item":
            for v_aux, adj_aux in g.graph.items():
                if vertex != v_aux and v_aux.type != "item":
                    print("Main vertex:", vertex)
                    print("Other vertex:", v_aux)
                    print("Main adj:", g.vertices[vertex])
                    print("Other adj:", g.vertices[v_aux],"\n")


                    if g2.has_vertex(vertex):
                        if g2.has_vertex(v_aux):
                            if g2.are_connected(vertex,v_aux):
                                similarity = compute_similarity(
                                    g.graph[vertex], g.graph[v_aux])
                                g2.add_edge(Edge(vertex, v_aux,similarity))
                        else:
                            g2.add_vertex(v_aux)
                            similarity = compute_similarity(
                                g.graph[vertex], g.graph[v_aux])
                            g2.add_edge(Edge(vertex, v_aux,similarity))
                            
                    else:
                        g2.add_vertex(vertex)
                        if g2.has_vertex(v_aux):
                            similarity = compute_similarity(
                                g.graph[vertex], g.graph[v_aux])
                            g2.add_edge(Edge(vertex, v_aux,similarity))
                        else:
                            g2.add_vertex(v_aux)
                            similarity = compute_similarity(
                                g.graph[vertex], g.graph[v_aux])
                            g2.add_edge(Edge(vertex, v_aux,similarity))
        
    g2_display = g2.display()
    with open("display_g2.txt", "w") as file:
    # Write the string variable to the file
        file.write(g2_display)
    return g2
                

def draw_graph(interactions, interactions_not5, n):
    g = do_graph(interactions, interactions_not5, n)
    dot_representation = g.get_dot_representation()
    count = set()
    interactions_n = 0
    for vertex, adjacent_vertices in g.graph.items():
        if vertex.type == "user":
            interactions_n += len(adjacent_vertices)
            count.add(vertex)
    print("Number of users:", len(count))
    print("Number of interactions from user with a match: ", interactions_n)
    # Open a file in write mode
    """ with open("g_display.txt", "w") as file:
    # Write the string variable to the file
        file.write(g_display) """
    # Open a file in write mode
    with open("graph.txt", "w") as file:
    # Write the string variable to the file
        file.write(dot_representation)

def draw_graph1235(i):
    g = do_graph1235(i)
    dot_representation = g.get_dot_representation()
    # Open a file in write mode
    with open("warm/data/graph1235.txt", "w") as file:
    # Write the string variable to the file
        file.write(dot_representation)

def coefficient_similarity(interactions):
    g = do_graph(interactions)
    g2 = do_users_graph(g)
    g2.display()

def stats_graph(interactions, interactions_not5, n = 10000):
    g = do_graph(interactions, interactions_not5, n)
    users_ls = []
    match_ls = []
    gender_ls = []
    genders1 = ["Male", "Male", "Female"]
    genders0 = ["Male", "Female", "Female"]
    for vertex, adjacent_vertices in g.graph.items():
        if vertex.type == "user":
            for adj_vertex, weight in adjacent_vertices:
                users_ls.append(vertex.id)
                if vertex.id%2 == 0:
                    gender_ls.append("Male")
                else:
                    gender_ls.append("Female")
                if g.are_connected(vertex, adj_vertex):
                    match_ls.append(1)
                else:
                    match_ls.append(0)
    data = {
        "Users": users_ls,
        "Match": match_ls,
        "Sex": gender_ls
    }

    sex_df = pd.DataFrame(data)
    return sex_df


def get_dict_matches(interactions1235):
    g = do_graph1235(interactions1235)
    notebook = defaultdict(int)

    """  for v1, adj1 in g.graph.items():
        if v1.type=="user":
            notebook[int(v1.id)]=0
            for v2, weight in adj1:
                if(v2.type == "item") & (v1 in g.vertices[v2]):
                    notebook[int(v1.id)]+=1
        else:
           for v2, weight in adj1: """

    for v1, adj1 in g.graph.items():
        if v1.type=="user":
            if int(v1.id) in notebook.keys():
                pass
            else:
                notebook[int(v1.id)]=0
        else:
           for v2, weight in adj1:
                    if int(v2.id) in notebook.keys():
                        notebook[int(v2.id)]+=1
                    else:
                        notebook[int(v2.id)]=1

    
             
        

               

    notebook = dict(sorted(notebook.items(), key=lambda item: item[1], reverse=True))
    return notebook
        
def get_user_item_matrix(interactions123, n = 3):
    replies = interactions123[(interactions123['interaction'] == n)]
    item_ids = replies['item'].unique().tolist()
    user_ids = replies['user'].unique().tolist()
    data = []

    for u in user_ids:
        user_preferences = []
        for i in item_ids:
            filtered_df = replies[(replies['user'] == u) & (replies['item'] == i)]
            if filtered_df.size == 0:
                user_preferences.append(0)
            else:
                user_preferences.append(1)
        data.append(user_preferences)

    df = pd.DataFrame(data, index=user_ids,columns= item_ids)

    return df


                       




    


#get_duplicates()
#plot_applications_user(train_set[(train_set['interaction'] == 3) | (train_set['interaction'] == 2) | (train_set['interaction'] == 1)])
#plot_users_companies()
#plot_match()

#get_interactions(train_set)
#get_interactions1235(train_set)
#get_interactions123(interactions1235)

#do_graph(interactions)
#draw_graph(interactions,interactions_not5, 100000)
#do_users_graph(do_graph(interactions[:100]))
#do_graph1235(interactions1235)
#draw_graph1235(interactions1235[:1000000])

#coefficient_similarity(interactions)
#stats_graph(interactions, interactions_not5,)
#funct_1(interactions, interactions_not5,)
#get_dict_matches(interactions1235)

print(get_user_item_matrix(interactions123, 3))


