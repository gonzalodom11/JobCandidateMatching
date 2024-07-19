class Graph:
    def __init__(self):
        self.graph = {}
        self.vertices = {}
        self.matches = 0
        

    def add_vertex(self, vertex):
        if not isinstance(vertex, Vertex):
            raise TypeError("vertex must be an instance of Vertex")
        if vertex not in self.vertices:
            self.vertices[vertex] = []
            self.graph[vertex] = []

    def add_edge(self, edge):
        if isinstance(edge, Edge):
            vertex1 = edge.vertex1
            vertex2 = edge.vertex2
            weight = edge.weight
            if vertex1 in self.graph and vertex2 in self.graph:
                self.graph[vertex1].append((vertex2, weight))
                self.vertices[vertex1].append(vertex2)
                if self.are_connected(vertex1, vertex2):
                    self.matches+=1
            else:
                raise ValueError("Both vertices must be in the graph")
        else:
            raise TypeError("edge must be an instance of Edge")
        

    def has_vertex(self, vertex):
        return (vertex in self.vertices) or (vertex in self.graph)

    def are_connected(self, vertex1, vertex2):
        if vertex1 in self.vertices and vertex2 in self.vertices:
            return (vertex2 in self.vertices[vertex1]) and (vertex1 in self.vertices[vertex2])
        else:
            raise ValueError("Both vertices must be in the graph")

    def total_vertices(self):
        return len(self.vertices)
    
    def num_vertices(self, vertex_type):
        return len([vertex for vertex, _ in self.graph.items() 
            if vertex.type == vertex_type])
        
    def display(self):
        dot = ""
        for vertex, adjacent_vertices in self.graph.items():
            #print(f"{vertex}:")
            dot += "\n"
            dot += f"{vertex}:"
            for adj_vertex, weight in adjacent_vertices:
                #print(f"  {adj_vertex} (Weight: {weight})")
                dot+=f"  {adj_vertex} (Weight: {weight})"
        return dot
    def get_dot_representation(self):
        dot = 'digraph G {\n'
        
        # Define attributes for user and item vertices
        user_attributes = 'color=blue'
        item_attributes = 'color=orange, penwidth=2'
        
        # Define layout attributes
        layout_attributes = 'layout=neato'
        count = 0
        # Add layout attribute
        dot += f'    {layout_attributes};\n'
        
        # Add node definitions
        for vertex in self.vertices:
            # Choose attributes based on vertex type
            if vertex.type == "user":
                attributes = user_attributes
            elif vertex.type == "item":
                attributes = item_attributes
            else:
                attributes = ''  # Default attributes
            
            # Add node definition with attributes
            dot += f'    "{vertex.id}_{vertex.type}" [label="{vertex.id} ({vertex.type})", {attributes}];\n'
        
        # Add edge definitions with weights
        for vertex, adjacent_vertices in self.graph.items():
            for adj_vertex, weight in adjacent_vertices:
                if self.are_connected(vertex, adj_vertex):
                    if vertex.type == "user":
                        count+=1
                        edge_attributes = 'color=green, penwidth=3, dir = both'
                        dot += f'    "{vertex.id}_{vertex.type}" -> "{adj_vertex.id}_{adj_vertex.type}" [{edge_attributes}];\n'
                else:
                    edge_attributes = 'color=black, penwidth=1'
                    dot += f'    "{vertex.id}_{vertex.type}" -> "{adj_vertex.id}_{adj_vertex.type}" [{edge_attributes}];\n'    
        
        dot += '}'
        print("Matches:", count)
        return dot
    
    # Original dot representation
    def get_dot_users_representation(self):
        dot = 'digraph G {\n'
        
        # Define attributes for user and item vertices
        user_attributes = 'color=blue'
        item_attributes = 'color=orange, penwidth=2'
        
        # Define layout attributes
        layout_attributes = 'layout=neato'
        
        # Add layout attribute
        dot += f'    {layout_attributes};\n'
        
        # Add node definitions
        for vertex in self.vertices:
            # Choose attributes based on vertex type
            if vertex.type == "user":
                attributes = user_attributes
            elif vertex.type == "item":
                attributes = item_attributes
            else:
                attributes = ''  # Default attributes
            
            # Add node definition with attributes
            dot += f'    "{vertex.id}_{vertex.type}" [label="{vertex.id} ({vertex.type})", {attributes}];\n'
        
        # Add edge definitions with weights
        for vertex, adjacent_vertices in self.graph.items():
            for adj_vertex, weight in adjacent_vertices:
                dot += f'    "{vertex.id}_{vertex.type}" -> "{adj_vertex.id}_{adj_vertex.type}"\n'
        
        dot += '}'
        return dot


class Vertex:
    def __init__(self, vertex_id, vertex_type):
        if vertex_type not in {"user", "item"}:
            raise ValueError("vertex_type must be 'user' or 'item'")
        self.id = vertex_id
        self.type = vertex_type

    def __repr__(self):
        return f"Vertex(id={self.id}, type='{self.type}')"

    def __eq__(self, other):
        return self.id == other.id and self.type == other.type

    def __hash__(self):
        return hash((self.id, self.type))

class Edge:
    def __init__(self, vertex1, vertex2, weight=0):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.weight = weight

