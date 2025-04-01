import tkinter as tk
from tkinter import ttk
import random
import matplotlib.animation
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import heapq
import matplotlib.animation as animation
import matplotlib.patches as patches
import time
from matplotlib.animation import FuncAnimation
import json
from tkinter import filedialog

class GrilleApp:

    # Initiating valriables
    def __init__(self, root):
        self.root = root
        self.root.title("Grid generator/finding shortest path")

        self.size = tk.IntVar(value=10) # Grid dimension
        self.obstacle_reset = True # To reset obstacles when you click on "reset grid"
        # Obstacle configuration
        self.obstacle_choice = None 
        self.obstacle_ratio = tk.DoubleVar(value=0.2) 
        self.obstacle_number = tk.IntVar(value=30) 

        self.grid_number = tk.IntVar(value=1) # Number of graph generated
        # Selection of modes
        self.selection_obstacle_mode = tk.StringVar(value="start")
        self.selection_shortest_path_mode = tk.StringVar(value="Dijkstra") # Shortest path mode selection
        self.selection_diagonal_mode = tk.StringVar(value="nondiagonal") # Diagonal or nondiagonal mode selection

        self.obstacles = set()
        # Speed of the animation
        self.speed_var = tk.DoubleVar(value=100) 
        self.speed_slider = tk.Scale(root, from_=10, to=500, resolution=10, orient="horizontal", label="Speed", variable=self.speed_var)
        self.speed_slider.pack()
        # Start and target nodes
        self.start = None
        self.target = None
        # Coloring of the gris for the animation
        self.path_patches = []
        self.confetti_patches = []
        self.confetti_velocities = []
        # Animation time
        self.ani_start_time = None
        # UI
        self.setup_ui()
        self.update_grille()
    
    # Display of the interface
    def setup_ui(self):
        # Principal frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(side="left", padx=10, pady=10, fill="y")

        # Size of the grid
        ttk.Label(control_frame, text="Size of the grid:").pack()
        ttk.Entry(control_frame, textvariable=self.size, width=5).pack()

        # Frame to show or hide obstacle mode
        self.obstacle_frame = tk.Frame(control_frame)
        self.obstacle_button = ttk.Button(control_frame, text="Obstacles options", command=self.toggle_obstacle_frame)
        self.obstacle_button.pack(pady=5)
        self.obstacle_frame.pack_forget() 

        # Obstacles section
        self.obstacle_mode = tk.StringVar(value="ratio") 
        ttk.Radiobutton(self.obstacle_frame, text="Obstacle ratio", variable=self.obstacle_mode, value="ratio", command=self.update_obstacle_inputs).pack()
        ttk.Radiobutton(self.obstacle_frame, text="Obstacle number", variable=self.obstacle_mode, value="number", command=self.update_obstacle_inputs).pack()
        
        # Ratio
        self.ratio_entry = ttk.Entry(self.obstacle_frame, textvariable=self.obstacle_ratio, width=5)
        ttk.Label(self.obstacle_frame, text="Obstacle ratio:").pack()
        self.ratio_entry.pack()
        
        # Number
        self.number_entry = ttk.Entry(self.obstacle_frame, textvariable=self.obstacle_number, width=5)
        ttk.Label(self.obstacle_frame, text="Obstacle number:").pack()
        self.number_entry.pack()

        # Generate grid
        ttk.Button(control_frame, text="Generate grid", command=self.update_grille).pack(pady=5)
        
        # Shortest path section
        ttk.Label(control_frame, text="Shortest path selection:").pack()
        ttk.Radiobutton(control_frame, text="Dijkstra", variable=self.selection_shortest_path_mode, value="Dijkstra").pack()
        ttk.Radiobutton(control_frame, text="A*", variable=self.selection_shortest_path_mode, value="Astar").pack()

        # Can go diagonal or not
        ttk.Radiobutton(control_frame, text="Non-diagonal", variable=self.selection_diagonal_mode, value="nondiagonal").pack()
        ttk.Radiobutton(control_frame, text="Diagonal", variable=self.selection_diagonal_mode, value="diagonal").pack()

        ttk.Button(control_frame, text="Launch shortest path", command=self.run_shortest_path).pack(pady=5)

        # Choose the start and the target nodes
        ttk.Label(control_frame, text="Start and target nodes:").pack()
        ttk.Radiobutton(control_frame, text="Start", variable=self.selection_obstacle_mode, value="start").pack()
        ttk.Radiobutton(control_frame, text="Target", variable=self.selection_obstacle_mode, value="target").pack()
        
        # Reset grid with the same obstacles
        ttk.Button(control_frame, text="Resest grid", command=self.reset_graph).pack(pady=5)
        
        # Bouton pour afficher/cacher les options de sauvegarde
        self.save_options_frame = tk.Frame(control_frame)
        self.save_options_button = ttk.Button(control_frame, text="Save", command=self.open_save_window)
        self.save_options_button.pack(pady=5)
        self.save_options_frame.pack_forget()  # Cacher la frame au départ

        # Number of grid to save
        ttk.Label(self.save_options_frame, text="Number of grid:").pack()
        self.number_grid_entry = ttk.Entry(self.save_options_frame, textvariable=self.grid_number, width=5)
        self.number_grid_entry.pack()

        # confirm the save
        self.confirm_save_button = ttk.Button(self.save_options_frame, text="confirm save", command=self.save_grid)
        self.confirm_save_button.pack()
    

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
    
    # To save multiple grids
    def open_save_window(self):
        # Create a new window
        save_window = tk.Toplevel(self.root)
        save_window.title("Save")

        # Field to enter the number of grids saved
        ttk.Label(save_window, text="Number of grid:").pack(pady=5)
        grid_number_entry = ttk.Entry(save_window, textvariable=self.grid_number, width=5)
        grid_number_entry.pack(pady=5)

        # Cancel and save buttons
        button_frame = ttk.Frame(save_window)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Cancel", command=save_window.destroy).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save", command=lambda: self.save_grid(save_window)).pack(side="left", padx=5)
    
    # Toggle to show/hide save 
    def toggle_save_options(self):
        if self.save_options_frame.winfo_ismapped():
            self.save_options_frame.pack_forget()
        else:
            self.save_options_frame.pack(pady=5)
    
    # Toggle to show/hide obstacle options
    def toggle_obstacle_frame(self):
        if self.obstacle_frame.winfo_ismapped():
            self.obstacle_frame.pack_forget()
        else:
            self.obstacle_frame.pack(pady=5)

    # Reset of the grid with the same obstacles
    def reset_graph(self):
        self.start = None # So that start and target nodes can be reseted
        self.target = None
        self.path_patches.clear() # clear all colored rectangles
        self.obstacle_reset = False # So that the obstacles dont reset
        self.update_target_start() # reset start and target nodes
        self.ax.clear()
        self.update_grille() # Reset the grid 
        self.canvas.draw() # Display the reseted grid
        self.ani_start_time = None

    # Update obstacle inputs
    def update_obstacle_inputs(self):
        # If ratio is activated, disables number
        if self.obstacle_mode.get() == "ratio":
            self.ratio_entry.config(state="normal")
            self.number_entry.config(state="disabled")
        # If number is activated, disables ratio
        elif self.obstacle_mode.get() == "number":
            self.ratio_entry.config(state="disabled")
            self.number_entry.config(state="normal")
        else:
            self.ratio_entry.config(state="disabled")
            self.number_entry.config(state="disabled")

    
    # Function for user to click on the chosen start and target nodes
    def on_click(self, event):
        # If the click is out of the graph (on the sides or on an obstacle), does nothing
        if event.xdata is None or event.ydata is None:
            return
        row, col = int(event.ydata), int(event.xdata) # get the coordinates of the click
        if (row, col) not in self.G.nodes or (row, col) in self.obstacles:
            return
        # Start node
        if self.selection_obstacle_mode.get() == "start":
            self.start = (row, col)
        # Target node
        elif self.selection_obstacle_mode.get() == "target":
            self.target = (row, col)

        self.update_target_start()


    # Generate the grid
    def run_generer_grille(self):
        self.grid, self.G = generer_grille(size=self.size.get(), obstacle_mode=self.obstacle_mode.get(), 
                                        obstacle_ratio=self.obstacle_ratio.get(), 
                                        obstacle_number=self.obstacle_number.get())
        return self.grid, self.G

    # To save multiple graphs
    def save_grid(self, save_window):
        # Create a file JSON
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")])
        if filename:
            grid_list = [] # List where grids will be stocked in
            for _ in range(self.grid_number.get()):
                grid, _ = self.generer_grille()
                grid_data = grid.tolist()  # Convert the grid in a list
                grid_list.append(grid_data) # Put the list in the grid list 

            with open(filename, 'w') as f:
                json.dump(grid_list, f)

            print(f"{len(grid_list)} grids saved as : {filename}")

        # Close the save window after saving
        save_window.destroy()

    # Update the display of the grid
    def update_grille(self):
        self.ax.clear()
        self.grid, self.G = self.run_generer_grille()
        n = self.grid.shape[0]
        self.obstacle_reset = True
        self.ax.imshow(self.grid, cmap="Greys", origin="upper")
        self.ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        self.ax.grid(which="minor", color="black", linewidth=0.5)
        self.ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        self.canvas.draw()


    # Update on the display of current start and target nodes
    def update_target_start(self):
        if hasattr(self, 'start_node') and self.start_node in self.ax.patches:
            self.start_node.remove()
        if hasattr(self, 'target_node') and self.target_node in self.ax.patches:
            self.target_node.remove()

       # Start node
        if self.start:
            self.start_node = self.ax.add_patch(patches.Rectangle(
                (self.start[1] - 0.5, self.start[0] - 0.5), 1, 1, facecolor="green", alpha=0.8))

        # Target node
        if self.target:
            self.target_node = self.ax.add_patch(patches.Rectangle(
                (self.target[1] - 0.5, self.target[0] - 0.5), 1, 1, facecolor="yellow", alpha=0.8))

        
        self.canvas.draw()


    # Function to run the selected shortest path algorithm
    def run_shortest_path(self):
        selected_algorithm = self.selection_shortest_path_mode.get()

        if selected_algorithm == "Dijkstra":
            print('dijkstra')
            self.run_dijkstra()
        elif selected_algorithm == "Astar":
            print("Astar")
            self.run_Astar()
        else:
            print('marche pas')


    # Function to run dijkstra if selected
    def run_dijkstra(self):
        print('dijkstra')
        if self.start is None or self.target is None:
            return
        self.evaluated_nodes, self.path_history = dijkstra_stepwise(G=self.G, start=self.start, target=self.target, diagonal_mode=self.selection_diagonal_mode.get())
        if self.evaluated_nodes is None:
            print("⚠️ Aucun chemin trouvé entre le point de départ et l'arrivée.")
        else:
            self.ani = FuncAnimation(self.fig, self.update_animation, frames=len(self.evaluated_nodes), 
                         interval=self.speed_var.get(), repeat=False)
            self.canvas.draw()

    # Function to run Astar if selected
    def run_Astar(self):
        print('Astar')
        if self.start is None or self.target is None:
            return
        self.evaluated_nodes, self.path_history = astar_stepwise(G=self.G, start=self.start, target=self.target, diagonal_mode=self.selection_diagonal_mode.get())
        self.ani = FuncAnimation(self.fig, self.update_animation, frames=len(self.evaluated_nodes), 
                         interval=self.speed_var.get(), repeat=False)
        self.canvas.draw()

    # Animation that recreates the exact steps of the shortest path algorithm
    def update_animation(self, frame):
        if self.ani_start_time is None: 
            self.ani_start_time = time.time()  
            print("Animation started...")

        if frame < len(self.evaluated_nodes) - 1:
            node = self.evaluated_nodes[frame]
            rect = patches.Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, facecolor="blue", alpha=0.6)
            self.ax.add_patch(rect)

        if frame < len(self.path_history):
            if hasattr(self, 'path_patches'):
                for patch in self.path_patches:
                    patch.remove()
                self.path_patches.clear() 

        if frame < len(self.path_history):
            self.path_patches = []
            for node in self.path_history[frame]:
                if node == self.target:
                    rect = patches.Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, facecolor="yellow", alpha=0.8)
                else:
                    rect = patches.Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, facecolor="red", alpha=0.8)
                    self.ax.add_patch(rect)
                    self.path_patches.append(rect)

        if frame == len(self.evaluated_nodes) - 1:
            ani_end_time = time.time()      
            ani_time = ani_end_time - self.ani_start_time
            print(f"Animation time :{ani_time:.4f} secondes") 
    
        # confetti effect when we reach the target node
        if frame == len(self.path_history) - 1:
        # creation of particules around the final node
            self.confetti_patches.clear()  
            self.confetti_velocities.clear()  
            for _ in range(50):  # 50 particules
                offset_x = random.uniform(-0.5, 0.5)
                offset_y = random.uniform(-0.5, 0.5)
                color = np.random.rand(3,)  # Random color
                confetti_rect = patches.Circle(
                   (self.target[1] + offset_x, self.target[0] + offset_y),
                    radius=0.1, facecolor=color, alpha=0.7
                )
                self.ax.add_patch(confetti_rect)
                self.confetti_patches.append(confetti_rect)
            
                # Define velocity of the particales
                velocity = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]  # Movement in all directions
                self.confetti_velocities.append(velocity)

        # Animation of the particales
        for i, patch in enumerate(self.confetti_patches):
            new_center = (
                patch.center[0] + self.confetti_velocities[i][0],  # X movement
                patch.center[1] + self.confetti_velocities[i][1]   # Y movement
            )
            patch.set_center(new_center)

        self.canvas.draw()

def generer_grille(size, obstacle_mode="ratio", obstacle_ratio=0.2, obstacle_number=20):
    n = size
    grid = np.zeros((n, n))  
    G = nx.grid_2d_graph(n, n)  

    obstacles = set()

    if obstacle_mode == "ratio": 
        num_obstacles = int(n * n * obstacle_ratio)
    else:
        num_obstacles = obstacle_number

    while len(obstacles) < num_obstacles:
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        if (x, y) != (0, 0) and (x, y) != (n-1, n-1):
            obstacles.add((x, y))
            grid[x, y] = 1  # Ajouter un obstacle
            if (x, y) in G:
                G.remove_node((x, y))
        
    return grid, G

# Dijkstra algorithm. The function returns a list of list of nodes to create the animation

import heapq
import time

def dijkstra_stepwise(G, start, target, diagonal_mode="nondiagonal"):
    start_time = time.time()
    distances = {node: float('inf') for node in G.nodes()}
    distances[start] = 0
    previous_nodes = {node: None for node in G.nodes()}
    evaluated_nodes = []
    path_to_current = []
    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node not in evaluated_nodes:
            evaluated_nodes.append(current_node)

        # Reconstruction du chemin actuel
        temp_path = []
        node = current_node
        while node is not None:
            temp_path.append(node)
            node = previous_nodes[node]
        temp_path.reverse()
        path_to_current.append(temp_path)

        if current_node == target:
            break

        if diagonal_mode == "diagonal":
            neighbors = list(get_neighbors_diagonal(current_node, G))
        else:
            neighbors = list(G.neighbors(current_node))

        for neighbor in neighbors:
            if neighbor not in evaluated_nodes:
                edge_weight = G[current_node][neighbor].get("weight", 1)  # Récupérer le poids réel
                new_distance = current_distance + edge_weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    if distances[target] == float('inf'):
        print("⚠️ Aucun chemin trouvé entre le point de départ et l'arrivée.")
        return None, None  # Retourner None pour indiquer l'absence de chemin

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of Dijkstra: {execution_time:.4f} secondes")
    return evaluated_nodes, path_to_current


# Astar algorihtm with Chebyshev heuristic. The function returns a list of lists of nodes to create the animation
def heuristic(current, target):
    return max(abs(current[0] - target[0]), abs(current[1] - target[1]))
        
def astar_stepwise(G, start, target, diagonal_mode="nondiagonal"):
    start_time = time.time()
    g_scores = {node: float('inf') for node in G.nodes()}
    g_scores[start] = 0
        
    f_scores = {node: float('inf') for node in G.nodes()}
    f_scores[start] = heuristic(start, target)
        
    previous_nodes = {node: None for node in G.nodes()}
    evaluated_nodes = []
    path_to_current = []
    priority_queue = [(f_scores[start], start)]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_f_score, current_node = heapq.heappop(priority_queue)
            
        if current_node not in evaluated_nodes:
            evaluated_nodes.append(current_node)
            
        temp_path = []
        node = current_node
        while node is not None:
            temp_path.append(node)
            node = previous_nodes[node]
        temp_path.reverse()
        path_to_current.append(temp_path)
            
        #print(f"Step {len(evaluated_nodes)}: Evaluated {current_node}, Path: {temp_path}")

        if current_node == target:
            break

        if diagonal_mode == "diagonal":
            neighbors = list(get_neighbors_diagonal(current_node, G))
        elif diagonal_mode == "nondiagonal":
            neighbors = list(G.neighbors(current_node))

        for neighbor in neighbors:
            move_cost = 1 if current_node[0] == neighbor[0] or current_node[1] == neighbor[1] else np.sqrt(2)
            tentative_g_score = g_scores[current_node] + move_cost
            if tentative_g_score < g_scores[neighbor]:
                previous_nodes[neighbor] = current_node
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = g_scores[neighbor] + heuristic(neighbor, target)
                heapq.heappush(priority_queue, (f_scores[neighbor], neighbor))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of A* : {execution_time:.4f} secondes")
    return evaluated_nodes, path_to_current
    
# Returns neighbors with allowed diagonal moves
def get_neighbors_diagonal(node, G):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),(1, 1), (-1, -1), (1, -1), (-1, 1) ]
    for dx, dy in directions:
            neighbor = (node[0] + dx, node[1] + dy)
            if neighbor in G.nodes():
                yield neighbor

def save_graph(G, output_base, copies=1):
    for i in range(copies):
        output_file = f"{output_base}_{i}.json"  
        data = {
            "nodes": list(G.nodes()),
            "edges": list(G.edges())
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Copie {i+1}/{copies} saved as '{output_file}'.")

import json
import numpy as np

def load_graph(file_path):
    print(file_path)
    if file_path.endswith(".json"):
        print('allo')
        with open(file_path, 'r') as f:
            graph = json.load(f)
        return {int(k): {int(neigh): weight for neigh, weight in v.items()} for k, v in graph.items()}  # Conversion en int

    elif file_path.endswith(".npz"):
        # Charger les données du fichier npz
        data = np.load(file_path, allow_pickle=True)
        print(f"Keys in the .npz file: {data.files}")  # Affiche les clés disponibles dans le fichier .npz

        # Assurer que 'adjacency_matrix' est bien la clé correcte
        adjacency_matrix = data['adjacency_matrix']
        node_indices = data['node_indices']
        vol_dims = data['vol_dims']
        
        G = nx.Graph()  # Utiliser nx.DiGraph() si le graphe est orienté

        # Ajouter les arêtes pondérées au graphe
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix[i])):
                if adjacency_matrix[i, j] > 0:  # S'il y a une connexion
                    G.add_edge(node_indices[i], node_indices[j], weight=adjacency_matrix[i, j])

        return G, node_indices, vol_dims
    else:
        raise ValueError("Unsupported file format. Use either .json or .npz")
if __name__ == "__main__":
    root = tk.Tk()
    app = GrilleApp(root)
    root.mainloop()
