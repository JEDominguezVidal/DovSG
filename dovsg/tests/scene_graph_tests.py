import pickle
import json
import torch
import numpy as np
import open_clip
from dovsg.perception.models.myclip import MyClip

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

base = "data_example/room1/memory/3_0.1_0.01_True_0.2_0.5/step_0"

# 1. Load view dataset (3D point cloud + voxel system)
with open(f"{base}/view_dataset.pkl", "rb") as f:
    view_dataset = pickle.load(f)

# 2. Load instance objects (list of 49 detected objects)
with open(f"{base}/instance_objects.pkl", "rb") as f:
    instance_objects = pickle.load(f)

# 3. Load scene graph (spatial relationships)
with open(f"{base}/instance_scene_graph.pkl", "rb") as f:
    scene_graph = pickle.load(f)

# 4. Load class names and colors
with open(f"{base}/classes_and_colors.json", "r") as f:
    classes_and_colors = json.load(f)

print(f"\nLoaded {len(instance_objects)} objects.")

# --- Helper to get object center ---
def get_object_center(obj):
    points = view_dataset.index_to_point(obj["indexes"])
    return points.mean(axis=0)

# Precompute centers for all objects
object_centers = {}
for obj in instance_objects:
    object_centers[obj['class_id']] = get_object_center(obj)

# --- 1. CLIP Search ---
print("\n=== 1. CLIP Search ===")
try:
    # Initialize CLIP (singleton)
    clip_model = MyClip(device=device)
    
    queries = ["a red pepper", "a green pepper", "a plate", "a table", "a toy"]
    print(f"Queries: {queries}")
    
    # Get text features
    text_features = clip_model.get_text_feature(queries) # [N_queries, 768]
    if isinstance(text_features, torch.Tensor):
        text_features = text_features.detach().cpu().numpy()
        
    # Stack object features
    obj_features = []
    w_class_ids = []
    for obj in instance_objects:
        if "clip_ft" in obj:
            ft = obj["clip_ft"]
            if isinstance(ft, torch.Tensor):
                ft = ft.detach().cpu().numpy()
            obj_features.append(ft)
            w_class_ids.append(obj['class_id'])
            
    if obj_features:
        obj_features = np.array(obj_features) # [N_objs, 768]
        
        # Compute similarity
        sims = text_features @ obj_features.T # [N_queries, N_objs]
        
        for i, query in enumerate(queries):
            print(f"Query: {query}")
            print(f"Similarities: {sims[i]}")

            print("Smilarities (class_id, similarity)")
            text=[]
            for j in range(len(sims[i])):
                # print(w_class_ids[j], sims[i, j])
                text.append((w_class_ids[j], sims[i, j]))
            print(text)
            best_class_idx = np.argmax(sims[i])
            best_obj_class_id = w_class_ids[best_class_idx]
            best_score = sims[i, best_class_idx]
            
            # Find class name for verification
            obj_class = "Unknown"
            for obj in instance_objects:
                if obj['class_id'] == best_obj_class_id:
                    # obj['class_name'] might be the human readable name, let's check keys again if needed.
                    # Previous output showed 'class_name' key exists.
                    # 'class_id' seems to be unique id like 'plate_27'
                    # Let's use 'class_name' if available, or fall back to 'class_id'
                    obj_class = obj.get('class_name', obj['class_id']) 
                    break
            
            print(f"Query: '{query}' -> Best Match: ID {best_obj_class_id} ({obj_class}), Score: {best_score:.4f}")
    else:
        print("No CLIP features found in objects.")

except Exception as e:
    print(f"CLIP search failed: {e}")
    import traceback
    traceback.print_exc()


# --- 2. Spatial Reasoning ---
print("\n=== 2. Spatial Reasoning ===")

# Example: Find nearest object to the first object found in CLIP search (or just the first object)
if len(instance_objects) > 0:
    ref_obj = instance_objects[0]
    ref_center = object_centers[ref_obj['class_id']]
    # Use class_name if possible for display
    ref_name = ref_obj.get('class_name', ref_obj['class_id'])
    print(f"Reference Object: ID {ref_obj['class_id']} ({ref_name}) at {ref_center}")
    
    min_dist = float('inf')
    nearest_class_id = None
    
    for obj in instance_objects:
        if obj['class_id'] == ref_obj['class_id']:
            continue
            
        dist = np.linalg.norm(object_centers[obj['class_id']] - ref_center)
        if dist < min_dist:
            min_dist = dist
            nearest_class_id = obj['class_id']
            
    if nearest_class_id is not None:
         # Find class name
        nearest_class = "Unknown"
        for obj in instance_objects:
            if obj['class_id'] == nearest_class_id:
                nearest_class = obj.get('class_name', obj['class_id'])
                break
        print(f"Nearest Object: ID {nearest_class_id} ({nearest_class}), Distance: {min_dist:.4f}")

# --- 3. Scene Graph Traversal ---
print("\n=== 3. Scene Graph Traversal ===")

output_lines = []
for node_id, node in scene_graph.object_nodes.items():
    if node.parent:
        output_lines.append(f"{node_id} is {node.parent_relation} {node.parent.node_id}")

# Print first 10 relationships
for line in output_lines[:10]:
    print(line)
    
# Find all objects on 'table' (assuming 'on' relationship and 'table' in parent class name)
# Note: node_id usually contains class name
tables = [nid for nid in scene_graph.object_nodes if 'table' in nid.lower()]
if tables:
    for table in tables:
        print(f"\nObjects on {table}:")
        parent_node = scene_graph.object_nodes[table]
        # Since children point to parent, we iterate all nodes to find those with this parent
        # Or checking if the node object has a 'children' attribute (it does based on inspection)
        if hasattr(parent_node, 'children') and parent_node.children:
            for child in parent_node.children.values():
                print(f" - {child.node_id}")
        else:
            print(" - None found via children attribute.")
        
# --- 4. Attribute Analysis ---
print("\n=== 4. Attribute Analysis ===")
if len(instance_objects) > 0:
    obj = instance_objects[0]
    print(f"Attributes for {obj['class_id']}:")
    for k, v in obj.items():
        if k not in ['indexes', 'clip_ft']: # Skip large arrays
            print(f" - {k}: {v}")

