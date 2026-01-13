import networkx as nx
import matplotlib.pyplot as plt
import logging
import pickle
import os
from typing import Dict, Any, List
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class PIDGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

        self.graph_path = os.path.join(settings.WORKING_DIR, "system_topology.gpickle")

    def build_graph(self, parsed_data: Dict[str, Any]):
        logger.info(f"Building topological graph for {parsed_data['filename']}...")
        
        # 1. Add Explicit Nodes (from Equipment list)
        for item in parsed_data['equipment']:
            self.graph.add_node(
                item['id'], 
                tag=item['tag'], 
                type=item['type'], 
                **item.get('attributes', {})
            )

        # 2. Add Edges (Connections)
        for conn in parsed_data['connections']:
            self.graph.add_edge(
                conn['source'], 
                conn['target'], 
                type=conn['type']
            )
            
        # 3. Cleanup: Find "Ghost Nodes" (nodes with no attributes) and give them defaults
        for n in self.graph.nodes():
            if 'tag' not in self.graph.nodes[n]:
                self.graph.nodes[n]['tag'] = f"Unknown_ID_{n}"
                self.graph.nodes[n]['type'] = "Implicit_Node"
            
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes.")
        return self.graph

    def save_graph(self):
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info(f"Graph topology saved to {self.graph_path}")

    def load_graph(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "rb") as f:
                self.graph = pickle.load(f)
            logger.info("Graph topology loaded successfully.")
        else:
            logger.warning("No graph file found. Please run ingestion first.")

    def generate_visualization(self, output_path: str = "data/processed/topology.svg"):
        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, skipping visualization.")
            return

        plt.figure(figsize=(14, 10))
        
        # Use spring layout
        pos = nx.spring_layout(self.graph, k=0.15, iterations=20)
        
        # Draw Nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=300, node_color='lightblue')
        
        # Draw Edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
        
        labels = {}
        for n, d in self.graph.nodes(data=True):
            node_type = d.get('type', '')
            
            # Skip labeling simple pipes to reduce clutter
            if 'Pipe' in node_type:
                continue
                
            # Use 'tag' if available, otherwise fallback to the ID (n)
            label_text = d.get('tag', str(n)) 
            
            # If the fallback ID is huge/ugly, truncate it
            if len(label_text) > 15: 
                label_text = label_text[:12] + "..."
                
            labels[n] = label_text

        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("System Topology")
        plt.axis('off')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format="svg")
        plt.close()
        logger.info(f"Visualization saved to {output_path}")

    def get_neighbors(self, node_tag: str) -> List[str]:
        """Tool for the agent: Find what is directly connected."""
        # We need to map Tag -> ID first (inverse lookup)
        tag_map = {data['tag']: node for node, data in self.graph.nodes(data=True)}
        
        if node_tag not in tag_map:
            return [f"Tag {node_tag} not found."]
            
        node_id = tag_map[node_tag]
        successors = list(self.graph.successors(node_id))
        predecessors = list(self.graph.predecessors(node_id))
        
        results = []
        for nid in successors:
            data = self.graph.nodes[nid]
            results.append(f"Flows INTO: {data.get('tag', 'Unknown')} ({data.get('type')})")
            
        for nid in predecessors:
            data = self.graph.nodes[nid]
            results.append(f"Receives FROM: {data.get('tag', 'Unknown')} ({data.get('type')})")
            
        return results

    def find_path(self, start_tag: str, end_tag: str) -> str:
        """Tool for the agent: Find flow path between two items."""
        tag_map = {data['tag']: node for node, data in self.graph.nodes(data=True)}
        
        if start_tag not in tag_map or end_tag not in tag_map:
            return "One or both tags not found."
            
        try:
            path_ids = nx.shortest_path(self.graph, tag_map[start_tag], tag_map[end_tag])
            path_names = [self.graph.nodes[nid].get('tag', 'Unknown') for nid in path_ids]
            return " -> ".join(path_names)
        except nx.NetworkXNoPath:
            return f"No physical connection found between {start_tag} and {end_tag}."
    
    def find_relevant_nodes(self, query_text: str) -> List[str]:
        """
        Scans the query text to find any valid Node IDs or Tags that exist in the graph.
        Returns a list of Node IDs found in the text.
        """
        found_ids = []
        
        # 1. Clean and tokenize the query
        clean_text = query_text.replace("?", "").replace(",", "").replace("!", "")
        tokens = set(clean_text.split())
        
        # 2. Iterate through graph nodes to find matches
        for node_id, data in self.graph.nodes(data=True):
            tag = data.get('tag', '')
            
            # Check if the Node ID is in the text (Exact Match)
            if str(node_id) in tokens:
                found_ids.append(node_id)
                continue
                
            # Check if the Tag is in the text (e.g., "P-101")
            if tag and tag in tokens:
                found_ids.append(node_id)
        
        logger.info(f"Smart Lookup found nodes: {found_ids}")
        return found_ids

    def get_neighbors_by_id(self, node_id: str) -> List[str]:
        """Direct neighbor lookup using ID instead of Tag."""
        if not self.graph.has_node(node_id):
            return [f"Node ID {node_id} not found."]
            
        data = self.graph.nodes[node_id]
        my_tag = data.get('tag', 'Unknown')
        my_type = data.get('type', 'Unknown')
        
        results = [f"Focus Entity: {my_tag} ({my_type}) [ID: {node_id}]"]
        
        # Successors (Downstream)
        for nid in self.graph.successors(node_id):
            n_data = self.graph.nodes[nid]
            results.append(f"-> Connects TO: {n_data.get('tag', 'Unknown')} ({n_data.get('type')}) [ID: {nid}]")
            
        # Predecessors (Upstream)
        for nid in self.graph.predecessors(node_id):
            n_data = self.graph.nodes[nid]
            results.append(f"<- Receives FROM: {n_data.get('tag', 'Unknown')} ({n_data.get('type')}) [ID: {nid}]")
            
        return results