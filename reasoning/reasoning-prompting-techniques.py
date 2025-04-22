"""
Structured Reasoning Techniques for LLMs
Implementations of CoT, ToT, GoT, and DoT prompting methods
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import json
from collections import deque
import heapq

class ReasoningTechniques:
    def __init__(self, llm_model, tokenizer):
        self.model = llm_model
        self.tokenizer = tokenizer
        self.max_length = 512
        
    def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        """Basic generation function for the LLM"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 1. Chain of Thought (CoT)
    def chain_of_thought(self, question: str, zero_shot: bool = True) -> str:
        """
        Chain of Thought prompting
        - zero_shot: If True, uses "Let's think step by step."
        - If False, expects few-shot examples to be included in the question
        """
        if zero_shot:
            prompt = f"{question}\n\nLet's think step by step."
        else:
            prompt = question  # Assumes examples are already included
        
        response = self.generate_response(prompt)
        return response
    
    # 2. Tree of Thoughts (ToT)
    def tree_of_thoughts(self, question: str, branching_factor: int = 3, max_depth: int = 3) -> str:
        """
        Tree of Thoughts implementation with BFS exploration
        """
        class ThoughtNode:
            def __init__(self, thought: str, parent=None, depth: int = 0):
                self.thought = thought
                self.parent = parent
                self.children = []
                self.depth = depth
                self.score = 0
        
        # Root node
        root = ThoughtNode(f"Question: {question}")
        queue = deque([root])
        best_leaf = None
        best_score = float('-inf')
        
        while queue and root.depth < max_depth:
            current_node = queue.popleft()
            
            # Generate multiple thought branches
            for _ in range(branching_factor):
                prompt = self._build_tot_prompt(current_node, question)
                thought = self.generate_response(prompt, max_tokens=50)
                
                # Create child node
                child = ThoughtNode(thought, current_node, current_node.depth + 1)
                current_node.children.append(child)
                
                # Self-evaluate the thought
                evaluation_prompt = f"Evaluate this reasoning step for solving '{question}':\n{thought}\n\nScore (0-10):"
                score_response = self.generate_response(evaluation_prompt, max_tokens=10)
                try:
                    child.score = float(score_response.strip())
                except:
                    child.score = 5.0  # Default score
                
                # Update best leaf if this is better
                if child.depth == max_depth and child.score > best_score:
                    best_score = child.score
                    best_leaf = child
                
                # Add to queue for further exploration
                if child.depth < max_depth:
                    queue.append(child)
        
        # Reconstruct the best path
        return self._reconstruct_tot_path(best_leaf)
    
    def _build_tot_prompt(self, node: 'ThoughtNode', question: str) -> str:
        """Build prompt for ToT by tracing back to root"""
        path = []
        current = node
        while current.parent is not None:
            path.append(current.thought)
            current = current.parent
        path.reverse()
        
        prompt = f"Question: {question}\n\n"
        prompt += "Previous thoughts:\n"
        for i, thought in enumerate(path, 1):
            prompt += f"{i}. {thought}\n"
        prompt += "\nNext thought:"
        return prompt
    
    def _reconstruct_tot_path(self, leaf_node: 'ThoughtNode') -> str:
        """Reconstruct the path from root to best leaf"""
        path = []
        current = leaf_node
        while current is not None:
            path.append(current.thought)
            current = current.parent
        path.reverse()
        
        result = "\n".join(path)
        result += f"\nFinal score: {leaf_node.score}"
        return result
    
    # 3. Graph of Thoughts (GoT)
    def graph_of_thoughts(self, question: str, num_iterations: int = 5) -> str:
        """
        Graph of Thoughts implementation with idea merging and feedback loops
        """
        class ThoughtGraph:
            def __init__(self):
                self.nodes = {}
                self.edges = {}
                self.node_id = 0
            
            def add_node(self, thought: str, node_type: str = "normal") -> int:
                self.node_id += 1
                self.nodes[self.node_id] = {
                    "thought": thought,
                    "type": node_type,
                    "score": 0
                }
                return self.node_id
            
            def add_edge(self, from_id: int, to_id: int, edge_type: str = "sequential"):
                if from_id not in self.edges:
                    self.edges[from_id] = []
                self.edges[from_id].append({
                    "to": to_id,
                    "type": edge_type
                })
        
        graph = ThoughtGraph()
        root_id = graph.add_node(f"Question: {question}", "root")
        current_nodes = [root_id]
        
        for iteration in range(num_iterations):
            new_nodes = []
            
            # Process each current node
            for node_id in current_nodes:
                node = graph.nodes[node_id]
                
                # Generate follow-up thoughts
                prompt = f"Based on: {node['thought']}\nGenerate a follow-up thought:"
                thought = self.generate_response(prompt, max_tokens=50)
                new_id = graph.add_node(thought)
                graph.add_edge(node_id, new_id, "sequential")
                new_nodes.append(new_id)
                
                # Merge ideas from multiple nodes
                if len(current_nodes) > 1 and iteration % 2 == 0:  # Merge every other iteration
                    merge_partner = current_nodes[0] if node_id != current_nodes[0] else current_nodes[1]
                    merge_prompt = f"Merge these ideas:\n1. {node['thought']}\n2. {graph.nodes[merge_partner]['thought']}\n\nMerged idea:"
                    merged_thought = self.generate_response(merge_prompt, max_tokens=50)
                    merge_id = graph.add_node(merged_thought, "merge")
                    graph.add_edge(node_id, merge_id, "merge")
                    graph.add_edge(merge_partner, merge_id, "merge")
                    new_nodes.append(merge_id)
                
                # Create feedback loop (self-critique)
                if iteration > 0:
                    critique_prompt = f"Critique this thought: {node['thought']}\n\nCritique:"
                    critique = self.generate_response(critique_prompt, max_tokens=50)
                    critique_id = graph.add_node(critique, "critique")
                    graph.add_edge(node_id, critique_id, "critique")
                    
                    # Refine based on critique
                    refine_prompt = f"Original: {node['thought']}\nCritique: {critique}\n\nRefined thought:"
                    refined_thought = self.generate_response(refine_prompt, max_tokens=50)
                    refine_id = graph.add_node(refined_thought, "refinement")
                    graph.add_edge(critique_id, refine_id, "refinement")
                    new_nodes.append(refine_id)
            
            current_nodes = new_nodes
        
        # Find best path through graph
        return self._find_best_graph_path(graph, root_id)
    
    def _find_best_graph_path(self, graph: 'ThoughtGraph', start_id: int) -> str:
        """Find the best path through the thought graph"""
        # For simplicity, return all nodes in order of creation
        result = "Graph of Thoughts Result:\n\n"
        for node_id, node in graph.nodes.items():
            result += f"[{node['type']}] {node['thought']}\n"
            if node_id in graph.edges:
                for edge in graph.edges[node_id]:
                    result += f"  -> [{edge['type']}] to node {edge['to']}\n"
            result += "\n"
        return result
    
    # 4. Diagram of Thought (DoT)
    def diagram_of_thought(self, question: str, max_iterations: int = 8) -> str:
        """
        Diagram of Thought implementation with role-specific reasoning phases
        """
        roles = {
            "proposer": "Propose a solution or idea:",
            "critic": "Critique the previous idea:",
            "refiner": "Refine based on the critique:",
            "verifier": "Verify the refined solution:"
        }
        
        diagram = []
        current_thought = f"Question: {question}"
        diagram.append({"role": "initial", "thought": current_thought})
        
        for i in range(max_iterations):
            role = list(roles.keys())[i % len(roles)]
            prompt = f"{current_thought}\n\n[{role.upper()}] {roles[role]}"
            response = self.generate_response(prompt, max_tokens=100)
            
            diagram.append({"role": role, "thought": response})
            current_thought = response
            
            # Early stopping if verifier confirms solution
            if role == "verifier" and "correct" in response.lower():
                break
        
        # Format the diagram as a readable output
        result = "Diagram of Thought Reasoning:\n\n"
        for i, node in enumerate(diagram):
            result += f"{i}. [{node['role'].upper()}] {node['thought']}\n"
        
        return result
    
    # Helper method to compare different techniques
    def compare_techniques(self, question: str) -> Dict[str, str]:
        """Compare all reasoning techniques on the same question"""
        results = {}
        
        print("Running Chain of Thought...")
        results["CoT"] = self.chain_of_thought(question)
        
        print("Running Tree of Thoughts...")
        results["ToT"] = self.tree_of_thoughts(question, branching_factor=2, max_depth=2)
        
        print("Running Graph of Thoughts...")
        results["GoT"] = self.graph_of_thoughts(question, num_iterations=3)
        
        print("Running Diagram of Thought...")
        results["DoT"] = self.diagram_of_thought(question, max_iterations=6)
        
        return results

# Example usage
def demo_reasoning_techniques():
    """Demonstrate the different reasoning techniques"""
    
    # Mock LLM model and tokenizer for demonstration
    class MockLLM:
        def generate(self, input_ids, **kwargs):
            # Simulate LLM response
            response_text = "This is a simulated response based on the input."
            return torch.tensor([[1] * 20])  # Mock tensor
    
    class MockTokenizer:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1] * len(text.split())])}
        
        def decode(self, token_ids, skip_special_tokens=True):
            return "Simulated response for the query."
    
    model = MockLLM()
    tokenizer = MockTokenizer()
    
    reasoner = ReasoningTechniques(model, tokenizer)
    
    # Test problem
    question = "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?"
    
    # Compare all techniques
    results = reasoner.compare_techniques(question)
    
    # Print results
    for technique, result in results.items():
        print(f"\n{'='*50}")
        print(f"{technique} Result:")
        print(f"{'='*50}")
        print(result)

if __name__ == "__main__":
    demo_reasoning_techniques()
