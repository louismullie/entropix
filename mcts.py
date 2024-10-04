import random
import logging
import numpy as np
import networkx as nx
from typing import List, Dict
import openai

# Set up logging to output to console at the DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DialogueState:
    def __init__(self, system_prompt: str, current_query: str, current_answer: str = ""):
        self.system_prompt = system_prompt
        self.current_query = current_query
        self.current_answer = current_answer

    def __str__(self):
        return (
            f"System: {self.system_prompt}\n"
            f"Current Query: {self.current_query}\n"
            f"Current Answer: {self.current_answer}"
        )

class MCTSNode:
    def __init__(self, state: DialogueState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class MCTS:
    def __init__(self, simulation_depth, exploration_weight, model):
        self.simulation_depth = simulation_depth
        self.exploration_weight = exploration_weight
        self.root = None
        self.graph = nx.Graph()
        self.node_labels = {}
        self.model = model
        self.completion_tokens = 0

    def select(self, node: MCTSNode) -> MCTSNode:
        logger.debug(f"Selecting node. Current node visits: {node.visits}, value: {node.value}")
        while node.children:
            node = max(
                node.children, 
                key=lambda c: c.value / (c.visits + 1e-8) + 
                              self.exploration_weight * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-8))
            )
            logger.debug(f"Selected child node. Visits: {node.visits}, Value: {node.value}")
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        logger.debug(f"Expanding node. Current state: {node.state}")
        if node.children:
            logger.debug("Node already expanded.")
            return random.choice(node.children)
        actions = self.generate_actions(node.state)
        logger.debug(f"Generated {len(actions)} possible answers")
        for i, action in enumerate(actions):
            new_state = self.apply_action(node.state, action)
            child = MCTSNode(new_state, parent=node)
            node.children.append(child)
            self.graph.add_edge(id(node), id(child))
            self.node_labels[id(child)] = f"Visits: {child.visits}\nValue: {child.value:.2f}"
            logger.debug(f"Created child node {i+1}. Answer: {action[:50]}...")
        if node.children:
            selected_child = random.choice(node.children)
            logger.debug(
                f"Randomly selected child node for simulation. Visits: {selected_child.visits}, Value: {selected_child.value}"
            )
            return selected_child
        else:
            logger.debug("No actions generated; returning the same node.")
            return node

    def simulate(self, node: MCTSNode) -> float:
        logger.debug(f"Simulating from node (evaluating answer)")
        value = self.evaluate_state(node.state)
        logger.debug(f"Simulation complete. Answer value: {value}")
        return value

    def backpropagate(self, node: MCTSNode, value: float):
        logger.debug(f"Starting backpropagation. Initial value: {value}")
        while node:
            node.visits += 1
            node.value += value
            self.node_labels[id(node)] = f"Visits: {node.visits}\nValue: {node.value:.2f}"
            logger.debug(f"Updated node. Visits: {node.visits}, New value: {node.value}")
            node = node.parent

    def search(self, initial_state: DialogueState, num_simulations: int) -> DialogueState:
        logger.debug(f"Starting MCTS search with {num_simulations} simulations")
        if not self.root:
            self.root = MCTSNode(initial_state)
            self.graph.add_node(id(self.root))
            self.node_labels[id(self.root)] = f"Root\nVisits: 0\nValue: 0.00"
            logger.debug("Created root node")
        
        for i in range(num_simulations):
            logger.debug(f"Starting simulation {i+1}")
            node = self.select(self.root)
            if node.visits == 0 or not node.children:
                node = self.expand(node)
            value = self.simulate(node)
            self.backpropagate(node, value)
                
        if not self.root.children:
            raise ValueError("No children were generated during the search.")
        
        best_child = max(self.root.children, key=lambda c: c.value / (c.visits + 1e-8))
        logger.debug(
            f"Search complete. Best child node: Visits: {best_child.visits}, Value: {best_child.value}"
        )
        return best_child.state

    def generate_actions(self, state: DialogueState) -> List[str]:
        logger.debug("Generating possible answers for the current query")
        messages = [
            {"role": "system", "content": state.system_prompt},
            {"role": "user", "content": state.current_query}
        ]
        n = 3  # Number of possible answers to generate
        logger.info(f"Requesting {n} possible answers from the model")
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=128,
                n=n,
                temperature=1.0
            )
            answers = [choice.message['content'].strip() for choice in response.choices]
            self.completion_tokens += response.usage.completion_tokens
            logger.info(f"Received {len(answers)} possible answers from the model")
            return answers
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            return []

    def apply_action(self, state: DialogueState, action: str) -> DialogueState:
        logger.info(f"Applying action (new possible answer): {action[:50]}...")
        return DialogueState(state.system_prompt, state.current_query, action)

    def evaluate_state(self, state: DialogueState) -> float:
        logger.info("Evaluating the assistant's answer")
        messages = [
            {"role": "system", "content": "You are an expert evaluator of assistant's answers."},
            {"role": "user", "content": state.current_query},
            {"role": "assistant", "content": state.current_answer},
            {"role": "user", "content": (
                "Please rate the assistant's answer on a scale from 0 to 1, where 0 is completely incorrect or unhelpful, "
                "and 1 is completely correct and helpful. Respond with only a number."
            )}
        ]
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=5,
                n=1,
                temperature=0.0
            )
            self.completion_tokens += response.usage.completion_tokens
            score_str = response.choices[0].message['content'].strip()
            score = float(score_str)
            score = max(0.0, min(score, 1.0))
            logger.info(f"Answer evaluation score: {score}")
            return score
        except Exception as e:
            logger.warning(f"Failed to parse evaluation score: {e}")
            return 0.5  # Default score

def chat_with_mcts(
    system_prompt: str, 
    initial_query: str, 
    model: str = "gpt-4o-mini", 
    num_simulations: int = 10, 
    exploration_weight: float = 1.4, 
    simulation_depth: int = 1
) -> str:
    logger.info("Starting MCTS search for the best answer")
    logger.info(
        f"Parameters: num_simulations={num_simulations}, exploration_weight={exploration_weight}, "
        f"simulation_depth={simulation_depth}"
    )
    mcts = MCTS(simulation_depth=simulation_depth, exploration_weight=exploration_weight, model=model)
    initial_state = DialogueState(system_prompt, initial_query)
    logger.info(f"Initial query: {initial_query}")
    final_state = mcts.search(initial_state, num_simulations)
    response = final_state.current_answer
    logger.info(f"MCTS search complete. Best answer: {response[:100]}...")
    return response, mcts.completion_tokens

# Example usage:
if __name__ == "__main__":
    # Make sure to set your OpenAI API key before running the code

    openai.api_key = ""
    system_prompt = "You are an assistant that provides helpful and accurate answers."
    initial_query = "Which number is larger, 9.9 or 9.11?"
    initial_query = "How many Rs are in raspberry?"
    response, tokens_used = chat_with_mcts(system_prompt, initial_query)
    print("Assistant's best answer:")
    print(response)
    print(f"Total tokens used: {tokens_used}")