import traceback
import numpy as np
from typing import List, Dict, Any

class CodeCandidate:
    def __init__(self, code: str, parent=None):
        self.code = code
        self.parent = parent
        self.transition_accuracy = 0.0
        self.num_refinements = 0
        self.error_trace = None

class CWMRefiner:
    def __init__(self, llm_client, max_attempts=5):
        self.client = llm_client
        self.max_attempts = max_attempts
        self.candidates: List[CodeCandidate] = []

    def refine(self, initial_code: str) -> str:
        """
        Implements the Tree Search refinement from CWM for tic-tac-toe.
        """
        root = CodeCandidate(initial_code)
        self.evaluate_candidate(root)
        self.candidates.append(root)

        for i in range(self.max_attempts):
            # Thompson sampling-like selection: 
            # Favor high accuracy or low refinement count 
            best_candidate = self._select_candidate()
            
            if best_candidate.transition_accuracy == 1.0:
                print(f"✅ [Refiner] Found candidate with 100% accuracy on Attempt {i}")
                return best_candidate.code

            print(f"🔄 [Refiner] Refining candidate (Acc: {best_candidate.transition_accuracy:.2f})")
            
            # Generate new tic-tac-toe code based on the error
            new_code = self.client.generate_code(
                template="tic_tac_toe",
                failed_tests=best_candidate.error_trace
            )
            
            new_node = CodeCandidate(new_code, parent=best_candidate)
            new_node.num_refinements = best_candidate.num_refinements + 1
            
            self.evaluate_candidate(new_node)
            self.candidates.append(new_node)

        # Return best code found if budget exhausted
        best = max(self.candidates, key=lambda c: c.transition_accuracy)
        print(f"⚠️ [Refiner] Budget exhausted. Returning best candidate (Acc: {best.transition_accuracy:.2f})")
        return best.code

    def evaluate_candidate(self, candidate: CodeCandidate):
        """
        Runs trajectory tests to measure 'Transition Accuracy'[cite: 185].
        """
        from generation.unit_tests import run_trajectory_test
        
        try:
            # We attempt to generate a valid game trajectory
            success = run_trajectory_test(candidate.code)
            if success:
                candidate.transition_accuracy = 1.0
                candidate.error_trace = None
            else:
                # Should not happen if run_trajectory_test throws on failure
                candidate.transition_accuracy = 0.5 
        except Exception as e:
            candidate.transition_accuracy = 0.0
            candidate.error_trace = traceback.format_exc()

    def _select_candidate(self) -> CodeCandidate:
        # Simple heuristic: Sort by accuracy desc, then refinements asc
        # This approximates the CWM selection policy 
        candidates = sorted(
            self.candidates, 
            key=lambda c: (c.transition_accuracy, -c.num_refinements), 
            reverse=True
        )
        return candidates[0]