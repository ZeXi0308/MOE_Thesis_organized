from typing import Optional, Literal
from collections import deque
import time
from functools import lru_cache 

class KnapSack():
    def __init__(self, 
                 block_size = int, 
                 total_available_blocks = int, 
                 solver: Literal['greedy', 'dp'] = 'greedy',
                 delta_t: int = 10,
                 ) -> None:
        
        if solver == 'dp':
            self.solver_func = KnapSack._dp_knapsack
        elif solver == 'greedy':
            self.solver_func = KnapSack._greedy_knapsack
        else:
            raise ValueError(f"Solver {self.solver} is not supported")

        self.delta_t = delta_t
        self.block_size = block_size 
        self.total_available_blocks = total_available_blocks 
        self.token_latency = 0.02
        self.max_num_preempt = 1
        self.unit_overhead = 3/90000 * self.block_size
        self.percentile_to_sacrifice = 0.1

    def pick_requests(self, running, waiting, swapped): 
        # Helper functions to get context block for list and individual requests
        def get_context_block_list(requests):
            return [get_context_block(r) for r in requests]
        
        @lru_cache(maxsize=150)
        def get_context_block(request):
            return round((request.get_len() + 1) / self.block_size + 0.5)
        
        if not waiting and not swapped:
            return running
        # Convert input lists to mutable lists (if not already)
        running = list(running)
        waiting = list(waiting)
        swapped = list(swapped)
        now = time.monotonic()

        # ============ online preemption ==============
        context_block_list = get_context_block_list(running + waiting + swapped) 
        num_req = len(running) + len(waiting) + len(swapped)
        slack_list = [r.get_slack(now) for r in running + waiting + swapped]
        index = int(self.percentile_to_sacrifice * num_req)
        slack_list = sorted(slack_list)
        threshold_value = slack_list[index]
        overhead_per_request = self.unit_overhead * sum(context_block_list) / num_req
        self.max_num_preempt = int( threshold_value // overhead_per_request) 
        if self.max_num_preempt < 1:
            return running + waiting + swapped

        # ============ online preemption ============== 
        run_value_list = [r.get_value(now, self.token_latency, self.delta_t, True) / get_context_block(r) for r in running] 
        pause_value_list = [r.get_value(now, self.token_latency, self.delta_t, False) / get_context_block(r) for r in waiting + swapped]
        threshold_value = max(pause_value_list, default=1)
        maybe_preempt = [(i, r) for i, r in enumerate(running) if run_value_list[i] <= threshold_value ][:self.max_num_preempt]

        # Use set for efficient index checking when keeping non-preempted running items
        preempt_indices = {i for i, _ in maybe_preempt}
        keep_running_idx = [i for i, _ in enumerate(running) if i not in preempt_indices]
        keep_running = [r for i, r in enumerate(running) if i not in preempt_indices]
        running = [r for _, r in maybe_preempt]
 
        available_blocks = self.total_available_blocks - sum([context_block_list[i] for i in keep_running_idx])
        context_block_list = [context_block_list[i] for i in preempt_indices] + context_block_list[-len(waiting + swapped):]
        value_list = [run_value_list[i] for i in preempt_indices] + pause_value_list 
        _, best_plan = self.solver_func(available_blocks, context_block_list, value_list)
        return [(running + waiting + swapped)[i] for i in best_plan] + keep_running

    def get_context_block(self, request):
        return round((request.get_len() + 1) / self.block_size + 0.5)

    def schedule_requests(self, budget, running, waiting, swapped, utilization, latency_function):
        # TODO: consider budget
        if utilization < 0.9:
            # # sort by value
            now = time.monotonic()
            waiting = sorted(waiting, key=lambda x: x.get_value(now, self.token_latency, self.delta_t, False)/self.get_context_block(x), reverse=True)
            return deque(waiting), deque(swapped), deque()
        
        new_running = self.pick_requests(running, waiting, swapped)
        seq_to_admit = [r for r in new_running if r in waiting]
        seq_to_swap_in = [r for r in new_running if r in swapped]
        seq_to_evict = [r for r in running if r not in new_running]
        
        return deque(seq_to_admit), deque(seq_to_swap_in), deque(seq_to_evict)

    @staticmethod
    def _greedy_knapsack(capacity: int,  weights: list, values: list) -> tuple:
        """
        Greedy algorithm for the fractional knapsack problem.

        Args:
        capacity (int): Maximum weight the knapsack can carry.
        weights (list): List of weights of the items.
        values (list): List of values of the items.

        Returns:
        tuple: Total value of the picked items and the list of indices of the picked items.
        """
        # Calculate value-to-weight ratio and sort items by this ratio in descending order
        items = [(v / w, w, v, i) for i, (v, w) in enumerate(zip(values, weights))]
        items.sort(reverse=True, key=lambda x: x[0]) 
        current_weight = current_value = 0
        picked_items = [] 
        for _, weight, value, index in items:
            if current_weight + weight <= capacity:  
                current_weight += weight
                current_value += value
                picked_items.append(index)
            else:
                # if current_weight / capacity < 0.9:
                #     continue
                # else:
                break
        return current_value, picked_items

    @staticmethod
    def _dp_knapsack(capacity: int, weights: list, values: list) -> tuple:
        """
        Dynamic programming solution for the 0/1 Knapsack problem.

        Args:
        capacity (int): Maximum weight the knapsack can carry.
        weights (list): List of weights of the items.
        values (list): List of values of the items.

        Returns:
        tuple: Maximum value of the picked items and the list of indices of the picked items.
        """
        n = len(weights)
        # dp[i][w] will store the maximum value with the first i items and weight limit w
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        # Fill the dp array
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:  # if the current item can fit in the remaining weight
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        # Traceback to find the items to include in the knapsack
        picked_items = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:  # Item i-1 is included
                picked_items.append(i - 1)
                w -= weights[i - 1]
        
        # The maximum value is stored in dp[n][capacity]
        max_value = dp[n][capacity]
        return max_value, picked_items[::-1]
