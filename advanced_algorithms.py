"""
Advanced Algorithmic Problems with Time and Space Complexity Analysis
====================================================================

This file contains implementations of various advanced algorithmic problems
with detailed comments explaining the approach and complexity analysis.
"""

def four_number_sum(array, target_sum):
    """
    Find all quadruplets in the array that sum up to the target sum.
    
    Approach: Use hash table to store pairs and their sums
    Time Complexity: O(n²) average, O(n³) worst case
    Space Complexity: O(n²)
    """
    all_pair_sums = {}
    quadruplets = []
    
    for i in range(1, len(array) - 1):
        for j in range(i + 1, len(array)):
            current_sum = array[i] + array[j]
            difference = target_sum - current_sum
            
            if difference in all_pair_sums:
                for pair in all_pair_sums[difference]:
                    quadruplets.append(pair + [array[i], array[j]])
        
        for k in range(0, i):
            current_sum = array[i] + array[k]
            if current_sum not in all_pair_sums:
                all_pair_sums[current_sum] = [[array[k], array[i]]]
            else:
                all_pair_sums[current_sum].append([array[k], array[i]])
    
    return quadruplets


def subarray_sort(array):
    """
    Find the smallest subarray that needs to be sorted to make the entire array sorted.
    
    Approach: Find the minimum and maximum values that are out of order
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    min_out_of_order = float("inf")
    max_out_of_order = float("-inf")
    
    for i in range(len(array)):
        num = array[i]
        if is_out_of_order(i, num, array):
            min_out_of_order = min(min_out_of_order, num)
            max_out_of_order = max(max_out_of_order, num)
    
    if min_out_of_order == float("inf"):
        return [-1, -1]
    
    subarray_left_idx = 0
    while min_out_of_order >= array[subarray_left_idx]:
        subarray_left_idx += 1
    
    subarray_right_idx = len(array) - 1
    while max_out_of_order <= array[subarray_right_idx]:
        subarray_right_idx -= 1
    
    return [subarray_left_idx, subarray_right_idx]


def is_out_of_order(i, num, array):
    """Helper function for subarray_sort"""
    if i == 0:
        return num > array[i + 1]
    if i == len(array) - 1:
        return num < array[i - 1]
    return num > array[i + 1] or num < array[i - 1]


def largest_range(array):
    """
    Find the largest range of consecutive integers in the array.
    
    Approach: Use hash table to track visited numbers
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    best_range = []
    longest_length = 0
    nums = {}
    
    for num in array:
        nums[num] = True
    
    for num in array:
        if not nums[num]:
            continue
        
        nums[num] = False
        current_length = 1
        left = num - 1
        right = num + 1
        
        while left in nums:
            nums[left] = False
            current_length += 1
            left -= 1
        
        while right in nums:
            nums[right] = False
            current_length += 1
            right += 1
        
        if current_length > longest_length:
            longest_length = current_length
            best_range = [left + 1, right - 1]
    
    return best_range


def min_rewards(scores):
    """
    Distribute minimum rewards to students based on their scores.
    Each student must get at least 1 reward and more than adjacent students with lower scores.
    
    Approach: Two-pass algorithm - left to right, then right to left
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    rewards = [1 for _ in scores]
    
    # Left to right pass
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            rewards[i] = rewards[i - 1] + 1
    
    # Right to left pass
    for i in range(len(scores) - 2, -1, -1):
        if scores[i] > scores[i + 1]:
            rewards[i] = max(rewards[i], rewards[i + 1] + 1)
    
    return sum(rewards)


def zigzag_traverse(array):
    """
    Traverse a 2D array in a zigzag pattern.
    
    Approach: Track direction and handle edge cases
    Time Complexity: O(n) where n is total number of elements
    Space Complexity: O(n) for result array
    """
    height = len(array) - 1
    width = len(array[0]) - 1
    result = []
    row, col = 0, 0
    going_down = True
    
    while not is_out_of_bounds(row, col, height, width):
        result.append(array[row][col])
        
        if going_down:
            if col == 0 or row == height:
                going_down = False
                if row == height:
                    col += 1
                else:
                    row += 1
            else:
                row += 1
                col -= 1
        else:
            if row == 0 or col == width:
                going_down = True
                if col == width:
                    row += 1
                else:
                    col += 1
            else:
                row -= 1
                col += 1
    
    return result


def is_out_of_bounds(row, col, height, width):
    """Helper function for zigzag_traverse"""
    return row < 0 or row > height or col < 0 or col > width


def longest_subarray_with_sum(array, target_sum):
    """
    Find the longest subarray that sums to the target sum.
    
    Approach: Use hash table to store cumulative sums
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    indices = {0: -1}
    longest_length = 0
    current_sum = 0
    
    for i, num in enumerate(array):
        current_sum += num
        
        if current_sum - target_sum in indices:
            longest_length = max(longest_length, i - indices[current_sum - target_sum])
        
        if current_sum not in indices:
            indices[current_sum] = i
    
    return longest_length


def knight_connection(knight_a, knight_b):
    """
    Find the minimum number of moves for a knight to reach from position a to b.
    
    Approach: BFS with chess board constraints
    Time Complexity: O(1) since board is fixed size
    Space Complexity: O(1)
    """
    from collections import deque
    
    def get_neighbors(pos):
        x, y = pos
        moves = [
            (x + 2, y + 1), (x + 2, y - 1),
            (x - 2, y + 1), (x - 2, y - 1),
            (x + 1, y + 2), (x + 1, y - 2),
            (x - 1, y + 2), (x - 1, y - 2)
        ]
        return [(nx, ny) for nx, ny in moves if 0 <= nx < 8 and 0 <= ny < 8]
    
    queue = deque([(knight_a, 0)])
    visited = {knight_a}
    
    while queue:
        pos, moves = queue.popleft()
        
        if pos == knight_b:
            return moves
        
        for neighbor in get_neighbors(pos):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, moves + 1))
    
    return -1


def count_squares(matrix):
    """
    Count the number of squares of 1s in a binary matrix.
    
    Approach: Dynamic programming - each cell represents the size of largest square ending at that cell
    Time Complexity: O(m*n)
    Space Complexity: O(m*n)
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                count += dp[i][j]
    
    return count


def same_bsts(array_one, array_two):
    """
    Check if two arrays represent the same BST.
    
    Approach: Recursive comparison with BST property
    Time Complexity: O(n²) worst case
    Space Complexity: O(d) where d is the depth of the tree
    """
    if len(array_one) != len(array_two):
        return False
    
    if len(array_one) == 0 and len(array_two) == 0:
        return True
    
    if array_one[0] != array_two[0]:
        return False
    
    left_one = get_smaller(array_one)
    left_two = get_smaller(array_two)
    right_one = get_bigger_or_equal(array_one)
    right_two = get_bigger_or_equal(array_two)
    
    return same_bsts(left_one, left_two) and same_bsts(right_one, right_two)


def get_smaller(array):
    """Helper function for same_bsts"""
    return [x for x in array[1:] if x < array[0]]


def get_bigger_or_equal(array):
    """Helper function for same_bsts"""
    return [x for x in array[1:] if x >= array[0]]


def validate_three_nodes(node_one, node_two, node_three):
    """
    Check if node_two is a descendant of node_one and node_three is a descendant of node_two.
    
    Approach: Check both directions of the relationship
    Time Complexity: O(h) where h is the height of the tree
    Space Complexity: O(1)
    """
    search_one = node_one
    search_two = node_three
    
    while True:
        found_three_from_one = search_one is node_three
        found_one_from_three = search_two is node_one
        found_node_two = search_one is node_two or search_two is node_two
        finished_searching = search_one is None and search_two is None
        
        if found_three_from_one or found_one_from_three or found_node_two or finished_searching:
            break
        
        if search_one is not None:
            search_one = search_one.left if node_two.value < search_one.value else search_one.right
        
        if search_two is not None:
            search_two = search_two.left if node_two.value < search_two.value else search_two.right
    
    found_node_from_other = search_one is node_three or search_two is node_one
    found_node_two = search_one is node_two or search_two is node_two
    
    if not found_node_two or found_node_from_other:
        return False
    
    return search_for_target(node_two, node_three if search_one is node_two else node_one)


def search_for_target(node, target):
    """Helper function for validate_three_nodes"""
    while node is not None and node is not target:
        node = node.left if target.value < node.value else node.right
    
    return node is target


"""
Advanced Algorithmic Problems - Part 2: Trees, Dynamic Programming, and Graphs
==============================================================================

This file contains implementations of tree algorithms, dynamic programming problems,
and graph algorithms with detailed comments and complexity analysis.
"""

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def repair_bst(tree):
    """
    Repair a BST where exactly two nodes have been swapped.
    
    Approach: Inorder traversal to find the two swapped nodes
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    def inorder_traversal(node, nodes):
        if node is None:
            return
        inorder_traversal(node.left, nodes)
        nodes.append(node)
        inorder_traversal(node.right, nodes)
    
    nodes = []
    inorder_traversal(tree, nodes)
    
    # Find the two nodes that are out of order
    first = None
    second = None
    
    for i in range(len(nodes) - 1):
        if nodes[i].value > nodes[i + 1].value:
            if first is None:
                first = nodes[i]
            second = nodes[i + 1]
    
    # Swap the values
    if first and second:
        first.value, second.value = second.value, first.value
    
    return tree


def sum_bsts(tree):
    """
    Calculate the sum of all values in a BST.
    
    Approach: Recursive traversal
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    if tree is None:
        return 0
    
    return tree.value + sum_bsts(tree.left) + sum_bsts(tree.right)


def max_path_sum_in_binary_tree(tree):
    """
    Find the maximum path sum in a binary tree.
    
    Approach: Recursive DFS with path tracking
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    def max_path_sum_helper(node):
        if node is None:
            return 0
        
        left_sum = max(0, max_path_sum_helper(node.left))
        right_sum = max(0, max_path_sum_helper(node.right))
        
        # Update global maximum
        max_path_sum_in_binary_tree.max_sum = max(
            max_path_sum_in_binary_tree.max_sum,
            node.value + left_sum + right_sum
        )
        
        return node.value + max(left_sum, right_sum)
    
    max_path_sum_in_binary_tree.max_sum = float('-inf')
    max_path_sum_helper(tree)
    return max_path_sum_in_binary_tree.max_sum


def find_nodes_distance_k(tree, target, k):
    """
    Find all nodes at distance k from the target node.
    
    Approach: BFS from target node with distance tracking
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    from collections import deque, defaultdict
    
    # Build parent map
    parent_map = {}
    def build_parent_map(node, parent=None):
        if node is None:
            return
        parent_map[node] = parent
        build_parent_map(node.left, node)
        build_parent_map(node.right, node)
    
    build_parent_map(tree)
    
    # BFS to find nodes at distance k
    queue = deque([(target, 0)])
    visited = {target}
    result = []
    
    while queue:
        node, distance = queue.popleft()
        
        if distance == k:
            result.append(node.value)
            continue
        
        # Add unvisited neighbors
        neighbors = [node.left, node.right, parent_map.get(node)]
        for neighbor in neighbors:
            if neighbor and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return result


def max_sum_increasing_subsequence(array):
    """
    Find the maximum sum of an increasing subsequence.
    
    Approach: Dynamic programming with binary search optimization
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    import bisect
    
    if not array:
        return 0
    
    # dp[i] represents the maximum sum ending at index i
    dp = [0] * len(array)
    dp[0] = array[0]
    
    for i in range(1, len(array)):
        dp[i] = array[i]
        for j in range(i):
            if array[j] < array[i]:
                dp[i] = max(dp[i], dp[j] + array[i])
    
    return max(dp)


def longest_common_subsequence(str1, str2):
    """
    Find the length of the longest common subsequence between two strings.
    
    Approach: Dynamic programming
    Time Complexity: O(m*n)
    Space Complexity: O(m*n)
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def min_number_of_jumps(array):
    """
    Find the minimum number of jumps needed to reach the end of the array.
    
    Approach: Greedy algorithm with reach tracking
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(array) <= 1:
        return 0
    
    jumps = 0
    current_reach = array[0]
    max_reach = array[0]
    
    for i in range(1, len(array)):
        if i > current_reach:
            jumps += 1
            current_reach = max_reach
        
        max_reach = max(max_reach, i + array[i])
        
        if current_reach >= len(array) - 1:
            return jumps + 1
    
    return -1


def water_area(heights):
    """
    Calculate the total water area trapped between bars.
    
    Approach: Two-pointer approach
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not heights:
        return 0
    
    left = 0
    right = len(heights) - 1
    left_max = heights[left]
    right_max = heights[right]
    water = 0
    
    while left < right:
        if heights[left] < heights[right]:
            if heights[left] >= left_max:
                left_max = heights[left]
            else:
                water += left_max - heights[left]
            left += 1
        else:
            if heights[right] >= right_max:
                right_max = heights[right]
            else:
                water += right_max - heights[right]
            right -= 1
    
    return water


def knapsack_problem(items, capacity):
    """
    Solve the 0/1 knapsack problem.
    
    Approach: Dynamic programming
    Time Complexity: O(n*capacity)
    Space Complexity: O(n*capacity)
    """
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        weight, value = items[i - 1]
        for w in range(capacity + 1):
            if weight <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]


def disk_stacking(disks):
    """
    Find the maximum height possible by stacking disks.
    
    Approach: Dynamic programming with sorting
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    # Sort disks by width, height, depth
    disks.sort(key=lambda x: (x[0], x[1], x[2]))
    
    heights = [disk[1] for disk in disks]
    sequences = [None for _ in disks]
    max_height_idx = 0
    
    for i in range(1, len(disks)):
        current_disk = disks[i]
        for j in range(i):
            other_disk = disks[j]
            if (other_disk[0] < current_disk[0] and 
                other_disk[1] < current_disk[1] and 
                other_disk[2] < current_disk[2]):
                if heights[i] <= heights[j] + current_disk[1]:
                    heights[i] = heights[j] + current_disk[1]
                    sequences[i] = j
        
        if heights[i] >= heights[max_height_idx]:
            max_height_idx = i
    
    return build_sequence(disks, sequences, max_height_idx)


def build_sequence(disks, sequences, current_idx):
    """Helper function for disk_stacking"""
    sequence = []
    while current_idx is not None:
        sequence.append(disks[current_idx])
        current_idx = sequences[current_idx]
    return list(reversed(sequence))


def numbers_in_pi(pi, numbers):
    """
    Find the minimum number of spaces needed to separate pi into valid numbers.
    
    Approach: Dynamic programming with memoization
    Time Complexity: O(n³ + m) where n is length of pi, m is total length of numbers
    Space Complexity: O(n + m)
    """
    numbers_table = {number: True for number in numbers}
    cache = {}
    
    def get_min_spaces(idx):
        if idx == len(pi):
            return -1
        
        if idx in cache:
            return cache[idx]
        
        min_spaces = float('inf')
        for i in range(idx, len(pi)):
            prefix = pi[idx:i + 1]
            if prefix in numbers_table:
                min_spaces_in_suffix = get_min_spaces(i + 1)
                if min_spaces_in_suffix == -1:
                    min_spaces = min(min_spaces, 0)
                else:
                    min_spaces = min(min_spaces, min_spaces_in_suffix + 1)
        
        cache[idx] = min_spaces if min_spaces != float('inf') else -1
        return cache[idx]
    
    return get_min_spaces(0)


def maximum_sum_submatrix(matrix, size):
    """
    Find the maximum sum of a submatrix of given size.
    
    Approach: 2D sliding window with prefix sums
    Time Complexity: O(m*n)
    Space Complexity: O(m*n)
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    
    # Calculate prefix sums
    prefix_sums = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows):
        for j in range(cols):
            prefix_sums[i + 1][j + 1] = (prefix_sums[i + 1][j] + 
                                        prefix_sums[i][j + 1] - 
                                        prefix_sums[i][j] + 
                                        matrix[i][j])
    
    max_sum = float('-inf')
    for i in range(size, rows + 1):
        for j in range(size, cols + 1):
            current_sum = (prefix_sums[i][j] - 
                          prefix_sums[i - size][j] - 
                          prefix_sums[i][j - size] + 
                          prefix_sums[i - size][j - size])
            max_sum = max(max_sum, current_sum)
    
    return max_sum


def maximize_expression(array):
    """
    Maximize the expression: A - B + C - D where A, B, C, D are array elements.
    
    Approach: Dynamic programming with state tracking
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(array) < 4:
        return 0
    
    # dp[i][j] represents max value using first i elements with j operators used
    dp = [[float('-inf')] * 4 for _ in range(len(array) + 1)]
    dp[0][0] = 0
    
    for i in range(1, len(array) + 1):
        for j in range(4):
            # Don't use current element
            dp[i][j] = dp[i - 1][j]
            
            # Use current element
            if j > 0:
                if j == 1:  # A
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + array[i - 1])
                elif j == 2:  # A - B
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] - array[i - 1])
                elif j == 3:  # A - B + C
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + array[i - 1])
                else:  # A - B + C - D
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] - array[i - 1])
    
    return dp[len(array)][3]


def dice_throws(n, faces, total):
    """
    Count the number of ways to roll n dice to get a total sum.
    
    Approach: Dynamic programming
    Time Complexity: O(n*total*faces)
    Space Complexity: O(n*total)
    """
    dp = [[0] * (total + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        for j in range(1, total + 1):
            for k in range(1, min(faces + 1, j + 1)):
                dp[i][j] += dp[i - 1][j - k]
    
    return dp[n][total]


def juice_bottling(prices):
    """
    Find the optimal way to bottle juice to maximize profit.
    
    Approach: Dynamic programming
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    n = len(prices)
    dp = [0] * (n + 1)
    cuts = [0] * (n + 1)
    
    for i in range(1, n + 1):
        max_val = 0
        for j in range(i):
            if prices[j] + dp[i - j - 1] > max_val:
                max_val = prices[j] + dp[i - j - 1]
                cuts[i] = j + 1
        dp[i] = max_val
    
    # Reconstruct the solution
    result = []
    remaining = n
    while remaining > 0:
        result.append(cuts[remaining])
        remaining -= cuts[remaining]
    
    return result


def dijkstra_algorithm(graph, start):
    """
    Find shortest paths from start vertex to all other vertices.
    
    Approach: Dijkstra's algorithm with priority queue
    Time Complexity: O((V + E) log V) with binary heap
    Space Complexity: O(V)
    """
    import heapq
    
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_vertex in visited:
            continue
        
        visited.add(current_vertex)
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances


def topological_sort(graph):
    """
    Perform topological sort on a directed acyclic graph.
    
    Approach: DFS with cycle detection
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    def dfs(node):
        if node in visiting:
            return False  # Cycle detected
        if node in visited:
            return True
        
        visiting.add(node)
        
        for neighbor in graph.get(node, []):
            if not dfs(neighbor):
                return False
        
        visiting.remove(node)
        visited.add(node)
        result.append(node)
        return True
    
    result = []
    visited = set()
    visiting = set()
    
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []  # Cycle detected
    
    return list(reversed(result))


def kruskal_algorithm(edges, num_vertices):
    """
    Find minimum spanning tree using Kruskal's algorithm.
    
    Approach: Union-Find with sorting
    Time Complexity: O(E log E)
    Space Complexity: O(V)
    """
    def find(parent, i):
        if parent[i] != i:
            parent[i] = find(parent, parent[i])
        return parent[i]
    
    def union(parent, rank, x, y):
        root_x = find(parent, x)
        root_y = find(parent, y)
        
        if root_x == root_y:
            return False
        
        if rank[root_x] < rank[root_y]:
            root_x, root_y = root_y, root_x
        
        parent[root_y] = root_x
        if rank[root_x] == rank[root_y]:
            rank[root_x] += 1
        
        return True
    
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    parent = list(range(num_vertices))
    rank = [0] * num_vertices
    mst = []
    
    for u, v, weight in edges:
        if union(parent, rank, u, v):
            mst.append((u, v, weight))
    
    return mst


def prim_algorithm(graph):
    """
    Find minimum spanning tree using Prim's algorithm.
    
    Approach: Priority queue with adjacency list
    Time Complexity: O(E log V)
    Space Complexity: O(V)
    """
    import heapq
    
    if not graph:
        return []
    
    start_vertex = next(iter(graph))
    visited = {start_vertex}
    edges = []
    pq = [(weight, start_vertex, neighbor) 
          for neighbor, weight in graph[start_vertex].items()]
    heapq.heapify(pq)
    
    while pq and len(visited) < len(graph):
        weight, u, v = heapq.heappop(pq)
        
        if v in visited:
            continue
        
        visited.add(v)
        edges.append((u, v, weight))
        
        for neighbor, weight in graph[v].items():
            if neighbor not in visited:
                heapq.heappush(pq, (weight, v, neighbor))
    
    return edges


def boggle_board(board, words):
    """
    Find all words from the dictionary that can be formed on the boggle board.
    
    Approach: Trie + DFS
    Time Complexity: O(nm * 8^s) where s is max word length
    Space Complexity: O(w*s) where w is number of words
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False
            self.word = ""
    
    def build_trie(words):
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.word = word
        return root
    
    def dfs(i, j, node, visited):
        if (i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or 
            visited[i][j] or board[i][j] not in node.children):
            return
        
        visited[i][j] = True
        node = node.children[board[i][j]]
        
        if node.is_end:
            result.add(node.word)
        
        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            dfs(i + di, j + dj, node, visited)
        
        visited[i][j] = False
    
    trie = build_trie(words)
    result = set()
    visited = [[False] * len(board[0]) for _ in range(len(board))]
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            dfs(i, j, trie, visited)
    
    return list(result)


def largest_island(matrix):
    """
    Find the size of the largest island in a binary matrix.
    
    Approach: DFS with area calculation
    Time Complexity: O(m*n)
    Space Complexity: O(m*n)
    """
    def dfs(i, j):
        if (i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or 
            matrix[i][j] == 0):
            return 0
        
        matrix[i][j] = 0  # Mark as visited
        area = 1
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            area += dfs(i + di, j + dj)
        
        return area
    
    max_area = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                max_area = max(max_area, dfs(i, j))
    
    return max_area


"""
Advanced Algorithmic Problems - Part 3: Linked Lists, Strings, Sorting, and More
===============================================================================

This file contains implementations of linked list algorithms, string algorithms,
sorting algorithms, and other advanced problems with detailed comments and complexity analysis.
"""

class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def continuous_median(numbers):
    """
    Maintain a running median of a stream of numbers.
    
    Approach: Use two heaps - max heap for lower half, min heap for upper half
    Time Complexity: O(log n) per insertion
    Space Complexity: O(n)
    """
    import heapq
    
    max_heap = []  # Lower half (max heap using negative values)
    min_heap = []  # Upper half (min heap)
    
    def add_number(num):
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)
        
        # Balance heaps
        if len(max_heap) > len(min_heap) + 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
    
    def get_median():
        if len(max_heap) == len(min_heap):
            return (-max_heap[0] + min_heap[0]) / 2
        else:
            return -max_heap[0]
    
    medians = []
    for num in numbers:
        add_number(num)
        medians.append(get_median())
    
    return medians


def sort_k_sorted_array(array, k):
    """
    Sort an array where each element is at most k positions away from its sorted position.
    
    Approach: Use min heap to maintain k+1 elements
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    """
    import heapq
    
    if k >= len(array):
        return sorted(array)
    
    min_heap = array[:k + 1]
    heapq.heapify(min_heap)
    
    result = []
    for i in range(k + 1, len(array)):
        result.append(heapq.heappop(min_heap))
        heapq.heappush(min_heap, array[i])
    
    while min_heap:
        result.append(heapq.heappop(min_heap))
    
    return result


def laptop_rentals(times):
    """
    Find the minimum number of laptops needed for rental times.
    
    Approach: Sort start and end times, use two pointers
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    start_times = sorted([time[0] for time in times])
    end_times = sorted([time[1] for time in times])
    
    laptop_count = 0
    max_laptops = 0
    start_idx = 0
    end_idx = 0
    
    while start_idx < len(start_times):
        if start_times[start_idx] < end_times[end_idx]:
            laptop_count += 1
            max_laptops = max(max_laptops, laptop_count)
            start_idx += 1
        else:
            laptop_count -= 1
            end_idx += 1
    
    return max_laptops


def find_loop(head):
    """
    Detect if a linked list has a cycle and return the node where the cycle begins.
    
    Approach: Floyd's Cycle-Finding Algorithm (Tortoise and Hare)
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return None
    
    slow = head
    fast = head
    
    # Find meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    
    if slow != fast:
        return None
    
    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow


def reverse_linked_list(head):
    """
    Reverse a linked list iteratively.
    
    Approach: Three pointers (prev, curr, next)
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    return prev


def merge_linked_lists(head_one, head_two):
    """
    Merge two sorted linked lists.
    
    Approach: Compare and link nodes
    Time Complexity: O(n + m)
    Space Complexity: O(1)
    """
    dummy = LinkedList(0)
    current = dummy
    
    while head_one and head_two:
        if head_one.value <= head_two.value:
            current.next = head_one
            head_one = head_one.next
        else:
            current.next = head_two
            head_two = head_two.next
        current = current.next
    
    current.next = head_one if head_one else head_two
    return dummy.next


def shift_linked_list(head, k):
    """
    Shift a linked list by k positions.
    
    Approach: Find length, adjust k, find new head and tail
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or k == 0:
        return head
    
    # Find length and tail
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    
    # Normalize k
    k = k % length
    if k < 0:
        k += length
    
    if k == 0:
        return head
    
    # Find new head and tail
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next
    
    new_head = new_tail.next
    new_tail.next = None
    tail.next = head
    
    return new_head


def lowest_common_manager(top_manager, report_one, report_two):
    """
    Find the lowest common manager of two reports in an organizational hierarchy.
    
    Approach: DFS with return values
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    def get_lowest_common_manager(node):
        if node is None:
            return None, 0
        
        reports_found = 0
        if node == report_one or node == report_two:
            reports_found += 1
        
        for direct_report in node.direct_reports:
            lcm, reports_in_subtree = get_lowest_common_manager(direct_report)
            if lcm is not None:
                return lcm, reports_in_subtree
            reports_found += reports_in_subtree
        
        if reports_found == 2:
            return node, reports_found
        
        return None, reports_found
    
    lcm, _ = get_lowest_common_manager(top_manager)
    return lcm


def interweaving_strings(one, two, three):
    """
    Check if string three can be formed by interweaving strings one and two.
    
    Approach: Dynamic programming with memoization
    Time Complexity: O(n*m)
    Space Complexity: O(n*m)
    """
    if len(three) != len(one) + len(two):
        return False
    
    def can_interweave(i, j, k, memo):
        if k == len(three):
            return True
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        result = False
        if i < len(one) and one[i] == three[k]:
            result = can_interweave(i + 1, j, k + 1, memo)
        
        if not result and j < len(two) and two[j] == three[k]:
            result = can_interweave(i, j + 1, k + 1, memo)
        
        memo[(i, j)] = result
        return result
    
    return can_interweave(0, 0, 0, {})


def solve_sudoku(board):
    """
    Solve a 9x9 Sudoku puzzle.
    
    Approach: Backtracking with constraint checking
    Time Complexity: O(9^(n²)) worst case
    Space Complexity: O(n²)
    """
    def is_valid(board, row, col, num):
        # Check row
        for x in range(9):
            if board[row][x] == num:
                return False
        
        # Check column
        for x in range(9):
            if board[x][col] == num:
                return False
        
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if board[i + start_row][j + start_col] == num:
                    return False
        
        return True
    
    def solve(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if solve(board):
                                return True
                            board[i][j] = 0
                    return False
        return True
    
    solve(board)
    return board


def generate_div_tags(number_of_tags):
    """
    Generate all valid combinations of opening and closing div tags.
    
    Approach: Backtracking with balance tracking
    Time Complexity: O(4^n / sqrt(n))
    Space Complexity: O(n)
    """
    def generate_tags(open_count, close_count, current, result):
        if open_count == 0 and close_count == 0:
            result.append(current)
            return
        
        if open_count > 0:
            generate_tags(open_count - 1, close_count, current + "<div>", result)
        
        if close_count > open_count:
            generate_tags(open_count, close_count - 1, current + "</div>", result)
    
    result = []
    generate_tags(number_of_tags, number_of_tags, "", result)
    return result


def ambiguous_measurements(measuring_cups, low, high):
    """
    Check if a target range can be measured using given measuring cups.
    
    Approach: Dynamic programming with memoization
    Time Complexity: O(n * (high - low))
    Space Complexity: O(high - low)
    """
    def can_measure_in_range(target_low, target_high, memo):
        if target_low <= 0 and target_high >= 0:
            return True
        
        if target_high < 0:
            return False
        
        key = (target_low, target_high)
        if key in memo:
            return memo[key]
        
        for cup_low, cup_high in measuring_cups:
            if can_measure_in_range(target_low - cup_high, target_high - cup_low, memo):
                memo[key] = True
                return True
        
        memo[key] = False
        return False
    
    return can_measure_in_range(low, high, {})


def shifted_binary_search(array, target):
    """
    Search for target in a shifted sorted array.
    
    Approach: Modified binary search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(array) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if array[mid] == target:
            return mid
        
        # Check if left half is sorted
        if array[left] <= array[mid]:
            if array[left] <= target < array[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if array[mid] < target <= array[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


def search_for_range(array, target):
    """
    Find the first and last occurrence of target in a sorted array.
    
    Approach: Binary search for first and last occurrence
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    def find_first(array, target):
        left, right = 0, len(array) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if array[mid] == target:
                result = mid
                right = mid - 1
            elif array[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def find_last(array, target):
        left, right = 0, len(array) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if array[mid] == target:
                result = mid
                left = mid + 1
            elif array[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    first = find_first(array, target)
    if first == -1:
        return [-1, -1]
    
    last = find_last(array, target)
    return [first, last]


def quickselect(array, k):
    """
    Find the kth smallest element in an unsorted array.
    
    Approach: Quickselect algorithm
    Time Complexity: O(n) average, O(n²) worst case
    Space Complexity: O(1)
    """
    def partition(left, right, pivot_index):
        pivot = array[pivot_index]
        array[pivot_index], array[right] = array[right], array[pivot_index]
        
        store_index = left
        for i in range(left, right):
            if array[i] < pivot:
                array[store_index], array[i] = array[i], array[store_index]
                store_index += 1
        
        array[right], array[store_index] = array[store_index], array[right]
        return store_index
    
    def select(left, right, k_smallest):
        if left == right:
            return array[left]
        
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if k_smallest == pivot_index:
            return array[k_smallest]
        elif k_smallest < pivot_index:
            return select(left, pivot_index - 1, k_smallest)
        else:
            return select(pivot_index + 1, right, k_smallest)
    
    return select(0, len(array) - 1, k - 1)


def index_equals_value(array):
    """
    Find the first index where array[index] == index.
    
    Approach: Binary search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(array) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if array[mid] == mid:
            result = mid
            right = mid - 1  # Look for earlier occurrence
        elif array[mid] < mid:
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def quick_sort(array):
    """
    Sort array using quicksort algorithm.
    
    Approach: Divide and conquer with pivot selection
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(log n) average
    """
    def partition(left, right):
        pivot = array[right]
        i = left - 1
        
        for j in range(left, right):
            if array[j] <= pivot:
                i += 1
                array[i], array[j] = array[j], array[i]
        
        array[i + 1], array[right] = array[right], array[i + 1]
        return i + 1
    
    def quick_sort_helper(left, right):
        if left < right:
            pivot_index = partition(left, right)
            quick_sort_helper(left, pivot_index - 1)
            quick_sort_helper(pivot_index + 1, right)
    
    quick_sort_helper(0, len(array) - 1)
    return array


def heap_sort(array):
    """
    Sort array using heapsort algorithm.
    
    Approach: Build max heap and extract elements
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    def heapify(n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and array[left] > array[largest]:
            largest = left
        
        if right < n and array[right] > array[largest]:
            largest = right
        
        if largest != i:
            array[i], array[largest] = array[largest], array[i]
            heapify(n, largest)
    
    # Build max heap
    n = len(array)
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    
    # Extract elements from heap
    for i in range(n - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        heapify(i, 0)
    
    return array


def radix_sort(array):
    """
    Sort array using radix sort algorithm.
    
    Approach: Sort by each digit from least to most significant
    Time Complexity: O(d * (n + k)) where d is max digits, k is base
    Space Complexity: O(n + k)
    """
    def counting_sort(array, exp):
        n = len(array)
        output = [0] * n
        count = [0] * 10
        
        # Count occurrences
        for i in range(n):
            index = (array[i] // exp) % 10
            count[index] += 1
        
        # Calculate positions
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        # Build output array
        for i in range(n - 1, -1, -1):
            index = (array[i] // exp) % 10
            output[count[index] - 1] = array[i]
            count[index] -= 1
        
        # Copy back to original array
        for i in range(n):
            array[i] = output[i]
    
    if not array:
        return array
    
    max_num = max(array)
    exp = 1
    
    while max_num // exp > 0:
        counting_sort(array, exp)
        exp *= 10
    
    return array


def shorten_path(path):
    """
    Shorten a Unix-style file path by resolving . and .. components.
    
    Approach: Stack-based approach
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not path:
        return "/"
    
    is_absolute = path[0] == "/"
    tokens = [token for token in path.split("/") if token and token != "."]
    stack = []
    
    for token in tokens:
        if token == "..":
            if stack and stack[-1] != "..":
                stack.pop()
            elif not is_absolute:
                stack.append(token)
        else:
            stack.append(token)
    
    result = "/".join(stack)
    return "/" + result if is_absolute else result


def largest_rectangle_under_skyline(buildings):
    """
    Find the area of the largest rectangle that can be formed under the skyline.
    
    Approach: Stack-based approach
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    max_area = 0
    i = 0
    
    while i < len(buildings):
        if not stack or buildings[stack[-1]] <= buildings[i]:
            stack.append(i)
            i += 1
        else:
            height = buildings[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
    
    while stack:
        height = buildings[stack.pop()]
        width = i if not stack else i - stack[-1] - 1
        max_area = max(max_area, height * width)
    
    return max_area


def longest_substring_without_duplication(string):
    """
    Find the longest substring without duplicate characters.
    
    Approach: Sliding window with hash set
    Time Complexity: O(n)
    Space Complexity: O(min(m, n)) where m is charset size
    """
    char_set = set()
    left = 0
    max_length = 0
    max_start = 0
    
    for right in range(len(string)):
        while string[right] in char_set:
            char_set.remove(string[left])
            left += 1
        
        char_set.add(string[right])
        
        if right - left + 1 > max_length:
            max_length = right - left + 1
            max_start = left
    
    return string[max_start:max_start + max_length]


def underscorify_substring(string, substring):
    """
    Add underscores around all occurrences of substring in string.
    
    Approach: Find all occurrences and merge overlapping ones
    Time Complexity: O(n*m) where n is string length, m is substring length
    Space Complexity: O(n)
    """
    def get_locations(string, substring):
        locations = []
        start_idx = 0
        
        while start_idx < len(string):
            next_idx = string.find(substring, start_idx)
            if next_idx != -1:
                locations.append([next_idx, next_idx + len(substring)])
                start_idx = next_idx + 1
            else:
                break
        
        return locations
    
    def collapse(locations):
        if not locations:
            return locations
        
        new_locations = [locations[0]]
        previous = new_locations[0]
        
        for i in range(1, len(locations)):
            current = locations[i]
            if current[0] <= previous[1]:
                previous[1] = current[1]
            else:
                new_locations.append(current)
                previous = current
        
        return new_locations
    
    locations = collapse(get_locations(string, substring))
    
    if not locations:
        return string
    
    result = []
    string_idx = 0
    location_idx = 0
    
    while string_idx < len(string):
        if location_idx < len(locations) and string_idx == locations[location_idx][0]:
            result.append("_")
        
        result.append(string[string_idx])
        
        if location_idx < len(locations) and string_idx == locations[location_idx][1] - 1:
            result.append("_")
            location_idx += 1
        
        string_idx += 1
    
    return "".join(result)


def pattern_matcher(pattern, string):
    """
    Check if string matches pattern where x and y can be any strings.
    
    Approach: Try different combinations of x and y
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    if len(pattern) > len(string):
        return []
    
    def get_new_pattern(pattern):
        if pattern[0] == 'x':
            return pattern
        else:
            return ''.join(['x' if char == 'y' else 'y' for char in pattern])
    
    def get_counts_and_first_y_pos(pattern):
        counts = {'x': 0, 'y': 0}
        first_y_pos = None
        
        for i, char in enumerate(pattern):
            counts[char] += 1
            if char == 'y' and first_y_pos is None:
                first_y_pos = i
        
        return counts, first_y_pos
    
    def build_candidates(string, x_len, y_len, first_y_pos):
        x = string[:x_len]
        y_start = first_y_pos * x_len
        y = string[y_start:y_start + y_len]
        return x, y
    
    def matches_pattern(string, pattern, x, y):
        result = []
        pattern_idx = 0
        
        for char in pattern:
            if char == 'x':
                result.append(x)
            else:
                result.append(y)
        
        return ''.join(result) == string
    
    pattern = get_new_pattern(pattern)
    counts, first_y_pos = get_counts_and_first_y_pos(pattern)
    
    if counts['y'] == 0:
        if len(string) % counts['x'] != 0:
            return []
        x_len = len(string) // counts['x']
        x = string[:x_len]
        if matches_pattern(string, pattern, x, ''):
            return [x, '']
        return []
    
    for x_len in range(1, len(string)):
        y_len = (len(string) - counts['x'] * x_len) / counts['y']
        if y_len <= 0 or y_len != int(y_len):
            continue
        
        y_len = int(y_len)
        x, y = build_candidates(string, x_len, y_len, first_y_pos)
        
        if matches_pattern(string, pattern, x, y):
            return [x, y]
    
    return []


def multi_string_search(big_string, small_strings):
    """
    Find which small strings are contained in the big string.
    
    Approach: Build suffix trie and search
    Time Complexity: O(b² + ns) where b is big string length, n is number of small strings, s is max small string length
    Space Complexity: O(b² + ns)
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.end_word = None
    
    def build_suffix_trie(string):
        root = TrieNode()
        for i in range(len(string)):
            node = root
            for j in range(i, len(string)):
                char = string[j]
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
        return root
    
    def contains_string(big_string, trie, string):
        node = trie
        for char in string:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    trie = build_suffix_trie(big_string)
    return [contains_string(big_string, trie, string) for string in small_strings]


def longest_most_frequent_prefix(strings):
    """
    Find the longest prefix that appears in the most strings.
    
    Approach: Build prefix trie and count occurrences
    Time Complexity: O(n*s) where n is number of strings, s is max string length
    Space Complexity: O(n*s)
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.count = 0
    
    def build_prefix_trie(strings):
        root = TrieNode()
        for string in strings:
            node = root
            for char in string:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.count += 1
        return root
    
    def find_longest_most_frequent(trie, total_strings):
        result = ""
        max_count = 0
        current = trie
        
        while current.children:
            best_char = None
            best_count = 0
            
            for char, child in current.children.items():
                if child.count > best_count:
                    best_count = child.count
                    best_char = char
            
            if best_count < total_strings // 2:
                break
            
            result += best_char
            current = current.children[best_char]
        
        return result
    
    trie = build_prefix_trie(strings)
    return find_longest_most_frequent(trie, len(strings))


def shortest_unique_prefixes(strings):
    """
    Find the shortest unique prefix for each string.
    
    Approach: Build prefix trie and find unique prefixes
    Time Complexity: O(n*s) where n is number of strings, s is max string length
    Space Complexity: O(n*s)
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.count = 0
            self.end_word = None
    
    def build_trie(strings):
        root = TrieNode()
        for string in strings:
            node = root
            for char in string:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.count += 1
            node.end_word = string
        return root
    
    def find_shortest_prefixes(trie, result):
        if trie.count == 1 and trie.end_word:
            result[trie.end_word] = trie.end_word
            return
        
        for char, child in trie.children.items():
            if child.count == 1 and child.end_word:
                result[child.end_word] = child.end_word
            else:
                find_shortest_prefixes(child, result)
    
    trie = build_trie(strings)
    result = {}
    find_shortest_prefixes(trie, result)
    return result


