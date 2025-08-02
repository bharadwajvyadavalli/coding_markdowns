"""
Advanced Algorithms Collection
This file contains implementations of various advanced algorithms with detailed comments
and time/space complexity analysis.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import heapq
import math


def apartment_hunting(blocks: List[Dict[str, bool]], reqs: List[str]) -> int:
    """
    Apartment Hunting Problem
    
    Find the optimal block to live in that minimizes the maximum distance to all required amenities.
    
    Args:
        blocks: List of dictionaries where each dict represents a block and contains
               boolean values for each amenity (True if present, False if not)
        reqs: List of required amenities
    
    Returns:
        Index of the optimal block to live in
    
    Time Complexity: O(B * R) where B = number of blocks, R = number of requirements
    Space Complexity: O(B * R) for storing distances
    """
    if not blocks or not reqs:
        return 0
    
    num_blocks = len(blocks)
    num_reqs = len(reqs)
    
    # For each requirement, find the minimum distance to each block
    min_distances = [[float('inf')] * num_blocks for _ in range(num_reqs)]
    
    # For each requirement
    for req_idx, req in enumerate(reqs):
        # Find blocks that have this requirement
        blocks_with_req = []
        for block_idx, block in enumerate(blocks):
            if block.get(req, False):
                blocks_with_req.append(block_idx)
        
        # Calculate minimum distance from each block to this requirement
        for block_idx in range(num_blocks):
            min_dist = float('inf')
            for req_block_idx in blocks_with_req:
                min_dist = min(min_dist, abs(block_idx - req_block_idx))
            min_distances[req_idx][block_idx] = min_dist
    
    # Find the block with minimum maximum distance
    optimal_block = 0
    min_max_distance = float('inf')
    
    for block_idx in range(num_blocks):
        max_distance = max(min_distances[req_idx][block_idx] for req_idx in range(num_reqs))
        if max_distance < min_max_distance:
            min_max_distance = max_distance
            optimal_block = block_idx
    
    return optimal_block


def calendar_matching(calendar1: List[Dict], daily_bounds1: Dict, 
                     calendar2: List[Dict], daily_bounds2: Dict, 
                     meeting_duration: int) -> List[Dict]:
    """
    Calendar Matching Problem
    
    Find available time slots for a meeting between two people given their calendars
    and daily availability bounds.
    
    Args:
        calendar1: List of meetings for person 1, each with 'start' and 'end' times
        daily_bounds1: Daily availability bounds for person 1
        calendar2: List of meetings for person 2, each with 'start' and 'end' times
        daily_bounds2: Daily availability bounds for person 2
        meeting_duration: Duration of the meeting in minutes
    
    Returns:
        List of available time slots for the meeting
    
    Time Complexity: O(C1 + C2) where C1, C2 are the number of meetings for each person
    Space Complexity: O(C1 + C2) for storing merged calendar
    """
    def time_to_minutes(time_str: str) -> int:
        """Convert time string (HH:MM) to minutes since midnight"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    def minutes_to_time(minutes: int) -> str:
        """Convert minutes since midnight to time string (HH:MM)"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    # Convert all times to minutes for easier comparison
    def convert_calendar(calendar: List[Dict], daily_bounds: Dict) -> List[List[int]]:
        converted = []
        for meeting in calendar:
            converted.append([
                time_to_minutes(meeting['start']),
                time_to_minutes(meeting['end'])
            ])
        return converted
    
    # Merge calendars and find available slots
    def merge_calendars(cal1: List[List[int]], bounds1: Dict,
                       cal2: List[List[int]], bounds2: Dict) -> List[List[int]]:
        # Add daily bounds as meetings
        start1 = time_to_minutes(bounds1['start'])
        end1 = time_to_minutes(bounds1['end'])
        start2 = time_to_minutes(bounds2['start'])
        end2 = time_to_minutes(bounds2['end'])
        
        # Use the later start time and earlier end time
        merged_start = max(start1, start2)
        merged_end = min(end1, end2)
        
        # Combine all meetings
        all_meetings = cal1 + cal2
        all_meetings.append([merged_start, merged_start])  # Start bound
        all_meetings.append([merged_end, merged_end])      # End bound
        
        # Sort by start time
        all_meetings.sort(key=lambda x: x[0])
        
        # Merge overlapping meetings
        merged = []
        for meeting in all_meetings:
            if not merged or meeting[0] > merged[-1][1]:
                merged.append(meeting)
            else:
                merged[-1][1] = max(merged[-1][1], meeting[1])
        
        return merged
    
    # Convert calendars
    cal1 = convert_calendar(calendar1, daily_bounds1)
    cal2 = convert_calendar(calendar2, daily_bounds2)
    
    # Merge calendars
    merged_calendar = merge_calendars(cal1, daily_bounds1, cal2, daily_bounds2)
    
    # Find available slots
    available_slots = []
    for i in range(len(merged_calendar) - 1):
        current_end = merged_calendar[i][1]
        next_start = merged_calendar[i + 1][0]
        
        if next_start - current_end >= meeting_duration:
            available_slots.append({
                'start': minutes_to_time(current_end),
                'end': minutes_to_time(next_start)
            })
    
    return available_slots


def waterfall_streams(array: List[List[float]], source: int) -> List[float]:
    """
    Waterfall Streams Problem
    
    Calculate the percentage of water that reaches each position at the bottom
    of a 2D array representing a waterfall with blocks.
    
    Args:
        array: 2D array where 0 represents water can flow, 1 represents a block
        source: Column index where water starts flowing from the top
    
    Returns:
        List of percentages representing how much water reaches each bottom position
    
    Time Complexity: O(W * H) where W = width, H = height of the array
    Space Complexity: O(W * H) for the DP table
    """
    if not array or not array[0]:
        return []
    
    height = len(array)
    width = len(array[0])
    
    # DP table to store water percentages at each position
    dp = [[0.0 for _ in range(width)] for _ in range(height)]
    
    # Start with 100% water at the source
    dp[0][source] = 100.0
    
    # Process each row
    for row in range(height - 1):
        for col in range(width):
            if dp[row][col] > 0:  # If there's water at this position
                current_water = dp[row][col]
                
                # Check if there's a block below
                if row + 1 < height and array[row + 1][col] == 0:
                    # Water can flow straight down
                    dp[row + 1][col] += current_water
                else:
                    # Water needs to split left and right
                    left_col = col - 1
                    right_col = col + 1
                    
                    # Try to flow left
                    while left_col >= 0 and array[row][left_col] == 0:
                        if row + 1 < height and array[row + 1][left_col] == 0:
                            dp[row + 1][left_col] += current_water / 2
                            break
                        left_col -= 1
                    
                    # Try to flow right
                    while right_col < width and array[row][right_col] == 0:
                        if row + 1 < height and array[row + 1][right_col] == 0:
                            dp[row + 1][right_col] += current_water / 2
                            break
                        right_col += 1
    
    # Return the bottom row
    return dp[height - 1]


def minimum_area_rectangle(points: List[List[int]]) -> int:
    """
    Minimum Area Rectangle Problem
    
    Find the minimum area of a rectangle that can be formed by any four points
    from the given list of points.
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        Minimum area of a rectangle, or 0 if no rectangle can be formed
    
    Time Complexity: O(N^2) where N is the number of points
    Space Complexity: O(N) for storing points in a set
    """
    if len(points) < 4:
        return 0
    
    # Convert points to set for O(1) lookup
    point_set = set()
    for x, y in points:
        point_set.add((x, y))
    
    min_area = float('inf')
    
    # Try all pairs of points as diagonal corners
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Check if these points can form diagonal corners of a rectangle
            if x1 != x2 and y1 != y2:
                # Check if the other two corners exist
                if (x1, y2) in point_set and (x2, y1) in point_set:
                    area = abs((x2 - x1) * (y2 - y1))
                    min_area = min(min_area, area)
    
    return min_area if min_area != float('inf') else 0


def line_through_points(points: List[List[int]]) -> int:
    """
    Line Through Points Problem
    
    Find the maximum number of points that can be placed on a single straight line.
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        Maximum number of points on a single line
    
    Time Complexity: O(N^2) where N is the number of points
    Space Complexity: O(N) for storing slopes
    """
    if len(points) <= 2:
        return len(points)
    
    def gcd(a: int, b: int) -> int:
        """Calculate greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def get_slope(p1: List[int], p2: List[int]) -> Tuple[int, int]:
        """Get slope as a reduced fraction (numerator, denominator)"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0:
            return (1, 0)  # Vertical line
        if dy == 0:
            return (0, 1)  # Horizontal line
        
        g = gcd(abs(dx), abs(dy))
        return (dy // g, dx // g)
    
    max_points = 1
    
    for i in range(len(points)):
        slopes = defaultdict(int)
        for j in range(len(points)):
            if i != j:
                slope = get_slope(points[i], points[j])
                slopes[slope] += 1
        
        if slopes:
            max_points = max(max_points, max(slopes.values()) + 1)
    
    return max_points


def right_smaller_than(array: List[int]) -> List[int]:
    """
    Right Smaller Than Problem
    
    For each element in the array, count how many elements to its right are smaller.
    
    Args:
        array: List of integers
    
    Returns:
        List where each element is the count of smaller elements to the right
    
    Time Complexity: O(N log N) using merge sort with counting
    Space Complexity: O(N) for the result array and temporary storage
    """
    if not array:
        return []
    
    n = len(array)
    result = [0] * n
    
    # Create list of (value, original_index) pairs
    indexed_array = [(array[i], i) for i in range(n)]
    
    def merge_sort(arr: List[Tuple[int, int]], start: int, end: int):
        """Merge sort with counting of inversions"""
        if start >= end:
            return
        
        mid = (start + end) // 2
        merge_sort(arr, start, mid)
        merge_sort(arr, mid + 1, end)
        merge(arr, start, mid, end)
    
    def merge(arr: List[Tuple[int, int]], start: int, mid: int, end: int):
        """Merge two sorted subarrays and count inversions"""
        left = arr[start:mid + 1]
        right = arr[mid + 1:end + 1]
        
        i = j = 0
        k = start
        
        while i < len(left) and j < len(right):
            if left[i][0] <= right[j][0]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                # Count inversions: all remaining elements in left are greater
                for idx in range(i, len(left)):
                    result[left[idx][1]] += 1
                j += 1
            k += 1
        
        # Copy remaining elements
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
    
    merge_sort(indexed_array, 0, n - 1)
    return result


def iterative_inorder_traversal(tree):
    """
    Iterative In-order Traversal Problem
    
    Perform in-order traversal of a binary tree using iteration instead of recursion.
    
    Args:
        tree: Binary tree node with left, right, and value attributes
    
    Returns:
        List of values in in-order traversal order
    
    Time Complexity: O(N) where N is the number of nodes
    Space Complexity: O(H) where H is the height of the tree (worst case O(N))
    """
    if not tree:
        return []
    
    result = []
    stack = []
    current = tree
    
    while current or stack:
        # Reach the leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.value)
        
        # Move to right subtree
        current = current.right
    
    return result


def flatten_binary_tree(root):
    """
    Flatten Binary Tree Problem
    
    Flatten a binary tree into a linked list in pre-order traversal order.
    The left pointer should be null and the right pointer should point to the next node.
    
    Args:
        root: Root of the binary tree
    
    Returns:
        Root of the flattened tree (linked list)
    
    Time Complexity: O(N) where N is the number of nodes
    Space Complexity: O(H) where H is the height of the tree (worst case O(N))
    """
    if not root:
        return None
    
    def flatten_helper(node):
        if not node:
            return None
        
        # Flatten left subtree
        left_tail = flatten_helper(node.left)
        
        # Flatten right subtree
        right_tail = flatten_helper(node.right)
        
        # If left subtree exists, insert it between current node and right subtree
        if node.left:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        
        # Return the tail of the flattened tree
        if right_tail:
            return right_tail
        elif left_tail:
            return left_tail
        else:
            return node
    
    flatten_helper(root)
    return root


def right_sibling_tree(root):
    """
    Right Sibling Tree Problem
    
    Given a binary tree, connect each node to its right sibling (next node at the same level).
    If there's no right sibling, the pointer should be null.
    
    Args:
        root: Root of the binary tree
    
    Returns:
        Root of the tree with right sibling pointers added
    
    Time Complexity: O(N) where N is the number of nodes
    Space Complexity: O(1) excluding the output tree
    """
    if not root:
        return None
    
    def connect_level(level_start):
        """Connect all nodes at the given level"""
        current = level_start
        
        while current:
            # Connect left child to right child if both exist
            if current.left and current.right:
                current.left.right = current.right
            
            # Connect right child to left child of next node at same level
            if current.right and current.right:
                # Find the next node at the same level
                next_node = current.right
                while next_node and not next_node.left and not next_node.right:
                    next_node = next_node.right
                
                if next_node:
                    if next_node.left:
                        current.right.right = next_node.left
                    elif next_node.right:
                        current.right.right = next_node.right
            
            current = current.right
    
    # Connect each level
    level_start = root
    while level_start:
        connect_level(level_start)
        
        # Find the start of the next level
        next_level_start = None
        current = level_start
        while current and not next_level_start:
            if current.left:
                next_level_start = current.left
            elif current.right:
                next_level_start = current.right
            else:
                current = current.right
        
        level_start = next_level_start
    
    return root


def all_kinds_of_node_depths(root):
    """
    All Kinds Of Node Depths Problem
    
    Calculate the sum of depths of all nodes in a binary tree.
    The depth of a node is the number of edges from the root to that node.
    
    Args:
        root: Root of the binary tree
    
    Returns:
        Sum of depths of all nodes
    
    Time Complexity: O(N) where N is the number of nodes
    Space Complexity: O(H) where H is the height of the tree (worst case O(N))
    """
    if not root:
        return 0
    
    def calculate_depths(node, depth=0):
        """Calculate depths recursively"""
        if not node:
            return 0
        
        # Current node's depth + depths of left and right subtrees
        return depth + calculate_depths(node.left, depth + 1) + calculate_depths(node.right, depth + 1)
    
    return calculate_depths(root)


def compare_leaf_traversal(tree1, tree2):
    """
    Compare Leaf Traversal Problem
    
    Compare the leaf traversal of two binary trees.
    
    Args:
        tree1: First binary tree
        tree2: Second binary tree
    
    Returns:
        True if both trees have the same leaf traversal, False otherwise
    
    Time Complexity: O(N1 + N2) where N1, N2 are the number of nodes in each tree
    Space Complexity: O(H1 + H2) where H1, H2 are the heights of the trees
    """
    def get_leaf_sequence(node, leaves):
        """Get leaf sequence using in-order traversal"""
        if not node:
            return
        
        # If it's a leaf node, add to sequence
        if not node.left and not node.right:
            leaves.append(node.value)
            return
        
        get_leaf_sequence(node.left, leaves)
        get_leaf_sequence(node.right, leaves)
    
    leaves1 = []
    leaves2 = []
    
    get_leaf_sequence(tree1, leaves1)
    get_leaf_sequence(tree2, leaves2)
    
    return leaves1 == leaves2


def max_profit_with_k_transactions(prices, k):
    """
    Max Profit With K Transactions Problem
    
    Find the maximum profit that can be achieved with at most k transactions.
    A transaction consists of buying one share and selling one share.
    
    Args:
        prices: List of stock prices
        k: Maximum number of transactions allowed
    
    Returns:
        Maximum profit achievable
    
    Time Complexity: O(N * K) where N is the number of prices, K is the number of transactions
    Space Complexity: O(N * K) for the DP table
    """
    if not prices or k == 0:
        return 0
    
    n = len(prices)
    
    # If k >= n/2, we can make unlimited transactions
    if k >= n // 2:
        profit = 0
        for i in range(1, n):
            if prices[i] > prices[i - 1]:
                profit += prices[i] - prices[i - 1]
        return profit
    
    # DP table: dp[i][j] = max profit with i transactions up to day j
    dp = [[0] * n for _ in range(k + 1)]
    
    for transaction in range(1, k + 1):
        max_profit_before = -prices[0]  # Max profit if we buy on day 0
        
        for day in range(1, n):
            # Don't make transaction on day 'day'
            dp[transaction][day] = dp[transaction][day - 1]
            
            # Make transaction: sell on day 'day' and buy on some previous day
            dp[transaction][day] = max(dp[transaction][day], 
                                     max_profit_before + prices[day])
            
            # Update max_profit_before for next iteration
            max_profit_before = max(max_profit_before, 
                                  dp[transaction - 1][day - 1] - prices[day])
    
    return dp[k][n - 1]


def palindrome_partitioning_min_cuts(string):
    """
    Palindrome Partitioning Min Cuts Problem
    
    Find the minimum number of cuts needed to partition a string into palindromes.
    
    Args:
        string: Input string
    
    Returns:
        Minimum number of cuts needed
    
    Time Complexity: O(N^2) where N is the length of the string
    Space Complexity: O(N^2) for the palindrome table
    """
    if not string:
        return 0
    
    n = len(string)
    
    # Create palindrome table
    is_palindrome = [[False] * n for _ in range(n)]
    
    # Single characters are palindromes
    for i in range(n):
        is_palindrome[i][i] = True
    
    # Check for palindromes of length 2 and more
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_palindrome[i][j] = (string[i] == string[j])
            else:
                is_palindrome[i][j] = (string[i] == string[j] and is_palindrome[i + 1][j - 1])
    
    # Calculate minimum cuts
    cuts = [0] * n
    
    for i in range(n):
        if is_palindrome[0][i]:
            cuts[i] = 0
        else:
            cuts[i] = float('inf')
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    cuts[i] = min(cuts[i], cuts[j] + 1)
    
    return cuts[n - 1]


def longest_increasing_subsequence(array):
    """
    Longest Increasing Subsequence Problem
    
    Find the length of the longest strictly increasing subsequence.
    
    Args:
        array: List of integers
    
    Returns:
        Length of the longest increasing subsequence
    
    Time Complexity: O(N log N) using binary search
    Space Complexity: O(N) for storing the sequence
    """
    if not array:
        return 0
    
    # Use patience sorting algorithm
    piles = []
    
    for num in array:
        # Find the leftmost pile where we can place this number
        left, right = 0, len(piles)
        while left < right:
            mid = (left + right) // 2
            if piles[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(piles):
            piles.append(num)
        else:
            piles[left] = num
    
    return len(piles)


def longest_string_chain(words):
    """
    Longest String Chain Problem
    
    Find the length of the longest word chain where each word is formed by adding
    exactly one letter to the previous word.
    
    Args:
        words: List of strings
    
    Returns:
        Length of the longest word chain
    
    Time Complexity: O(N * L^2) where N is the number of words, L is the max word length
    Space Complexity: O(N) for the DP table
    """
    if not words:
        return 0
    
    # Sort words by length
    words.sort(key=len)
    
    # DP table: dp[word] = length of longest chain ending with this word
    dp = {}
    max_chain_length = 1
    
    for word in words:
        dp[word] = 1
        
        # Try removing each character to find predecessor
        for i in range(len(word)):
            predecessor = word[:i] + word[i + 1:]
            if predecessor in dp:
                dp[word] = max(dp[word], dp[predecessor] + 1)
        
        max_chain_length = max(max_chain_length, dp[word])
    
    return max_chain_length


def square_of_zeroes(matrix):
    """
    Square of Zeroes Problem
    
    Find the largest square of zeroes in a binary matrix.
    
    Args:
        matrix: 2D binary matrix where 0 represents zero, 1 represents one
    
    Returns:
        Size of the largest square of zeroes (side length)
    
    Time Complexity: O(N^3) where N is the size of the matrix
    Space Complexity: O(N^2) for the DP table
    """
    if not matrix or not matrix[0]:
        return 0
    
    n = len(matrix)
    
    # Create DP table for consecutive zeros
    zeros_right = [[0] * n for _ in range(n)]
    zeros_below = [[0] * n for _ in range(n)]
    
    # Fill zeros_right table
    for i in range(n):
        count = 0
        for j in range(n - 1, -1, -1):
            if matrix[i][j] == 0:
                count += 1
            else:
                count = 0
            zeros_right[i][j] = count
    
    # Fill zeros_below table
    for j in range(n):
        count = 0
        for i in range(n - 1, -1, -1):
            if matrix[i][j] == 0:
                count += 1
            else:
                count = 0
            zeros_below[i][j] = count
    
    # Find largest square
    max_size = 0
    
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 0:
                # Try different square sizes
                for size in range(1, min(n - i, n - j) + 1):
                    # Check if we can form a square of this size
                    if (zeros_right[i][j] >= size and 
                        zeros_below[i][j] >= size and
                        zeros_right[i + size - 1][j] >= size and
                        zeros_below[i][j + size - 1] >= size):
                        max_size = max(max_size, size)
    
    return max_size


def knuth_morris_pratt_algorithm(string, substring):
    """
    Knuth—Morris—Pratt Algorithm
    
    Find all occurrences of a substring in a string using the KMP algorithm.
    
    Args:
        string: The main string to search in
        substring: The substring to search for
    
    Returns:
        List of starting indices where the substring is found
    
    Time Complexity: O(N + M) where N is the length of the string, M is the length of the substring
    Space Complexity: O(M) for the failure function
    """
    if not string or not substring:
        return []
    
    def compute_lps(pattern):
        """Compute the longest proper prefix which is also suffix"""
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    lps = compute_lps(substring)
    result = []
    
    i = j = 0
    while i < len(string):
        if substring[j] == string[i]:
            i += 1
            j += 1
        
        if j == len(substring):
            result.append(i - j)
            j = lps[j - 1]
        elif i < len(string) and substring[j] != string[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return result


def a_star_algorithm(grid, start, goal):
    """
    A* Algorithm
    
    Find the shortest path from start to goal in a grid using A* search.
    
    Args:
        grid: 2D grid where 0 represents walkable cells, 1 represents obstacles
        start: Starting position (row, col)
        goal: Goal position (row, col)
    
    Returns:
        List of positions representing the shortest path, or empty list if no path exists
    
    Time Complexity: O(N * log N) where N is the number of cells in the grid
    Space Complexity: O(N) for the priority queue and visited set
    """
    if not grid or not grid[0]:
        return []
    
    rows, cols = len(grid), len(grid[0])
    
    def heuristic(pos1, pos2):
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(pos):
        """Get valid neighboring positions"""
        row, col = pos
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                grid[new_row][new_col] == 0):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    # Priority queue: (f_score, current_pos, g_score, path)
    open_set = [(heuristic(start, goal), start, 0, [start])]
    visited = set()
    
    while open_set:
        f_score, current, g_score, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                new_g_score = g_score + 1
                new_f_score = new_g_score + heuristic(neighbor, goal)
                new_path = path + [neighbor]
                
                heapq.heappush(open_set, (new_f_score, neighbor, new_g_score, new_path))
    
    return []  # No path found


def rectangle_mania(coords):
    """
    Rectangle Mania Problem
    
    Count the number of rectangles that can be formed by the given coordinates.
    Rectangles must have sides parallel to the x and y axes.
    
    Args:
        coords: List of [x, y] coordinates
    
    Returns:
        Number of rectangles that can be formed
    
    Time Complexity: O(N^2) where N is the number of coordinates
    Space Complexity: O(N) for storing coordinates in a set
    """
    if len(coords) < 4:
        return 0
    
    # Convert coordinates to set for O(1) lookup
    coord_set = set()
    for x, y in coords:
        coord_set.add((x, y))
    
    count = 0
    
    # Try all pairs of points as diagonal corners
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            
            # Check if these points can form diagonal corners of a rectangle
            if x1 != x2 and y1 != y2:
                # Check if the other two corners exist
                if (x1, y2) in coord_set and (x2, y1) in coord_set:
                    count += 1
    
    return count


def detect_arbitrage(exchange_rates):
    """
    Detect Arbitrage Problem
    
    Detect if there's an arbitrage opportunity in a currency exchange rate matrix.
    An arbitrage opportunity exists if we can start with one unit of currency and
    end up with more than one unit after a series of exchanges.
    
    Args:
        exchange_rates: 2D matrix where exchange_rates[i][j] is the rate from currency i to j
    
    Returns:
        True if arbitrage opportunity exists, False otherwise
    
    Time Complexity: O(N^3) using Floyd-Warshall algorithm
    Space Complexity: O(N^2) for the distance matrix
    """
    if not exchange_rates or not exchange_rates[0]:
        return False
    
    n = len(exchange_rates)
    
    # Convert rates to logarithms to convert multiplication to addition
    # and detect negative cycles
    log_rates = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if exchange_rates[i][j] > 0:
                log_rates[i][j] = -math.log(exchange_rates[i][j])
            else:
                log_rates[i][j] = float('inf')
    
    # Floyd-Warshall algorithm to find shortest paths
    dist = [row[:] for row in log_rates]
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Check for negative cycles (arbitrage opportunities)
    for i in range(n):
        if dist[i][i] < 0:
            return True
    
    return False


def two_edge_connected_graph(edges, vertices):
    """
    Two-Edge-Connected Graph Problem
    
    Check if a graph is 2-edge-connected (removing any single edge doesn't disconnect the graph).
    
    Args:
        edges: List of [u, v] representing edges
        vertices: Number of vertices in the graph
    
    Returns:
        True if the graph is 2-edge-connected, False otherwise
    
    Time Complexity: O(V + E) where V is the number of vertices, E is the number of edges
    Space Complexity: O(V + E) for the adjacency list and visited arrays
    """
    if vertices <= 1:
        return True
    
    # Build adjacency list
    graph = [[] for _ in range(vertices)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Check if graph is connected
    visited = [False] * vertices
    
    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    dfs(0)
    
    # If not connected, it's not 2-edge-connected
    if not all(visited):
        return False
    
    # Check for bridges using Tarjan's algorithm
    disc = [-1] * vertices  # Discovery times
    low = [-1] * vertices   # Lowest vertex reachable
    time = [0]
    bridges = []
    
    def find_bridges(u, parent):
        disc[u] = low[u] = time[0]
        time[0] += 1
        
        for v in graph[u]:
            if disc[v] == -1:  # Not visited
                find_bridges(v, u)
                low[u] = min(low[u], low[v])
                
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent:  # Back edge
                low[u] = min(low[u], disc[v])
    
    find_bridges(0, -1)
    
    return len(bridges) == 0


def airport_connections(airports, routes, starting_airport):
    """
    Airport Connections Problem
    
    Find the minimum number of new routes needed to make all airports reachable
    from the starting airport.
    
    Args:
        airports: List of airport codes
        routes: List of [from_airport, to_airport] representing existing routes
        starting_airport: The starting airport code
    
    Returns:
        Minimum number of new routes needed
    
    Time Complexity: O(A + R) where A is the number of airports, R is the number of routes
    Space Complexity: O(A + R) for the graph representation
    """
    # Build adjacency list
    graph = {airport: [] for airport in airports}
    for from_airport, to_airport in routes:
        graph[from_airport].append(to_airport)
    
    # Find strongly connected components using Kosaraju's algorithm
    def dfs1(node, visited, stack):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs1(neighbor, visited, stack)
        stack.append(node)
    
    def dfs2(node, visited, component):
        visited.add(node)
        component.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs2(neighbor, visited, component)
    
    # First DFS to fill stack
    visited = set()
    stack = []
    for airport in airports:
        if airport not in visited:
            dfs1(airport, visited, stack)
    
    # Build reverse graph
    reverse_graph = {airport: [] for airport in airports}
    for from_airport, to_airport in routes:
        reverse_graph[to_airport].append(from_airport)
    
    # Second DFS to find SCCs
    visited = set()
    sccs = []
    while stack:
        node = stack.pop()
        if node not in visited:
            component = set()
            dfs2(node, visited, component)
            sccs.append(component)
    
    # Find which SCCs are reachable from starting airport
    starting_scc = None
    for scc in sccs:
        if starting_airport in scc:
            starting_scc = scc
            break
    
    if not starting_scc:
        return len(airports) - 1  # Need to connect to all other airports
    
    # Count unreachable SCCs
    unreachable_sccs = 0
    for scc in sccs:
        if scc != starting_scc:
            unreachable_sccs += 1
    
    return unreachable_sccs


def merge_sorted_arrays(arrays):
    """
    Merge Sorted Arrays Problem
    
    Merge multiple sorted arrays into a single sorted array.
    
    Args:
        arrays: List of sorted arrays
    
    Returns:
        Single sorted array containing all elements
    
    Time Complexity: O(N log K) where N is total number of elements, K is number of arrays
    Space Complexity: O(N) for the result array
    """
    if not arrays:
        return []
    
    # Use min heap to merge arrays
    heap = []
    result = []
    
    # Initialize heap with first element from each array
    for i, array in enumerate(arrays):
        if array:
            heapq.heappush(heap, (array[0], i, 0))
    
    # Merge arrays
    while heap:
        value, array_idx, element_idx = heapq.heappop(heap)
        result.append(value)
        
        # Add next element from the same array if available
        if element_idx + 1 < len(arrays[array_idx]):
            next_element = arrays[array_idx][element_idx + 1]
            heapq.heappush(heap, (next_element, array_idx, element_idx + 1))
    
    return result


class LRUCache:
    """
    LRU Cache Implementation
    
    Least Recently Used (LRU) cache with O(1) time complexity for get and put operations.
    
    Time Complexity: O(1) for both get and put operations
    Space Complexity: O(C) where C is the capacity of the cache
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = None
        self.tail = None
    
    def get(self, key):
        if key not in self.cache:
            return -1
        
        # Move to front (most recently used)
        self._move_to_front(key)
        return self.cache[key]['value']
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing key
            self.cache[key]['value'] = value
            self._move_to_front(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                self._remove_lru()
            
            self.cache[key] = {'value': value, 'prev': None, 'next': None}
            self._add_to_front(key)
    
    def _add_to_front(self, key):
        if not self.head:
            self.head = self.tail = key
        else:
            self.cache[key]['next'] = self.head
            self.cache[self.head]['prev'] = key
            self.head = key
    
    def _remove_from_list(self, key):
        node = self.cache[key]
        if node['prev']:
            self.cache[node['prev']]['next'] = node['next']
        else:
            self.head = node['next']
        
        if node['next']:
            self.cache[node['next']]['prev'] = node['prev']
        else:
            self.tail = node['prev']
    
    def _move_to_front(self, key):
        self._remove_from_list(key)
        self._add_to_front(key)
    
    def _remove_lru(self):
        if self.tail:
            del self.cache[self.tail]
            if self.head == self.tail:
                self.head = self.tail = None
            else:
                self.tail = self.cache[self.tail]['prev']
                self.cache[self.tail]['next'] = None


def rearrange_linked_list(head, k):
    """
    Rearrange Linked List Problem
    
    Rearrange a linked list so that all nodes with values less than k come before
    nodes with values equal to k, which come before nodes with values greater than k.
    
    Args:
        head: Head of the linked list
        k: Pivot value
    
    Returns:
        Head of the rearranged linked list
    
    Time Complexity: O(N) where N is the number of nodes
    Space Complexity: O(1) using three separate lists
    """
    if not head:
        return None
    
    # Create three separate lists for less, equal, and greater values
    less_head = less_tail = None
    equal_head = equal_tail = None
    greater_head = greater_tail = None
    
    current = head
    while current:
        next_node = current.next
        current.next = None
        
        if current.value < k:
            if not less_head:
                less_head = less_tail = current
            else:
                less_tail.next = current
                less_tail = current
        elif current.value == k:
            if not equal_head:
                equal_head = equal_tail = current
            else:
                equal_tail.next = current
                equal_tail = current
        else:
            if not greater_head:
                greater_head = greater_tail = current
            else:
                greater_tail.next = current
                greater_tail = current
        
        current = next_node
    
    # Connect the three lists
    if less_tail:
        less_tail.next = equal_head if equal_head else greater_head
    if equal_tail:
        equal_tail.next = greater_head
    
    return less_head if less_head else (equal_head if equal_head else greater_head)


def linked_list_palindrome(head):
    """
    Linked List Palindrome Problem
    
    Check if a linked list is a palindrome.
    
    Args:
        head: Head of the linked list
    
    Returns:
        True if the linked list is a palindrome, False otherwise
    
    Time Complexity: O(N) where N is the number of nodes
    Space Complexity: O(1) using fast and slow pointers
    """
    if not head or not head.next:
        return True
    
    # Find the middle of the linked list
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse the second half
    second_half = reverse_linked_list(slow.next)
    slow.next = None
    
    # Compare first and second halves
    first = head
    second = second_half
    
    while first and second:
        if first.value != second.value:
            return False
        first = first.next
        second = second.next
    
    # Restore the original list
    slow.next = reverse_linked_list(second_half)
    
    return True


def reverse_linked_list(head):
    """Helper function to reverse a linked list"""
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev


def zip_linked_list(head):
    """
    Zip Linked List Problem
    
    Rearrange a linked list in a zip pattern: first node, last node, second node, 
    second-to-last node, and so on.
    
    Args:
        head: Head of the linked list
    
    Returns:
        Head of the zipped linked list
    
    Time Complexity: O(N) where N is the number of nodes
    Space Complexity: O(1) using fast and slow pointers
    """
    if not head or not head.next:
        return head
    
    # Find the middle of the linked list
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse the second half
    second_half = reverse_linked_list(slow.next)
    slow.next = None
    
    # Merge the two halves in zip pattern
    first = head
    second = second_half
    
    while first and second:
        first_next = first.next
        second_next = second.next
        
        first.next = second
        second.next = first_next
        
        first = first_next
        second = second_next
    
    return head


def node_swap(head):
    """
    Node Swap Problem
    
    Swap every two adjacent nodes in a linked list.
    
    Args:
        head: Head of the linked list
    
    Returns:
        Head of the linked list with adjacent nodes swapped
    
    Time Complexity: O(N) where N is the number of nodes
    Space Complexity: O(1) using iterative approach
    """
    if not head or not head.next:
        return head
    
    # Create a dummy node to handle the head case
    dummy = type('Node', (), {'value': 0, 'next': head})()
    prev = dummy
    
    while prev.next and prev.next.next:
        first = prev.next
        second = prev.next.next
        
        # Swap the nodes
        first.next = second.next
        second.next = first
        prev.next = second
        
        # Move to the next pair
        prev = first
    
    return dummy.next


def number_of_binary_tree_topologies(n):
    """
    Number Of Binary Tree Topologies Problem
    
    Calculate the number of different binary tree topologies possible with n nodes.
    
    Args:
        n: Number of nodes
    
    Returns:
        Number of different binary tree topologies
    
    Time Complexity: O(N^2) using dynamic programming
    Space Complexity: O(N) for the DP table
    """
    if n <= 1:
        return 1
    
    # DP table: dp[i] = number of topologies with i nodes
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
    
    return dp[n]


def non_attacking_queens(n):
    """
    Non-Attacking Queens Problem
    
    Find the number of ways to place n queens on an n x n chessboard so that
    no two queens attack each other.
    
    Args:
        n: Size of the chessboard
    
    Returns:
        Number of valid queen placements
    
    Time Complexity: O(N!) in worst case, but much better with pruning
    Space Complexity: O(N) for the recursion stack
    """
    def is_safe(board, row, col):
        """Check if placing a queen at (row, col) is safe"""
        # Check row
        for j in range(col):
            if board[row][j] == 1:
                return False
        
        # Check upper diagonal
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        
        # Check lower diagonal
        for i, j in zip(range(row, n, 1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        
        return True
    
    def solve_n_queens(board, col):
        """Solve N-Queens using backtracking"""
        if col >= n:
            return 1
        
        count = 0
        for row in range(n):
            if is_safe(board, row, col):
                board[row][col] = 1
                count += solve_n_queens(board, col + 1)
                board[row][col] = 0
        
        return count
    
    board = [[0] * n for _ in range(n)]
    return solve_n_queens(board, 0)


def median_of_two_sorted_arrays(nums1, nums2):
    """
    Median Of Two Sorted Arrays Problem
    
    Find the median of two sorted arrays.
    
    Args:
        nums1: First sorted array
        nums2: Second sorted array
    
    Returns:
        Median of the two arrays
    
    Time Complexity: O(log(min(M, N))) where M, N are the lengths of the arrays
    Space Complexity: O(1)
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        # Partition nums1
        partition_x = (left + right) // 2
        partition_y = (m + n + 1) // 2 - partition_x
        
        # Find the four elements around the partition
        max_left_x = float('-inf') if partition_x == 0 else nums1[partition_x - 1]
        min_right_x = float('inf') if partition_x == m else nums1[partition_x]
        
        max_left_y = float('-inf') if partition_y == 0 else nums2[partition_y - 1]
        min_right_y = float('inf') if partition_y == n else nums2[partition_y]
        
        # Check if partition is correct
        if max_left_x <= min_right_y and max_left_y <= min_right_x:
            # Found the correct partition
            if (m + n) % 2 == 0:
                return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2
            else:
                return max(max_left_x, max_left_y)
        elif max_left_x > min_right_y:
            right = partition_x - 1
        else:
            left = partition_x + 1
    
    return 0.0


def optimal_assembly_line(tasks, workers):
    """
    Optimal Assembly Line Problem
    
    Assign tasks to workers to minimize the maximum time any worker spends.
    
    Args:
        tasks: List of task durations
        workers: Number of workers available
    
    Returns:
        Minimum maximum time any worker will spend
    
    Time Complexity: O(N * log(SUM)) where N is number of tasks, SUM is sum of all task times
    Space Complexity: O(1)
    """
    if not tasks or workers <= 0:
        return 0
    
    def can_assign(tasks, workers, max_time):
        """Check if tasks can be assigned with given max_time"""
        current_worker_time = 0
        workers_needed = 1
        
        for task in tasks:
            if task > max_time:
                return False
            
            if current_worker_time + task <= max_time:
                current_worker_time += task
            else:
                workers_needed += 1
                current_worker_time = task
                
                if workers_needed > workers:
                    return False
        
        return True
    
    left, right = max(tasks), sum(tasks)
    
    while left < right:
        mid = (left + right) // 2
        if can_assign(tasks, workers, mid):
            right = mid
        else:
            left = mid + 1
    
    return left


def merge_sort(arr):
    """
    Merge Sort Implementation
    
    Sort an array using the merge sort algorithm.
    
    Args:
        arr: List to be sorted
    
    Returns:
        Sorted list
    
    Time Complexity: O(N log N)
    Space Complexity: O(N) for temporary arrays
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left, right):
    """Helper function to merge two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def count_inversions(arr):
    """
    Count Inversions Problem
    
    Count the number of inversions in an array.
    An inversion is a pair (i, j) where i < j and arr[i] > arr[j].
    
    Args:
        arr: List of integers
    
    Returns:
        Number of inversions
    
    Time Complexity: O(N log N) using merge sort
    Space Complexity: O(N) for temporary arrays
    """
    def merge_sort_with_count(arr):
        if len(arr) <= 1:
            return arr, 0
        
        mid = len(arr) // 2
        left, left_inversions = merge_sort_with_count(arr[:mid])
        right, right_inversions = merge_sort_with_count(arr[mid:])
        
        merged, split_inversions = merge_with_count(left, right)
        total_inversions = left_inversions + right_inversions + split_inversions
        
        return merged, total_inversions
    
    def merge_with_count(left, right):
        result = []
        i = j = 0
        inversions = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                inversions += len(left) - i
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result, inversions
    
    _, count = merge_sort_with_count(arr)
    return count


def largest_park(land):
    """
    Largest Park Problem
    
    Find the largest rectangular area that can be used for a park in a 2D grid.
    The park must be completely surrounded by buildings.
    
    Args:
        land: 2D grid where 0 represents empty land, 1 represents buildings
    
    Returns:
        Area of the largest possible park
    
    Time Complexity: O(R * C) where R is number of rows, C is number of columns
    Space Complexity: O(C) for the height array
    """
    if not land or not land[0]:
        return 0
    
    rows, cols = len(land), len(land[0])
    heights = [0] * cols
    max_area = 0
    
    for row in range(rows):
        # Update heights for current row
        for col in range(cols):
            if land[row][col] == 0:
                heights[col] += 1
            else:
                heights[col] = 0
        
        # Calculate largest rectangle in histogram
        max_area = max(max_area, largest_rectangle_in_histogram(heights))
    
    return max_area


def largest_rectangle_in_histogram(heights):
    """Helper function to find largest rectangle in histogram"""
    stack = []
    max_area = 0
    i = 0
    
    while i < len(heights):
        if not stack or heights[stack[-1]] <= heights[i]:
            stack.append(i)
            i += 1
        else:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
    
    while stack:
        height = heights[stack.pop()]
        width = i if not stack else i - stack[-1] - 1
        max_area = max(max_area, height * width)
    
    return max_area


def smallest_substring_containing(big_string, small_string):
    """
    Smallest Substring Containing Problem
    
    Find the smallest substring of big_string that contains all characters from small_string.
    
    Args:
        big_string: The main string to search in
        small_string: The string whose characters must be contained
    
    Returns:
        Smallest substring containing all characters from small_string
    
    Time Complexity: O(N + M) where N is length of big_string, M is length of small_string
    Space Complexity: O(M) for the character count maps
    """
    if not big_string or not small_string:
        return ""
    
    # Count characters in small_string
    target_counts = {}
    for char in small_string:
        target_counts[char] = target_counts.get(char, 0) + 1
    
    # Sliding window approach
    left = 0
    min_left = 0
    min_length = float('inf')
    current_counts = {}
    formed = 0
    required = len(target_counts)
    
    for right in range(len(big_string)):
        char = big_string[right]
        current_counts[char] = current_counts.get(char, 0) + 1
        
        if char in target_counts and current_counts[char] == target_counts[char]:
            formed += 1
        
        while left <= right and formed == required:
            if right - left + 1 < min_length:
                min_length = right - left + 1
                min_left = left
            
            char = big_string[left]
            current_counts[char] -= 1
            
            if char in target_counts and current_counts[char] < target_counts[char]:
                formed -= 1
            
            left += 1
    
    return big_string[min_left:min_left + min_length] if min_length != float('inf') else ""


def longest_balanced_substring(string):
    """
    Longest Balanced Substring Problem
    
    Find the length of the longest balanced substring (equal number of opening and closing brackets).
    
    Args:
        string: String containing only '(' and ')'
    
    Returns:
        Length of the longest balanced substring
    
    Time Complexity: O(N) where N is the length of the string
    Space Complexity: O(N) for the stack
    """
    if not string:
        return 0
    
    stack = [-1]  # Initialize with -1 to handle edge cases
    max_length = 0
    
    for i, char in enumerate(string):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_length = max(max_length, i - stack[-1])
    
    return max_length


def strings_made_up_of_strings(big_string, small_strings):
    """
    Strings Made Up Of Strings Problem
    
    Check if big_string can be constructed by concatenating some of the small_strings
    in any order, with each small_string used at most once.
    
    Args:
        big_string: The target string to construct
        small_strings: List of strings that can be used
    
    Returns:
        True if big_string can be constructed, False otherwise
    
    Time Complexity: O(N * M) where N is length of big_string, M is number of small_strings
    Space Complexity: O(N) for the DP table
    """
    if not big_string:
        return True
    
    if not small_strings:
        return False
    
    n = len(big_string)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for string in small_strings:
            if (i >= len(string) and 
                dp[i - len(string)] and 
                big_string[i - len(string):i] == string):
                dp[i] = True
                break
    
    return dp[n]


# Test cases for the implemented algorithms
if __name__ == "__main__":
    print("=== Advanced Algorithms Test Suite ===\n")
    
    # Test Apartment Hunting
    blocks = [
        {"gym": False, "school": True, "store": False},
        {"gym": True, "school": False, "store": False},
        {"gym": True, "school": True, "store": False},
        {"gym": False, "school": True, "store": False},
        {"gym": False, "school": True, "store": True}
    ]
    reqs = ["gym", "school", "store"]
    print(f"Apartment Hunting Result: {apartment_hunting(blocks, reqs)}")
    
    # Test Calendar Matching
    calendar1 = [{"start": "9:00", "end": "10:30"}, {"start": "12:00", "end": "13:00"}]
    daily_bounds1 = {"start": "9:00", "end": "20:00"}
    calendar2 = [{"start": "10:00", "end": "11:30"}, {"start": "12:30", "end": "14:30"}]
    daily_bounds2 = {"start": "10:00", "end": "18:30"}
    meeting_duration = 30
    print(f"Calendar Matching Result: {calendar_matching(calendar1, daily_bounds1, calendar2, daily_bounds2, meeting_duration)}")
    
    # Test Waterfall Streams
    array = [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    source = 3
    print(f"Waterfall Streams Result: {waterfall_streams(array, source)}")
    
    # Test Minimum Area Rectangle
    points = [[1, 5], [5, 1], [4, 2], [2, 4], [2, 2], [1, 2], [4, 5], [2, 5], [-1, -2]]
    print(f"Minimum Area Rectangle Result: {minimum_area_rectangle(points)}")
    
    # Test Line Through Points
    points = [[1, 1], [2, 2], [3, 3], [0, 0], [-2, 2], [3, 1]]
    print(f"Line Through Points Result: {line_through_points(points)}")
    
    # Test Right Smaller Than
    array = [8, 5, 11, -1, 3, 4, 2]
    print(f"Right Smaller Than Result: {right_smaller_than(array)}")
    
    # Test KMP Algorithm
    string = "ABABCABCABAB"
    substring = "ABC"
    print(f"KMP Algorithm Result: {knuth_morris_pratt_algorithm(string, substring)}")
    
    # Test A* Algorithm
    grid = [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    start = (0, 0)
    goal = (4, 4)
    print(f"A* Algorithm Result: {a_star_algorithm(grid, start, goal)}")
    
    # Test Rectangle Mania
    coords = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    print(f"Rectangle Mania Result: {rectangle_mania(coords)}")
    
    # Test Detect Arbitrage
    exchange_rates = [
        [1.0, 0.5, 0.25],
        [2.0, 1.0, 0.5],
        [4.0, 2.0, 1.0]
    ]
    print(f"Detect Arbitrage Result: {detect_arbitrage(exchange_rates)}")
    
    # Test Two-Edge-Connected Graph
    edges = [[0, 1], [1, 2], [2, 0], [1, 3], [3, 4], [4, 1]]
    vertices = 5
    print(f"Two-Edge-Connected Graph Result: {two_edge_connected_graph(edges, vertices)}")
    
    # Test Airport Connections
    airports = ["BGI", "CDG", "DEL", "DOH", "DSM", "EWR", "EYW", "HND", "ICN", "JFK", "LGA", "LHR", "ORD", "SAN", "SFO", "SIN", "TLV", "BUD"]
    routes = [["DSM", "ORD"], ["ORD", "BGI"], ["BGI", "LGA"], ["SIN", "CDG"], ["CDG", "SIN"], ["CDG", "BUD"], ["DEL", "DOH"], ["DEL", "CDG"], ["TLV", "DEL"], ["EWR", "HND"], ["HND", "ICN"], ["HND", "JFK"], ["ICN", "JFK"], ["JFK", "LGA"], ["EYW", "LHR"], ["LHR", "SFO"], ["SFO", "SAN"], ["SFO", "DSM"], ["SAN", "EYW"]]
    starting_airport = "LGA"
    print(f"Airport Connections Result: {airport_connections(airports, routes, starting_airport)}")
    
    # Test Merge Sorted Arrays
    arrays = [[1, 3, 5], [2, 4, 6], [0, 7, 8]]
    print(f"Merge Sorted Arrays Result: {merge_sorted_arrays(arrays)}")
    
    # Test LRU Cache
    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    print(f"LRU Cache Get(1): {lru.get(1)}")
    lru.put(3, 3)
    print(f"LRU Cache Get(2): {lru.get(2)}")
    
    # Test Longest Increasing Subsequence
    array = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"Longest Increasing Subsequence Result: {longest_increasing_subsequence(array)}")
    
    # Test Longest String Chain
    words = ["a", "b", "ba", "bca", "bda", "bdca"]
    print(f"Longest String Chain Result: {longest_string_chain(words)}")
    
    # Test Square of Zeroes
    matrix = [
        [1, 1, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    print(f"Square of Zeroes Result: {square_of_zeroes(matrix)}")
    
    # Test Palindrome Partitioning Min Cuts
    string = "aab"
    print(f"Palindrome Partitioning Min Cuts Result: {palindrome_partitioning_min_cuts(string)}")
    
    # Test Max Profit With K Transactions
    prices = [3, 2, 6, 5, 0, 3]
    k = 2
    print(f"Max Profit With K Transactions Result: {max_profit_with_k_transactions(prices, k)}")
    
    # Test Number of Binary Tree Topologies
    n = 3
    print(f"Number of Binary Tree Topologies Result: {number_of_binary_tree_topologies(n)}")
    
    # Test Non-Attacking Queens
    n = 4
    print(f"Non-Attacking Queens Result: {non_attacking_queens(n)}")
    
    # Test Median of Two Sorted Arrays
    nums1 = [1, 3]
    nums2 = [2]
    print(f"Median of Two Sorted Arrays Result: {median_of_two_sorted_arrays(nums1, nums2)}")
    
    # Test Optimal Assembly Line
    tasks = [3, 5, 1, 7, 2, 4]
    workers = 3
    print(f"Optimal Assembly Line Result: {optimal_assembly_line(tasks, workers)}")
    
    # Test Merge Sort
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Merge Sort Result: {merge_sort(arr)}")
    
    # Test Count Inversions
    arr = [2, 4, 1, 3, 5]
    print(f"Count Inversions Result: {count_inversions(arr)}")
    
    # Test Largest Park
    land = [
        [1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    print(f"Largest Park Result: {largest_park(land)}")
    
    # Test Smallest Substring Containing
    big_string = "ADOBECODEBANC"
    small_string = "ABC"
    print(f"Smallest Substring Containing Result: {smallest_substring_containing(big_string, small_string)}")
    
    # Test Longest Balanced Substring
    string = "(()))"
    print(f"Longest Balanced Substring Result: {longest_balanced_substring(string)}")
    
    # Test Strings Made Up Of Strings
    big_string = "leetcode"
    small_strings = ["leet", "code"]
    print(f"Strings Made Up Of Strings Result: {strings_made_up_of_strings(big_string, small_strings)}")
    
    print("\n=== All tests completed! ===") 