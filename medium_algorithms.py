"""
Comprehensive Algorithms Collection
==================================
This file contains implementations of various algorithms with detailed comments
and time/space complexity analysis.
"""

# ============================================================================
# ARRAY ALGORITHMS
# ============================================================================

def three_number_sum(array, target_sum):
    """
    Find all triplets in the array that sum up to the target sum.
    
    Time Complexity: O(n²) - Two nested loops
    Space Complexity: O(n) - To store the result triplets
    
    Args:
        array: List of integers
        target_sum: Target sum to find
    
    Returns:
        List of triplets that sum to target_sum
    """
    array.sort()  # O(n log n)
    triplets = []
    
    for i in range(len(array) - 2):
        left = i + 1
        right = len(array) - 1
        
        while left < right:
            current_sum = array[i] + array[left] + array[right]
            
            if current_sum == target_sum:
                triplets.append([array[i], array[left], array[right]])
                left += 1
                right -= 1
            elif current_sum < target_sum:
                left += 1
            else:
                right -= 1
    
    return triplets


def smallest_difference(array_one, array_two):
    """
    Find the pair of numbers (one from each array) whose absolute difference is closest to zero.
    
    Time Complexity: O(n log n + m log m) - Sorting both arrays
    Space Complexity: O(1) - Only storing the result pair
    
    Args:
        array_one: First array of integers
        array_two: Second array of integers
    
    Returns:
        List with two numbers that have the smallest difference
    """
    array_one.sort()  # O(n log n)
    array_two.sort()  # O(m log m)
    
    idx_one = 0
    idx_two = 0
    smallest = float("inf")
    current = float("inf")
    smallest_pair = []
    
    while idx_one < len(array_one) and idx_two < len(array_two):
        first_num = array_one[idx_one]
        second_num = array_two[idx_two]
        
        if first_num < second_num:
            current = second_num - first_num
            idx_one += 1
        elif second_num < first_num:
            current = first_num - second_num
            idx_two += 1
        else:
            return [first_num, second_num]
        
        if smallest > current:
            smallest = current
            smallest_pair = [first_num, second_num]
    
    return smallest_pair


def move_element_to_end(array, to_move):
    """
    Move all instances of a specified integer to the end of the array.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - In-place modification
    
    Args:
        array: List of integers
        to_move: Integer to move to the end
    
    Returns:
        Modified array with specified element moved to end
    """
    left = 0
    right = len(array) - 1
    
    while left < right:
        # Find the rightmost element that's not to_move
        while left < right and array[right] == to_move:
            right -= 1
        
        # If left element is to_move, swap with right
        if array[left] == to_move:
            array[left], array[right] = array[right], array[left]
        
        left += 1
    
    return array


def is_monotonic(array):
    """
    Check if an array is monotonic (entirely non-increasing or non-decreasing).
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        array: List of integers
    
    Returns:
        Boolean indicating if array is monotonic
    """
    if len(array) <= 2:
        return True
    
    direction = array[1] - array[0]
    
    for i in range(2, len(array)):
        if direction == 0:
            direction = array[i] - array[i - 1]
            continue
        
        if breaks_direction(direction, array[i - 1], array[i]):
            return False
    
    return True


def breaks_direction(direction, previous, current):
    """
    Helper function to check if current element breaks the monotonic direction.
    """
    difference = current - previous
    
    if direction > 0:
        return difference < 0
    return difference > 0


def spiral_traverse(array):
    """
    Traverse a 2D array in a spiral order starting from the top-left corner.
    
    Time Complexity: O(n) - Visit each element exactly once
    Space Complexity: O(n) - To store the result array
    
    Args:
        array: 2D array (matrix)
    
    Returns:
        List of elements in spiral order
    """
    if len(array) == 0:
        return []
    
    result = []
    start_row, end_row = 0, len(array) - 1
    start_col, end_col = 0, len(array[0]) - 1
    
    while start_row <= end_row and start_col <= end_col:
        # Traverse right
        for col in range(start_col, end_col + 1):
            result.append(array[start_row][col])
        
        # Traverse down
        for row in range(start_row + 1, end_row + 1):
            result.append(array[row][end_col])
        
        # Traverse left
        for col in range(end_col - 1, start_col - 1, -1):
            if start_row == end_row:
                break
            result.append(array[end_row][col])
        
        # Traverse up
        for row in range(end_row - 1, start_row, -1):
            if start_col == end_col:
                break
            result.append(array[row][start_col])
        
        start_row += 1
        end_row -= 1
        start_col += 1
        end_col -= 1
    
    return result


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_three_number_sum():
    """Test the three number sum function."""
    array = [12, 3, 1, 2, -6, 5, -8, 6]
    target = 0
    result = three_number_sum(array, target)
    print(f"Three Number Sum: {result}")
    # Expected: [[-8, 2, 6], [-8, 3, 5], [-6, 1, 5]]


def test_smallest_difference():
    """Test the smallest difference function."""
    array_one = [-1, 5, 10, 20, 28, 3]
    array_two = [26, 134, 135, 15, 17]
    result = smallest_difference(array_one, array_two)
    print(f"Smallest Difference: {result}")
    # Expected: [28, 26]


def test_move_element_to_end():
    """Test the move element to end function."""
    array = [2, 1, 2, 2, 2, 3, 4, 2]
    to_move = 2
    result = move_element_to_end(array, to_move)
    print(f"Move Element To End: {result}")
    # Expected: [1, 3, 4, 2, 2, 2, 2, 2]


def test_is_monotonic():
    """Test the monotonic array function."""
    array = [-1, -5, -10, -1100, -1100, -1101, -1102, -9001]
    result = is_monotonic(array)
    print(f"Is Monotonic: {result}")
    # Expected: True


def test_spiral_traverse():
    """Test the spiral traverse function."""
    array = [
        [1, 2, 3, 4],
        [12, 13, 14, 5],
        [11, 16, 15, 6],
        [10, 9, 8, 7]
    ]
    result = spiral_traverse(array)
    print(f"Spiral Traverse: {result}")
    # Expected: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


# ============================================================================
# MORE ARRAY ALGORITHMS
# ============================================================================

def longest_peak(array):
    """
    Find the length of the longest peak in an array.
    A peak is defined as adjacent integers that are strictly increasing until they reach a tip,
    at which point they become strictly decreasing.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        array: List of integers
    
    Returns:
        Length of the longest peak
    """
    longest_peak_length = 0
    i = 1
    
    while i < len(array) - 1:
        is_peak = array[i - 1] < array[i] and array[i] > array[i + 1]
        
        if not is_peak:
            i += 1
            continue
        
        # Find the left boundary of the peak
        left_idx = i - 2
        while left_idx >= 0 and array[left_idx] < array[left_idx + 1]:
            left_idx -= 1
        
        # Find the right boundary of the peak
        right_idx = i + 2
        while right_idx < len(array) and array[right_idx] < array[right_idx - 1]:
            right_idx += 1
        
        current_peak_length = right_idx - left_idx - 1
        longest_peak_length = max(longest_peak_length, current_peak_length)
        i = right_idx
    
    return longest_peak_length


def array_of_products(array):
    """
    Return an array where each element is equal to the product of every other number in the input array.
    
    Time Complexity: O(n) - Two passes through the array
    Space Complexity: O(n) - To store the result array
    
    Args:
        array: List of integers
    
    Returns:
        Array where each element is product of all other elements
    """
    products = [1 for _ in range(len(array))]
    
    # Calculate left products
    left_running_product = 1
    for i in range(len(array)):
        products[i] = left_running_product
        left_running_product *= array[i]
    
    # Calculate right products and multiply with left products
    right_running_product = 1
    for i in reversed(range(len(array))):
        products[i] *= right_running_product
        right_running_product *= array[i]
    
    return products


def first_duplicate_value(array):
    """
    Find the first duplicate value in an array where values are between 1 and n.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - Using the array itself as a hash set
    
    Args:
        array: List of integers with values 1 to n
    
    Returns:
        First duplicate value found, or -1 if no duplicates
    """
    for value in array:
        abs_value = abs(value)
        if array[abs_value - 1] < 0:
            return abs_value
        array[abs_value - 1] *= -1
    
    return -1


def merge_overlapping_intervals(intervals):
    """
    Merge overlapping intervals in a list of intervals.
    
    Time Complexity: O(n log n) - Sorting the intervals
    Space Complexity: O(n) - To store the merged intervals
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if len(intervals) <= 1:
        return intervals
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged_intervals = [intervals[0]]
    
    for current_interval in intervals[1:]:
        previous_interval = merged_intervals[-1]
        
        # If current interval overlaps with previous, merge them
        if current_interval[0] <= previous_interval[1]:
            merged_intervals[-1] = [
                previous_interval[0],
                max(previous_interval[1], current_interval[1])
            ]
        else:
            merged_intervals.append(current_interval)
    
    return merged_intervals


def best_seat(seats):
    """
    Find the best seat in a row where 0 represents an empty seat and 1 represents an occupied seat.
    The best seat is the one that maximizes the distance to the closest person.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        seats: List of 0s and 1s representing empty and occupied seats
    
    Returns:
        Index of the best seat, or -1 if no empty seats
    """
    best_seat = -1
    max_space = 0
    
    left = 0
    while left < len(seats):
        right = left
        while right < len(seats) and seats[right] == 0:
            right += 1
        
        available_space = right - left
        if available_space > max_space:
            max_space = available_space
            best_seat = (left + right - 1) // 2
        
        left = right + 1
    
    return best_seat


def zero_sum_subarray(array):
    """
    Check if there exists a subarray that sums to zero.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(n) - To store prefix sums
    
    Args:
        array: List of integers
    
    Returns:
        Boolean indicating if zero sum subarray exists
    """
    sums = {0: -1}  # sum: index
    current_sum = 0
    
    for i, num in enumerate(array):
        current_sum += num
        
        if current_sum in sums:
            return True
        
        sums[current_sum] = i
    
    return False


def missing_numbers(nums):
    """
    Find all missing numbers in an array containing numbers from 1 to n.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - Using the array itself as a hash set
    
    Args:
        nums: List of integers with values 1 to n
    
    Returns:
        List of missing numbers
    """
    # Mark numbers as negative to indicate they exist
    for num in nums:
        abs_num = abs(num)
        if abs_num <= len(nums):
            nums[abs_num - 1] = -abs(nums[abs_num - 1])
    
    # Find positive numbers (missing numbers)
    missing = []
    for i, num in enumerate(nums):
        if num > 0:
            missing.append(i + 1)
    
    return missing


def majority_element(array):
    """
    Find the majority element in an array (appears more than n/2 times).
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        array: List of integers
    
    Returns:
        Majority element
    """
    count = 0
    candidate = None
    
    for num in array:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    
    return candidate


def sweet_and_savory(dishes, target):
    """
    Find the best sweet and savory dish combination that comes closest to the target.
    
    Time Complexity: O(n log n) - Sorting the dishes
    Space Complexity: O(n) - To separate sweet and savory dishes
    
    Args:
        dishes: List of integers (negative for sweet, positive for savory)
        target: Target sum to achieve
    
    Returns:
        List with best sweet and savory dish combination
    """
    sweet_dishes = [dish for dish in dishes if dish < 0]
    savory_dishes = [dish for dish in dishes if dish > 0]
    
    sweet_dishes.sort(reverse=True)  # Sort in descending order
    savory_dishes.sort()  # Sort in ascending order
    
    best_pair = [0, 0]
    best_diff = float('inf')
    
    sweet_idx = 0
    savory_idx = 0
    
    while sweet_idx < len(sweet_dishes) and savory_idx < len(savory_dishes):
        current_sum = sweet_dishes[sweet_idx] + savory_dishes[savory_idx]
        
        if current_sum <= target:
            current_diff = target - current_sum
            if current_diff < best_diff:
                best_diff = current_diff
                best_pair = [sweet_dishes[sweet_idx], savory_dishes[savory_idx]]
            savory_idx += 1
        else:
            sweet_idx += 1
    
    return best_pair


# ============================================================================
# BINARY SEARCH TREE ALGORITHMS
# ============================================================================

class BST:
    """Binary Search Tree node class."""
    
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def bst_construction(values):
    """
    Construct a Binary Search Tree from a list of values.
    
    Time Complexity: O(n²) - In worst case (unbalanced tree)
    Space Complexity: O(n) - To store the tree nodes
    
    Args:
        values: List of integers to insert into BST
    
    Returns:
        Root of the constructed BST
    """
    if not values:
        return None
    
    root = BST(values[0])
    
    for value in values[1:]:
        insert_into_bst(root, value)
    
    return root


def insert_into_bst(root, value):
    """Helper function to insert a value into BST."""
    if value < root.value:
        if root.left is None:
            root.left = BST(value)
        else:
            insert_into_bst(root.left, value)
    else:
        if root.right is None:
            root.right = BST(value)
        else:
            insert_into_bst(root.right, value)


def validate_bst(tree):
    """
    Check if a binary tree is a valid Binary Search Tree.
    
    Time Complexity: O(n) - Visit each node once
    Space Complexity: O(h) - Height of the tree (recursion stack)
    
    Args:
        tree: Root of the binary tree
    
    Returns:
        Boolean indicating if tree is valid BST
    """
    return validate_bst_helper(tree, float('-inf'), float('inf'))


def validate_bst_helper(tree, min_val, max_val):
    """Helper function for BST validation."""
    if tree is None:
        return True
    
    if tree.value < min_val or tree.value >= max_val:
        return False
    
    left_is_valid = validate_bst_helper(tree.left, min_val, tree.value)
    right_is_valid = validate_bst_helper(tree.right, tree.value, max_val)
    
    return left_is_valid and right_is_valid


def bst_traversal(tree, traversal_type="inorder"):
    """
    Perform different types of tree traversals.
    
    Time Complexity: O(n) - Visit each node once
    Space Complexity: O(n) - To store the result array
    
    Args:
        tree: Root of the binary tree
        traversal_type: Type of traversal ("inorder", "preorder", "postorder")
    
    Returns:
        List of values in the specified traversal order
    """
    result = []
    
    if traversal_type == "inorder":
        inorder_traversal(tree, result)
    elif traversal_type == "preorder":
        preorder_traversal(tree, result)
    elif traversal_type == "postorder":
        postorder_traversal(tree, result)
    
    return result


def inorder_traversal(tree, result):
    """Helper function for inorder traversal."""
    if tree is not None:
        inorder_traversal(tree.left, result)
        result.append(tree.value)
        inorder_traversal(tree.right, result)


def preorder_traversal(tree, result):
    """Helper function for preorder traversal."""
    if tree is not None:
        result.append(tree.value)
        preorder_traversal(tree.left, result)
        preorder_traversal(tree.right, result)


def postorder_traversal(tree, result):
    """Helper function for postorder traversal."""
    if tree is not None:
        postorder_traversal(tree.left, result)
        postorder_traversal(tree.right, result)
        result.append(tree.value)


def min_height_bst(array):
    """
    Construct a Binary Search Tree with minimum height from a sorted array.
    
    Time Complexity: O(n) - Visit each element once
    Space Complexity: O(n) - To store the tree nodes
    
    Args:
        array: Sorted array of integers
    
    Returns:
        Root of the minimum height BST
    """
    return min_height_bst_helper(array, 0, len(array) - 1)


def min_height_bst_helper(array, start_idx, end_idx):
    """Helper function for constructing minimum height BST."""
    if end_idx < start_idx:
        return None
    
    mid_idx = (start_idx + end_idx) // 2
    bst = BST(array[mid_idx])
    
    bst.left = min_height_bst_helper(array, start_idx, mid_idx - 1)
    bst.right = min_height_bst_helper(array, mid_idx + 1, end_idx)
    
    return bst


def find_kth_largest_value_in_bst(tree, k):
    """
    Find the kth largest value in a Binary Search Tree.
    
    Time Complexity: O(h + k) - Where h is height of tree
    Space Complexity: O(h) - Height of the tree (recursion stack)
    
    Args:
        tree: Root of the BST
        k: kth largest value to find
    
    Returns:
        kth largest value in the BST
    """
    result = []
    reverse_inorder_traversal(tree, result, k)
    return result[-1] if result else None


def reverse_inorder_traversal(tree, result, k):
    """Helper function for reverse inorder traversal."""
    if tree is None or len(result) >= k:
        return
    
    reverse_inorder_traversal(tree.right, result, k)
    if len(result) < k:
        result.append(tree.value)
    reverse_inorder_traversal(tree.left, result, k)


def reconstruct_bst(preorder_values):
    """
    Reconstruct a Binary Search Tree from its preorder traversal.
    
    Time Complexity: O(n) - Visit each value once
    Space Complexity: O(n) - To store the tree nodes
    
    Args:
        preorder_values: List of values in preorder traversal
    
    Returns:
        Root of the reconstructed BST
    """
    if not preorder_values:
        return None
    
    root_value = preorder_values[0]
    root = BST(root_value)
    
    # Find the index where right subtree starts
    right_subtree_start = len(preorder_values)
    for i in range(1, len(preorder_values)):
        if preorder_values[i] >= root_value:
            right_subtree_start = i
            break
    
    # Construct left and right subtrees
    root.left = reconstruct_bst(preorder_values[1:right_subtree_start])
    root.right = reconstruct_bst(preorder_values[right_subtree_start:])
    
    return root


def invert_binary_tree(tree):
    """
    Invert a binary tree (mirror image).
    
    Time Complexity: O(n) - Visit each node once
    Space Complexity: O(h) - Height of the tree (recursion stack)
    
    Args:
        tree: Root of the binary tree
    
    Returns:
        Root of the inverted binary tree
    """
    if tree is None:
        return None
    
    # Swap left and right children
    tree.left, tree.right = tree.right, tree.left
    
    # Recursively invert subtrees
    invert_binary_tree(tree.left)
    invert_binary_tree(tree.right)
    
    return tree


def binary_tree_diameter(tree):
    """
    Find the diameter of a binary tree (longest path between any two nodes).
    
    Time Complexity: O(n) - Visit each node once
    Space Complexity: O(h) - Height of the tree (recursion stack)
    
    Args:
        tree: Root of the binary tree
    
    Returns:
        Diameter of the binary tree
    """
    diameter = [0]  # Use list to store diameter (mutable)
    calculate_height(tree, diameter)
    return diameter[0]


def calculate_height(tree, diameter):
    """Helper function to calculate height and update diameter."""
    if tree is None:
        return 0
    
    left_height = calculate_height(tree.left, diameter)
    right_height = calculate_height(tree.right, diameter)
    
    # Update diameter if current path is longer
    diameter[0] = max(diameter[0], left_height + right_height)
    
    return max(left_height, right_height) + 1


def find_successor(tree, node):
    """
    Find the in-order successor of a given node in a BST.
    
    Time Complexity: O(h) - Height of the tree
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        tree: Root of the BST
        node: Node to find successor for
    
    Returns:
        Successor node or None
    """
    if node.right is not None:
        # Successor is the leftmost node in right subtree
        current = node.right
        while current.left is not None:
            current = current.left
        return current
    
    # Successor is the closest ancestor where node is in left subtree
    successor = None
    current = tree
    
    while current is not None:
        if node.value < current.value:
            successor = current
            current = current.left
        elif node.value > current.value:
            current = current.right
        else:
            break
    
    return successor


def height_balanced_binary_tree(tree):
    """
    Check if a binary tree is height-balanced.
    
    Time Complexity: O(n) - Visit each node once
    Space Complexity: O(h) - Height of the tree (recursion stack)
    
    Args:
        tree: Root of the binary tree
    
    Returns:
        Boolean indicating if tree is height-balanced
    """
    return get_tree_info(tree).is_balanced


class TreeInfo:
    """Helper class to store tree information."""
    def __init__(self, is_balanced, height):
        self.is_balanced = is_balanced
        self.height = height


def get_tree_info(tree):
    """Helper function to get tree information."""
    if tree is None:
        return TreeInfo(True, -1)
    
    left_subtree_info = get_tree_info(tree.left)
    right_subtree_info = get_tree_info(tree.right)
    
    is_balanced = (left_subtree_info.is_balanced and 
                   right_subtree_info.is_balanced and
                   abs(left_subtree_info.height - right_subtree_info.height) <= 1)
    
    height = max(left_subtree_info.height, right_subtree_info.height) + 1
    
    return TreeInfo(is_balanced, height)


def merge_binary_trees(tree1, tree2):
    """
    Merge two binary trees by adding corresponding nodes.
    
    Time Complexity: O(min(n1, n2)) - Where n1, n2 are number of nodes
    Space Complexity: O(min(h1, h2)) - Height of the smaller tree
    
    Args:
        tree1: Root of first binary tree
        tree2: Root of second binary tree
    
    Returns:
        Root of merged binary tree
    """
    if tree1 is None:
        return tree2
    if tree2 is None:
        return tree1
    
    tree1.value += tree2.value
    tree1.left = merge_binary_trees(tree1.left, tree2.left)
    tree1.right = merge_binary_trees(tree1.right, tree2.right)
    
    return tree1


def symmetrical_tree(tree):
    """
    Check if a binary tree is symmetrical (mirror image of itself).
    
    Time Complexity: O(n) - Visit each node once
    Space Complexity: O(h) - Height of the tree (recursion stack)
    
    Args:
        tree: Root of the binary tree
    
    Returns:
        Boolean indicating if tree is symmetrical
    """
    if tree is None:
        return True
    
    return trees_are_mirrored(tree.left, tree.right)


def trees_are_mirrored(left, right):
    """Helper function to check if two trees are mirrored."""
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    
    return (left.value == right.value and
            trees_are_mirrored(left.left, right.right) and
            trees_are_mirrored(left.right, right.left))


def split_binary_tree(tree):
    """
    Check if a binary tree can be split into two trees with equal sums.
    
    Time Complexity: O(n) - Visit each node once
    Space Complexity: O(h) - Height of the tree (recursion stack)
    
    Args:
        tree: Root of the binary tree
    
    Returns:
        Boolean indicating if tree can be split
    """
    total_sum = get_tree_sum(tree)
    
    # If total sum is odd, cannot split equally
    if total_sum % 2 != 0:
        return False
    
    target_sum = total_sum // 2
    can_split = [False]
    
    check_subtree_sum(tree, target_sum, can_split)
    return can_split[0]


def get_tree_sum(tree):
    """Helper function to get sum of all nodes in tree."""
    if tree is None:
        return 0
    return tree.value + get_tree_sum(tree.left) + get_tree_sum(tree.right)


def check_subtree_sum(tree, target_sum, can_split):
    """Helper function to check if any subtree has target sum."""
    if tree is None:
        return 0
    
    current_sum = (tree.value + 
                   check_subtree_sum(tree.left, target_sum, can_split) +
                   check_subtree_sum(tree.right, target_sum, can_split))
    
    if current_sum == target_sum:
        can_split[0] = True
    
    return current_sum

"""
Comprehensive Algorithms Collection - Part 2
===========================================
This file contains implementations of dynamic programming, graph algorithms,
and other advanced algorithms with detailed comments and complexity analysis.
"""

# ============================================================================
# DYNAMIC PROGRAMMING ALGORITHMS
# ============================================================================

def max_subset_sum_no_adjacent(array):
    """
    Find the maximum sum of non-adjacent elements in an array.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        array: List of integers
    
    Returns:
        Maximum sum of non-adjacent elements
    """
    if not array:
        return 0
    if len(array) == 1:
        return array[0]
    
    second = array[0]
    first = max(array[0], array[1])
    
    for i in range(2, len(array)):
        current = max(first, second + array[i])
        second = first
        first = current
    
    return first


def number_of_ways_to_make_change(n, denoms):
    """
    Find the number of ways to make change for amount n using given denominations.
    
    Time Complexity: O(nd) - Where d is number of denominations
    Space Complexity: O(n) - To store the DP array
    
    Args:
        n: Target amount
        denoms: List of available denominations
    
    Returns:
        Number of ways to make change
    """
    ways = [0 for _ in range(n + 1)]
    ways[0] = 1
    
    for denom in denoms:
        for amount in range(1, n + 1):
            if denom <= amount:
                ways[amount] += ways[amount - denom]
    
    return ways[n]


def min_number_of_coins_for_change(n, denoms):
    """
    Find the minimum number of coins needed to make change for amount n.
    
    Time Complexity: O(nd) - Where d is number of denominations
    Space Complexity: O(n) - To store the DP array
    
    Args:
        n: Target amount
        denoms: List of available coin denominations
    
    Returns:
        Minimum number of coins needed, or -1 if impossible
    """
    num_coins = [float('inf') for _ in range(n + 1)]
    num_coins[0] = 0
    
    for denom in denoms:
        for amount in range(denom, n + 1):
            num_coins[amount] = min(num_coins[amount], num_coins[amount - denom] + 1)
    
    return num_coins[n] if num_coins[n] != float('inf') else -1


def levenshtein_distance(str1, str2):
    """
    Calculate the Levenshtein distance between two strings.
    
    Time Complexity: O(mn) - Where m, n are lengths of strings
    Space Complexity: O(mn) - To store the DP matrix
    
    Args:
        str1: First string
        str2: Second string
    
    Returns:
        Levenshtein distance between the strings
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n]


def number_of_ways_to_traverse_graph(width, height):
    """
    Find the number of ways to traverse a graph from top-left to bottom-right.
    
    Time Complexity: O(wh) - Where w, h are width and height
    Space Complexity: O(wh) - To store the DP matrix
    
    Args:
        width: Width of the graph
        height: Height of the graph
    
    Returns:
        Number of ways to traverse the graph
    """
    dp = [[1] * width for _ in range(height)]
    
    for i in range(1, height):
        for j in range(1, width):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[height - 1][width - 1]


def kadane_algorithm(array):
    """
    Find the maximum subarray sum using Kadane's algorithm.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        array: List of integers
    
    Returns:
        Maximum subarray sum
    """
    max_current = max_global = array[0]
    
    for i in range(1, len(array)):
        max_current = max(array[i], max_current + array[i])
        max_global = max(max_global, max_current)
    
    return max_global


# ============================================================================
# GRAPH ALGORITHMS
# ============================================================================

def stable_internships(interns, teams):
    """
    Find stable internship assignments using Gale-Shapley algorithm.
    
    Time Complexity: O(n²) - Where n is number of interns/teams
    Space Complexity: O(n²) - To store preferences and assignments
    
    Args:
        interns: List of intern preferences for teams
        teams: List of team preferences for interns
    
    Returns:
        List of stable internship assignments
    """
    n = len(interns)
    intern_assignments = [-1] * n
    team_assignments = [-1] * n
    intern_proposals = [0] * n
    
    # Continue until all interns are assigned
    while -1 in intern_assignments:
        for intern in range(n):
            if intern_assignments[intern] == -1:
                # Find next team to propose to
                team = interns[intern][intern_proposals[intern]]
                
                if team_assignments[team] == -1:
                    # Team is free, accept proposal
                    intern_assignments[intern] = team
                    team_assignments[team] = intern
                else:
                    # Team prefers new intern over current
                    current_intern = team_assignments[team]
                    if teams[team].index(intern) < teams[team].index(current_intern):
                        intern_assignments[intern] = team
                        team_assignments[team] = intern
                        intern_assignments[current_intern] = -1
                    else:
                        intern_proposals[intern] += 1
    
    return [[intern, team] for intern, team in enumerate(intern_assignments)]


class UnionFind:
    """Union-Find data structure for graph algorithms."""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        return True


def single_cycle_check(array):
    """
    Check if a single cycle exists in an array where each element represents a jump.
    
    Time Complexity: O(n) - Visit each element at most once
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        array: List of integers representing jumps
    
    Returns:
        Boolean indicating if single cycle exists
    """
    n = len(array)
    visited = [False] * n
    current_idx = 0
    visited_count = 0
    
    while visited_count < n:
        if visited[current_idx]:
            return False
        
        visited[current_idx] = True
        visited_count += 1
        
        # Calculate next index
        jump = array[current_idx]
        next_idx = (current_idx + jump) % n
        
        # Handle negative jumps
        if next_idx < 0:
            next_idx += n
        
        current_idx = next_idx
    
    return current_idx == 0


def breadth_first_search(graph, start):
    """
    Perform breadth-first search on a graph.
    
    Time Complexity: O(V + E) - Where V is vertices, E is edges
    Space Complexity: O(V) - To store visited nodes and queue
    
    Args:
        graph: Adjacency list representation of graph
        start: Starting node
    
    Returns:
        List of nodes in BFS order
    """
    visited = set()
    queue = [start]
    result = []
    
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            result.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return result


def river_sizes(matrix):
    """
    Find the sizes of all rivers in a matrix where 1 represents water and 0 represents land.
    
    Time Complexity: O(wh) - Where w, h are width and height
    Space Complexity: O(wh) - To store visited cells
    
    Args:
        matrix: 2D matrix representing land and water
    
    Returns:
        List of river sizes
    """
    if not matrix:
        return []
    
    rows, cols = len(matrix), len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]
    river_sizes = []
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1 and not visited[i][j]:
                size = dfs_river(matrix, i, j, visited)
                river_sizes.append(size)
    
    return river_sizes


def dfs_river(matrix, i, j, visited):
    """Helper function for river size calculation using DFS."""
    if (i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or
        matrix[i][j] == 0 or visited[i][j]):
        return 0
    
    visited[i][j] = True
    size = 1
    
    # Check all 4 directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for di, dj in directions:
        size += dfs_river(matrix, i + di, j + dj, visited)
    
    return size


def youngest_common_ancestor(top_ancestor, descendant_one, descendant_two):
    """
    Find the youngest common ancestor of two descendants in a family tree.
    
    Time Complexity: O(d) - Where d is depth of the tree
    Space Complexity: O(d) - To store the path to root
    
    Args:
        top_ancestor: Root of the family tree
        descendant_one: First descendant
        descendant_two: Second descendant
    
    Returns:
        Youngest common ancestor
    """
    def get_depth(descendant):
        depth = 0
        while descendant != top_ancestor:
            depth += 1
            descendant = descendant.ancestor
        return depth
    
    def get_ancestor_at_depth(descendant, depth):
        while depth > 0:
            descendant = descendant.ancestor
            depth -= 1
        return descendant
    
    depth_one = get_depth(descendant_one)
    depth_two = get_depth(descendant_two)
    
    if depth_one > depth_two:
        descendant_one = get_ancestor_at_depth(descendant_one, depth_one - depth_two)
    elif depth_two > depth_one:
        descendant_two = get_ancestor_at_depth(descendant_two, depth_two - depth_one)
    
    while descendant_one != descendant_two:
        descendant_one = descendant_one.ancestor
        descendant_two = descendant_two.ancestor
    
    return descendant_one


def remove_islands(matrix):
    """
    Remove islands (1s that are not connected to the border) from a matrix.
    
    Time Complexity: O(wh) - Where w, h are width and height
    Space Complexity: O(wh) - To store visited cells
    
    Args:
        matrix: 2D matrix where 1 represents land and 0 represents water
    
    Returns:
        Matrix with islands removed
    """
    if not matrix:
        return matrix
    
    rows, cols = len(matrix), len(matrix[0])
    
    # Mark all 1s connected to border
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and matrix[i][j] == 1:
                mark_connected(matrix, i, j)
    
    # Remove islands (unmarked 1s)
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                matrix[i][j] = 0
            elif matrix[i][j] == 2:
                matrix[i][j] = 1
    
    return matrix


def mark_connected(matrix, i, j):
    """Helper function to mark connected land cells."""
    if (i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or
        matrix[i][j] != 1):
        return
    
    matrix[i][j] = 2  # Mark as connected
    
    # Check all 4 directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for di, dj in directions:
        mark_connected(matrix, i + di, j + dj)


def cycle_in_graph(edges):
    """
    Check if a cycle exists in a directed graph.
    
    Time Complexity: O(V + E) - Where V is vertices, E is edges
    Space Complexity: O(V) - To store visited and recursion stack
    
    Args:
        edges: Adjacency list representation of directed graph
    
    Returns:
        Boolean indicating if cycle exists
    """
    n = len(edges)
    visited = [False] * n
    recursion_stack = [False] * n
    
    for i in range(n):
        if not visited[i]:
            if has_cycle_dfs(edges, i, visited, recursion_stack):
                return True
    
    return False


def has_cycle_dfs(edges, node, visited, recursion_stack):
    """Helper function for cycle detection using DFS."""
    visited[node] = True
    recursion_stack[node] = True
    
    for neighbor in edges[node]:
        if not visited[neighbor]:
            if has_cycle_dfs(edges, neighbor, visited, recursion_stack):
                return True
        elif recursion_stack[neighbor]:
            return True
    
    recursion_stack[node] = False
    return False


def minimum_passes_of_matrix(matrix):
    """
    Find the minimum number of passes needed to convert all negative numbers to positive.
    
    Time Complexity: O(wh * passes) - Where w, h are width and height
    Space Complexity: O(wh) - To store the matrix
    
    Args:
        matrix: 2D matrix with positive and negative numbers
    
    Returns:
        Minimum number of passes needed, or -1 if impossible
    """
    if not matrix:
        return -1
    
    rows, cols = len(matrix), len(matrix[0])
    passes = 0
    
    while True:
        converted = False
        next_queue = []
        
        # Find all positive numbers
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] > 0:
                    next_queue.append((i, j))
        
        if not next_queue:
            # Check if any negative numbers remain
            for i in range(rows):
                for j in range(cols):
                    if matrix[i][j] < 0:
                        return -1
            break
        
        # Convert adjacent negative numbers
        for i, j in next_queue:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if (0 <= ni < rows and 0 <= nj < cols and 
                    matrix[ni][nj] < 0):
                    matrix[ni][nj] = abs(matrix[ni][nj])
                    converted = True
        
        if not converted:
            break
        
        passes += 1
    
    return passes


def two_colorable(edges):
    """
    Check if a graph is two-colorable (bipartite).
    
    Time Complexity: O(V + E) - Where V is vertices, E is edges
    Space Complexity: O(V) - To store colors
    
    Args:
        edges: Adjacency list representation of undirected graph
    
    Returns:
        Boolean indicating if graph is two-colorable
    """
    n = len(edges)
    colors = [-1] * n
    
    for i in range(n):
        if colors[i] == -1:
            if not is_bipartite_dfs(edges, i, 0, colors):
                return False
    
    return True


def is_bipartite_dfs(edges, node, color, colors):
    """Helper function for bipartite check using DFS."""
    colors[node] = color
    
    for neighbor in edges[node]:
        if colors[neighbor] == -1:
            if not is_bipartite_dfs(edges, neighbor, 1 - color, colors):
                return False
        elif colors[neighbor] == color:
            return False
    
    return True


def task_assignment(k, tasks):
    """
    Assign tasks to workers optimally.
    
    Time Complexity: O(n log n) - Sorting the tasks
    Space Complexity: O(n) - To store the result
    
    Args:
        k: Number of workers
        tasks: List of task durations
    
    Returns:
        List of task assignments
    """
    # Create pairs of (task_duration, original_index)
    task_pairs = [(duration, i) for i, duration in enumerate(tasks)]
    task_pairs.sort()  # Sort by duration
    
    assignments = []
    left, right = 0, len(task_pairs) - 1
    
    while left < right:
        assignments.append([task_pairs[left][1], task_pairs[right][1]])
        left += 1
        right -= 1
    
    return assignments


def valid_starting_city(distances, fuel, mpg):
    """
    Find a valid starting city for a circular route.
    
    Time Complexity: O(n) - Single pass through the cities
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        distances: List of distances between cities
        fuel: List of fuel available at each city
        mpg: Miles per gallon
    
    Returns:
        Index of valid starting city
    """
    n = len(distances)
    miles_remaining = 0
    starting_city = 0
    
    for i in range(n):
        miles_remaining += fuel[i] * mpg - distances[i]
        
        if miles_remaining < 0:
            miles_remaining = 0
            starting_city = i + 1
    
    return starting_city


# ============================================================================
# HEAP ALGORITHMS
# ============================================================================

class MinHeap:
    """Min Heap implementation."""
    
    def __init__(self, array):
        self.heap = self.build_heap(array)
    
    def build_heap(self, array):
        """Build a min heap from an array."""
        first_parent_idx = (len(array) - 2) // 2
        for current_idx in reversed(range(first_parent_idx + 1)):
            self.sift_down(current_idx, len(array) - 1, array)
        return array
    
    def sift_down(self, current_idx, end_idx, heap):
        """Sift down operation for heap."""
        child_one_idx = current_idx * 2 + 1
        while child_one_idx <= end_idx:
            child_two_idx = current_idx * 2 + 2 if current_idx * 2 + 2 <= end_idx else -1
            
            if child_two_idx != -1 and heap[child_two_idx] < heap[child_one_idx]:
                idx_to_swap = child_two_idx
            else:
                idx_to_swap = child_one_idx
            
            if heap[idx_to_swap] < heap[current_idx]:
                self.swap(current_idx, idx_to_swap, heap)
                current_idx = idx_to_swap
                child_one_idx = current_idx * 2 + 1
            else:
                return
    
    def sift_up(self, current_idx, heap):
        """Sift up operation for heap."""
        parent_idx = (current_idx - 1) // 2
        while current_idx > 0 and heap[current_idx] < heap[parent_idx]:
            self.swap(current_idx, parent_idx, heap)
            current_idx = parent_idx
            parent_idx = (current_idx - 1) // 2
    
    def peek(self):
        """Get the minimum element without removing it."""
        return self.heap[0]
    
    def remove(self):
        """Remove and return the minimum element."""
        self.swap(0, len(self.heap) - 1, self.heap)
        value_to_remove = self.heap.pop()
        self.sift_down(0, len(self.heap) - 1, self.heap)
        return value_to_remove
    
    def insert(self, value):
        """Insert a new value into the heap."""
        self.heap.append(value)
        self.sift_up(len(self.heap) - 1, self.heap)
    
    def swap(self, i, j, heap):
        """Swap elements at indices i and j."""
        heap[i], heap[j] = heap[j], heap[i]


# ============================================================================
# LINKED LIST ALGORITHMS
# ============================================================================

class LinkedList:
    """Linked List node class."""
    
    def __init__(self, value):
        self.value = value
        self.next = None


def linked_list_construction(values):
    """
    Construct a linked list from a list of values.
    
    Time Complexity: O(n) - Where n is number of values
    Space Complexity: O(n) - To store the linked list nodes
    
    Args:
        values: List of values to create linked list from
    
    Returns:
        Head of the linked list
    """
    if not values:
        return None
    
    head = LinkedList(values[0])
    current = head
    
    for value in values[1:]:
        current.next = LinkedList(value)
        current = current.next
    
    return head


def remove_kth_node_from_end(head, k):
    """
    Remove the kth node from the end of a linked list.
    
    Time Complexity: O(n) - Where n is length of linked list
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        head: Head of the linked list
        k: kth node from end to remove
    """
    first = head
    second = head
    counter = 1
    
    while counter <= k:
        second = second.next
        counter += 1
    
    if second is None:
        head.value = head.next.value
        head.next = head.next.next
        return
    
    while second.next is not None:
        second = second.next
        first = first.next
    
    first.next = first.next.next


def sum_of_linked_lists(linked_list_one, linked_list_two):
    """
    Add two numbers represented as linked lists.
    
    Time Complexity: O(max(n, m)) - Where n, m are lengths of lists
    Space Complexity: O(max(n, m)) - To store the result
    
    Args:
        linked_list_one: First linked list representing a number
        linked_list_two: Second linked list representing a number
    
    Returns:
        Head of linked list representing the sum
    """
    dummy = LinkedList(0)
    current = dummy
    carry = 0
    
    while linked_list_one or linked_list_two or carry:
        val_one = linked_list_one.value if linked_list_one else 0
        val_two = linked_list_two.value if linked_list_two else 0
        
        total = val_one + val_two + carry
        carry = total // 10
        
        current.next = LinkedList(total % 10)
        current = current.next
        
        linked_list_one = linked_list_one.next if linked_list_one else None
        linked_list_two = linked_list_two.next if linked_list_two else None
    
    return dummy.next


def merging_linked_lists(linked_list_one, linked_list_two):
    """
    Find the intersection point of two linked lists.
    
    Time Complexity: O(n + m) - Where n, m are lengths of lists
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        linked_list_one: First linked list
        linked_list_two: Second linked list
    
    Returns:
        Intersection node or None
    """
    def get_length(head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    
    len_one = get_length(linked_list_one)
    len_two = get_length(linked_list_two)
    
    # Make both lists start at the same distance from intersection
    if len_one > len_two:
        for _ in range(len_one - len_two):
            linked_list_one = linked_list_one.next
    else:
        for _ in range(len_two - len_one):
            linked_list_two = linked_list_two.next
    
    # Find intersection
    while linked_list_one and linked_list_two:
        if linked_list_one == linked_list_two:
            return linked_list_one
        linked_list_one = linked_list_one.next
        linked_list_two = linked_list_two.next
    
    return None

"""
Comprehensive Algorithms Collection - Part 3
===========================================
This file contains implementations of recursion, backtracking, string algorithms,
and other advanced algorithms with detailed comments and complexity analysis.
"""

# ============================================================================
# RECURSION AND BACKTRACKING ALGORITHMS
# ============================================================================

def permutations(array):
    """
    Generate all permutations of an array.
    
    Time Complexity: O(n!) - Factorial of array length
    Space Complexity: O(n!) - To store all permutations
    
    Args:
        array: List of elements to permute
    
    Returns:
        List of all permutations
    """
    if len(array) == 0:
        return []
    if len(array) == 1:
        return [array]
    
    result = []
    
    for i in range(len(array)):
        current = array[i]
        remaining = array[:i] + array[i+1:]
        
        for perm in permutations(remaining):
            result.append([current] + perm)
    
    return result


def powerset(array):
    """
    Generate all subsets (powerset) of an array.
    
    Time Complexity: O(2^n) - Exponential time
    Space Complexity: O(2^n) - To store all subsets
    
    Args:
        array: List of elements
    
    Returns:
        List of all subsets
    """
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(array)):
            current.append(array[i])
            backtrack(i + 1, current)
            current.pop()
    
    result = []
    backtrack(0, [])
    return result


def phone_number_mnemonics(phone_number):
    """
    Generate all possible mnemonics for a phone number.
    
    Time Complexity: O(4^n * n) - Where n is length of phone number
    Space Complexity: O(4^n * n) - To store all mnemonics
    
    Args:
        phone_number: String representing phone number
    
    Returns:
        List of all possible mnemonics
    """
    digit_to_letters = {
        '0': ['0'],
        '1': ['1'],
        '2': ['a', 'b', 'c'],
        '3': ['d', 'e', 'f'],
        '4': ['g', 'h', 'i'],
        '5': ['j', 'k', 'l'],
        '6': ['m', 'n', 'o'],
        '7': ['p', 'q', 'r', 's'],
        '8': ['t', 'u', 'v'],
        '9': ['w', 'x', 'y', 'z']
    }
    
    def backtrack(index, current):
        if index == len(phone_number):
            result.append(''.join(current))
            return
        
        digit = phone_number[index]
        for letter in digit_to_letters[digit]:
            current.append(letter)
            backtrack(index + 1, current)
            current.pop()
    
    result = []
    backtrack(0, [])
    return result


def staircase_traversal(height, max_steps):
    """
    Find the number of ways to climb a staircase with given height and max steps.
    
    Time Complexity: O(k^n) - Where k is max_steps, n is height
    Space Complexity: O(n) - Recursion stack depth
    
    Args:
        height: Height of the staircase
        max_steps: Maximum number of steps that can be taken at once
    
    Returns:
        Number of ways to climb the staircase
    """
    def climb(current_height):
        if current_height == 0:
            return 1
        if current_height < 0:
            return 0
        
        ways = 0
        for step in range(1, max_steps + 1):
            ways += climb(current_height - step)
        
        return ways
    
    return climb(height)


def blackjack_probability(target, starting_hand):
    """
    Calculate the probability of reaching target in blackjack.
    
    Time Complexity: O(target) - Dynamic programming approach
    Space Complexity: O(target) - To store DP array
    
    Args:
        target: Target score to reach
        starting_hand: Current hand value
    
    Returns:
        Probability of reaching target
    """
    if starting_hand > target:
        return 0.0
    if starting_hand + 4 >= target:
        return 1.0
    
    # Probability of drawing each card (1-10)
    card_probabilities = [0.1] * 9 + [0.4]  # 1-9: 0.1 each, 10: 0.4
    
    def calculate_probability(current_hand):
        if current_hand > target:
            return 0.0
        if current_hand + 4 >= target:
            return 1.0
        
        total_prob = 0.0
        for card_value, prob in enumerate(card_probabilities, 1):
            if current_hand + card_value <= target:
                total_prob += prob * calculate_probability(current_hand + card_value)
        
        return total_prob
    
    return calculate_probability(starting_hand)


def reveal_minesweeper(board, row, col):
    """
    Reveal cells in a minesweeper board starting from given position.
    
    Time Complexity: O(wh) - Where w, h are width and height
    Space Complexity: O(wh) - To store visited cells
    
    Args:
        board: 2D matrix representing minesweeper board
        row: Starting row
        col: Starting column
    
    Returns:
        Updated board after revealing cells
    """
    if board[row][col] == 'M':
        board[row][col] = 'X'
        return board
    
    rows, cols = len(board), len(board[0])
    queue = [(row, col)]
    
    while queue:
        r, c = queue.pop(0)
        
        if board[r][c] != 'E':
            continue
        
        # Count adjacent mines
        mine_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    board[nr][nc] == 'M'):
                    mine_count += 1
        
        if mine_count == 0:
            board[r][c] = 'B'
            # Add adjacent cells to queue
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        board[nr][nc] == 'E'):
                        queue.append((nr, nc))
        else:
            board[r][c] = str(mine_count)
    
    return board


def search_in_sorted_matrix(matrix, target):
    """
    Search for a target value in a sorted matrix.
    
    Time Complexity: O(n + m) - Where n, m are matrix dimensions
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        matrix: 2D sorted matrix
        target: Value to search for
    
    Returns:
        [row, col] of target, or [-1, -1] if not found
    """
    if not matrix:
        return [-1, -1]
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return [row, col]
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    
    return [-1, -1]


def three_number_sort(array, order):
    """
    Sort an array containing only three distinct values according to given order.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(1) - In-place sorting
    
    Args:
        array: Array with three distinct values
        order: Order to sort the values [first, second, third]
    
    Returns:
        Sorted array
    """
    first_value, second_value, third_value = order
    
    first_idx = 0
    second_idx = 0
    third_idx = len(array) - 1
    
    while second_idx <= third_idx:
        value = array[second_idx]
        
        if value == first_value:
            array[first_idx], array[second_idx] = array[second_idx], array[first_idx]
            first_idx += 1
            second_idx += 1
        elif value == second_value:
            second_idx += 1
        else:
            array[second_idx], array[third_idx] = array[third_idx], array[second_idx]
            third_idx -= 1
    
    return array


# ============================================================================
# STACK ALGORITHMS
# ============================================================================

class MinMaxStack:
    """Stack that tracks minimum and maximum values."""
    
    def __init__(self):
        self.stack = []
        self.min_max_stack = []
    
    def peek(self):
        """Get the top element without removing it."""
        return self.stack[-1] if self.stack else None
    
    def pop(self):
        """Remove and return the top element."""
        if not self.stack:
            return None
        
        self.min_max_stack.pop()
        return self.stack.pop()
    
    def push(self, number):
        """Add a number to the top of the stack."""
        new_min_max = {"min": number, "max": number}
        
        if self.min_max_stack:
            last_min_max = self.min_max_stack[-1]
            new_min_max["min"] = min(last_min_max["min"], number)
            new_min_max["max"] = max(last_min_max["max"], number)
        
        self.min_max_stack.append(new_min_max)
        self.stack.append(number)
    
    def get_min(self):
        """Get the minimum value in the stack."""
        return self.min_max_stack[-1]["min"] if self.min_max_stack else None
    
    def get_max(self):
        """Get the maximum value in the stack."""
        return self.min_max_stack[-1]["max"] if self.min_max_stack else None


def balanced_brackets(string):
    """
    Check if a string has balanced brackets.
    
    Time Complexity: O(n) - Single pass through the string
    Space Complexity: O(n) - To store the stack
    
    Args:
        string: String containing brackets
    
    Returns:
        Boolean indicating if brackets are balanced
    """
    opening_brackets = "([{"
    closing_brackets = ")]}"
    matching_brackets = {")": "(", "]": "[", "}": "{"}
    
    stack = []
    
    for char in string:
        if char in opening_brackets:
            stack.append(char)
        elif char in closing_brackets:
            if not stack:
                return False
            
            if stack.pop() != matching_brackets[char]:
                return False
    
    return len(stack) == 0


def sunset_views(buildings, direction):
    """
    Find buildings that have a sunset view.
    
    Time Complexity: O(n) - Single pass through buildings
    Space Complexity: O(n) - To store buildings with sunset view
    
    Args:
        buildings: List of building heights
        direction: "EAST" or "WEST"
    
    Returns:
        List of indices of buildings with sunset view
    """
    if not buildings:
        return []
    
    if direction == "EAST":
        buildings = buildings[::-1]
    
    result = []
    max_height = 0
    
    for i, height in enumerate(buildings):
        if height > max_height:
            result.append(i)
            max_height = height
    
    if direction == "EAST":
        result = [len(buildings) - 1 - idx for idx in result]
    
    return result


def best_digits(number, num_digits):
    """
    Find the best digits by removing k digits to get the largest possible number.
    
    Time Complexity: O(n) - Where n is length of number
    Space Complexity: O(n) - To store the result
    
    Args:
        number: String representing a number
        num_digits: Number of digits to remove
    
    Returns:
        Largest possible number after removing digits
    """
    stack = []
    digits_to_remove = num_digits
    
    for digit in number:
        while stack and digits_to_remove > 0 and stack[-1] < digit:
            stack.pop()
            digits_to_remove -= 1
        stack.append(digit)
    
    # Remove remaining digits from the end if needed
    while digits_to_remove > 0:
        stack.pop()
        digits_to_remove -= 1
    
    return ''.join(stack)


def sort_stack(stack):
    """
    Sort a stack using only stack operations.
    
    Time Complexity: O(n²) - Where n is number of elements
    Space Complexity: O(n) - To store temporary stack
    
    Args:
        stack: Stack to sort
    
    Returns:
        Sorted stack
    """
    if not stack:
        return stack
    
    temp_stack = []
    
    while stack:
        temp = stack.pop()
        
        while temp_stack and temp_stack[-1] > temp:
            stack.append(temp_stack.pop())
        
        temp_stack.append(temp)
    
    return temp_stack


def next_greater_element(array):
    """
    Find the next greater element for each element in the array.
    
    Time Complexity: O(n) - Single pass through the array
    Space Complexity: O(n) - To store the result
    
    Args:
        array: List of integers
    
    Returns:
        List where each element is the next greater element
    """
    result = [-1] * len(array)
    stack = []
    
    for i in range(len(array)):
        while stack and array[stack[-1]] < array[i]:
            result[stack.pop()] = array[i]
        stack.append(i)
    
    return result


def reverse_polish_notation(tokens):
    """
    Evaluate a reverse polish notation expression.
    
    Time Complexity: O(n) - Where n is number of tokens
    Space Complexity: O(n) - To store the stack
    
    Args:
        tokens: List of tokens (numbers and operators)
    
    Returns:
        Result of the expression
    """
    stack = []
    
    for token in tokens:
        if token in "+-*/":
            b = stack.pop()
            a = stack.pop()
            
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            elif token == "/":
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]


def colliding_asteroids(asteroids):
    """
    Simulate asteroid collisions.
    
    Time Complexity: O(n) - Where n is number of asteroids
    Space Complexity: O(n) - To store the stack
    
    Args:
        asteroids: List of asteroid sizes (positive = right, negative = left)
    
    Returns:
        List of remaining asteroids after collisions
    """
    stack = []
    
    for asteroid in asteroids:
        while stack and asteroid < 0 and stack[-1] > 0:
            if abs(asteroid) > stack[-1]:
                stack.pop()
            elif abs(asteroid) == stack[-1]:
                stack.pop()
                break
            else:
                break
        else:
            stack.append(asteroid)
    
    return stack


# ============================================================================
# STRING ALGORITHMS
# ============================================================================

def longest_palindromic_substring(string):
    """
    Find the longest palindromic substring in a string.
    
    Time Complexity: O(n²) - Where n is length of string
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        string: Input string
    
    Returns:
        Longest palindromic substring
    """
    if len(string) <= 1:
        return string
    
    start = 0
    max_length = 1
    
    for i in range(len(string)):
        # Check odd length palindromes
        len1 = expand_around_center(string, i, i)
        # Check even length palindromes
        len2 = expand_around_center(string, i, i + 1)
        
        max_len = max(len1, len2)
        if max_len > max_length:
            start = i - (max_len - 1) // 2
            max_length = max_len
    
    return string[start:start + max_length]


def expand_around_center(string, left, right):
    """Helper function for palindrome expansion."""
    while left >= 0 and right < len(string) and string[left] == string[right]:
        left -= 1
        right += 1
    return right - left - 1


def group_anagrams(words):
    """
    Group words that are anagrams of each other.
    
    Time Complexity: O(n * k * log k) - Where n is number of words, k is max word length
    Space Complexity: O(n * k) - To store the result
    
    Args:
        words: List of words
    
    Returns:
        List of grouped anagrams
    """
    anagram_groups = {}
    
    for word in words:
        sorted_word = ''.join(sorted(word))
        if sorted_word in anagram_groups:
            anagram_groups[sorted_word].append(word)
        else:
            anagram_groups[sorted_word] = [word]
    
    return list(anagram_groups.values())


def valid_ip_addresses(string):
    """
    Generate all valid IP addresses from a string of digits.
    
    Time Complexity: O(1) - Constant time as IP has fixed format
    Space Complexity: O(1) - Constant number of valid IPs
    
    Args:
        string: String of digits
    
    Returns:
        List of valid IP addresses
    """
    def is_valid_part(part):
        if len(part) > 3 or len(part) == 0:
            return False
        if len(part) > 1 and part[0] == '0':
            return False
        return 0 <= int(part) <= 255
    
    def backtrack(start, parts):
        if len(parts) == 4:
            if start == len(string):
                result.append('.'.join(parts))
            return
        
        for i in range(1, 4):
            if start + i <= len(string):
                part = string[start:start + i]
                if is_valid_part(part):
                    backtrack(start + i, parts + [part])
    
    result = []
    backtrack(0, [])
    return result


def reverse_words_in_string(string):
    """
    Reverse the order of words in a string.
    
    Time Complexity: O(n) - Where n is length of string
    Space Complexity: O(n) - To store the result
    
    Args:
        string: Input string
    
    Returns:
        String with words reversed
    """
    # Split by spaces and reverse
    words = string.split()
    return ' '.join(reversed(words))


def minimum_characters_for_words(words):
    """
    Find minimum characters needed to write all words.
    
    Time Complexity: O(n * k) - Where n is number of words, k is max word length
    Space Complexity: O(c) - Where c is number of unique characters
    
    Args:
        words: List of words
    
    Returns:
        List of characters needed
    """
    char_counts = {}
    
    for word in words:
        word_counts = {}
        for char in word:
            word_counts[char] = word_counts.get(char, 0) + 1
        
        for char, count in word_counts.items():
            char_counts[char] = max(char_counts.get(char, 0), count)
    
    result = []
    for char, count in char_counts.items():
        result.extend([char] * count)
    
    return result


def one_edit(string_one, string_two):
    """
    Check if two strings are one edit away from each other.
    
    Time Complexity: O(n) - Where n is length of shorter string
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        string_one: First string
        string_two: Second string
    
    Returns:
        Boolean indicating if strings are one edit away
    """
    len_one, len_two = len(string_one), len(string_two)
    
    if abs(len_one - len_two) > 1:
        return False
    
    if len_one == len_two:
        # Check for one replacement
        differences = 0
        for i in range(len_one):
            if string_one[i] != string_two[i]:
                differences += 1
                if differences > 1:
                    return False
        return True
    else:
        # Check for one insertion/deletion
        shorter = string_one if len_one < len_two else string_two
        longer = string_two if len_one < len_two else string_one
        
        i = j = 0
        found_difference = False
        
        while i < len(shorter) and j < len(longer):
            if shorter[i] != longer[j]:
                if found_difference:
                    return False
                found_difference = True
                j += 1
            else:
                i += 1
                j += 1
        
        return True


# ============================================================================
# TRIE ALGORITHMS
# ============================================================================

class SuffixTrie:
    """Suffix Trie implementation."""
    
    def __init__(self, string):
        self.root = {}
        self.end_symbol = "*"
        self.populate_suffix_trie_from(string)
    
    def populate_suffix_trie_from(self, string):
        """Build the suffix trie from a string."""
        for i in range(len(string)):
            self.insert_substring_starting_at(i, string)
    
    def insert_substring_starting_at(self, i, string):
        """Insert substring starting at index i."""
        node = self.root
        for j in range(i, len(string)):
            letter = string[j]
            if letter not in node:
                node[letter] = {}
            node = node[letter]
        node[self.end_symbol] = True
    
    def contains(self, string):
        """Check if string is contained in the trie."""
        node = self.root
        for letter in string:
            if letter not in node:
                return False
            node = node[letter]
        return self.end_symbol in node
