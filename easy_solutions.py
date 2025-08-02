"""
Comprehensive Coding Solutions
=============================

This file contains solutions to 30 common coding problems with detailed comments
and time/space complexity analysis.

Author: AI Assistant
Date: 2024
"""

import math
from typing import List, Dict, Set, Optional, Union, Any


# ============================================================================
# ARRAY AND STRING PROBLEMS
# ============================================================================

def two_number_sum(array: List[int], target_sum: int) -> List[int]:
    """
    Find two numbers in the array that sum to target_sum.
    
    Time Complexity: O(n) - we traverse the array once
    Space Complexity: O(n) - we store at most n elements in the hash set
    """
    # Use a hash set to store numbers we've seen
    seen = set()
    
    for num in array:
        # Calculate the complement needed to reach target_sum
        complement = target_sum - num
        
        # If we've seen the complement, we found our pair
        if complement in seen:
            return [complement, num]
        
        # Add current number to seen set
        seen.add(num)
    
    # If no pair found, return empty array
    return []


def two_number_sum_sorted(array: List[int], target_sum: int) -> List[int]:
    """
    Alternative solution using sorting.
    
    Time Complexity: O(n log n) - sorting dominates
    Space Complexity: O(1) - only using a few variables
    """
    array.sort()
    left = 0
    right = len(array) - 1
    
    while left < right:
        current_sum = array[left] + array[right]
        
        if current_sum == target_sum:
            return [array[left], array[right]]
        elif current_sum < target_sum:
            left += 1
        else:
            right -= 1
    
    return []


def is_valid_subsequence(array: List[int], sequence: List[int]) -> bool:
    """
    Check if sequence is a subsequence of array.
    
    Time Complexity: O(n) - we traverse the main array once
    Space Complexity: O(1) - we only use a few variables
    """
    # Pointer to track position in sequence
    seq_idx = 0
    
    # Traverse the main array
    for value in array:
        # If we've found all elements in sequence, we're done
        if seq_idx == len(sequence):
            break
        
        # If current value matches sequence element, move to next
        if sequence[seq_idx] == value:
            seq_idx += 1
    
    # Return True if we've processed all elements in sequence
    return seq_idx == len(sequence)


def sorted_squared_array(array: List[int]) -> List[int]:
    """
    Return sorted array of squares of input array.
    
    Time Complexity: O(n) - we traverse the array once
    Space Complexity: O(n) - we create a new array of same size
    """
    # Initialize result array with zeros
    result = [0 for _ in array]
    
    # Use two pointers approach since array is sorted
    left = 0
    right = len(array) - 1
    
    # Fill result array from right to left (largest to smallest)
    for i in range(len(array) - 1, -1, -1):
        left_value = abs(array[left])
        right_value = abs(array[right])
        
        if left_value > right_value:
            result[i] = left_value * left_value
            left += 1
        else:
            result[i] = right_value * right_value
            right -= 1
    
    return result


def tournament_winner(competitions: List[List[str]], results: List[int]) -> str:
    """
    Determine the tournament winner based on competitions and results.
    
    Time Complexity: O(n) - we traverse competitions array once
    Space Complexity: O(k) - where k is number of unique teams
    """
    # Dictionary to store team scores
    scores = {}
    
    # Process each competition
    for i, competition in enumerate(competitions):
        home_team, away_team = competition
        result = results[i]
        
        # Determine winner (0 means away team won, 1 means home team won)
        winning_team = home_team if result == 1 else away_team
        
        # Update score
        if winning_team in scores:
            scores[winning_team] += 3
        else:
            scores[winning_team] = 3
    
    # Find team with highest score
    winner = max(scores.keys(), key=lambda team: scores[team])
    return winner


def non_constructible_change(coins: List[int]) -> int:
    """
    Find minimum amount of change that cannot be created.
    
    Time Complexity: O(n log n) - sorting dominates
    Space Complexity: O(1) - only using a few variables
    """
    # Sort coins to process them in ascending order
    coins.sort()
    
    # Track the current change we can create
    current_change_created = 0
    
    for coin in coins:
        # If current coin is greater than what we can create + 1,
        # then we found our answer
        if coin > current_change_created + 1:
            return current_change_created + 1
        
        # Add current coin to what we can create
        current_change_created += coin
    
    # If we can create all amounts up to sum, return sum + 1
    return current_change_created + 1


def transpose_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Transpose the given matrix.
    
    Time Complexity: O(n*m) - we visit each element once
    Space Complexity: O(n*m) - we create a new matrix
    """
    # Get dimensions
    rows = len(matrix)
    cols = len(matrix[0])
    
    # Create new matrix with swapped dimensions
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    
    # Fill the transposed matrix
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    
    return transposed


# ============================================================================
# BINARY SEARCH TREE PROBLEMS
# ============================================================================

class BST:
    """Binary Search Tree node class."""
    
    def __init__(self, value: int):
        self.value = value
        self.left = None
        self.right = None


def find_closest_value_in_bst(tree: BST, target: int) -> int:
    """
    Find the closest value to target in BST.
    
    Time Complexity: O(h) - where h is height of tree
    Space Complexity: O(h) - due to recursion stack
    """
    return find_closest_value_in_bst_helper(tree, target, float('inf'))


def find_closest_value_in_bst_helper(tree: BST, target: int, closest: int) -> int:
    """
    Helper function for finding closest value.
    """
    if tree is None:
        return closest
    
    # Update closest if current node is closer
    if abs(target - closest) > abs(target - tree.value):
        closest = tree.value
    
    # Navigate BST based on target value
    if target < tree.value:
        return find_closest_value_in_bst_helper(tree.left, target, closest)
    elif target > tree.value:
        return find_closest_value_in_bst_helper(tree.right, target, closest)
    else:
        return closest


def branch_sums(root: BST) -> List[int]:
    """
    Calculate sum of all branches in binary tree.
    
    Time Complexity: O(n) - we visit each node once
    Space Complexity: O(h) - where h is height of tree (recursion stack)
    """
    sums = []
    calculate_branch_sums(root, 0, sums)
    return sums


def calculate_branch_sums(node: BST, running_sum: int, sums: List[int]) -> None:
    """
    Helper function to calculate branch sums recursively.
    """
    if node is None:
        return
    
    # Add current node value to running sum
    new_running_sum = running_sum + node.value
    
    # If leaf node, add sum to results
    if node.left is None and node.right is None:
        sums.append(new_running_sum)
        return
    
    # Recursively process left and right children
    calculate_branch_sums(node.left, new_running_sum, sums)
    calculate_branch_sums(node.right, new_running_sum, sums)


def node_depths(root: BST) -> int:
    """
    Calculate sum of depths of all nodes in binary tree.
    
    Time Complexity: O(n) - we visit each node once
    Space Complexity: O(h) - where h is height of tree (recursion stack)
    """
    return node_depths_helper(root, 0)


def node_depths_helper(node: BST, depth: int) -> int:
    """
    Helper function to calculate node depths recursively.
    """
    if node is None:
        return 0
    
    # Current depth + depths of left and right subtrees
    return depth + node_depths_helper(node.left, depth + 1) + node_depths_helper(node.right, depth + 1)


# ============================================================================
# EXPRESSION TREE AND GRAPH PROBLEMS
# ============================================================================

class BinaryTree:
    """Binary Tree node class for expression trees."""
    
    def __init__(self, value: int):
        self.value = value
        self.left = None
        self.right = None


def evaluate_expression_tree(tree: BinaryTree) -> int:
    """
    Evaluate a binary expression tree.
    
    Time Complexity: O(n) - we visit each node once
    Space Complexity: O(h) - where h is height of tree (recursion stack)
    """
    # If leaf node, return the value
    if tree.left is None and tree.right is None:
        return tree.value
    
    # Recursively evaluate left and right subtrees
    left_value = evaluate_expression_tree(tree.left)
    right_value = evaluate_expression_tree(tree.right)
    
    # Apply the operator
    if tree.value == -1:  # Addition
        return left_value + right_value
    elif tree.value == -2:  # Subtraction
        return left_value - right_value
    elif tree.value == -3:  # Division
        return left_value // right_value
    elif tree.value == -4:  # Multiplication
        return left_value * right_value
    
    return 0


class Node:
    """Node class for graph traversal."""
    
    def __init__(self, name: str):
        self.children = []
        self.name = name

    def add_child(self, name: str) -> 'Node':
        self.children.append(Node(name))
        return self

    def depth_first_search(self, array: List[str]) -> List[str]:
        """
        Perform depth-first search and return array of node names.
        
        Time Complexity: O(v + e) - where v is vertices, e is edges
        Space Complexity: O(v) - to store the result array
        """
        # Add current node to array
        array.append(self.name)
        
        # Recursively process all children
        for child in self.children:
            child.depth_first_search(array)
        
        return array


# ============================================================================
# GREEDY ALGORITHMS
# ============================================================================

def minimum_waiting_time(queries: List[int]) -> int:
    """
    Calculate minimum total waiting time for queries.
    
    Time Complexity: O(n log n) - sorting dominates
    Space Complexity: O(1) - only using a few variables
    """
    # Sort queries to process shortest first
    queries.sort()
    
    total_waiting_time = 0
    queries_left = len(queries)
    
    for duration in queries:
        # Number of queries that will wait for this duration
        queries_left -= 1
        total_waiting_time += duration * queries_left
    
    return total_waiting_time


def class_photos(red_shirt_heights: List[int], blue_shirt_heights: List[int]) -> bool:
    """
    Determine if students can be arranged in two rows for photo.
    
    Time Complexity: O(n log n) - sorting dominates
    Space Complexity: O(1) - only using a few variables
    """
    # Sort both arrays
    red_shirt_heights.sort()
    blue_shirt_heights.sort()
    
    # Determine which row should be in front
    first_row_color = 'RED' if red_shirt_heights[0] < blue_shirt_heights[0] else 'BLUE'
    
    # Check if arrangement is possible
    for i in range(len(red_shirt_heights)):
        red_height = red_shirt_heights[i]
        blue_height = blue_shirt_heights[i]
        
        if first_row_color == 'RED':
            if red_height >= blue_height:
                return False
        else:
            if blue_height >= red_height:
                return False
    
    return True


def tandem_bicycle(red_shirt_speeds: List[int], blue_shirt_speeds: List[int], fastest: bool) -> int:
    """
    Calculate total speed of tandem bicycles.
    
    Time Complexity: O(n log n) - sorting dominates
    Space Complexity: O(1) - only using a few variables
    """
    # Sort both arrays
    red_shirt_speeds.sort()
    blue_shirt_speeds.sort()
    
    total_speed = 0
    
    if fastest:
        # For maximum speed, pair fastest with fastest
        for i in range(len(red_shirt_speeds)):
            total_speed += max(red_shirt_speeds[i], blue_shirt_speeds[len(blue_shirt_speeds) - 1 - i])
    else:
        # For minimum speed, pair fastest with slowest
        for i in range(len(red_shirt_speeds)):
            total_speed += max(red_shirt_speeds[i], blue_shirt_speeds[i])
    
    return total_speed


def optimal_freelancing(jobs: List[Dict[str, int]]) -> int:
    """
    Find optimal schedule for freelancing jobs.
    
    Time Complexity: O(n log n) - sorting dominates
    Space Complexity: O(1) - only using a few variables
    """
    # Sort jobs by payment (highest first)
    jobs.sort(key=lambda x: x['payment'], reverse=True)
    
    # Track which days are taken
    taken = [False] * 7
    
    total_payment = 0
    
    for job in jobs:
        # Try to schedule job on latest possible day
        for day in range(min(job['deadline'], 7) - 1, -1, -1):
            if not taken[day]:
                taken[day] = True
                total_payment += job['payment']
                break
    
    return total_payment


# ============================================================================
# LINKED LIST PROBLEMS
# ============================================================================

class LinkedList:
    """Linked List node class."""
    
    def __init__(self, value: int):
        self.value = value
        self.next = None


def remove_duplicates_from_linked_list(linked_list: LinkedList) -> LinkedList:
    """
    Remove duplicates from linked list.
    
    Time Complexity: O(n) - we traverse the list once
    Space Complexity: O(1) - only using a few variables
    """
    current_node = linked_list
    
    while current_node is not None:
        # Look ahead to find next distinct node
        next_distinct_node = current_node.next
        while next_distinct_node is not None and next_distinct_node.value == current_node.value:
            next_distinct_node = next_distinct_node.next
        
        # Remove duplicates by linking to next distinct node
        current_node.next = next_distinct_node
        current_node = next_distinct_node
    
    return linked_list


def middle_node(linked_list: LinkedList) -> LinkedList:
    """
    Find middle node of linked list.
    
    Time Complexity: O(n) - we traverse the list once
    Space Complexity: O(1) - only using a few variables
    """
    # Use two pointers: slow and fast
    slow = linked_list
    fast = linked_list
    
    # Fast pointer moves twice as fast as slow pointer
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    
    return slow


# ============================================================================
# DYNAMIC PROGRAMMING
# ============================================================================

def get_nth_fib(n: int) -> int:
    """
    Calculate nth Fibonacci number.
    
    Time Complexity: O(n) - we calculate each Fibonacci number once
    Space Complexity: O(1) - only using a few variables
    """
    if n == 1:
        return 0
    elif n == 2:
        return 1
    
    # Use iterative approach to avoid stack overflow
    prev = 0
    current = 1
    
    for i in range(3, n + 1):
        next_fib = prev + current
        prev = current
        current = next_fib
    
    return current


def get_nth_fib_recursive(n: int) -> int:
    """
    Recursive solution for nth Fibonacci (less efficient).
    
    Time Complexity: O(2^n) - exponential due to repeated calculations
    Space Complexity: O(n) - recursion stack depth
    """
    if n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return get_nth_fib_recursive(n - 1) + get_nth_fib_recursive(n - 2)


def product_sum(array: List[Union[int, List]], multiplier: int = 1) -> int:
    """
    Calculate product sum of special array.
    
    Time Complexity: O(n) - where n is total number of elements including nested ones
    Space Complexity: O(d) - where d is maximum depth of nesting
    """
    total = 0
    
    for element in array:
        if isinstance(element, list):
            # Recursively calculate product sum for nested arrays
            total += product_sum(element, multiplier + 1)
        else:
            # Add element multiplied by current depth
            total += element
    
    return total * multiplier


# ============================================================================
# SEARCHING AND SORTING
# ============================================================================

def binary_search(array: List[int], target: int) -> int:
    """
    Perform binary search to find target in sorted array.
    
    Time Complexity: O(log n) - we halve the search space each iteration
    Space Complexity: O(1) - only using a few variables
    """
    left = 0
    right = len(array) - 1
    
    while left <= right:
        # Calculate middle index
        middle = (left + right) // 2
        
        # Check if target is at middle
        if array[middle] == target:
            return middle
        elif array[middle] < target:
            # Target is in right half
            left = middle + 1
        else:
            # Target is in left half
            right = middle - 1
    
    # Target not found
    return -1


def find_three_largest_numbers(array: List[int]) -> List[int]:
    """
    Find three largest numbers in array.
    
    Time Complexity: O(n) - we traverse the array once
    Space Complexity: O(1) - we only store 3 values
    """
    # Initialize with negative infinity
    three_largest = [float('-inf'), float('-inf'), float('-inf')]
    
    for num in array:
        update_largest(three_largest, num)
    
    return three_largest


def update_largest(three_largest: List[int], num: int) -> None:
    """
    Helper function to update three largest numbers.
    """
    if num > three_largest[2]:
        shift_and_update(three_largest, num, 2)
    elif num > three_largest[1]:
        shift_and_update(three_largest, num, 1)
    elif num > three_largest[0]:
        shift_and_update(three_largest, num, 0)


def shift_and_update(array: List[int], num: int, idx: int) -> None:
    """
    Helper function to shift and update array.
    """
    for i in range(idx + 1):
        if i == idx:
            array[i] = num
        else:
            array[i] = array[i + 1]


def bubble_sort(array: List[int]) -> List[int]:
    """
    Sort array using bubble sort algorithm.
    
    Time Complexity: O(n²) - nested loops
    Space Complexity: O(1) - in-place sorting
    """
    is_sorted = False
    counter = 0
    
    while not is_sorted:
        is_sorted = True
        
        # Compare adjacent elements and swap if needed
        for i in range(len(array) - 1 - counter):
            if array[i] > array[i + 1]:
                # Swap elements
                array[i], array[i + 1] = array[i + 1], array[i]
                is_sorted = False
        
        counter += 1
    
    return array


def insertion_sort(array: List[int]) -> List[int]:
    """
    Sort array using insertion sort algorithm.
    
    Time Complexity: O(n²) - nested loops
    Space Complexity: O(1) - in-place sorting
    """
    # Start from second element
    for i in range(1, len(array)):
        j = i
        
        # Move current element to its correct position
        while j > 0 and array[j] < array[j - 1]:
            # Swap elements
            array[j], array[j - 1] = array[j - 1], array[j]
            j -= 1
    
    return array


def selection_sort(array: List[int]) -> List[int]:
    """
    Sort array using selection sort algorithm.
    
    Time Complexity: O(n²) - nested loops
    Space Complexity: O(1) - in-place sorting
    """
    # Find minimum element and place it at beginning
    for i in range(len(array)):
        min_idx = i
        
        # Find minimum element in unsorted portion
        for j in range(i + 1, len(array)):
            if array[j] < array[min_idx]:
                min_idx = j
        
        # Swap minimum element with current position
        if min_idx != i:
            array[i], array[min_idx] = array[min_idx], array[i]
    
    return array


# ============================================================================
# STRING MANIPULATION
# ============================================================================

def is_palindrome(string: str) -> bool:
    """
    Check if string is palindrome.
    
    Time Complexity: O(n) - we traverse half the string
    Space Complexity: O(1) - only using a few variables
    """
    # Use two pointers approach
    left = 0
    right = len(string) - 1
    
    while left < right:
        # Compare characters from both ends
        if string[left] != string[right]:
            return False
        left += 1
        right -= 1
    
    return True


def caesar_cipher_encryptor(string: str, key: int) -> str:
    """
    Encrypt string using Caesar cipher.
    
    Time Complexity: O(n) - we traverse the string once
    Space Complexity: O(n) - we create a new string
    """
    # Handle key wrapping
    key = key % 26
    
    # Convert string to list for easier manipulation
    result = []
    
    for char in string:
        # Get ASCII value
        ascii_value = ord(char)
        
        # Calculate new ASCII value
        new_ascii_value = ascii_value + key
        
        # Handle wrapping around alphabet
        if new_ascii_value > ord('z'):
            new_ascii_value = ord('a') + (new_ascii_value - ord('z') - 1)
        
        # Convert back to character
        result.append(chr(new_ascii_value))
    
    return ''.join(result)


def run_length_encoding(string: str) -> str:
    """
    Encode string using run-length encoding.
    
    Time Complexity: O(n) - we traverse the string once
    Space Complexity: O(n) - worst case when no consecutive characters
    """
    # Handle edge case
    if not string:
        return ""
    
    result = []
    current_char = string[0]
    current_count = 1
    
    # Process string starting from second character
    for i in range(1, len(string)):
        if string[i] == current_char and current_count < 9:
            current_count += 1
        else:
            # Add current run to result
            result.append(str(current_count) + current_char)
            current_char = string[i]
            current_count = 1
    
    # Add final run
    result.append(str(current_count) + current_char)
    
    return ''.join(result)


def common_characters(strings: List[str]) -> List[str]:
    """
    Find characters common to all strings.
    
    Time Complexity: O(n*m) - where n is number of strings, m is max string length
    Space Complexity: O(c) - where c is number of unique characters
    """
    # Get character counts for first string
    char_counts = get_char_counts(strings[0])
    
    # Update counts based on other strings
    for string in strings[1:]:
        string_char_counts = get_char_counts(string)
        
        # Update counts to minimum
        for char in char_counts:
            if char in string_char_counts:
                char_counts[char] = min(char_counts[char], string_char_counts[char])
            else:
                char_counts[char] = 0
    
    # Build result array
    result = []
    for char, count in char_counts.items():
        result.extend([char] * count)
    
    return result


def get_char_counts(string: str) -> Dict[str, int]:
    """
    Helper function to get character counts.
    """
    char_counts = {}
    for char in string:
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] = 1
    return char_counts


def generate_document(characters: str, document: str) -> bool:
    """
    Check if document can be generated from characters.
    
    Time Complexity: O(n + m) - where n is length of characters, m is length of document
    Space Complexity: O(c) - where c is number of unique characters
    """
    # Get character counts
    char_counts = {}
    
    # Count characters available
    for char in characters:
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] = 1
    
    # Check if we can generate document
    for char in document:
        if char not in char_counts or char_counts[char] == 0:
            return False
        char_counts[char] -= 1
    
    return True


def first_non_repeating_character(string: str) -> int:
    """
    Find first non-repeating character in string.
    
    Time Complexity: O(n) - we traverse the string twice
    Space Complexity: O(1) - we store at most 26 characters (alphabet)
    """
    # Count character frequencies
    char_counts = {}
    
    for char in string:
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] = 1
    
    # Find first character with count 1
    for i, char in enumerate(string):
        if char_counts[char] == 1:
            return i
    
    return -1


def semordnilap(words: List[str]) -> List[List[str]]:
    """
    Find all semordnilap pairs in words array.
    
    Time Complexity: O(n*m) - where n is number of words, m is average word length
    Space Complexity: O(n) - to store the set of words
    """
    # Use set for O(1) lookup
    words_set = set(words)
    semordnilap_pairs = []
    
    for word in words:
        # Create reverse of current word
        reverse = word[::-1]
        
        # Check if reverse exists and is different from current word
        if reverse in words_set and reverse != word:
            # Add pair (avoid duplicates by checking if current word is lexicographically smaller)
            if word < reverse:
                semordnilap_pairs.append([word, reverse])
    
    return semordnilap_pairs