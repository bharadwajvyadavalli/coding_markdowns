# Comprehensive Algorithms Collection

This repository contains implementations of 70+ algorithms with detailed comments and time/space complexity analysis. The algorithms are organized into three Python files for better readability and maintenance.

## 📁 File Organization

### `algorithms.py` - Part 1
**Array Algorithms & Binary Search Trees**
- Three Number Sum
- Smallest Difference
- Move Element To End
- Monotonic Array
- Spiral Traverse
- Longest Peak
- Array Of Products
- First Duplicate Value
- Merge Overlapping Intervals
- Best Seat
- Zero Sum Subarray
- Missing Numbers
- Majority Element
- Sweet And Savory
- BST Construction
- Validate BST
- BST Traversal
- Min Height BST
- Find Kth Largest Value In BST
- Reconstruct BST
- Invert Binary Tree
- Binary Tree Diameter
- Find Successor
- Height Balanced Binary Tree
- Merge Binary Trees
- Symmetrical Tree
- Split Binary Tree

### `algorithms_part2.py` - Part 2
**Dynamic Programming, Graphs, Heaps & Linked Lists**
- Max Subset Sum No Adjacent
- Number Of Ways To Make Change
- Min Number Of Coins For Change
- Levenshtein Distance
- Number Of Ways To Traverse Graph
- Kadane's Algorithm
- Stable Internships
- Union Find
- Single Cycle Check
- Breadth-first Search
- River Sizes
- Youngest Common Ancestor
- Remove Islands
- Cycle In Graph
- Minimum Passes Of Matrix
- Two-Colorable
- Task Assignment
- Valid Starting City
- Min Heap Construction
- Linked List Construction
- Remove Kth Node From End
- Sum of Linked Lists
- Merging Linked Lists

### `algorithms_part3.py` - Part 3
**Recursion, Backtracking, Stacks & Strings**
- Permutations
- Powerset
- Phone Number Mnemonics
- Staircase Traversal
- Blackjack Probability
- Reveal Minesweeper
- Search In Sorted Matrix
- Three Number Sort
- Min Max Stack Construction
- Balanced Brackets
- Sunset Views
- Best Digits
- Sort Stack
- Next Greater Element
- Reverse Polish Notation
- Colliding Asteroids
- Longest Palindromic Substring
- Group Anagrams
- Valid IP Addresses
- Reverse Words In String
- Minimum Characters For Words
- One Edit
- Suffix Trie Construction

## 🚀 Quick Start

### Running All Tests
```bash
# Run all algorithm tests
python algorithms.py
python algorithms_part2.py
python algorithms_part3.py
```

### Using Individual Algorithms
```python
# Import specific algorithms
from algorithms import three_number_sum, smallest_difference
from algorithms_part2 import max_subset_sum_no_adjacent, kadane_algorithm
from algorithms_part3 import permutations, longest_palindromic_substring

# Example usage
array = [12, 3, 1, 2, -6, 5, -8, 6]
target = 0
result = three_number_sum(array, target)
print(f"Three Number Sum: {result}")
```

## 📊 Algorithm Categories

### Array Manipulation
- **Two Pointers**: Three Number Sum, Smallest Difference, Move Element To End
- **Sliding Window**: Longest Peak, Best Seat
- **Prefix Sum**: Zero Sum Subarray, Array Of Products
- **Sorting**: Monotonic Array, Merge Overlapping Intervals

### Binary Search Trees
- **Construction**: BST Construction, Min Height BST, Reconstruct BST
- **Traversal**: BST Traversal (Inorder, Preorder, Postorder)
- **Validation**: Validate BST, Height Balanced Binary Tree
- **Operations**: Find Kth Largest, Find Successor, Invert Binary Tree

### Dynamic Programming
- **1D DP**: Max Subset Sum No Adjacent, Number Of Ways To Make Change
- **2D DP**: Levenshtein Distance, Number Of Ways To Traverse Graph
- **Optimization**: Min Number Of Coins For Change, Kadane's Algorithm

### Graph Algorithms
- **Traversal**: Breadth-first Search, River Sizes
- **Cycle Detection**: Single Cycle Check, Cycle In Graph
- **Connectivity**: Remove Islands, Two-Colorable
- **Matching**: Stable Internships

### Data Structures
- **Heaps**: Min Heap Construction
- **Linked Lists**: Linked List Construction, Remove Kth Node, Sum of Lists
- **Stacks**: Min Max Stack, Balanced Brackets, Sort Stack
- **Tries**: Suffix Trie Construction

### Recursion & Backtracking
- **Combinatorics**: Permutations, Powerset, Phone Number Mnemonics
- **Path Finding**: Staircase Traversal
- **Game Theory**: Blackjack Probability, Reveal Minesweeper

### String Algorithms
- **Palindrome**: Longest Palindromic Substring
- **Anagrams**: Group Anagrams
- **Parsing**: Valid IP Addresses, Reverse Words In String
- **Edit Distance**: One Edit, Minimum Characters For Words

## ⏱️ Time & Space Complexity

Each algorithm includes detailed complexity analysis:

```python
def example_algorithm(array):
    """
    Example algorithm with complexity analysis.
    
    Time Complexity: O(n) - Single pass through array
    Space Complexity: O(1) - Only using a few variables
    
    Args:
        array: List of integers
    
    Returns:
        Result of the algorithm
    """
    # Implementation here
    pass
```

## 🧪 Testing

Each file includes comprehensive test functions that demonstrate:
- Basic functionality
- Edge cases
- Expected outputs
- Performance characteristics

### Running Tests
```python
# Test specific categories
test_array_algorithms()
test_bst_algorithms()
test_dynamic_programming()
test_graph_algorithms()
test_recursion_algorithms()
test_string_algorithms()
```

## 📈 Performance Considerations

### Time Complexity Ranges
- **O(1)**: Constant time operations
- **O(log n)**: Binary search, tree operations
- **O(n)**: Linear time algorithms
- **O(n log n)**: Sorting-based algorithms
- **O(n²)**: Quadratic time algorithms
- **O(2^n)**: Exponential time (powerset, some DP)
- **O(n!)**: Factorial time (permutations)

### Space Complexity Considerations
- **O(1)**: In-place algorithms
- **O(n)**: Linear space for result storage
- **O(n²)**: Matrix-based algorithms
- **O(2^n)**: Exponential space for combinatorial results

## 🔧 Customization

### Adding New Algorithms
1. Choose the appropriate file based on category
2. Add the algorithm with proper documentation
3. Include time/space complexity analysis
4. Add test cases
5. Update this README

### Modifying Existing Algorithms
- Maintain the same function signature
- Update complexity analysis if changes affect performance
- Ensure test cases still pass
- Update documentation if needed

## 📚 Learning Resources

This collection is designed to complement:
- **AlgoExpert**: Advanced algorithm problems
- **LeetCode**: Competitive programming
- **Interview Preparation**: Technical interviews
- **Computer Science Education**: Algorithm concepts

## 🤝 Contributing

Feel free to:
- Add new algorithms
- Improve existing implementations
- Add more test cases
- Enhance documentation
- Optimize performance

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Coding! 🚀**