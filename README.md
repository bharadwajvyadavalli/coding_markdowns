# Comprehensive Algorithms Collection

This repository contains implementations of 200+ algorithms organized by difficulty level, with detailed comments and time/space complexity analysis. The algorithms are organized into four Python files for better readability and maintenance.

## 📁 File Organization

### `easy_solutions.py` - Easy Level Algorithms
**30 fundamental algorithms covering basic data structures and common problems**
- **Array & String Problems**: Two Number Sum, Valid Subsequence, Sorted Squared Array, Tournament Winner
- **Binary Search Trees**: Find Closest Value, Branch Sums, Node Depths, Evaluate Expression Tree
- **Graphs**: Depth First Search
- **Greedy Algorithms**: Minimum Waiting Time, Class Photos, Tandem Bicycle, Optimal Freelancing
- **Linked Lists**: Remove Duplicates, Middle Node
- **Dynamic Programming**: Nth Fibonacci (iterative & recursive), Product Sum
- **Search**: Binary Search, Find Three Largest Numbers
- **Sorting**: Bubble Sort, Insertion Sort, Selection Sort
- **String Manipulation**: Palindrome Check, Caesar Cipher, Run Length Encoding, Common Characters
- **Document Processing**: Generate Document, First Non-Repeating Character, Semordnilap

### `medium_algorithms.py` - Medium Level Algorithms
**70+ intermediate algorithms covering advanced data structures and techniques**
- **Array Algorithms**: Three Number Sum, Smallest Difference, Move Element To End, Monotonic Array, Spiral Traverse, Longest Peak, Array Of Products, First Duplicate Value, Merge Overlapping Intervals, Best Seat, Zero Sum Subarray, Missing Numbers, Majority Element, Sweet And Savory
- **Binary Search Trees**: BST Construction, Validate BST, BST Traversal, Min Height BST, Find Kth Largest Value, Reconstruct BST, Invert Binary Tree, Binary Tree Diameter, Find Successor, Height Balanced Binary Tree, Merge Binary Trees, Symmetrical Tree, Split Binary Tree
- **Dynamic Programming**: Max Subset Sum No Adjacent, Number Of Ways To Make Change, Min Number Of Coins For Change, Levenshtein Distance, Number Of Ways To Traverse Graph, Kadane's Algorithm
- **Graph Algorithms**: Stable Internships, Union Find, Single Cycle Check, Breadth-first Search, River Sizes, Youngest Common Ancestor, Remove Islands, Cycle In Graph, Minimum Passes Of Matrix, Two-Colorable, Task Assignment, Valid Starting City
- **Data Structures**: Min Heap Construction, Linked List Construction, Remove Kth Node From End, Sum of Linked Lists, Merging Linked Lists
- **Recursion & Backtracking**: Permutations, Powerset, Phone Number Mnemonics, Staircase Traversal, Blackjack Probability, Reveal Minesweeper
- **Matrix & Stack**: Search In Sorted Matrix, Three Number Sort, Min Max Stack Construction, Balanced Brackets, Sunset Views, Best Digits, Sort Stack, Next Greater Element, Reverse Polish Notation, Colliding Asteroids
- **String Algorithms**: Longest Palindromic Substring, Group Anagrams, Valid IP Addresses, Reverse Words In String, Minimum Characters For Words, One Edit, Suffix Trie Construction

### `advanced_algorithms.py` - Advanced Level Algorithms
**60+ complex algorithms covering advanced techniques and optimization**
- **Array & Matrix**: Four Number Sum, Subarray Sort, Largest Range, Min Rewards, Zigzag Traverse, Longest Subarray With Sum, Count Squares
- **Binary Search Trees**: Same BSTs, Validate Three Nodes, Repair BST, Sum BSTs, Max Path Sum In Binary Tree, Find Nodes Distance K
- **Dynamic Programming**: Max Sum Increasing Subsequence, Longest Common Subsequence, Min Number Of Jumps, Water Area, Knapsack Problem, Disk Stacking, Numbers In Pi, Maximum Sum Submatrix, Maximize Expression, Dice Throws, Juice Bottling
- **Graph Algorithms**: Dijkstra's Algorithm, Topological Sort, Kruskal's Algorithm, Prim's Algorithm, Boggle Board, Largest Island
- **Linked Lists**: Continuous Median, Sort K Sorted Array, Laptop Rentals, Find Loop, Reverse Linked List, Merge Linked Lists, Shift Linked List
- **Tree Algorithms**: Lowest Common Manager, Interweaving Strings, Solve Sudoku, Generate Div Tags, Ambiguous Measurements
- **Search & Sort**: Shifted Binary Search, Search For Range, Quickselect, Index Equals Value, Quick Sort, Heap Sort, Radix Sort
- **String & Path**: Shorten Path, Largest Rectangle Under Skyline, Longest Substring Without Duplication, Underscorify Substring, Pattern Matcher, Multi String Search, Longest Most Frequent Prefix, Shortest Unique Prefixes

### `very_advanced_algorithms.py` - Very Advanced Level Algorithms
**40+ expert-level algorithms covering cutting-edge techniques**
- **Optimization Problems**: Apartment Hunting, Calendar Matching, Waterfall Streams, Minimum Area Rectangle, Line Through Points, Right Smaller Than
- **Tree Traversals**: Iterative Inorder Traversal, Flatten Binary Tree, Right Sibling Tree, All Kinds Of Node Depths, Compare Leaf Traversal
- **Advanced Dynamic Programming**: Max Profit With K Transactions, Palindrome Partitioning Min Cuts, Longest Increasing Subsequence, Longest String Chain
- **Matrix & Geometry**: Square Of Zeroes, Rectangle Mania
- **String Algorithms**: Knuth-Morris-Pratt Algorithm, Longest Balanced Substring, Smallest Substring Containing, Strings Made Up Of Strings
- **Graph Algorithms**: A* Algorithm, Detect Arbitrage, Two Edge Connected Graph, Airport Connections
- **Data Structures**: Merge Sorted Arrays, LRU Cache, Rearrange Linked List, Linked List Palindrome, Reverse Linked List, Zip Linked List, Node Swap
- **Combinatorics**: Number Of Binary Tree Topologies, Non Attacking Queens
- **Advanced Search**: Median Of Two Sorted Arrays, Optimal Assembly Line
- **Sorting & Analysis**: Merge Sort, Count Inversions, Largest Park, Largest Rectangle In Histogram

## 🚀 Quick Start

### Running All Tests
```bash
# Run all algorithm tests
python easy_solutions.py
python medium_algorithms.py
python advanced_algorithms.py
python very_advanced_algorithms.py
```

### Using Individual Algorithms
```python
# Import specific algorithms
from easy_solutions import two_number_sum, binary_search
from medium_algorithms import three_number_sum, validate_bst
from advanced_algorithms import four_number_sum, dijkstra_algorithm
from very_advanced_algorithms import apartment_hunting, calendar_matching

# Example usage
array = [12, 3, 1, 2, -6, 5, -8, 6]
target = 0
result = three_number_sum(array, target)
print(f"Three Number Sum: {result}")
```

## 📊 Algorithm Categories by Difficulty

### Easy Level (30 algorithms)
- **Basic Data Structures**: Arrays, Strings, BSTs, Linked Lists
- **Simple Algorithms**: Two pointers, basic traversal, greedy approaches
- **Common Problems**: Search, sorting, string manipulation
- **Time Complexity**: Mostly O(n) to O(n²)
- **Space Complexity**: Mostly O(1) to O(n)

### Medium Level (70+ algorithms)
- **Advanced Data Structures**: Heaps, Graphs, Stacks, Tries
- **Complex Techniques**: Dynamic programming, backtracking, graph traversal
- **Optimization Problems**: Path finding, scheduling, matching
- **Time Complexity**: O(n log n) to O(n³)
- **Space Complexity**: O(n) to O(n²)

### Advanced Level (60+ algorithms)
- **Expert Techniques**: Advanced DP, graph algorithms, tree manipulation
- **Optimization**: Complex state management, memoization, pruning
- **Specialized Algorithms**: Geometry, matrix operations, advanced sorting
- **Time Complexity**: O(n log n) to O(2^n)
- **Space Complexity**: O(n) to O(2^n)

### Very Advanced Level (40+ algorithms)
- **Cutting-Edge Techniques**: Advanced optimization, complex graph algorithms
- **Research-Level Problems**: NP-hard problems, advanced combinatorics
- **Specialized Domains**: Computational geometry, advanced string processing
- **Time Complexity**: O(n log n) to O(n!)
- **Space Complexity**: O(n) to O(n!)

## ⏱️ Time & Space Complexity Analysis

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

### Time Complexity Ranges by Difficulty
- **Easy**: O(1) to O(n²) - Linear and quadratic time algorithms
- **Medium**: O(n log n) to O(n³) - Sorting-based and cubic algorithms
- **Advanced**: O(n log n) to O(2^n) - Exponential time algorithms
- **Very Advanced**: O(n log n) to O(n!) - Factorial time algorithms

### Space Complexity Considerations
- **Easy**: O(1) to O(n) - Mostly constant and linear space
- **Medium**: O(n) to O(n²) - Linear and quadratic space
- **Advanced**: O(n) to O(2^n) - Exponential space for combinatorial results
- **Very Advanced**: O(n) to O(n!) - Factorial space for complex problems

## 🔧 Customization

### Adding New Algorithms
1. Choose the appropriate file based on difficulty level
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