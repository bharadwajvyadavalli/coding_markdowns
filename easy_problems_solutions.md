# Coding Problems Solutions

## Two Number Sum

### 🧠 Problem Summary
Given an array of integers and a target sum, find two numbers that add up to the target. Return the indices of these numbers.

**Constraints:**
- Each input has exactly one solution
- Cannot use the same element twice
- Return indices in any order

**Examples:**
- `[2, 7, 11, 15], target = 9` → `[0, 1]` (2 + 7 = 9)
- `[3, 2, 4], target = 6` → `[1, 2]` (2 + 4 = 6)

### ⚡ Optimal Strategy
Use a hash map to store complements. For each number, check if its complement (target - current) exists in the map. If found, return both indices.

**Time Complexity:** O(n) - single pass through array
**Space Complexity:** O(n) - hash map storage

### 🔍 Pattern/Technique
**Hashing** - Using a hash map for O(1) lookups to find complements efficiently.

### ✅ Clean Code (Python)
```python
def twoNumberSum(array, targetSum):
    seen = {}
    for i, num in enumerate(array):
        complement = targetSum - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

---

## Validate Subsequence

### 🧠 Problem Summary
Given two arrays, determine if the second array is a subsequence of the first. A subsequence means elements appear in the same order but not necessarily consecutively.

**Constraints:**
- Both arrays are non-empty
- Subsequence elements must appear in order

**Examples:**
- `[5, 1, 22, 25, 6, -1, 8, 10], [1, 6, -1, 10]` → `True`
- `[5, 1, 22, 25, 6, -1, 8, 10], [22, 25, 6]` → `True`
- `[5, 1, 22, 25, 6, -1, 8, 10], [1, 6, -1, 11]` → `False`

### ⚡ Optimal Strategy
Use two pointers: one for the main array, one for the subsequence. Traverse the main array and advance the subsequence pointer when matches are found.

**Time Complexity:** O(n) - single pass through main array
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Two Pointers** - Using pointers to track progress through both arrays simultaneously.

### ✅ Clean Code (Python)
```python
def isValidSubsequence(array, sequence):
    seq_idx = 0
    for value in array:
        if seq_idx == len(sequence):
            break
        if sequence[seq_idx] == value:
            seq_idx += 1
    return seq_idx == len(sequence)
```

---

## Sorted Squared Array

### 🧠 Problem Summary
Given a sorted array of integers, return a new array with all values squared and sorted in ascending order.

**Constraints:**
- Input array is sorted in ascending order
- Can contain negative numbers

**Examples:**
- `[1, 2, 3, 5, 6, 8, 9]` → `[1, 4, 9, 25, 36, 64, 81]`
- `[-2, -1, 0, 1, 2]` → `[0, 1, 1, 4, 4]`

### ⚡ Optimal Strategy
Use two pointers from both ends since the largest squares will come from the most negative or most positive numbers. Compare squares and fill result array from right to left.

**Time Complexity:** O(n) - single pass through array
**Space Complexity:** O(n) - new result array

### 🔍 Pattern/Technique
**Two Pointers** - Using pointers from both ends to handle negative numbers efficiently.

### ✅ Clean Code (Python)
```python
def sortedSquaredArray(array):
    result = [0] * len(array)
    left, right = 0, len(array) - 1
    
    for i in range(len(array) - 1, -1, -1):
        if abs(array[left]) > abs(array[right]):
            result[i] = array[left] ** 2
            left += 1
        else:
            result[i] = array[right] ** 2
            right -= 1
    
    return result
```

---

## Tournament Winner

### 🧠 Problem Summary
Given competitions between teams and results, determine the tournament winner. Each competition has two teams and a result (0 = away team won, 1 = home team won). Award 3 points for each win.

**Constraints:**
- At least one competition
- No ties in competitions
- Team names are strings

**Examples:**
- `[["HTML", "C#"], ["C#", "Python"], ["Python", "HTML"]], [0, 0, 1]` → `"Python"`

### ⚡ Optimal Strategy
Use a hash map to track points for each team. Iterate through competitions and results, awarding 3 points to the winning team.

**Time Complexity:** O(n) - single pass through competitions
**Space Complexity:** O(k) - where k is number of teams

### 🔍 Pattern/Technique
**Hashing** - Using a hash map to efficiently track and update team scores.

### ✅ Clean Code (Python)
```python
def tournamentWinner(competitions, results):
    scores = {}
    best_team = ""
    best_score = 0
    
    for i, (home, away) in enumerate(competitions):
        winner = home if results[i] == 1 else away
        scores[winner] = scores.get(winner, 0) + 3
        
        if scores[winner] > best_score:
            best_score = scores[winner]
            best_team = winner
    
    return best_team
```

---

## Non-Constructible Change

### 🧠 Problem Summary
Given an array of positive integers representing coin denominations, find the minimum amount of change that cannot be created using any combination of the coins.

**Constraints:**
- All coins are positive integers
- Can use any number of each coin

**Examples:**
- `[1, 2, 5]` → `4` (can make 1, 2, 3, 5, 6, 7, 8... but not 4)
- `[5, 7, 1, 1, 2, 3, 22]` → `20`

### ⚡ Optimal Strategy
Sort the array first. For each coin, if it's greater than the current change + 1, then current change + 1 is the answer. Otherwise, add the coin to the current change.

**Time Complexity:** O(n log n) - sorting dominates
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Greedy Algorithm** - Making locally optimal choices (using smallest coins first) to find the global solution.

### ✅ Clean Code (Python)
```python
def nonConstructibleChange(coins):
    coins.sort()
    current_change = 0
    
    for coin in coins:
        if coin > current_change + 1:
            return current_change + 1
        current_change += coin
    
    return current_change + 1
```

---

## Transpose Matrix

### 🧠 Problem Summary
Given a 2D matrix, return its transpose. The transpose of a matrix is formed by turning rows into columns and columns into rows.

**Constraints:**
- Matrix is not empty
- All rows have the same length

**Examples:**
- `[[1, 2, 3], [4, 5, 6]]` → `[[1, 4], [2, 5], [3, 6]]`
- `[[1, 2], [3, 4], [5, 6]]` → `[[1, 3, 5], [2, 4, 6]]`

### ⚡ Optimal Strategy
Create a new matrix with dimensions swapped (rows become columns). Copy elements from original matrix to new positions.

**Time Complexity:** O(m*n) - must visit each element once
**Space Complexity:** O(m*n) - new matrix storage

### 🔍 Pattern/Technique
**Matrix Manipulation** - Swapping dimensions and reorganizing elements systematically.

### ✅ Clean Code (Python)
```python
def transposeMatrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    result = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    
    return result
```

---

## Find Closest Value In BST

### 🧠 Problem Summary
Given a BST and a target value, find the value in the BST that is closest to the target.

**Constraints:**
- BST is valid
- If two values are equally close, return the smaller one

**Examples:**
- Target: `12`, BST with values `10, 15, 22, 13, 14, 5, 2, 1` → `13`

### ⚡ Optimal Strategy
Use BST property: if target < current node, go left; if target > current node, go right. Keep track of closest value found so far.

**Time Complexity:** O(h) - where h is height of tree (O(log n) for balanced BST)
**Space Complexity:** O(h) - recursion stack space

### 🔍 Pattern/Technique
**Tree Traversal** - Using BST properties to eliminate half the tree at each step.

### ✅ Clean Code (Python)
```python
def findClosestValueInBst(tree, target):
    def helper(node, target, closest):
        if node is None:
            return closest
        
        if abs(target - closest) > abs(target - node.value):
            closest = node.value
        
        if target < node.value:
            return helper(node.left, target, closest)
        elif target > node.value:
            return helper(node.right, target, closest)
        else:
            return closest
    
    return helper(tree, target, tree.value)
```

---

## Branch Sums

### 🧠 Problem Summary
Given a binary tree, return an array of sums of all values along each path from root to leaf.

**Constraints:**
- Tree is valid
- Return sums in order from leftmost to rightmost leaf

**Examples:**
- Tree with paths: `1->2->4`, `1->2->5`, `1->3->6` → `[7, 8, 10]`

### ⚡ Optimal Strategy
Use DFS to traverse all paths from root to leaf. At each node, add current value to running sum. When reaching a leaf, add sum to result array.

**Time Complexity:** O(n) - visit each node once
**Space Complexity:** O(h) - recursion stack space, where h is height

### 🔍 Pattern/Technique
**Depth-First Search (DFS)** - Exploring complete paths from root to leaf before backtracking.

### ✅ Clean Code (Python)
```python
def branchSums(root):
    sums = []
    
    def calculateBranchSums(node, runningSum, sums):
        if node is None:
            return
        
        newRunningSum = runningSum + node.value
        
        if node.left is None and node.right is None:
            sums.append(newRunningSum)
            return
        
        calculateBranchSums(node.left, newRunningSum, sums)
        calculateBranchSums(node.right, newRunningSum, sums)
    
    calculateBranchSums(root, 0, sums)
    return sums
```

---

## Node Depths

### 🧠 Problem Summary
Given a binary tree, calculate the sum of all node depths. The depth of a node is the distance from the root to that node.

**Constraints:**
- Tree is valid
- Root has depth 0

**Examples:**
- Tree with 3 levels: root (depth 0), 2 children (depth 1), 4 grandchildren (depth 2) → `0 + 1 + 1 + 2 + 2 + 2 + 2 = 10`

### ⚡ Optimal Strategy
Use DFS to traverse the tree, passing current depth to each recursive call. Sum up all depths.

**Time Complexity:** O(n) - visit each node once
**Space Complexity:** O(h) - recursion stack space

### 🔍 Pattern/Technique
**Depth-First Search (DFS)** - Traversing tree and accumulating depth information.

### ✅ Clean Code (Python)
```python
def nodeDepths(root):
    def calculateNodeDepths(node, depth):
        if node is None:
            return 0
        
        return depth + calculateNodeDepths(node.left, depth + 1) + calculateNodeDepths(node.right, depth + 1)
    
    return calculateNodeDepths(root, 0)
```

---

## Evaluate Expression Tree

### 🧠 Problem Summary
Given a binary tree representing a mathematical expression, evaluate the result. Leaf nodes contain numbers, internal nodes contain operators.

**Constraints:**
- Valid expression tree
- Operators: +, -, *, /
- Division is integer division

**Examples:**
- Tree: `*` with children `+` (left) and `2` (right), where `+` has children `3` and `4` → `(3+4)*2 = 14`

### ⚡ Optimal Strategy
Use post-order traversal (left, right, root) to evaluate expressions. When reaching an operator node, apply the operator to results from left and right subtrees.

**Time Complexity:** O(n) - visit each node once
**Space Complexity:** O(h) - recursion stack space

### 🔍 Pattern/Technique
**Tree Traversal (Post-order)** - Evaluating children before applying operators at parent nodes.

### ✅ Clean Code (Python)
```python
def evaluateExpressionTree(tree):
    if tree.value >= 0:
        return tree.value
    
    leftValue = evaluateExpressionTree(tree.left)
    rightValue = evaluateExpressionTree(tree.right)
    
    if tree.value == -1:
        return leftValue + rightValue
    elif tree.value == -2:
        return leftValue - rightValue
    elif tree.value == -3:
        return leftValue // rightValue
    else:
        return leftValue * rightValue
```

---

## Depth-first Search

### 🧠 Problem Summary
Given a graph represented as an adjacency list and a starting node, return an array of nodes in depth-first search order.

**Constraints:**
- Graph is connected
- Nodes are strings
- Return nodes in order they're visited

**Examples:**
- Graph: `A->B,C`, `B->D,E`, `C->F`, starting at `A` → `["A", "B", "D", "E", "C", "F"]`

### ⚡ Optimal Strategy
Use a stack (or recursion) to explore as far as possible along each branch before backtracking. Keep track of visited nodes to avoid cycles.

**Time Complexity:** O(V + E) - visit each vertex and edge once
**Space Complexity:** O(V) - visited set and recursion stack

### 🔍 Pattern/Technique
**Depth-First Search (DFS)** - Exploring deep into graph before backtracking.

### ✅ Clean Code (Python)
```python
def depthFirstSearch(array, graph):
    def dfs(node):
        array.append(node)
        for neighbor in graph[node]:
            if neighbor not in array:
                dfs(neighbor)
    
    dfs(array[0])
    return array
```

---

## Minimum Waiting Time

### 🧠 Problem Summary
Given an array of query execution times, find the minimum total waiting time. Waiting time for a query is the sum of execution times of all queries that come before it.

**Constraints:**
- All execution times are positive
- Queries can be executed in any order

**Examples:**
- `[3, 2, 1, 2, 6]` → `17` (optimal order: `[1, 2, 2, 3, 6]`)

### ⚡ Optimal Strategy
Sort the array in ascending order. The optimal strategy is to execute shorter queries first, minimizing waiting time for longer queries.

**Time Complexity:** O(n log n) - sorting dominates
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Greedy Algorithm** - Always choosing the shortest remaining query to minimize total waiting time.

### ✅ Clean Code (Python)
```python
def minimumWaitingTime(queries):
    queries.sort()
    totalWaitingTime = 0
    
    for i, duration in enumerate(queries):
        queriesLeft = len(queries) - (i + 1)
        totalWaitingTime += duration * queriesLeft
    
    return totalWaitingTime
```

---

## Class Photos

### 🧠 Problem Summary
Arrange students in two rows for a class photo. Each student has a height. The student in the back row must be taller than the student directly in front of them.

**Constraints:**
- Same number of students in each row
- All students must be in exactly one row

**Examples:**
- Red shirts: `[5, 8, 1, 3, 4]`, Blue shirts: `[6, 9, 2, 4, 5]` → `True` (can arrange)

### ⚡ Optimal Strategy
Sort both arrays. Put the tallest student from one team in the back row, then alternate placing students ensuring back row is always taller.

**Time Complexity:** O(n log n) - sorting dominates
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Greedy Algorithm** - Making optimal local choices (tallest students in back) to achieve global solution.

### ✅ Clean Code (Python)
```python
def classPhotos(redShirtHeights, blueShirtHeights):
    redShirtHeights.sort(reverse=True)
    blueShirtHeights.sort(reverse=True)
    
    firstRowColor = "RED" if redShirtHeights[0] < blueShirtHeights[0] else "BLUE"
    
    for i in range(len(redShirtHeights)):
        redHeight = redShirtHeights[i]
        blueHeight = blueShirtHeights[i]
        
        if firstRowColor == "RED":
            if redHeight >= blueHeight:
                return False
        else:
            if blueHeight >= redHeight:
                return False
    
    return True
```

---

## Tandem Bicycle

### 🧠 Problem Summary
Given arrays of speeds for red and blue shirt cyclists, pair them up for tandem bicycles. Each tandem has one person from each team. Find the maximum or minimum total speed.

**Constraints:**
- Same number of cyclists in each team
- Each cyclist can only be used once

**Examples:**
- Red: `[5, 5, 3, 9, 2]`, Blue: `[3, 6, 7, 2, 1]`, fastest: `True` → `32` (max total)

### ⚡ Optimal Strategy
For maximum speed: sort one array ascending, other descending, then pair. For minimum speed: sort both ascending and pair.

**Time Complexity:** O(n log n) - sorting dominates
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Greedy Algorithm** - Optimal pairing strategy based on whether we want maximum or minimum total.

### ✅ Clean Code (Python)
```python
def tandemBicycle(redShirtSpeeds, blueShirtSpeeds, fastest):
    redShirtSpeeds.sort()
    blueShirtSpeeds.sort()
    
    if fastest:
        blueShirtSpeeds.reverse()
    
    totalSpeed = 0
    for i in range(len(redShirtSpeeds)):
        totalSpeed += max(redShirtSpeeds[i], blueShirtSpeeds[i])
    
    return totalSpeed
```

---

## Optimal Freelancing

### 🧠 Problem Summary
Given a list of jobs with deadlines and payments, find the maximum profit you can earn. You can work on at most one job per day.

**Constraints:**
- Each job takes exactly one day
- Must complete job by its deadline
- Can work on at most one job per day

**Examples:**
- Jobs: `[{"deadline": 2, "payment": 100}, {"deadline": 1, "payment": 19}, {"deadline": 2, "payment": 27}]` → `127`

### ⚡ Optimal Strategy
Sort jobs by payment in descending order. For each job, try to schedule it on the latest available day before its deadline.

**Time Complexity:** O(n log n) - sorting dominates
**Space Complexity:** O(n) - to track scheduled days

### 🔍 Pattern/Technique
**Greedy Algorithm** - Always choosing the highest paying job that can still be scheduled.

### ✅ Clean Code (Python)
```python
def optimalFreelancing(jobs):
    jobs.sort(key=lambda x: x["payment"], reverse=True)
    timeline = [False] * 7
    
    profit = 0
    for job in jobs:
        deadline = min(job["deadline"], 7)
        for i in range(deadline - 1, -1, -1):
            if not timeline[i]:
                timeline[i] = True
                profit += job["payment"]
                break
    
    return profit
```

---

## Remove Duplicates From Linked List

### 🧠 Problem Summary
Given a sorted linked list, remove all duplicate nodes, keeping only one instance of each value.

**Constraints:**
- Linked list is sorted
- Return the head of the modified list

**Examples:**
- `1 -> 1 -> 2 -> 3 -> 3` → `1 -> 2 -> 3`

### ⚡ Optimal Strategy
Use two pointers: current and next. If current value equals next value, skip the next node by updating current.next.

**Time Complexity:** O(n) - single pass through list
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Two Pointers** - Using pointers to traverse and modify the linked list in-place.

### ✅ Clean Code (Python)
```python
def removeDuplicatesFromLinkedList(linkedList):
    current = linkedList
    
    while current is not None and current.next is not None:
        if current.value == current.next.value:
            current.next = current.next.next
        else:
            current = current.next
    
    return linkedList
```

---

## Middle Node

### 🧠 Problem Summary
Given a linked list, return the middle node. If there are two middle nodes, return the second one.

**Constraints:**
- Linked list has at least one node
- Return the actual node, not just the value

**Examples:**
- `1 -> 2 -> 3 -> 4 -> 5` → node with value `3`
- `1 -> 2 -> 3 -> 4` → node with value `3`

### ⚡ Optimal Strategy
Use two pointers: slow and fast. Fast pointer moves twice as fast as slow pointer. When fast reaches the end, slow will be at the middle.

**Time Complexity:** O(n) - single pass through list
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Two Pointers (Fast/Slow)** - Using different speeds to find the middle efficiently.

### ✅ Clean Code (Python)
```python
def middleNode(linkedList):
    slow = linkedList
    fast = linkedList
    
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

---

## Nth Fibonacci

### 🧠 Problem Summary
Calculate the nth Fibonacci number. F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2).

**Constraints:**
- n is a non-negative integer
- Handle large values efficiently

**Examples:**
- `n = 6` → `8` (sequence: 0, 1, 1, 2, 3, 5, 8)

### ⚡ Optimal Strategy
Use iterative approach with two variables to track previous two numbers. Avoid recursion to prevent stack overflow.

**Time Complexity:** O(n) - linear time
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Dynamic Programming (Iterative)** - Building solution bottom-up to avoid redundant calculations.

### ✅ Clean Code (Python)
```python
def getNthFib(n):
    if n <= 1:
        return 0
    elif n == 2:
        return 1
    
    prev, curr = 0, 1
    for i in range(3, n + 1):
        prev, curr = curr, prev + curr
    
    return curr
```

---

## Product Sum

### 🧠 Problem Summary
Calculate the product sum of a special array. A special array contains integers and/or other special arrays. The product sum is the sum of all elements multiplied by their depth.

**Constraints:**
- Arrays can be nested
- Depth starts at 1

**Examples:**
- `[5, 2, [7, -1], 3, [6, [-13, 8], 4]]` → `12` (5*1 + 2*1 + (7-1)*2 + 3*1 + (6+(-13+8)*3+4)*2)

### ⚡ Optimal Strategy
Use recursion with a depth parameter. For each element, if it's a list, recursively calculate its product sum with increased depth.

**Time Complexity:** O(n) - where n is total number of elements
**Space Complexity:** O(d) - where d is maximum depth

### 🔍 Pattern/Technique
**Recursion** - Handling nested structures by recursively processing sub-arrays.

### ✅ Clean Code (Python)
```python
def productSum(array, depth=1):
    total = 0
    
    for element in array:
        if isinstance(element, list):
            total += productSum(element, depth + 1)
        else:
            total += element
    
    return total * depth
```

---

## Binary Search

### 🧠 Problem Summary
Given a sorted array and a target value, return the index of the target if found, otherwise return -1.

**Constraints:**
- Array is sorted in ascending order
- All elements are unique

**Examples:**
- `[0, 1, 21, 33, 45, 45, 61, 71, 72, 73], 33` → `3`
- `[1, 5, 23, 111], 35` → `-1`

### ⚡ Optimal Strategy
Use binary search: compare target with middle element, eliminate half the array based on comparison, repeat until found or array is empty.

**Time Complexity:** O(log n) - halves search space each iteration
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Binary Search** - Dividing search space in half at each step for logarithmic time complexity.

### ✅ Clean Code (Python)
```python
def binarySearch(array, target):
    left, right = 0, len(array) - 1
    
    while left <= right:
        middle = (left + right) // 2
        
        if array[middle] == target:
            return middle
        elif array[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    
    return -1
```

---

## Find Three Largest Numbers

### 🧠 Problem Summary
Given an array of integers, return the three largest numbers in sorted order.

**Constraints:**
- Array has at least 3 integers
- Return in ascending order

**Examples:**
- `[141, 1, 17, -7, -17, -27, 18, 541, 8, 7, 7]` → `[18, 141, 541]`

### ⚡ Optimal Strategy
Use three variables to track the three largest numbers. For each element, update the three largest if necessary.

**Time Complexity:** O(n) - single pass through array
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Linear Scan** - Maintaining running maximums while traversing the array once.

### ✅ Clean Code (Python)
```python
def findThreeLargestNumbers(array):
    threeLargest = [float('-inf'), float('-inf'), float('-inf')]
    
    for num in array:
        updateLargest(threeLargest, num)
    
    return threeLargest

def updateLargest(threeLargest, num):
    if num > threeLargest[2]:
        shiftAndUpdate(threeLargest, num, 2)
    elif num > threeLargest[1]:
        shiftAndUpdate(threeLargest, num, 1)
    elif num > threeLargest[0]:
        shiftAndUpdate(threeLargest, num, 0)

def shiftAndUpdate(array, num, idx):
    for i in range(idx + 1):
        if i == idx:
            array[i] = num
        else:
            array[i] = array[i + 1]
```

---

## Bubble Sort

### 🧠 Problem Summary
Sort an array using the bubble sort algorithm. Repeatedly step through the list, compare adjacent elements, and swap them if they are in the wrong order.

**Constraints:**
- Sort in ascending order
- Modify array in-place

**Examples:**
- `[8, 5, 2, 9, 5, 6, 3]` → `[2, 3, 5, 5, 6, 8, 9]`

### ⚡ Optimal Strategy
Use nested loops. Outer loop tracks sorted portion, inner loop compares adjacent elements and swaps if needed.

**Time Complexity:** O(n²) - nested loops
**Space Complexity:** O(1) - in-place sorting

### 🔍 Pattern/Technique
**Bubble Sort** - Repeatedly swapping adjacent elements to "bubble" larger elements to the end.

### ✅ Clean Code (Python)
```python
def bubbleSort(array):
    isSorted = False
    counter = 0
    
    while not isSorted:
        isSorted = True
        for i in range(len(array) - 1 - counter):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                isSorted = False
        counter += 1
    
    return array
```

---

## Insertion Sort

### 🧠 Problem Summary
Sort an array using insertion sort. Build the final sorted array one item at a time by repeatedly inserting a new element into the sorted portion.

**Constraints:**
- Sort in ascending order
- Modify array in-place

**Examples:**
- `[8, 5, 2, 9, 5, 6, 3]` → `[2, 3, 5, 5, 6, 8, 9]`

### ⚡ Optimal Strategy
For each element starting from index 1, insert it into the correct position in the sorted portion to its left.

**Time Complexity:** O(n²) - worst case, but O(n) for nearly sorted arrays
**Space Complexity:** O(1) - in-place sorting

### 🔍 Pattern/Technique
**Insertion Sort** - Building sorted array incrementally by inserting each element in its correct position.

### ✅ Clean Code (Python)
```python
def insertionSort(array):
    for i in range(1, len(array)):
        j = i
        while j > 0 and array[j] < array[j - 1]:
            array[j], array[j - 1] = array[j - 1], array[j]
            j -= 1
    
    return array
```

---

## Selection Sort

### 🧠 Problem Summary
Sort an array using selection sort. Find the minimum element in the unsorted portion and swap it with the first element of the unsorted portion.

**Constraints:**
- Sort in ascending order
- Modify array in-place

**Examples:**
- `[8, 5, 2, 9, 5, 6, 3]` → `[2, 3, 5, 5, 6, 8, 9]`

### ⚡ Optimal Strategy
Use nested loops. Outer loop tracks sorted portion, inner loop finds minimum in unsorted portion and swaps.

**Time Complexity:** O(n²) - always quadratic regardless of input
**Space Complexity:** O(1) - in-place sorting

### 🔍 Pattern/Technique
**Selection Sort** - Repeatedly selecting the minimum element and placing it in the correct position.

### ✅ Clean Code (Python)
```python
def selectionSort(array):
    for i in range(len(array)):
        minIdx = i
        for j in range(i + 1, len(array)):
            if array[j] < array[minIdx]:
                minIdx = j
        
        if minIdx != i:
            array[i], array[minIdx] = array[minIdx], array[i]
    
    return array
```

---

## Palindrome Check

### 🧠 Problem Summary
Given a string, determine if it's a palindrome (reads the same forwards and backwards).

**Constraints:**
- String can contain letters, numbers, and special characters
- Case-insensitive
- Ignore non-alphanumeric characters

**Examples:**
- `"race a car"` → `False`
- `"A man, a plan, a canal: Panama"` → `True`

### ⚡ Optimal Strategy
Use two pointers from both ends. Skip non-alphanumeric characters and compare characters (case-insensitive).

**Time Complexity:** O(n) - single pass through string
**Space Complexity:** O(1) - constant extra space

### 🔍 Pattern/Technique
**Two Pointers** - Using pointers from both ends to compare characters efficiently.

### ✅ Clean Code (Python)
```python
def isPalindrome(string):
    left, right = 0, len(string) - 1
    
    while left < right:
        while left < right and not string[left].isalnum():
            left += 1
        while left < right and not string[right].isalnum():
            right -= 1
        
        if string[left].lower() != string[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True
```

---

## Caesar Cipher Encryptor

### 🧠 Problem Summary
Given a string and a key, shift each letter by the key positions in the alphabet. Wrap around if necessary.

**Constraints:**
- String contains only lowercase letters
- Key is a non-negative integer
- Wrap around alphabet (z -> a)

**Examples:**
- `"xyz", 2` → `"zab"`
- `"abc", 52` → `"abc"` (52 % 26 = 0)

### ⚡ Optimal Strategy
For each character, calculate new position using modulo arithmetic to handle wrap-around. Convert to new character.

**Time Complexity:** O(n) - single pass through string
**Space Complexity:** O(n) - new string storage

### 🔍 Pattern/Technique
**String Manipulation** - Using modulo arithmetic to handle circular shifts.

### ✅ Clean Code (Python)
```python
def caesarCipherEncryptor(string, key):
    newLetters = []
    newKey = key % 26
    
    for letter in string:
        newLetterCode = ord(letter) + newKey
        if newLetterCode <= 122:
            newLetters.append(chr(newLetterCode))
        else:
            newLetters.append(chr(96 + newLetterCode % 122))
    
    return "".join(newLetters)
```

---

## Run-Length Encoding

### 🧠 Problem Summary
Given a string, compress it using run-length encoding. Replace consecutive identical characters with the character followed by its count.

**Constraints:**
- String is non-empty
- Counts greater than 9 should be split into multiple digits

**Examples:**
- `"AAAAAAAAAAAAABBCCCCDD"` → `"9A4A2B4C2D"`
- `"a"` → `"1a"`

### ⚡ Optimal Strategy
Use two pointers to track current character and count consecutive occurrences. Build result string as we go.

**Time Complexity:** O(n) - single pass through string
**Space Complexity:** O(n) - result string storage

### 🔍 Pattern/Technique
**String Manipulation** - Using pointers to count consecutive characters and build compressed string.

### ✅ Clean Code (Python)
```python
def runLengthEncoding(string):
    encodedStringCharacters = []
    currentRunLength = 1
    
    for i in range(1, len(string)):
        currentCharacter = string[i]
        previousCharacter = string[i - 1]
        
        if currentCharacter != previousCharacter or currentRunLength == 9:
            encodedStringCharacters.append(str(currentRunLength))
            encodedStringCharacters.append(previousCharacter)
            currentRunLength = 0
        
        currentRunLength += 1
    
    encodedStringCharacters.append(str(currentRunLength))
    encodedStringCharacters.append(string[len(string) - 1])
    
    return "".join(encodedStringCharacters)
```

---

## Common Characters

### 🧠 Problem Summary
Given an array of strings, find the common characters that appear in all strings (with multiplicity).

**Constraints:**
- All strings contain only lowercase letters
- Return characters in alphabetical order

**Examples:**
- `["bella", "label", "roller"]` → `["e", "l", "l"]`
- `["cool", "lock", "cook"]` → `["c", "o"]`

### ⚡ Optimal Strategy
Count character frequencies in each string. For each character, find the minimum count across all strings and add that many to the result.

**Time Complexity:** O(n*m) - where n is number of strings, m is average string length
**Space Complexity:** O(k) - where k is number of unique characters

### 🔍 Pattern/Technique
**Character Counting** - Using frequency arrays to track character counts across multiple strings.

### ✅ Clean Code (Python)
```python
def commonCharacters(strings):
    characterCounts = {}
    for string in strings:
        uniqueStringCharacters = set(string)
        for character in uniqueStringCharacters:
            if character not in characterCounts:
                characterCounts[character] = 0
            characterCounts[character] += 1
    
    finalCharacters = []
    for character, count in characterCounts.items():
        if count == len(strings):
            finalCharacters.append(character)
    
    return finalCharacters
```

---

## Generate Document

### 🧠 Problem Summary
Given a string of characters and a document string, determine if you can generate the document using the available characters.

**Constraints:**
- Can only use each character once
- Case-sensitive

**Examples:**
- `"Bste!hetsi ogEAxpelrt x ", "AlgoExpert is the Best!"` → `True`
- `"A", "a"` → `False`

### ⚡ Optimal Strategy
Count character frequencies in both strings. Check if document can be created by comparing frequencies.

**Time Complexity:** O(n + m) - where n and m are lengths of strings
**Space Complexity:** O(k) - where k is number of unique characters

### 🔍 Pattern/Technique
**Character Counting** - Using hash maps to track character frequencies and compare availability.

### ✅ Clean Code (Python)
```python
def generateDocument(characters, document):
    characterCounts = {}
    
    for character in characters:
        if character not in characterCounts:
            characterCounts[character] = 0
        characterCounts[character] += 1
    
    for character in document:
        if character not in characterCounts or characterCounts[character] == 0:
            return False
        characterCounts[character] -= 1
    
    return True
```

---

## First Non-Repeating Character

### 🧠 Problem Summary
Given a string, find the first character that appears only once.

**Constraints:**
- String contains only lowercase letters
- Return -1 if no such character exists

**Examples:**
- `"abcdcaf"` → `1` (index of 'b')
- `"faadabcbbebdf"` → `6` (index of 'c')

### ⚡ Optimal Strategy
Count character frequencies in first pass, then find first character with frequency 1 in second pass.

**Time Complexity:** O(n) - two passes through string
**Space Complexity:** O(1) - fixed size character count array

### 🔍 Pattern/Technique
**Character Counting** - Using frequency tracking to identify unique characters.

### ✅ Clean Code (Python)
```python
def firstNonRepeatingCharacter(string):
    characterFrequencies = {}
    
    for character in string:
        characterFrequencies[character] = characterFrequencies.get(character, 0) + 1
    
    for idx in range(len(string)):
        character = string[idx]
        if characterFrequencies[character] == 1:
            return idx
    
    return -1
```

---

## Semordnilap

### 🧠 Problem Summary
Given an array of unique strings, find all pairs of strings that are semordnilaps (palindromes when read backwards).

**Constraints:**
- All strings are unique
- Return pairs in any order
- Don't include duplicates

**Examples:**
- `["diaper", "abc", "test", "cba", "repaid"]` → `[["diaper", "repaid"], ["abc", "cba"]]`

### ⚡ Optimal Strategy
Use a hash set to store seen strings. For each string, check if its reverse exists in the set.

**Time Complexity:** O(n*m) - where n is number of strings, m is average string length
**Space Complexity:** O(n*m) - hash set storage

### 🔍 Pattern/Technique
**Hashing** - Using a hash set for efficient lookups of reversed strings.

### ✅ Clean Code (Python)
```python
def semordnilap(words):
    wordSet = set(words)
    semordnilapPairs = []
    
    for word in words:
        reverse = word[::-1]
        if reverse in wordSet and reverse != word:
            semordnilapPairs.append([word, reverse])
            wordSet.remove(word)
            wordSet.remove(reverse)
    
    return semordnilapPairs
``` 