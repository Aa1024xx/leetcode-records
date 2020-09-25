## Two Sum ##
#### 1. Two Sum ####
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

###### 思路 ######
```
HashMap: num -> sum - num 
HashMap: key: sum, value: index
```
###### 代码 ######
```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0){
            return new int[2];
        }
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++){
            if (map.containsKey(target - nums[i])){
                return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i], i);
        }
        return new int[2];
    }
}
```
###### 复杂度 ######
```
Time complexity: O(n)
Space complexity: O(n)
```

###### 思路 ######
```
Brute Force
```
###### 代码 ######
```
public int[] twoSum(int[] nums, int target) {
    for (int i = 0; i < nums.length; i++) {
        for (int j = i + 1; j < nums.length; j++) {
            if (nums[j] == target - nums[i]) {
                return new int[] { i, j };
            }
        }
    }
    throw new IllegalArgumentException("No two sum solution");
}
```
###### 复杂度 ######
```
Time complexity: O(n^2)
Space complexity: O(1)
```
#### Two Sum II - Input array is sorted ####
Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

###### 思路 ######
```
Two pointers
TC: O(n), SC: O(1)
```

###### 代码 ######
```
public class Solution {
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length <= 1){
            return new int[2];
        }
        int left = 0, right = nums.length - 1;
        while (left < right){
            if (nums[left] + nums[right] == target){
                return new int[]{left + 1, right + 1};
            }else if (nums[left] + nums[right] < target){
                left++;
            }else {
                right--;
            }
        }
        return new int[2];
    }
}
```

#### Two Sum III - Data structure design ####
Design and implement a TwoSum class. It should support the following operations: add and find.

add - Add the number to an internal data structure.
find - Find if there exists any pair of numbers which sum is equal to the value.

###### Example ######
```
add(1); add(3); add(5);
find(4) // return true
find(7) // return false
```
###### 分析 ######
```
使用哈希表的方法是最快的: add: O(1), find: O(n)

HashMap: key: number, value: 次数
list: 记录 unique number
考虑两数是否相同
```

###### 代码 ######
```
public class TwoSum {
    private List<Integer> list = new ArrayList<>();
    private Map<Integer, Integer> map = new HashMap<>();

    public void add(int number) {
        if (map.containsKey(number)){
            map.put(number, map.get(number) + 1);
        }else {
            map.put(number, 1);
            list.add(number);
        }
    }

    public boolean find(int value) {
        for (int i = 0; i < list.size(); i++){
            int num1 = list.get(i);
            int num2 = value - num1;
            if ((num1 != num2 && map.containsKey(num2)) || (num1 == num2 && map.get(num1) > 1)){
                return true;
            }
        }
        return false;
    }
}
```

#### Two Sum IV - Input is a BST ####
Given a binary search tree and a number n, find two numbers in the tree that sums up to n.

###### Example ######
```
Example 1
Input: 
{4,2,5,1,3}
3
Output: [1,2] (or [2,1])
Explanation:
binary search tree:
    4
   / \
  2   5
 / \
1   3

Example 2
Input: 
{4,2,5,1,3}
5
Output: [2,3] (or [3,2])
```
###### 分析 ######
```
Inorder遍历， 再双指针 -> TC: O(n), SC: O(n)
```
###### 代码 ######
```
public class Solution {
    public int[] twoSum(TreeNode root, int n) {
        if (root == null){
            return null;
        }
        
        List<Integer> inorder = new ArrayList<>();
        inorderTraverse(root, inorder);
        
        int left = 0, right = inorder.size() - 1;
        while (left < right){
            if (inorder.get(left) + inorder.get(right) < n){
                left++;
            }else if (inorder.get(left) + inorder.get(right) > n){
                right--;
            }else {
                return new int[]{inorder.get(left), inorder.get(right)};
            }
        }
        
        return null;
    }
    
    private void inorderTraverse(TreeNode root, List<Integer> path){
        if (root == null){
            return;
        }
        inorderTraverse(root.left, path);
        path.add(root.val);
        inorderTraverse(root.right, path);
    }
}
```

#### Two Sum - Difference equals to target ####
Given an sorted array of integers, find two numbers that their difference equals to a target value.
return a list with two number like [num1, num2] that the difference of num1 and num2 equals to target value, and num1 is less thannum2.

###### Example ######
```
Example 1
Input: nums = [2, 7, 15, 24], target = 5 
Output: [2, 7] 
Explanation:
(7 - 2 = 5)

Example 2
Input: nums = [1, 1], target = 0
Output: [1, 1] 
Explanation:
(1 - 1 = 0)
```

###### 分析 ######
```
由于数组有序，可以用双指针来做
对于双指针i,j,当num[j]-num[i]< target时，说明j太小，于是我们将j++，直到num[j]-num[i] >= target，若num[j]-num[i] > target，我们将i++，若num[j]-num[i] = target说明我们找到答案
TC: O(n), SC: O(1)
```

###### 代码 ######
```
public class Solution {
    public int[] twoSum7(int[] nums, int target) {
        if (nums == null || nums.length < 2){
            return new int[2];
        }
        
        target = Math.abs(target);
        int j = 1;
        for (int i = 0; i < nums.length; i++){
            j = Math.max(i + 1, j);
            while (j < nums.length && nums[j] - nums[i] < target){
                j++;
            }
            if (j >= nums.length){
                break;
            }
            if (nums[j] - nums[i] == target){
                return new int[]{nums[i], nums[j]};
            }
        }
        
        
        return new int[2];
    }
}
```
#### Two Sum - Closest to target ####
Given an array nums of n integers, find two integers in nums such that the sum is closest to a given number, target.

Return the absolute value of difference between the sum of the two integers and the target.

###### Example ######
```
Example 1
Input:  nums = [-1, 2, 1, -4] and target = 4
Output: 1
Explanation:
The minimum difference is 1. (4 - (2 + 1) = 1).

Example 2
Input:  nums = [-1, -1, -1, -4] and target = 4
Output: 6
Explanation:
The minimum difference is 6. (4 - (- 1 - 1) = 6).
```
###### 分析 ######
```
数组排序， 再双指针
TC: O(nlogn) SC: O(1)
```

###### 代码 ######
```
public class Solution {
    public int twoSumClosest(int[] nums, int target) {
        if (nums == null || nums.length == 0){
            return -1;
        }
        
        int ans = Integer.MAX_VALUE;
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1;
        while (left < right){
            if (nums[left] + nums[right] == target){
                return 0;
            }else if (nums[left] + nums[right] < target){
                ans = Math.min(ans, target - (nums[left] + nums[right]));
                left++;
            }else {
                ans = Math.min(ans, nums[left] + nums[right] - target);
                right--;
            }
        }
        return ans;
    }
}
```

#### Two Sum - Unique pairs ####
Given an array of integers, find how many unique pairs in the array such that their sum is equal to a specific target number. Please return the number of pairs.

###### Example ######
```
Example 1
Input: nums = [1,1,2,45,46,46], target = 47 
Output: 2
Explanation:

1 + 46 = 47
2 + 45 = 47

Example 2
Input: nums = [1,1], target = 2 
Output: 1
Explanation:
1 + 1 = 2
```
###### 分析 ######
```
数组排序， 再双指针
TC: O(nlogn) SC: O(1)
注意去重
```
###### 代码 ######
```
public class Solution {
    public int twoSum6(int[] nums, int target) {
        if (nums == null || nums.length < 2){
            return 0;
        }
        
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1;
        int count = 0;
        while (left < right){
            if (nums[left] + nums[right] == target){
                count++;
                left++;
                right--;
                
                while (left < right && nums[left] == nums[left - 1]){
                    left++;
                }
                if (left >= right){
                    break;
                }
                while (left < right && nums[right] == nums[right + 1]){
                    right--;
                }
            }else if (nums[left] + nums[right] < target){
                left++;
            }else {
                right--;
            }
        }
        
        return count;
    }
}
```

#### Two Sum - Less than or equal to target ####
Given an array of integers, find how many pairs in the array such that their sum is less than or equal to a specific target number. Please return the number of pairs.

###### Example ######
```
Example 1
Input: nums = [2, 7, 11, 15], target = 24. 
Output: 5. 
Explanation:
2 + 7 < 24
2 + 11 < 24
2 + 15 < 24
7 + 11 < 24
7 + 15 < 24

Example 2
Input: nums = [1], target = 1. 
Output: 0. 
```

###### 分析 ######
```
数组排序， 再双指针
TC: O(nlogn) SC: O(1)
```

###### 代码 ######
```
public class Solution {
    public int twoSum5(int[] nums, int target) {
        if (nums == null || nums.length < 2){
            return 0;
        }
        
        int count = 0;
        int left = 0, right = nums.length -1;
        Arrays.sort(nums);
        
        while (left < right){
            while (nums[left] + nums[right] > target){
                right--;
            }
            if (right <= left){
                break;
            }
            count += right - left;
            left++;
        }
        
        return count;
    }
}
```

#### Two Sum - Greater than target ####
Given an array of integers, find how many pairs in the array such that their sum is bigger than a specific target number. Please return the number of pairs.

###### Example ######
```
Example 1
Input: [2, 7, 11, 15], target = 24
Output: 1
Explanation: 11 + 15 is the only pair.

Example 2
Input: [1, 1, 1, 1], target = 1
Output: 6
```

###### 分析 ######
```
数组排序， 再双指针
TC: O(nlogn) SC: O(1)
```

###### 代码 ######
```
public class Solution {
    public int twoSum2(int[] nums, int target) {
        if (nums == null || nums.length < 2){
            return 0;
        }
        
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1;
        int ans = 0;
        while (left < right){
            while (left < right && nums[left] + nums[right] <= target){
                left++;
            }
            
            // left -> right - 1 ==> (right - 1 - left + 1 = right - left) 
            ans += right - left;
            right--;
        }
        
        return ans;
    }
}
```

#### Two Sum VII ####
Given an array of integers that is already sorted in ascending absolute order, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are zero-based.

You are not allowed to sort this array.

###### Example ######
```
Example 1
Input: 
[0,-1,2,-3,4]
1
Output: [[1,2],[3,4]]
Explanation: nums[1] + nums[2] = -1 + 2 = 1, nums[3] + nums[4] = -3 + 4 = 1
You can return [[3,4],[1,2]], the system will automatically help you sort it to [[1,2],[3,4]]. But [[2,1],[3,4]] is invaild.
```

###### 分析 ######
```
two pointer left 指向最小值，right 指向最大值。两指针分别移动，求和判断即可。
TC: O(n), SC: O(1)
```

###### 代码 ######
```
public class Solution {
    public List<List<Integer>> twoSumVII(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null || nums.length == 0){
            return result;
        }
        
        int left = 0, right = 0;
        for (int i = 0; i < nums.length; i++){
            if (nums[i] < nums[left]){
                left = i;
            }
            if (nums[i] > nums[right]){
                right = i;
            }
        }
        
        while (nums[left] < nums[right]){
            if (nums[left] + nums[right] < target){
                left = nextLeft(nums, left);
                if (left == -1){
                    break;
                }
            }else if (nums[left] + nums[right] > target){
                right = nextRight(nums, right);
                if (right == -1){
                    break;
                }
            }else {
                List<Integer> tmp = new ArrayList<>();
                if (left < right){
                    tmp.add(left);
                    tmp.add(right);
                }else {
                    tmp.add(right);
                    tmp.add(left);
                }
                result.add(tmp);
                
                left = nextLeft(nums, left);
                right = nextRight(nums, right);
                if (left == -1 || right == -1){
                    break;
                }
            }
        }
        
        return result;
    }
    
    private int nextLeft(int[] nums, int index){
        if (nums[index] < 0){
            for (int i = index - 1; i >= 0; i--){
                if (nums[i] < 0){
                    return i;
                }
            }
            for (int i = 0; i < nums.length; i++){
                if (nums[i] >= 0){
                    return i;
                }
            }
            return -1;
        }
        for (int i = index + 1; i < nums.length; i++){
            if (nums[i] >= 0){
                return i;
            }
        }
        return -1;
    }
    
    private int nextRight(int[] nums, int index){
        if (nums[index] > 0){
            for (int i = index - 1; i >= 0; i--){
                if (nums[i] > 0){
                    return i;
                }
            }
            for (int i = 0; i < nums.length; i++){
                if (nums[i] <= 0){
                    return i;
                }
            }
            return -1;
        }
        for (int i = index + 1; i < nums.length; i++){
            if (nums[i] <= 0){
                return i;
            }
        }
        return -1;
    }
}
```