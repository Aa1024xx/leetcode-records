# Maximum Subarray #

## 53. Maximum Subarray ##
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

###### Example ######
```
example 1
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

```
example 2
Input: nums = [1]
Output: 1
```

```
example 3
Input: nums = [-2147483647]
Output: -2147483647
```

###### 思路 ######
```
subarray 含有 nums[i] 的局部最大值 vs. 全局最大值

遍历数组的subarray maximum : max(nums[i], leftMaxSubSum + nums[i])
```

###### 代码 ######
```
class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        if (nums == null || n == 0){
            return 0;
        }
        
        int max = nums[0];
        int maxCurr = nums[0];
        
        for (int i = 1; i < n; i++){
            maxCurr = Math.max(nums[i], maxCurr + nums[i]);
            max = Math.max(maxCurr, max);
        }
        
        return max;
    }
}
```

## Maximum Subarray II ##
Given an array of integers, find two non-overlapping subarrays which have the largest sum.
The number in each subarray should be contiguous.
Return the largest sum.

给定一个整数数组，找出两个 不重叠 子数组使得它们的和最大。
每个子数组的数字在数组中的位置应该是连续的。
返回最大的和。

###### 思路 ######
```
maximum subarray ->
prefixSum, postSum
```

###### 代码 ######
```
public class Solution {
    /*
     * @param nums: A list of integers
     * @return: An integer denotes the sum of max two non-overlapping subarrays
     */
    public int maxTwoSubArrays(List<Integer> nums) {
        List<Integer> leftSubSum = new ArrayList<>(nums);
        List<Integer> rightSubSum = new ArrayList<>(nums);
        
        int n = nums.size();
        if (nums == null || n == 0){
            return 0;
        }
        
        for (int i = 1; i < n; i++){
            leftSubSum.set(i, Math.max(nums.get(i), leftSubSum.get(i - 1) + nums.get(i)));
        }
        for (int i = n - 2; i >= 0; i--){
            rightSubSum.set(i, Math.max(nums.get(i), rightSubSum.get(i + 1) + nums.get(i)));
        }
        
        List<Integer> prefixSub = new ArrayList<>(leftSubSum);
        List<Integer> postSub = new ArrayList<>(rightSubSum);
        
        for (int i = 1; i < n; i++){
            prefixSub.set(i, Math.max(prefixSub.get(i), prefixSub.get(i - 1)));
        }
        for (int i = n - 2; i >= 0; i--){
            postSub.set(i, Math.max(postSub.get(i), postSub.get(i + 1)));
        }
        
        int result = Integer.MIN_VALUE;
        for (int i = 0; i < n - 1; i++){
            result = Math.max(result, prefixSub.get(i) + postSub.get(i + 1));
        }
        return result;
    }
}
```


