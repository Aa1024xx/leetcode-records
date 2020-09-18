### 42. Trapping Rain Water ###
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

###### example ######
```
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

###### 思路 ######
```
two pointers
leftMax[], rightMax

int H = Math.min(leftMax[i], rightMax);
area += H > height[i] ? H - height[i] : 0;

leftMax[i + 1] = Math.max(height[i], leftMax[i]); // i starts from 0, leftMax[0] = 0;
这里注意height[i] 是从0开始的的

rightMax = 0
```

###### 代码 ######
```
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        if (height == null || n == 0){
            return 0;
        }
        
        int[] leftMax = new int[n + 1]; 
        leftMax[0] = 0;
        for (int i = 0; i < n; i++){
            leftMax[i + 1] = Math.max(height[i], leftMax[i]);
        }
        
        int rightMax = 0;
        int area = 0;
        for (int i = n - 1; i >= 0; i--){
            int H = Math.min(leftMax[i], rightMax);
            area += H > height[i] ? H - height[i] : 0;
            rightMax = Math.max(rightMax, height[i]);
        }
        
        return area;
    }
}
```

### 238. Product of Array Except Self ###
Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Constraint: It's guaranteed that the product of the elements of any prefix or suffix of the array (including the whole array) fits in a 32 bit integer.

Note: Please solve it without division and in O(n).


###### example ######
```
Input:  [1,2,3,4]
Output: [24,12,8,6]
```

###### 思路 ######
```
表演： O(n) space complexity: prefixProduct[], postProduct[] -> result
follow up: O(1) space complexity: reuslt[] Arrays.fill(result, 1), int prefixProduct, postProduct
```

###### 代码 ######
```
class Solution {
    public int[] productExceptSelf(int[] nums) {

        // The length of the input array
        int length = nums.length;

        // The left and right arrays as described in the algorithm
        int[] L = new int[length];
        int[] R = new int[length];

        // Final answer array to be returned
        int[] answer = new int[length];

        // L[i] contains the product of all the elements to the left
        // Note: for the element at index '0', there are no elements to the left,
        // so L[0] would be 1
        L[0] = 1;
        for (int i = 1; i < length; i++) {

            // L[i - 1] already contains the product of elements to the left of 'i - 1'
            // Simply multiplying it with nums[i - 1] would give the product of all
            // elements to the left of index 'i'
            L[i] = nums[i - 1] * L[i - 1];
        }

        // R[i] contains the product of all the elements to the right
        // Note: for the element at index 'length - 1', there are no elements to the right,
        // so the R[length - 1] would be 1
        R[length - 1] = 1;
        for (int i = length - 2; i >= 0; i--) {

            // R[i + 1] already contains the product of elements to the right of 'i + 1'
            // Simply multiplying it with nums[i + 1] would give the product of all
            // elements to the right of index 'i'
            R[i] = nums[i + 1] * R[i + 1];
        }

        // Constructing the answer array
        for (int i = 0; i < length; i++) {
            // For the first element, R[i] would be product except self
            // For the last element of the array, product except self would be L[i]
            // Else, multiple product of all elements to the left and to the right
            answer[i] = L[i] * R[i];
        }

        return answer;
    }
}
```

```
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        if (nums == null || n == 0){
            return new int[]{};
        }
        
        int[] result = new int[n];
        Arrays.fill(result, 1);
        
        int prefixProduct = 1;
        int postProduct = 1;
        
        for (int i = 0; i < n; i++){
            result[i] *= prefixProduct;
            prefixProduct *= nums[i];
        }
        
        for (int i = n - 1; i >= 0; i--){
            result[i] *= postProduct;
            postProduct *= nums[i];
        }
        
        return result;
    }
}
```

