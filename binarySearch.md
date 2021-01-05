#### 69.Sqrt(x) ####
###### 思路 ###### 
```
接对答案可能存在的区间进行二分
判断区间的时候一个小技巧： mid * mid == x 中使用乘法可能会溢出，写成 mid == x / mid 即可防止溢出，不需要使用long或者BigInteger
```
###### 代码 ###### 
```
class Solution {
    public int mySqrt(int x) {
        if (x < 0) throw new IllegalArgumentException();
        if (x <= 1){
            return x;
        }
        
        int start = 1, end = x;
        while (start + 1 < end){
            int mid = start + (end - start) / 2;
            if (mid == x / mid){
                return mid;
            }else if (mid < x / mid){
                start = mid;
            }else {
                end = mid;
            }
        }
        
        if (end > x / end){
            return start;
        }
        return end;
    }
}
```
##### Follow Up #####
###### 描述 ######
```
实现 double sqrt(double x) 并且 x >= 0。
``` 
###### 思路 ###### 
```
二分法 
牛顿法
```
###### 代码 ###### 
```
public class Solution {
    public double sqrt(double x) {
        // Write your code here
        double l = 0; 
        double r = Math.max(x, 1.0);
        double eps = 1e-12;
        
        while (l + eps < r) {
            double mid = l + (r - l) / 2;
            if (mid * mid < x) {
                l = mid;
            } else {
                r = mid;
            }
        }
        
        return l;
    }
}

public class Solution {
    public double sqrt(double x) {
        // Write your code here
        double res = 1.0;
        double eps = 1e-12;

        while(Math.abs(res * res - x) > eps) {
            res = (res + x / res) / 2;
        }

        return res;
    }
}
```

#### 69.Find First and Last Position of Element in Sorted Array ####
###### 思路 ###### 
```
二分法: 做两次二分，分别确定左右边界
```
###### 代码 ###### 
```
class Solution {
    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0){
            return new int[]{-1, -1};
        }
        
        int leftIndex = findFirstPos(nums, target);
        int rightIndex = findLastPos(nums, target);
        if (leftIndex != -1 && rightIndex != -1){
            return new int[]{leftIndex, rightIndex};
        }
        
        return new int[]{-1, -1};
    }
    
    private int findFirstPos(int[] nums, int target){
        int left = 0, right = nums.length - 1;
        while (left + 1 < right){
            int mid = left + (right - left) / 2;
            if (nums[mid] >= target){
                right = mid;
            }else {
                left = mid;
            }
        }
        
        if (nums[left] == target){
            return left;
        }
        if (nums[right] == target){
            return right;
        }
        
        return -1;
    }
    
    private int findLastPos(int[] nums, int target){
        int left = 0, right = nums.length - 1;
        while (left + 1 < right){
            int mid = left + (right - left) / 2;
            if (nums[mid] <= target){
                left = mid;
            }else {
                right = mid;
            }
        }
        
        if (nums[right] == target){
            return right;
        }
        if (nums[left] == target){
            return left;
        }
        
        return -1;
    }
}
```

#### 81.Search in Rotated Sorted Array II ####
###### 思路 ###### 
```
暴力求解
```
###### 代码 ###### 
```
public class Solution {
    // 这个问题在面试中不会让实现完整程序
    // 只需要举出能够最坏情况的数据是 [1,1,1,1... 1] 里有一个0即可。
    // 在这种情况下是无法使用二分法的，复杂度是O(n)
    // 因此写个for循环最坏也是O(n)，那就写个for循环就好了
    // 如果你觉得，不是每个情况都是最坏情况，你想用二分法解决不是最坏情况的情况，那你就写一个二分吧。
    // 反正面试考的不是你在这个题上会不会用二分法。这个题的考点是你想不想得到最坏情况。
    public boolean search(int[] A, int target) {
        for (int i = 0; i < A.length; i ++) {
            if (A[i] == target) {
                return true;
            }
        }
        return false;
    }
}
```

#### 33.Search in Rotated Sorted Array ####
###### 思路 ###### 
```
二分法
```
###### 代码 ###### 
```
class Solution {
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0){
            return -1;
        }
        
        int start = 0, end = nums.length - 1;
        while (start + 1 < end){
            int mid = start + (end - start) / 2;
            if (nums[mid] == target){
                return mid;
            }
            if (nums[start] < nums[mid]){
                if (nums[start] <= target && target <= nums[mid]){
                    end = mid;
                }else {
                    start = mid;
                }
            }else {
                if (nums[mid] <= target && target <= nums[end]){
                    start = mid;
                }else {
                    end = mid;
                }
            }
        }
        
        if (nums[start] == target){
            return start;
        }
        if (nums[end] == target){
            return end;
        }
        return -1;
    }
}
```

#### 154.Find Minimum in Rotated Sorted Array II ####
###### 思路 ###### 
```
二分法
153寻找旋转排序数组中的最小值的follow-up题，区别是数组中可能会出现重复元素。我们依旧旋转数组的特性，用改进后的二分查找来解决，平均时间复杂度为O(logN)。因为重复元素的出现，所以分类讨论的条件会更精细。
```
###### 代码 ###### 
```
class Solution {
    public int findMin(int[] nums) {
        int start = 0, end = nums.length - 1;
        while (start + 1 < end){
            int mid = start + (end - start) / 2;
            if (nums[mid] == nums[end]){
                // if mid equals to end, that means it's fine to remove end
                // the smallest element won't be removed
                end--;
            }else if (nums[mid] < nums[end]){
                end = mid;
            }else {
                start = mid;
            }
        }
        
        if (nums[start] <= nums[end]){
            return nums[start];
        }
        return nums[end];
    }
}
```

#### 153.Find Minimum in Rotated Sorted Array II ####
###### 思路 ###### 
```
每次都和 end 去比
```
###### 代码 ###### 
```
class Solution {
    public int findMin(int[] nums) {
        int start = 0, end = nums.length - 1;
        while (start + 1 < end){
            int mid = start + (end - start) / 2;
            if (nums[mid] > nums[end]){
                start = mid;
            }else {
                end = mid;
            }
        }
        return Math.min(nums[start], nums[end]);
    }
}
```

#### 540.Single Element in a Sorted Array ####
###### 思路 ###### 
```
用二分法进行查找，mid向下取最近的偶数，如果nums[mid]不等于nums[mid + 1]，说明唯一的数在start ~ mid中，否则就在mid + 2 ~ end中
```
###### 代码 ###### 
```
class Solution {
    public int singleNonDuplicate(int[] nums) {
        int start = 0, end = nums.length - 1;
        while (start + 1 < end){
            int mid = start + (end - start) / 2;
            if (mid % 2 == 1){
                mid--;
            }
            if (nums[mid] != nums[mid + 1]){
                end = mid;
            }else {
                start = mid + 2;
            }
        }
        
        if ((start > 0 && nums[start] == nums[start - 1]) || ((start + 1 < nums.length)&& nums[start] == nums[start + 1])){
            return nums[end];
        }
        return nums[start];
    }
}
```

#### 4.Median of Two Sorted Arrays ####
###### 思路 ###### 
```
二分法
TC: O(log(range) * (log(n) + log(m))), 其中 range 为最小和最大的整数之间的范围
```
###### 代码 ###### 
```
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int m = nums2.length;
        if ((n + m) % 2 == 1){
            return (double)findKth(nums1, nums2, (n + m) / 2 + 1);
        }
        return (double)(findKth(nums1, nums2, (n + m) / 2) + findKth(nums1, nums2, (n + m) / 2 + 1)) / 2;
    }
    
    private int findKth(int[] A, int[] B, int K){
        if (A == null || A.length == 0){
            return B[K - 1];
        }
        if (B == null || B.length == 0){
            return A[K - 1];
        }
        
        int n = A.length;
        int m = B.length;
        int left = Math.min(A[0], B[0]);
        int right = Math.max(A[n - 1], B[m - 1]);
        
        while (left + 1 < right){
            int mid = left + (right - left) / 2;
            if (count(A, mid) + count(B, mid) < K){
                left = mid;
            }else{
                right = mid;
            }
        }
        
        if (count(A, left) + count(B, left) >= K){
            return left;
        }
        return right;
    }
    
    private int count(int[] nums, int target){
        int ans = 0;
        for (int num : nums){
            if (num <= target){
                ans++;
            }
        }
        return ans;
    }
}
```