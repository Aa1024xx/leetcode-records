#### 167. Two Sum II - Input array is sorted ####
###### 思路 ###### 
```
双向指针，一头一尾 （因为数组是升序的），相向而行。如果两数之和加起来> target, 说明right指针的数取得太大了， right--, 相反则left++, == target 返回答案。 （assuming there is only one pair of result）
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

##### I, III, IV #####
#### 1. Two Sum ####
###### 思路 ###### 
```
用map记录所有当前查找过的num[i]，存下target-num[i]
如果存在num[j]在map中说明存在一对i,j的使得num[i],num[j]和为target, 即找到了答案
HashMap: num -> sum - num 
HashMap: key: num, value: index
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

#### 170. Two Sum III - Data structure design ####
###### 思路 ###### 
```
用HashMap来存储数据（key是num，value是num的个数）。add的时间复杂度是O(1), find的时间复杂的是O(n)，空间复杂度O(n)。 需要注意的是当在map里找到value - num时要判断下这个值是否与num相等。如果不相等可以直接返回true，如果相等要确认num在map里不止一个
HashMap: key: number, value: 次数
list: 记录 unique number
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

#### 653. Two Sum IV - Input is a BST ####
###### 思路 ###### 
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
##### Follow Up ####
#### Two Sum - Difference equals to target ####
###### 描述 ###### 
```
Given an sorted array of integers, find two numbers that their difference equals to a target value. return a list with two number like [num1, num2] that the difference of num1 and num2 equals to target value, and num1 is less than num2.
```
###### 思路 ###### 
```
由于数组有序，可以用双指针来做
对于双指针i,j,当num[j]-num[i]< target时，说明j太小，于是我们将j++，直到num[j]-num[i] >= target
若num[j]-num[i] > target，我们将i++
若num[j]-num[i] = target说明我们找到答案
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
###### 描述 ###### 
```
Given an array nums of n integers, find two integers in nums such that the sum is closest to a given number, target.
Return the absolute value of difference between the sum of the two integers and the target.
```
###### 思路 ###### 
```
将数组排序后，双指针求解 
第一个指针初始位置在最前面，从前往后走 第二个指针初始位置在最后面，从后往前走， 在这个过程中迭代更新答案
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
###### 描述 ###### 
```
Given an array of integers, find how many unique pairs in the array such that their sum is equal to a specific target number. Please return the number of pairs.
```
###### 思路 ###### 
```
利用双指针的方法，扫描排序完的数组即可
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
###### 描述 ###### 
```
Given an array of integers, find how many pairs in the array such that their sum is less than or equal to a specific target number. Please return the number of pairs.
```
###### 思路 ###### 
```
排序后双指针即可
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
###### 描述 ###### 
```
Given an array of integers, find how many pairs in the array such that their sum is bigger than a specific target number. Please return the number of pairs.
```
###### 思路 ###### 
```
排序后使用两根指针进行遍历
遍历时的优化: l 左移时可以用二分/倍增加速.
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

public class Solution {
    public int twoSum2(int[] nums, int target) {
        // Edge cases
        if (nums == null || nums.length < 2) return 0;
        
        // Binary search method
        Arrays.sort(nums);
        
        int count = 0;
        for (int i = 1; i < nums.length; i++) {
            int first = bsFirstLarger(nums, i - 1, target - nums[i]);
            if (first != -1) count += i - first;
        }
        
        return count;
    }
    
    private int bsFirstLarger(int[] nums, int end, int target) {
        int start = 0;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] > target) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (nums[start] > target) return start;
        if (nums[end] > target) return end;
        return -1;
    }
}
```

#### Follow Up ####
###### 描述 ###### 
```
给定一个已经 按绝对值升序排列 的数组，找到两个数使他们加起来的和等于特定数。

函数应该返回这两个数的下标，index1必须小于index2。注意返回的值是0-based。

不能对该数组进行排序。
```
###### 思路 ###### 
```
two pointer: left 指向最小值，right 指向最大值。两指针分别移动，求和判断即可。
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

#### 88. Merge Sorted Array ####
###### 思路 ###### 
```
涉及两个有序数组合并,设置i和j双指针,分别从两个数组的尾部向头部移动,并判断Ai和Bj的大小关系,从而保证最终数组有序,同时每次index从尾部向头部移动。
```
###### 代码 ###### 
```
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1, j = n - 1;
        int index = m + n - 1;
        while (i >= 0 && j >= 0){
            if (nums1[i] > nums2[j]){
                nums1[index--] = nums1[i--];
            }else {
                nums1[index--] = nums2[j--];
            }
        }
        
        while (i >= 0){
            nums1[index--] = nums1[i--];
        }
        while (j >= 0){
            nums1[index--] = nums2[j--];
        }
    }
}
```

##### Follow Up #####
###### 描述 ###### 
```
合并两个有序升序的整数数组A和B变成一个新的数组。新数组也要有序。
```
###### 思路 ###### 
```
使用两个指针分别对数组从小到大遍历，每次取二者中较小的放在新数组中。 直到某个指针先到结尾，另一个数组中剩余的数字直接放在新数组后面。
```
###### 代码 ###### 
```
class Solution {
    public int[] mergeSortedArray(int[] A, int[] B) {
        if (A == null || A.length == 0) {
            return B;
        }
        if (B == null || B.length == 0){
            return A;
        }
        
        int[] result = new int[A.length + B.length];
        int i = 0, j = 0, index = 0;
        
        while (i < A.length && j < B.length) {
            if (A[i] < B[j]) {
                result[index++] = A[i++];
            } else {
                result[index++] = B[j++];
            }
        }
        
        while (i < A.length) {
            result[index++] = A[i++];
        }
        while (j < B.length) {
            result[index++] = B[j++];
        }
        
        return result;
    }
}
```

##### Follow Up #####
###### 描述 ###### 
```
将 k 个有序数组合并为一个大的有序数组
```
###### 思路 ###### 
```
使用优先队列的方法 TC等于O(NlogK)，N 是所有元素个数
初始将所有数组的首个元素入堆, 并记录入堆的元素是属于哪个数组的.
每次取出堆顶元素, 并放入该元素所在数组的下一个元素.
```
###### 代码 ###### 
```
class Element {
    public int row, col, val;
    Element(int row, int col, int val) {
        this.row = row;
        this.col = col;
        this.val = val;
    }
}

public class Solution {
    private Comparator<Element> ElementComparator = new Comparator<Element>() {
        public int compare(Element left, Element right) {
            return left.val - right.val;
        }
    };
    
    public int[] mergekSortedArrays(int[][] arrays) {
        if (arrays == null) {
            return new int[0];
        }
        
        int total_size = 0;
        Queue<Element> Q = new PriorityQueue<Element>(
            arrays.length, ElementComparator);
            
        for (int i = 0; i < arrays.length; i++) {
            if (arrays[i].length > 0) {
                Element elem = new Element(i, 0, arrays[i][0]);
                Q.add(elem);
                total_size += arrays[i].length;
            }
        }
        
        int[] result = new int[total_size];
        int index = 0;
        while (!Q.isEmpty()) {
            Element elem = Q.poll();
            result[index++] = elem.val;
            if (elem.col + 1 < arrays[elem.row].length) {
                elem.col += 1;
                elem.val = arrays[elem.row][elem.col];
                Q.add(elem);
            }
        }
        
        return result;
    }
}
```

#### 142. Linked List Cycle II ####
###### 思路 ###### 
```
一快一慢一起走，两个碰头慢回头，一步走，一步走。再相遇，就是goal。 （朋友一生一起走，那些日子不再有， 一句话，一辈子，一生情，一杯酒）
”水中的鱼“证明：http://fisherlei.blogspot.com/2013/11/leetcode-linked-list-cycle-ii-solution.html
```
###### 代码 ###### 
```
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null){
            return null;
        }
        
        ListNode fast = head;
        ListNode slow = head;
        
        while (fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            
            if (fast == slow){
                break;
            }
        }
        
        if (fast == null || fast.next == null){
            return null;
        }
        
        slow = head;
        while (fast != slow){
            fast = fast.next;
            slow = slow.next;
        }
        
        return slow;
    }
}
```

##### Follow Up #####
#### 141. Linked List Cycle I ####
###### 代码 ###### 
```
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null){
            return false;
        }
        
        ListNode fast = head;
        ListNode slow = head;
        
        while (fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            
            if (fast == slow){
                return true;
            }
        }
        
        return false;
    }
}
```

#### 76. Minimum Window Substring ####
###### 思路 ###### 
```
滑窗法：指针left和right分别指向窗口两端，从左向右滑动，实施维护这个窗口。
```
###### 代码 ###### 
```
class Solution {
    public String minWindow(String ss, String tt) {
        if (ss == null || ss.length() == 0 || tt == null || tt.length() == 0){
            return "";
        }
        
        char[] s = ss.toCharArray();
        char[] t = tt.toCharArray();
        int[] cntS = new int[256]; 
        // number of appearances for each character in the window 
        int[] cntT = new int[256];
        int K = 0; // number of T's unique chracters
        
        for (char c : t){
            cntT[c]++;
            if (cntT[c] == 1){
                K++;
            }
        }
        
        int now = 0;
        int ansl = -1, ansr = -1;
        int l, r = 0;
        for (l = 0; l < s.length; l++){
            while (r < s.length && now < K){
                cntS[s[r]]++;
                if (cntS[s[r]] == cntT[s[r]]){
                    now++;
                }
                r++;
            }
            if (now == K){
                if (ansl == -1 || r - l < ansr - ansl){
                    ansl = l;
                    ansr = r;
                }
            }
            
            --cntS[s[l]];
            if (cntS[s[l]] == cntT[s[l]] - 1){
                now--;
            }
        }
        
        return ansl == -1 ? "" : ss.substring(ansl, ansr);
    }
}
```

#### Follow Up ####
###### 描述 ###### 
```
给你一个字符串 S、一个字符串 T，S是循环的，请在字符串 S 里面找出：包含 T 所有字母的最小子串。
```
###### 思路 ###### 
```
String str = s + s;
```
###### 代码 ###### 
```
public class Solution {
    public String minWindowII(String s, String target) {
        String source = s + s;
        int[] cntS = new int[256];
        int[] cntT = new int[256];
        
        int K = 0;
        for (char c : target.toCharArray()){
            cntT[c]++;
            if (cntT[c] == 1){
                K++;
            }
        }
        
        int ansl = -1, ansr = -1;
        int now = 0;
        for (int l = 0, r = 0; l < source.length(); l++){
            while (r < source.length() && now < K){
                cntS[source.charAt(r)]++;
                if (cntS[source.charAt(r)] == cntT[source.charAt(r)]){
                    now++;
                }
                r++;
            }
            if (now == K){
                if (ansl == -1 || r - l < ansr - ansl){
                    ansl = l;
                    ansr = r;
                }
            }
            cntS[source.charAt(l)]--;
            if (cntS[source.charAt(l)] == cntT[source.charAt(l)] - 1){
                now--;
            }
        }
        
        return ansl == -1 ? "" : source.substring(ansl, ansr);
    }
}
```

#### 633. Sum of Square Numbers  ####
###### 思路 ###### 
```
双指针，起点为0，终点为sqrt(c)，剩下的操作其实和two sum的双指针法差不多
```
###### 代码 ###### 
```
class Solution {
    public boolean judgeSquareSum(int c) {
        if (c < 0){
            return false;
        }
        
        int l = 0, r = (int)Math.sqrt(c);
        while (l <= r){
            if (l * l + r * r < c){
                l++;
            }else if (l * l + r * r == c){
                return true;
            }else {
                r--;
            }
        }
        return false;
    }
}
```

#### 680. Valid Palindrome II  ####
###### 思路 ###### 
```
双指针: 从两头走到中间，发现第一对不一样的字符之后，要么删左边的，要么删右边的。

删除N个字符的通用模板
```
###### 代码 ###### 
```
class Solution {
    public boolean validPalindrome(String s) {
        int l = 0, r = s.length() - 1;
        while (l < r){
            if (s.charAt(l) != s.charAt(r)){
                break;
            }
            l++;
            r--;
        }
        if (l >= r){
            return true;
        }
        
        return isSubPalindrome(s, l + 1, r) || isSubPalindrome(s, l, r - 1);
    }
    
    private boolean isSubPalindrome(String s, int start, int end){
        if (start > end){
            return false;
        }
        while (start < end){
            if (s.charAt(start) != s.charAt(end)){
                return false;
            }
            start++;
            end--;
        }
        return true;
    }
}

class Solution {
    public boolean validPalindrome(String s) {
        return validate(s, 0, s.length() - 1, 0, 1);
    }
    
    private boolean validate(String s, int i, int j, int delCount, int N){
        if (delCount > N){
            return false;
        }
        
        while (i < j){
            if (s.charAt(i) != s.charAt(j)){
                delCount++;
                return validate(s, i + 1, j, delCount, N) || validate(s, i, j - 1, delCount, N);
            }
            i++;
            j--;
        }
        return i >= j;
    }
}
```

#### 125.Valid Palindrome ####
###### 思路 ###### 
```
双指针遍历，遇到非有效字符直接跳过。
```
###### 代码 ###### 
```
class Solution {
    public boolean isPalindrome(String s) {
        if (s == null || s.length() == 0){
            return true;
        }
        
        int l = 0, r = s.length() - 1;
        while (l < r){
            while(l < s.length() && !isValid(s.charAt(l))){
                l++;
            }
            if (l == s.length()){
                return true;
            }
            
            while (r >= 0 && !isValid(s.charAt(r))){
                r--;
            }
            
            if (Character.toLowerCase(s.charAt(l)) != Character.toLowerCase(s.charAt(r))){
                return false;
            }
            l++;
            r--;
        }
        return l >= r;
    }
    
    private boolean isValid(char c){
        return Character.isLetter(c) || Character.isDigit(c);
    }
}
```

#### 524.Longest Word in Dictionary through Deleting ####
###### 思路 ###### 
```
遍历字典，对于长度长于符合条件的最长子串的长度，或长度相等且字典序小于其的字符串，与给定字符串进行比对，若满足条件，则成为新的最长子串，直至扫描字典完毕
```
###### 代码 ###### 
```
class Solution {
    public String findLongestWord(String s, List<String> d) {
        String answer = "";
        for (String word : d){
            if (word.length() < answer.length() || (word.length() == answer.length() && word.compareTo(answer) > 0)){
                continue;
            }
            int i = 0, j = 0;
            while (i < s.length() && j < word.length() && s.length() - i >= word.length() - j){
                if (word.charAt(j) == s.charAt(i)){
                    j++;
                }
                i++;
            }
            if (j == word.length()){
                answer = word;
            }
        }
        return answer;
    }
}
```

#### 340.Longest Substring with At Most K Distinct Characters ####
###### 思路 ###### 
```
同向双指针: 在字符串上移动滑动窗口，保证窗口内有不超过 k 个不同字符，同时在每一步更新最大子串长度。
- 如果字符串为空或者 k 是零的话返回 0。
- 设置指针为字符串开头 left = 0 和 right = 0，初始化最大子串长度 maxLen = 1。
- 当 right < N 时：
  - 将当前字符 s[right] 加入哈希表并且向右移动 right 指针。 
  - 如果哈希表包含 k + 1 个不同字符，在哈希表中移除最左出现的字符(s[left])，右移动 left 指针使得滑动窗口只包含 k 个不同字符。 
  - 更新 maxLen = max(maxLen, right - left)。
```
###### 代码 ###### 
```
class Solution {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        if (s == null || s.length() == 0 || k == 0){
            return 0;
        }
        
        int left = 0, right = 0;
        int distinct = 0;
        int[] charSet = new int[256];
        int ans = 0;
        
        while (right < s.length()){
            if (charSet[s.charAt(right)] == 0){
                distinct++;
            }
            charSet[s.charAt(right)]++;
            right++;
            
            while (distinct > k){
                charSet[s.charAt(left)]--;
                if (charSet[s.charAt(left)] == 0){
                    distinct--;
                }
                left++;
            }
            
            ans = Math.max(ans, right - left);
        }
        return ans;
    }
}
```