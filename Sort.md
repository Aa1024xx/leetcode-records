#### 215.Kth Largest Element in an Array ####
###### 思路 ###### 
```
快速排序
```
###### 代码 ###### 
```
class Solution {
    public int findKthLargest(int[] nums, int k) {
        return qucikSort(nums, 0, nums.length - 1, k);
    }
    
    private int qucikSort(int[] nums, int start, int end, int k){
        int left = start;
        int right = end;
        int pivot = nums[left + (right - left) / 2];
        
        while (left <= right){
            while (left <= right && nums[left] > pivot){
                left++;
            }
            while (left <= right && nums[right] < pivot){
                right--;
            }
            if (left <= right){
                swap(nums, left, right);
                left++;
                right--;
            }
        }
        
        if (start + k - 1 <= right){
            return qucikSort(nums, start, right, k);
        }
        if (start + k - 1 >= left){
            return qucikSort(nums, left, end, k - (left - 1 - start + 1));
        }
        return nums[right + 1];
    }
    
    private void swap(int[] nums, int left, int right){
        int tmp = nums[left];
        nums[left] = nums[right];
        nums[right] = tmp;
    }
}

class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>((n1, n2) -> (n1 - n2));
        for (int num : nums){
            pq.offer(num);
            if (pq.size() > k){
                pq.poll();
            }
        }
        return pq.peek();
    }
}
```

#### 347.Top K Frequent Elements ####
###### 思路 ###### 
```
最小堆: 因为最终需要返回前 k 个频率最大的元素，通过维护一个元素数目为 k 的最小堆，每次都将新的元素与堆顶端的元素（堆中频率最小的元素）进行比较，如果新的元素的频率比堆顶端的元素大，则弹出堆顶端的元素，将新的元素添加进堆中。最终，堆中的 k 个元素即为前 k 个高频元素。

桶排: 将数组中的元素按照出现频次进行分组，即出现频次为 i 的元素存放在第 i 个桶。最后，从桶中逆序取出前 k 个元素。
```
###### 代码 ###### 
```
class Pair {
    int val, freq;
    public Pair(int val, int freq){
        this.val = val;
        this.freq = freq;
    }
}

class PairComparator implements Comparator<Pair>{
    public int compare(Pair a, Pair b){
        if (a.freq != b.freq){
            return a.freq - b.freq;
        }
        return b.val - a.val;
    }
}
    
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0){
            return new int[]{};
        }
        
        PriorityQueue<Pair> pq = new PriorityQueue<>(k, new PairComparator());
        Map<Integer, Integer> map = new HashMap<>();
        
        for (int num : nums){
            if (map.containsKey(num)){
                map.put(num, map.get(num) + 1);
            }else {
                map.put(num, 1);
            }
        }
        
        for (Map.Entry<Integer,Integer> entry : map.entrySet()){
            pq.offer(new Pair(entry.getKey(), entry.getValue()));
            if (pq.size() > k){
                pq.poll();
            }
        }
        
        int[] result = new int[k];
        while (!pq.isEmpty()){
            result[--k] = pq.poll().val;
        }
        return result;
    }
}
```

#### 451.Sort Characters By Frequency ####
###### 思路 ###### 
```
先统计后排序，最后组装字符串
```
###### 代码 ###### 
```
class Pair {
    Character c;
    int freq;
    public Pair(Character c, int freq){
        this.c = c;
        this.freq = freq;
    }
}

class PairComparator implements Comparator<Pair>{
    public int compare(Pair a, Pair b){
        if (a.freq != b.freq){
            return b.freq - a.freq;
        }
        return a.c - b.c;
    }
}
class Solution {
    public String frequencySort(String s) {
        if (s == null || s.length() == 0){
            return "";
        }
        
        PriorityQueue<Pair> pq = new PriorityQueue<>(s.length(), new PairComparator());
        Map<Character, Integer> map = new HashMap<>();
        for (char ch : s.toCharArray()){
            if (map.containsKey(ch)){
                map.put(ch, map.get(ch) + 1);
            }else {
                map.put(ch, 1);
            }
        }
        
        for (Map.Entry<Character,Integer> entry : map.entrySet()){
            pq.offer(new Pair(entry.getKey(), entry.getValue()));
        }
        
        StringBuilder sb = new StringBuilder();
        while (!pq.isEmpty()){
            int i = 0;
            while (i < pq.peek().freq){
                sb.append(pq.peek().c);
                i++;
            }
            pq.poll();
        }
        
        return new String(sb);
    }
}
```

#### 75.Sort Colors ####
###### 思路 ###### 
```
使用一次扫描的办法。 设立三根指针，left, index, right。定义如下规则：
- left 的左侧都是 0（不含 left）
- right 的右侧都是 2（不含 right）
index 从左到右扫描每个数，如果碰到 0 就丢给 left，碰到 2 就丢给 right。碰到 1 就跳过不管
```
###### 代码 ###### 
```
class Solution {
    public void sortColors(int[] nums) {
        int left = 0, index = 0, right = nums.length - 1;
        while(index <= right){
            if (nums[index] == 0){
                nums[index] = nums[left];
                nums[left] = 0;
                left++;
                index++;
            }else if (nums[index] == 2){
                nums[index] = nums[right];
                nums[right] = 2;
                right--;
            }else {
                index++;
            }
        }
    }
}
```

#### Follow Up ####
###### 描述 ###### 
```
给定一个有n个对象（包括k种不同的颜色，并按照1到k进行编号）的数组，将对象进行分类使相同颜色的对象相邻，并按照1,2，...k的顺序进行排序。
```
###### 思路 ###### 
```
运使用rainbowSort，或者说是改动过的quickSort，运用分治的思想，不断将当前需要处理的序列分成两个更小的序列处理。
思路与quickSort大致相同，每次选定一个中间的颜色，这个中间的颜色用给出的k来决定，将小于等于中间的颜色的就放到左边，大于中间颜色的就放到右边，然后分别再递归左右两半。
```
###### 代码 ###### 
```
public class Solution {
    public void sortColors2(int[] colors, int k) {
        if (colors == null || colors.length < 2) {

            return;

        }
        sort(colors, 0, colors.length - 1, 1, k);

    }

    

    private void sort(int[] colors, int start, int end, int colorFrom, int colorTo) {

        //若处理区间长度为小于等于1或颜色区间长度为1，则不需要再进行处理

        if (start >= end || colorFrom == colorTo) {

            return;

        }

        //设置左右指针以及中间的颜色

        int left = start;

        int right = end;

        int colorMid = colorFrom + (colorTo - colorFrom) / 2;

        while (left <= right) {

            //找到左侧大于中间颜色的位置

            while (left <= right && colors[left] <= colorMid) {

                left++;

            }

            //找到右侧小于等于中间颜色的位置

            while (left <= right && colors[right] > colorMid) {

                right--;

            }

            //交换左右指针指向的颜色

            if (left <= right) {

                int temp = colors[left];

                colors[left] = colors[right];

                colors[right] = temp;

            }

        }

        //继续递归处理左右两半序列

        sort(colors, start, right, colorFrom, colorMid);

        sort(colors, left, end, colorMid + 1, colorTo);

    }

}
```