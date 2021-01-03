#### 455. Assign Cookies ####
###### 思路 ###### 
```
对于一块饼干j,将它分给一个满足 g[i] <= s[j] 且 g[i] 最大的孩子
```
###### 代码 ###### 
```
class Solution {
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0, j = 0;
        for (; i < g.length && j < s.length; j++){
            if (g[i] <= s[j]){
                i++;
            }
        }
        return i;
    }
}
```

#### 135. Candy ####
###### 思路 ###### 
```
初始假定给每个小孩都分一颗糖果, 即设定一个全1的数组.
修正这个数组:
从左到右遍历, 如果发现一个小孩左边的小孩比自己的 rating 低, 那么把这个小孩的糖果数设为他左边的小孩糖果数 + 1
这时我们分配的糖果已经满足了: 评分更高的小孩比他左边的小孩获得更多的糖果. 然后我们再从右往左遍历一次就可以了.
但是这时还应该注意一点: 当一个小孩右边的小孩比自己的 rating 低时, 这个小孩的糖果可能已经比他右边的小孩多了, 而且可能多不止一个, 这时应该保留他的糖果数目不变.
```
###### 代码 ###### 
```
class Solution {
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0){
            return 0;
        }
        
        int[] candies = new int[ratings.length];
        Arrays.fill(candies, 1);
        
        for (int i = 1; i < ratings.length; i++){
            if (ratings[i - 1] < ratings[i]){
                candies[i] = candies[i - 1] + 1;
            }
        }
        
        for (int i = ratings.length - 2; i >= 0; i--){
            if (ratings[i + 1] < ratings[i] && candies[i] <= candies[i + 1]){
                candies[i] = candies[i + 1] + 1;
            }
        }
        
        int sum = 0;
        for (int candy : candies){
            sum += candy;
        }
        return sum;
    }
}
```

##### Follow Up #####
###### 描述 ######
```
-得分高的孩子比他们的邻居得到更多的糖果
-评级相同、相邻的孩子得到相同的糖果
```
###### 思路 ###### 
```
从左到右遍历，满足条件
从右往左遍历，满足条件
```
###### 代码 ###### 
```
class Solution {
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0){
            return 0;
        }
        
        int[] candies = new int[ratings.length];
        
        for (int i = 0; i < ratings.length; i++){
            if (i == 0 || ratings[i - 1] > ratings[i]){
                candies[i] = 1;
            }else if (ratings[i - 1] == ratings[i]){
                candies[i] = candies[i - 1];
            }else {
                candies[i] = candies[i - 1] + 1;
            }
        }

        for (int i = ratings.length - 2; i >= 0; i--){
            if (ratings[i] > ratings[i + 1]){
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
            }else if (ratings[i] == ratings[i + 1]){
                candies[i] = Math.max(candies[i], candies[i + 1]);
            }
        }
        
        int sum = 0;
        for (int candy : candies){
            sum += candy;
        }
        return sum;
    }
}
```

#### 435. Non-overlapping Intervals ####
###### 思路 ###### 
```
将所有的区间按照end从小到大的顺序排序, 参考值ref初始等于最小end值
对于当前区间，若 ref <= start，则 ref 更新为 end，列入不重叠区间个数计数 + 1 
需要移除的区间个数 = 总区间数 - 不重叠区间个数
```
###### 代码 ###### 
```
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals == null || intervals.length == 0){
            return 0;
        }
        
        Arrays.sort(intervals, new Comparator<int[]>(){
            public int compare(int[] a, int[] b){
                return a[1] - b[1];
            }
        });
        
        int count = 1;
        int ref = intervals[0][1];
        for (int i = 1; i < intervals.length; i++){
            if (ref <= intervals[i][0]){
                ref = intervals[i][1];
                count++;
            }
        }
        return intervals.length - count;
    }
}
```

#### 605. Can Place Flowers ####
###### 思路 ###### 
```
模拟种植过程
```
###### 代码 ###### 
```
class Solution {
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        if (flowerbed == null || flowerbed.length == 0){
            return n == 0;
        }
        
        int canPlantNum = 0;
        for (int i = 0; i < flowerbed.length; i++){
            if (flowerbed[i] == 0 && (flowerbed[i - 1] == 0 || i == 0) && (flowerbed[i + 1] == 0 || i == flowerbed.length - 1)){
                canPlantNum++;
            }
            if (canPlantNum >= n){
                return true;
            }
        }
        return false;
    }
}
```

#### 452. Minimum Number of Arrows to Burst Balloons ####
###### 思路 ###### 
```
将气球按照end从小到大进行排序，将第一个气球的end作为参考ref
从左到右遍历，若当前气球的起点小于等于参考点ref则continue
若当前气球的起点大于参考点，则将参考点ref更新为当前气球的终点，同时答案+1
```
###### 代码 ###### 
```
class Solution {
    public int findMinArrowShots(int[][] points) {
        if (points == null || points.length == 0){
            return 0;
        }

        // 针对test case input: [[-2147483646,-2147483645],[2147483646,2147483647]]
        // integer overflow
        Arrays.sort(points, new Comparator<int[]>(){
            public int compare(int[] a, int[] b){
                if (a[1] < b[1]){
                    return -1;
                }else if (a[1] == b[1]){
                    return 0;
                }else {
                    return 1;
                }
            }
        });
        
        int ref = points[0][1];
        int count = 1;
        for (int i = 1; i < points.length; i++){
            System.out.println(ref + " " + count);
            if (ref < points[i][0]){
                count++;
                ref = points[i][1];
            }
            System.out.println(ref + " " + count);
        }
        
        
        return count;
        
    }
}
```

#### 763. Partition Labels ####
###### 思路 ###### 
```
为了能很快的找到任意字符的最右下标，需要提前遍历一边字符串，并记录最右下标
再次遍历字符串S，用left和right表示当前子串的左边界和右边界，扩展当前的右边界right=max(right，当前字符的最右下标)
如果已经遍历到了right位置，这时我们就可切出一个子串，这个子串的下标是从left到right（包括right），之后再设置left为下一个字符的下标
重复上述操作，直到遍历完S    
```
###### 代码 ###### 
```
class Solution {
    public List<Integer> partitionLabels(String S) {
        List<Integer> result = new ArrayList<>();
        if (S == null || S.length() == 0){
            return result;
        }
        
        // find last appearance
        int[] lastAppearance = new int[26];
        for (int i = 0; i < S.length(); i++){
            lastAppearance[S.charAt(i) - 'a'] = i;
        }
        
        // left, right border to part labels
        int left = 0, right = 0;
        for (int i = 0; i < S.length(); i++){
            right = Math.max(lastAppearance[S.charAt(i) - 'a'], right);
            if (i == right){
                result.add(right - left + 1);
                left = i + 1;
            }
        }
        return result;
    }
}
```

#### 122. Best Time to Buy and Sell Stock II ####
###### 思路 ###### 
```
不限制交易次数
只要相邻的两天股票的价格是上升的, 就进行一次交易, 获得一定利润   
```
###### 代码 ###### 
```
class Solution {
    public int maxProfit(int[] prices) {
        int profit = 0;
        for (int i = 0; i < prices.length - 1; i++){
            if (prices[i] < prices[i + 1]){
                profit += prices[i + 1] - prices[i];
            }
        }
        return profit;
    }
}
```

##### I, III, IV #####
#### 121. Best Time to Buy and Sell Stock I ####
###### 思路 ###### 
```
ONCE at most
Record the max profit until now 
```
###### 代码 ###### 
```
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0){
            return 0;
        }
        
        int ans = 0;
        int[] profit = new int[prices.length];
        for(int i = 1; i < prices.length; i++){
            profit[i] = Math.max(profit[i], profit[i - 1] + prices[i] - prices[i - 1]);
            ans = Math.max(ans, profit[i]);
        }
        return ans;
    }
}
```

#### 123. Best Time to Buy and Sell Stock III ####
###### 思路 ###### 
```
twice at most
最多完成k笔交易时的最大利润, k = 2
```
###### 代码 ###### 
```
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int K = 2;
        
        if (prices == null || n == 0 || K == 0){
            return 0;
        }
        
        int[][] f = new int[K + 1][n];
        // f[i][j] = max(f[i][j - 1], prices[j] - prices[m] + f[i - 1][m]);
        // f[i][j] = max(f[i][j - 1], prices[j] + maxDiff);
        // maxDiff = max(maxDiff, f[i - 1][j - 1] - prices[j - 1]);
        for (int i = 1; i < f.length; i++){
            int maxDiff = -prices[0];
            for (int j = 1; j < f[0].length; j++){
                maxDiff = Math.max(maxDiff, f[i - 1][j - 1] - prices[j]);
                f[i][j] = Math.max(f[i][j - 1], prices[j] + maxDiff);
            }
        }
        return f[K][n - 1];
    }
}

class Solution {
    public int maxProfit(int[] prices) {
        int buy1 = Integer.MIN_VALUE, buy2 = Integer.MIN_VALUE;
        int sell1 = 0, sell2 = 0;
        for (int price : prices){
            buy1 = Math.max(buy1, -price);
            sell1 = Math.max(sell1, price + buy1);
            buy2 = Math.max(buy2, sell1 - price);
            sell2 = Math.max(sell2, price + buy2);
        }
        return sell2;
    }
}
```

#### 188. Best Time to Buy and Sell Stock IV ####
###### 思路 ###### 
```
最多完成k笔交易时的最大利润
```
###### 代码 ###### 
```
class Solution {
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        int K = k;
        
        if (prices == null || n == 0 || K == 0){
            return 0;
        }
        
        int[][] f = new int[K + 1][n];
        // f[i][j] = max(f[i][j - 1], prices[j] - prices[m] + f[i - 1][m]);
        // f[i][j] = max(f[i][j - 1], prices[j] + maxDiff);
        // maxDiff = max(maxDiff, f[i - 1][j - 1] - prices[j - 1]);
        for (int i = 1; i < f.length; i++){
            int maxDiff = -prices[0];
            for (int j = 1; j < f[0].length; j++){
                maxDiff = Math.max(maxDiff, f[i - 1][j - 1] - prices[j]);
                f[i][j] = Math.max(f[i][j - 1], prices[j] + maxDiff);
            }
        }
        return f[K][n - 1];
    }
}
```

##### Follow Up #####
###### 描述 ######
```
给出一个股票n天的价格，每天最多只能进行一次交易，可以选择买入一支股票或卖出一支股票或放弃交易，输出能够达到的最大利润值
```
###### 思路 ###### 
```
从左往右遍历，若当前价格大于之前遇到的最低价，则做交易。同时把在heap里用卖出价代替买入价，即将当前价格压入队列（假设当前价格为b,要被弹出的元素是a,后面有一个c元素，如果那时a还在，作为最低价，差值为c-a，而这里已经被b拿去做了差值，所以b得压入队列，因为c-b+b-a = c-a），弹出之前的最低价,可以利用优先队列来使之前的价格有序
- 用优先队列存储当前遇到过的价格
- 每日的新价格 与历史最低价比较
- 若比最低价高，则弹出最低价，同时更新答案，即加上差值
- 压入当前价格
```
###### 代码 ###### 
```
public class Solution {
    public int getAns(int[] prices) {
        if (prices == null || prices.length == 0){
            return 0;
        }
        
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        int profit = 0;
        for (int price : prices){
            if (pq.size() > 0 && price > pq.peek()){
                profit += price - pq.poll();
                pq.offer(price); // 用b替换掉a
            }
            pq.offer(price);
        }
        return profit;
    }
}
```

#### 406. Queue Reconstruction by Height ####
###### 思路 ###### 
```
将身高从高到低排序后依次插入即可。对于某个人h,k来说，插入答案的第k位。
```
###### 代码 ###### 
```
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>(){
            public int compare(int[] a, int[] b){
                if (a[0] == b[0]){
                    return a[1] - b[1];
                }else {
                    return b[0] - a[0];
                }
            }
        });
        
        List<int[]> list = new ArrayList<>();
        for (int[] person : people){
            list.add(person[1], person);
        }
        return list.toArray(new int[people.length][2]);
    }
}
```

#### 406. Queue Reconstruction by Height ####
###### 思路 ###### 
```
从头开始模拟，碰到第一次nums[i]<nums[i-1]时，我们需要修改nums[i]或者nums[i-1]来保证数组的不下降
有两种情况：
- nums[i]<nums[i-2]，比如3,4,2这样的情况，当前nums[i]=2。此时我们只能将nums[i]修改为4，才能在满足题意的条件下保证数组不下降，修改后为3,4,4
- nums[i]>=nums[i-2]，比如3,5,4，当前nums[i]=4。此时我们可以将nums[i-1]修改为4，修改后为3,4,4
```
###### 代码 ###### 
```
class Solution {
    public boolean checkPossibility(int[] nums) {
        int count = 0;
        for (int i = 1; i < nums.length; i++){
            if (nums[i - 1] > nums[i]){
                count++;
                if (i >= 2 && nums[i] < nums[i - 2]){
                    nums[i] = nums[i - 1];
                }else {
                    nums[i - 1] = nums[i];
                }
            }
        }
        return count <= 1;
    }
}
```