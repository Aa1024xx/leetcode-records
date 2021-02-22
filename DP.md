#### 70.Climbing Stairs ####
###### 思路 ###### 
```
方案数= 最后一步走1阶的方案数 + 最后一步走2阶的方案数
```
###### 代码 ###### 
```
class Solution {
    public int climbStairs(int n) {
        if (n <= 1) {
            return n;
        }
        
        int last = 1, lastLast = 1;
        int now  = 0;
        for (int i = 2; i <= n; i++) {
            now = last + lastLast;
            lastLast = last;
            last = now;
        }
        return now;
    }
}
```

##### Follow Up #####
#### 描述 ####
```
一个小孩爬一个 n 层台阶的楼梯。他可以每次跳 1 步， 2 步 或者 3 步。实现一个方法来统计总共有多少种不同的方式爬到最顶层的台阶。
```
###### 思路 ###### 
```
DP:
每一步的答案可以由三步转移而来。 初始状态和状态转移方程如下： 
- f0 = 1, f1 = 1, f2 = 2;
- fn = f_n-1 + f_n-2 + f_n-3 (n >= 3)
```
###### 代码 ###### 
```
public class Solution {
    public int climbStairs2(int n) {
        if (n <= 1){
            return 1;
        }
        if (n == 2){
            return 2;
        }
        int a = 1, b = 1, c = 2;
        for (int i = 3; i < n + 1; i ++){
            int next = a + b + c;
            a = b;
            b = c;
            c = next;
        }
        return c;
    }
}
```

#### 描述 ####
```
小明准备爬上一个n个台阶的楼梯，当他位于第i级台阶时，他可以往上走1至num[i]级台阶。问小明有多少种爬完楼梯的方法？由于答案可能会很大，所以返回答案对1e9+7取模即可。
```
###### 思路 ###### 
```
首先维护一个差分序列dp[],差分序列的前i项的前缀和sumi就相当于走上第i级台阶的方案数 状态转移的过程就是对于当前位置i,dpi+1+=dpi,dp[i+numi+1]-=dpi,同时维护sumi=sumi-1+dpi，注意中间过程要取模 最后sumn即为答案，由于答案较大，可以使用longlong(C++),long(Java).
```
###### 代码 ###### 
```
public class Solution {
    int[] dp = new int[1000000+2];
    public long Solve(int n, int[] num) {
        // Write your code here
        int mod = (int)(1e9 + 7);
        for (int i = 0; i <= n; i++) dp[i] = 0;
        dp[0] = 1;dp[1] = -1;
        int res = 0;
        for (int i = 0; i < n; i++) {
            res = (res + dp[i]) % mod;
            int L = i + 1, R = Math.min(n + 1, i + num[i] + 1);
            dp[L] = (dp[L] + res) % mod;
            dp[R] = (dp[R] - res) % mod;
        }
        res = (res + dp[n]) % mod;
        if (res < 0) res += mod;
        return res;
    }
}
```

#### 198.House Robber ####
###### 思路 ###### 
```
设 dpi 表示前i家房子最多收益, 答案是 dpn
dp[i] = max(dp[i-1], dp[i-2] + A[i-1])
```
###### 代码 ###### 
```
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        
        long[] res = new long[n + 1];
        res[0] = 0;
        res[1] = nums[0];
        for (int i = 2; i <= n; i++) {
            res[i] = Math.max(res[i - 1], res[i - 2] + nums[i - 1]);
        }
        return (int)res[n];
    }
}
```

##### Follow Up #####
#### 213.House Robber II ####
###### 思路 ###### 
```
DP: 考虑前若干个房子，记录抢最后一个房子或者不抢最后一个房子能抢到的最多的钱 然后交叉更新
```
###### 代码 ###### 
```
class Solution {
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        return Math.max(robber1(nums, 0, nums.length - 2), robber1(nums, 1, nums.length - 1));
    }
    
    private int robber1(int[] nums, int s, int e) {
        int[] res = new int[2];
        if (s == e) {
            return nums[e];
        }
        if (s + 1 == e) {
            return Math.max(nums[s], nums[e]);
        }
        res[s % 2] = nums[s];
        res[(s + 1) % 2] = Math.max(nums[s], nums[s + 1]);
        
        for (int i = s + 2; i <= e; i++) {
            res[i % 2] = Math.max(res[(i - 1) % 2], res[(i - 2) % 2] + nums[i]);
        }
        return res[e % 2];
    }
}
```

#### 337.House Robber III ####
###### 思路 ###### 
```
DP: 首先枚举是否抢第一个房子 考虑前若干个房子，记录抢最后一个房子或者不抢最后一个房子能抢到的最多的钱 然后交叉更新
```
###### 代码 ###### 
```
class Solution {
    public int rob(TreeNode root) {
        int[] ans = dp(root);
        return Math.max(ans[0], ans[1]);
    }
    
    private int[] dp(TreeNode root) {
        if (root == null) {
            return new int[]{0, 0};
        }
        
        int[] left = dp(root.left);
        int[] right = dp(root.right);
        int[] now = new int[2];
        now[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        now[1] = left[0] + right[0] + root.val;
        return now;
    }
}
```

#### 413.Arithmetic Slices ####
###### 思路 ###### 
```
设dpi表示以ai结尾的最长的等差数列长度。 转移方程为
dp[i] = dp[i - 1] + 1 if a[i] - a[i - 1] == a[i - 1] - a[i - 2]
else: dp[i] = 2, 最后的答案为 sum(dp[i] - 2)
```
###### 代码 ###### 
```
class Solution {
    public int numberOfArithmeticSlices(int[] A) {
        if (A == null || A.length <= 2) {
            return 0;
        }
        
        int len = A.length;
        int[] dp = new int[len];
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2; i < len; i++) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
                dp[i] = dp[i - 1] + 1;
            }else {
                dp[i] = 2;
            }
        }
        
        int ans = 0;
        for (int num : dp) {
            if (num > 2) {
                ans += num - 2;
            }
        }
        return ans;
    }
}
```

##### Follow Up #####
#### 446.Arithmetic Slices II - Subsequence ####
###### 思路 ###### 
```
设定状态:
f[i][d]  表示以 A 中下标为 i 的数结尾的, 公差为 d 的等差切片的数量
f2[i][d] 表示以 A 中下标为 i 的数结尾的, 长度为 2 的, 公差为 d 的等差切片的数量
(f2记录的严格来说并不能称为等差切片, 因为长度小于 3, 只是为了描述方便)
状态转移:
对于 i, 枚举 0 <= j < i, d = A[i] - A[j]
f[i][d] += f2[j][d] + f[j][d]   // 以 j 结尾的长度为2的等差切片, 增加了 A[i] 就会变成真正的等差切片
f2[i][d] += 1                   // A[j], A[i] 组成一个长度为 2 的等差切片
答案就是 f 内所有整数的和. 由于公差 d 可能是负数, 可能非常大而且不连续, 所以我们采用哈希表存储.
```
###### 代码 ###### 
```
class Solution {
    public int numberOfArithmeticSlices(int[] A) {
        if (A == null || A.length <= 2) {
            return 0;
        }
        
        Map<Integer, Integer>[] map = new Map[A.length];
        int ans = 0;
        for (int i = 0; i < A.length; i++) {
            map[i] = new HashMap<Integer, Integer>();
            for (int j = 0; j < i; j++) {
                if (Math.abs((long)(A[i] - A[j])) > Integer.MAX_VALUE) {
                    continue;
                }
                int d = A[i] - A[j];
                int mapId = map[i].getOrDefault(d, 0);
                int mapJd = map[j].getOrDefault(d, 0);
                mapId += mapJd + 1;
                map[i].put(d, mapId);
                ans += mapJd;
            }
        }
        return ans;
    }
}
```

#### 64.Minimum Path Sum ####
###### 思路 ###### 
```
DP: f[i][j] 表示从(0,0)到(i,j)的最小数字和

f[0][0] = grid[0][0]
f[i][0] = f[i - 1][0] + grid[i][0] 
f[0][j] = f[0][j - 1] + grid[0][j]

f[i][j] = min{f[i - 1][j], f[i][j - 1]]} + grid[i][j]

=> f[m - 1][n - 1]
```
###### 代码 ###### 
```
class Solution {
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        if (grid[0] == null || grid[0].length == 0) {
            return 0;
        }
        
        int m = grid.length;
        int n = grid[0].length;
        int[][] f = new int[m][n];
        
        f[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            f[i][0] = grid[i][0] + f[i - 1][0];
        }
        for (int j = 1; j < n; j++) {
            f[0][j] = grid[0][j] + f[0][j - 1];
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                f[i][j] = Math.min(f[i - 1][j], f[i][j - 1]) + grid[i][j];
            }
        }
        
        return f[m - 1][n - 1];
    }
}
```


##### Follow Up #####
#### 描述 ####
```
给出一个 n * m的矩阵，每个点有一个权值，从矩阵左下走到右上(可以走四个方向)，让你找到一条路径 使得该路径所路过的权值和最小，输出最小权值和。
```
###### 思路 ###### 
```
DFS
```
###### 代码 ###### 
```
public class Solution {
    int[] dx = {0, 1, -1, 0};
    int[] dy = {1, 0, 0, -1};
    public int minPathSumII(int[][] matrix) {
        // Write your code here
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] distance = new int[n][m];
        for(int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                distance[i][j] = Integer.MAX_VALUE;
            }
        }
        distance[n - 1][0] = matrix[n - 1][0];
        dfs(n - 1, 0, distance, matrix, matrix[n - 1][0], 0, m - 1);
        return distance[0][m - 1];
    }
    private void dfs(int stx, int sty, int[][] dis, int[][] matrix, int cursum, int edx, int edy) {
        dis[stx][sty] = cursum;
        int n = matrix.length;
        int m = matrix[0].length;
        for (int i = 0; i < 4; i++) {
            int nx = stx + dx[i];
            int ny = sty + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m || dis[nx][ny] < cursum + matrix[nx][ny]) {
                continue;
            }
            dfs(nx, ny, dis, matrix, cursum + matrix[nx][ny], 0, m - 1);
        }
    }
}
```

#### 描述 ####
```
给定一个只含整数的 m x n 网格，找到一条从左上角到右下角的可以使数字和最小的路径。
```
###### 思路 ###### 
```
DP
```
###### 代码 ###### 
```
public class Solution {
    public int minimumPathSumIII(int[][] grid) {
        int dp[][] = new int[1010][1010];
        int n = grid.length;
        int m = grid[0].length;
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++){
                    if(i == 0 && j == 0) dp[i][j] = grid[0][0];
                else if(i == 0) dp[i][j] = dp[i][j-1] + grid[i][j];
                else if(j == 0) dp[i][j] = dp[i - 1][j] + grid[i][j];
                else{
                    dp[i][j] = Math.min(dp[i][j - 1] + grid[i][j], dp[i - 1][j] + grid[i][j]);
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + grid[i][j], dp[i][j]);
                }
            }
        return dp[n - 1][m - 1];
    }
}
```

#### 542.01 Matrix ####
###### 思路 ###### 
```
从左上出发推一遍。 从右下出发推一遍。
```
###### 代码 ###### 
```
class Solution {
    public int[][] updateMatrix(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] dp = new int[n][m];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (matrix[i][j] == 0) {
                    dp[i][j] = 0;
                }else {
                    dp[i][j] = Integer.MAX_VALUE - 2000;
                }
                if (i > 0) {
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][j] + 1);
                }
                if (j > 0) {
                    dp[i][j] = Math.min(dp[i][j], dp[i][j - 1] + 1);
                }
            }
        }
        
        for (int i = n - 1; i >= 0; i--) {
            for (int j = m - 1; j >= 0; j--) {
                if (dp[i][j] > 0) {
                    if (i < n - 1) {
                        dp[i][j] = Math.min(dp[i][j], dp[i + 1][j] + 1);
                    }
                    if (j < m - 1) {
                        dp[i][j] = Math.min(dp[i][j], dp[i][j + 1] + 1);
                    }
                }
            }
        }
        
        return dp;
    }
}
```

#### 221.Maximal Square ####
###### 思路 ###### 
```
设dp[i][j] 为以(i, j)为右下角的最大正方形的边长
- dp[i][0] = dp[0][j] = matrix[i][j]
- dp[i][j] = 0 if matrix[i][j] = 0
- if matrix[i][j] = 1, dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
- ans = max(dp[i][j])^2
```
###### 代码 ###### 
```
class Solution {
    public int maximalSquare(char[][] matrix) {
        int ans = 0;
        int n = matrix.length;
        int m;
        if (n > 0) {
            m = matrix[0].length;
        }else {
            return ans;
        }
        
        int[][] dp = new int[n][m];
        for (int i = 0; i < n; i++) {
            dp[i][0] = matrix[i][0] - '0';
            ans = Math.max(ans, dp[i][0]);
        }
        for (int j = 0; j < m; j++) {
            dp[0][j] = matrix[0][j] - '0';
            ans = Math.max(ans, dp[0][j]);
        }
        
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                if (matrix[i][j] == '0') {
                    dp[i][j] = 0;
                }else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
                ans = Math.max(ans, dp[i][j]);
            }
        }
        
        return ans * ans;
    }
}
```

##### Follow Up #####
#### 描述 ####
```
给出一个只有 0 和 1 组成的二维矩阵，找出最大的一个子矩阵，使得这个矩阵对角线上全是 1 ，其他位置全是 0 .
```
###### 思路 ###### 
```
dp[i][j] = 1 + min {leftZeros[i][j], upZeros[i][j], dp[i - 1][j - 1] }
leftZeros[i][j] 表示在matrix[i][j]左边连续0的最大数目
upZeros[i][j] 表示在matrix[i][j]上面连续0的最大数目
要求一个矩阵斜对角全为1，其余为0的矩阵，除了右下角，那么他的底边肯定全为0，斜边全为1，右侧边全为0，三者长度相同才能保证这个矩阵是满足要求的，那么我们取三者中的最小值，就能保证所选的矩阵是一个满足条件的矩阵
```
###### 代码 ###### 
```
public class Solution {
    public int maxSquare2(int[][] matrix) {
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0){
            return 0;
        }
        int n = matrix.length; int m = matrix[0].length;
        int[][] leftZeros = new int[n][m];
        int[][] upZeros = new int[n][m];
        // 初始化leftZeros和upZeros，统计每个位置的左边和上面有多少连续0
        for(int i = 0; i < n; i++){
            leftZeros[i][0] = 0;
        }
        for(int j = 0; j < m; j++){
            upZeros[0][j] = 0;
        }
        for(int i = 0; i < n; i++){
            for(int j = 1; j < m; j++){
                if(matrix[i][j - 1] == 0){
                    leftZeros[i][j] = leftZeros[i][j - 1] + 1;
                }    
                else{
                    leftZeros[i][j] = 0;
                }
            }
        }
        for(int i = 1; i < n; i ++){
            for(int j = 0; j < m; j++){
                if(matrix[i - 1][j] == 0){
                    upZeros[i][j] = upZeros[i - 1][j] + 1;
                }
                else{
                    upZeros[i][j] = 0;
                }                
            }
        }

        int[][] dp = new int[n][m];
        //初始化dp数组
        for(int i = 0; i < n; i++){
            dp[i][0] = matrix[i][0];    
        }
        for(int j = 0; j < m; j++){
            dp[0][j] = matrix[0][j];
        }
        //状态转移。记录每个点对角线到此点的最大长度。
        for(int i = 1; i < n; i++){
            for(int j = 1; j < m; j++){
                if(matrix[i][j] == 0){
                    dp[i][j] = 0;
                }
                else{
                    dp[i][j] = Math.min(Math.min(leftZeros[i][j], upZeros[i][j]), dp[i - 1][j - 1]) + 1;
                }
            }
        }
        int max = 0;
        //遍历dp查找最大矩阵的对角线长度
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                max = Math.max(max, dp[i][j]);    
            }
        }
        //答案就是最长对角线长度的平方
        return max * max;
    }
}
```

#### 279.Perfect Squares ####
###### 思路 ###### 
```
DP:
首先我们对dp数组赋予初值，对于每个完全平方数的f=1。
利用记忆化搜索来完成查找。对于i，我们考虑i的前继状态，也就是哪几个状态可以加上一个完全平方数抵达i。
对于所有能够抵达i的状态，取他们的最小值+1即可。
f[i] = min(f[i], f[i - j * j] + 1)

Math:
根据四平方和定理，一个数能被四个数之内的平方和表示。
我们直接循环，判断这个数能否被1~到3个数的平方和表示即可，若不行，直接输出4
```
###### 代码 ###### 
```
class Solution {
    public int numSquares(int n) {
        int[] f = new int[n + 1];
        Arrays.fill(f, Integer.MAX_VALUE);
        for (int i = 0; i * i <= n; i++) {
            f[i * i] = 1;
        }
        for (int i = 0; i <= n; i++) {
            for (int j = 1; j * j <= i; j++) {
                f[i] = Math.min(f[i], f[i - j * j] + 1);
            }
        }
        return f[n];
    }
}

class Solution {
    public int numSquares(int n) {
         while (n % 4 == 0)

            n /= 4;

        if (n % 8 == 7)

            return 4;

        for (int i = 0; i * i <= n; ++i) {

            int j = (int)Math.sqrt(n * 1.0 - i * i);

            if (i * i + j * j == n) {

                int res = 0;

                if (i > 0)

                    res += 1;

                if (j > 0)

                    res += 1;

                return res;

            }

        }

        return 3;
    }
}
```

#### 91.Decode Ways ####
###### 思路 ###### 
```
DP: dp[i] 表示字符串的前i位解码有多少种解码方式
- dp[0] = dp[1] = 1
- if s[i - 1] 是1到9，dp[i] += dp[i - 1]
- if s[i - 1] 和 s[i - 1] 表示的数是10-26， dp[i] += dp[i - 2]
- 若以上两种情况都不满足，直接返回答案 0
- 若 s 以 0 开头，直接返回 0
```
###### 代码 ###### 
```
class Solution {
    public int numDecodings(String s) {
        if (s == null || s.length() == 0 || s.charAt(0) == '0') {
            return 0;
        }
        
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            if (s.charAt(i - 1) != '0') {
                dp[i] += dp[i - 1];
            }
            if (s.charAt(i - 2) != '0' && (s.charAt(i - 2) - '0') * 10 + (s.charAt(i - 1) - '0') <= 26) {
                dp[i] += dp[i - 2];
            }
            if (dp[i] == 0) {
                return 0;
            }
        }
        return dp[n];
    }
}
```

##### Follow Up #####
#### 639.Decode Ways II ####
###### 思路 ###### 
```
解法大同小异，最后两位的组合比较麻烦，情况分为 (星,星) (星,数字) (数字,星) (数字，数字) 带 星 的都需要考虑一共有多少种可能。
```
###### 代码 ###### 
```
class Solution {
    public int numDecodings(String ss) {
        char[] s = ss.toCharArray();
        int n = ss.length();
        long MOD = 1000000007;
        
        long[] f= new long[n + 1];
        f[0] = 1;
        for (int i = 1; i <= n; i++) {
            f[i] = f[i - 1] * cnt1(s[i - 1]);
            if (i > 1) {
                f[i] += f[i - 2] * cnt2(s[i - 2], s[i - 1]);
            }
            f[i] %= MOD;
        }
        return (int)f[n];
    }
    
    private int cnt1(char c) {
        if (c == '0') {
            return 0;
        }
        if (c != '*') {
            return 1;
        }
        return 9;
    }
    
    private int cnt2(char c2, char c1) {
        if (c2 == '0') {
            return 0;
        }
        
        if (c2 == '1') {
            if (c1 == '*') {
                return 9;
            }
            return 1;
        }
        
        if (c2 == '2') {
            if (c1 == '*') {
                return 6;
            }
            if (c1 <= '6') {
                return 1;
            }
            return 0;
        }
        
        if (c2 >= '3' && c2 <= '9') {
            return 0;
        }
        // c2 == '*'
        if (c1 >= '0' && c1 <= '6') {
            return 2;
        }
        if (c1 >= '7' && c1 <= '9') {
            return 1;
        }
        
        return 15;
    }
}
```

#### 139.Word Break ####
###### 思路 ###### 
```
DP: dp[i] 表示前 i 个字符是否能够被划分为若干个单词
dp[i] = if dp[j] && j + 1 ~ i 是一个单词
```
###### 代码 ###### 
```
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.length() == 0) {
            return true;
        }
        int maxLen = 0;
        for (String word : wordDict) {
            maxLen = Math.max(maxLen, word.length());
        }
        
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        
        for (int i = 1; i <= n; i++) {
            for (int l = 1; l <= maxLen; l++) {
                if (i < l) {
                    break;
                }
                if (!dp[i - l]) {
                    continue;
                }
                String word = s.substring(i - l, i);
                if (wordDict.contains(word)) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
}
```

##### Follow Up #####
#### 140.Word Break II ####
###### 思路 ###### 
```
DFS: 利用fi记录以i为起点的每个片段的终点j，并且片段要在字典中，然后从0位置开始搜索，每次给当前片段加上空格，然后以当前片段的末尾作为下一次搜索的头部，避免不必要的搜索。
```
###### 代码 ###### 
```
class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
        Map<String, List<String>> done = new HashMap<>();
        done.put("", new ArrayList<String>());
        done.get("").add("");
        
        return dfs(s, wordDict, done);
    }
    
    private List<String> dfs(String s, List<String> wordDict, Map<String, List<String>> done) {
        if (done.containsKey(s)) {
            return done.get(s);
        }
        List<String> res = new ArrayList<String>();
        
        for (int len = 1; len <= s.length(); len++) {
            String s1 = s.substring(0, len);
            String s2 = s.substring(len);
            
            if (wordDict.contains(s1)) {
                List<String> s22 = dfs(s2, wordDict, done);
                for (String str : s22) {
                    if (str.equals("")) {
                        res.add(s1);
                    }else {
                        res.add(s1 + " " + str);
                    }
                }
            }
        }
        done.put(s, res);
        return res;
    }
}
```

#### 描述 ####
```
给出一个单词表和一条去掉所有空格的句子，根据给出的单词表添加空格, 返回可以构成的句子的数量, 保证构成的句子中所有的单词都可以在单词表中找到.
```
###### 思路 ###### 
```
DP: dp[i][j]表示从字典dict中组成子串str[i:j+1]有多少种方法
dp[i][j] = Sum(dp[i][k] * dp[k + 1][j])
```
###### 代码 ###### 
```
public class Solution {
    public int wordBreak3(String s, Set<String> dict) {
        if (s == null ||s.length() == 0 || dict == null || dict.size() == 0) {
            return 0;
        }
        //将字符全部转化为小写，并将dict转换成hash_stet存储，降低判断子串存在性的时间复杂度
        s = s.toLowerCase();
        Set<String> set = new HashSet<String>();
        for (String word : dict) {
            String str = word.toLowerCase();
            set.add(str);
        }
        
        //dp[i]表示s[0:i](不含s[i])的拆分方法数
        int len = s.length();
        int[] dp = new int[len + 1];

        //dp[0]表示空串的拆分方法数
        dp[0] = 1;
        
        for (int i = 0; i < len; i++) {
            for (int j = i; j < len; j++) {
                //若存在匹配，则进行状态转移
                if (set.contains(s.substring(i, j + 1))) {
                    dp[j + 1] += dp[i];
                }
            }
        }
        return dp[len];
    }
}

```

#### 300.Longest Increasing Subsequence ####
###### 思路 ###### 
```
DP: dp[i] 表示以第i个数字为结尾的最长上升子序列的长度。 
对于每个数字，枚举前面所有小于自己的数字 j，dp[i] = max{dp[j]} + 1. 如果没有比自己小的，dp[i] = 1;
```
###### 代码 ###### 
```
class Solution {
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        
        int ans = 0;
        for (int num : dp) {
            ans = Math.max(ans, num);
        }
        return ans;
    }
}
```

#### 1143.Longest Common Subsequence ####
###### 思路 ###### 
```
DP: 将 X 和 Y 的最长公共子序列记为LCS(X,Y)
- 若 Xn = Ym, 即X的最后一个元素与Y的最后一个元素相同，这说明该元素一定位于公共子序列中, 现在只需要找：LCS(Xn-1，Ym-1)。
- 若 Xn != Ym, LCS = max( LCS(Xn-1，Ym), LCS(Xn，Ym-1) )   
```
###### 代码 ###### 
```
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int n = text1.length();
        int m = text2.length();
        int[][] dp = new int[n + 1][m + 1];
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - 1] + 1);
                }
            }
        }
        
        return dp[n][m];
    }
}
```

##### Follow Up #####
#### 描述 ####
```
给定两个序列 'P'和'Q'。你的任务是，我们可以对这对'P'这个序列修改不超过k个元素到任意的值，并要求两个修改后序列的最长公共子序列最长。
```
###### 思路 ###### 
```
DP: dp[m + 1][n + 1][k + 1] 当前值还有修改的机会次数 k 次 
```
###### 代码 ###### 
```
public class Solution {
    public int longestCommonSubsequence2(int[] P, int[] Q, int k) {
        int m = P.length;
        int n = Q.length;
        int[][][] dp = new int[m + 1][n + 1][k + 1];
        for (int i = 1; i < m + 1; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                if (P[i - 1] == Q[j - 1])
                    dp[i][j][0] = dp[i - 1][j - 1][0] + 1;
                else
                    dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i][j - 1][0]);
            }
        }

        for (int i = 1; i < m + 1; ++i) 
        {
            for (int j = 1; j < n + 1; ++j) 
            {
                for (int g = 1; g < k + 1; ++g) 
                {
                    if (P[i - 1] != Q[j - 1])
                        dp[i][j][g] = Math.max(dp[i - 1][j][g], Math.max(dp[i][j - 1][g], dp[i - 1][j - 1][g - 1] + 1));
                    else
                        dp[i][j][g] = Math.max(dp[i - 1][j][g], Math.max(dp[i][j - 1][g], dp[i - 1][j - 1][g] + 1));
                }
            }
        }
        return dp[m][n][k];
    }
}
```

#### 描述 ####
```
给出1-n的两个排列P1和P2，求它们的最长公共子序列。请将复杂度控制在O(nlogn)
```
###### 思路 ###### 
```
找出B数组里的每一个数在A数组里的位置，然后把这些位置放到一个新的数组C。
C的最长上升子序列，就是A和B的最长公共子序列。
剩下的就是照搬 Longest Increasing Subsequence 的二分法代码。
举例：
    0   1   2   3   4   5   6   7   8
A   6   9   4   2   8   1   3   5   7
B   8   1   2   4   5   3   7   9   6
C   4   5   3   2   7   6   8   1   0
C的最长上升子序列是4568，对应A数组相应位置的8137，就是A和B的最长公共子序列。
```
###### 代码 ###### 
```
class Solution:
    """
    @param A: 
    @param B: 
    @return: nothing
    """
    def longestCommonSubsequenceIII(self, A, B):
        indexes = {num : index for index, num in enumerate(A)}
        C = [indexes[num] for num in B]
        return self.longestIncreasingSubsequence(C)
    
    def longestIncreasingSubsequence(self, nums):
        ret = [sys.maxsize] * len(nums)
        for num in nums:
            index = self.binary_search(ret, num)
            ret[index] = num
        LIS = len(nums) - 1
        while LIS >= 0 and ret[LIS] == sys.maxsize:
            LIS -= 1
        return LIS + 1
    
    def binary_search(self, arr, num):
        start, end = 0, len(arr) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if arr[mid] >= num:
                end = mid
            else:
                start = mid
        if arr[start] >= num:
            return start
        return end
```

#### 416.Partition Equal Subset Sum ####
###### 思路 ###### 
```
等价与背包问题，能否背到总价值的一半。 01背包即可
```
###### 代码 ###### 
```
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % 2 == 1) {
            return false;
        }
        sum /= 2;
        boolean[] dp = new boolean[sum + 1];
        Arrays.fill(dp, false);
        dp[0] = true;
        for (int i = 0; i < nums.length; i++) {
            for (int j = sum; j >= nums[i]; j--) {
                dp[j] |= dp[j - nums[i]];
            }
        }
        return dp[sum];
    }
}
```

#### 474.Ones and Zeroes ####
###### 思路 ###### 
```
DP: dp[i][j][k]表示前 i 个字符串，使用 j 个 0, k 个 1 最多能选择的个数
dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - cnt_0(str[i])][k - cnt_1(str[i])])
```
###### 代码 ###### 
```
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int[][][] dp = new int[strs.length + 1][m + 1][n + 1];
        for (int i = 1; i <= strs.length; i++) {
            int[] cost = count(strs[i - 1]);
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= n; k++) {
                    if (j >= cost[0] && k >= cost[1]) {
                        dp[i][j][k] = Math.max(dp[i - 1][j][k], dp[i - 1][j - cost[0]][k - cost[1]] + 1);
                    }else {
                        dp[i][j][k] = dp[i - 1][j][k];
                    }
                }
            }
        }
        return dp[strs.length][m][n];
    }
    
    private int[] count(String str) {
        int[] cost = new int[2];
        for (int i = 0; i < str.length(); i++) {
            cost[str.charAt(i) - '0']++;
        }
        return cost;
    }
}

优化空间

class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        if (strs == null || strs.length == 0) {
            return 0;
        }
        
        int len = strs.length;
        int[] cnt0 = new int[len];
        int[] cnt1 = new int[len];
        int i, j, k;
        for (i = 0; i < len; i++) {
            char[] s = strs[i].toCharArray();
            cnt0[i] = cnt1[i] = 0;
            for (j = 0; j < s.length; j++) {
                if (s[j] == '0') {
                    cnt0[i]++;
                }else {
                    cnt1[i]++;
                }
            }
        }
        
        int[][] f = new int[m + 1][n + 1];
        for (i = 0; i <= len; i++) {
            for (j = m; j >= 0; j--) {
                for (k = n; k >= 0; k--) {
                    if (i == 0) {
                        f[j][k] = 0;
                        continue;
                    }
                    if (j >= cnt0[i - 1] && k >= cnt1[i - 1]) {
                        f[j][k] = Math.max(f[j][k], f[j - cnt0[i - 1]][k - cnt1[i - 1]] + 1);
                    }
                }
            }
        }
        return f[m][n];
    }
}
```

#### 322.Coin Change ####
###### 思路 ###### 
```
完全背包问题, DP: 设dpi表示使用前i个硬币，总金额为j时需要的最少硬币数量。
dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - k * coin[i]] + k) (0 <= k * coin[i] <= j)
```
###### 代码 ###### 
```
class Solution {
    public int coinChange(int[] coins, int amount) {
        final int INF = 0x3f3f3f3f;
        int[] f = new int[amount + 1];
        int n = coins.length;
        f[0] = 0;
        for (int i = 1; i <= amount; i++) {
            f[i] = Integer.MAX_VALUE;
            for (int j = 0; j < n; j++) {
                if (i >= coins[j] && f[i - coins[j]] != Integer.MAX_VALUE) {
                    f[i] = Math.min(f[i], f[i - coins[j]] + 1);
                }
            }
        }
        return f[amount] == Integer.MAX_VALUE ? -1 : f[amount];
    }
}
```

##### Follow Up #####
#### 518.Coin Change II ####
###### 思路 ###### 
```
DP: dp[i]表示能否取到两个集合之差为i的情况
dp[i] += dp[i - coins[k]];
```
###### 代码 ###### 
```
class Solution {
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] += dp[j - coins[i]];
            }
        }
        return dp[amount];
    }
}
```

#### 描述 ####
```
给出不同面额的硬币，每一个面额的硬币数目，以及一个总金额. 写一个方法来计算给出的总金额可以换取的最少的硬币数量. 如果已有硬币的任意组合均无法与总金额面额相等, 那么返回 -1.
```
###### 思路 ###### 
```
参考 322 : dp[ i - k * coins[i]] + k
```
###### 代码 ###### 
```
class Solution:
    def coin_change(self, coins, nums, amount):
        dp = [0] + [sys.maxsize]*amount 
        for c, num in zip(coins, nums):
            for j in range(1, num+1):
                for i in reversed(range(c*num, amount+1)):
                    dp[i] = min(dp[i], num + dp[i-c*j])
        return dp[amount] if dp[amount] != sys.maxsize else -1
```

#### 72.Edit Distance ####
###### 思路 ###### 
```
二维DP：f[i][j]为 word1 前 i 个字符到 word2 的前 j 个字符的转化的最小步
状态转移方程：假设对于f[i][j]以前的之都已知，考虑f[i][j]的情形。
若 word1[i] = word2[j]， 那么说明只要 word1 的前 i - 1 个能转换到 word2 的前 j - 1 个即可， 所以 f[i][j] = f[i - 1][j - 1]
反之，若不等，我们就要考虑以下情形了:
- 给word1插入一个和word2最后的字母相同的字母，这时word1和word2的最后一个字母就一样了，此时编辑距离等于1（插入操作） + 插入前的word1到word2去掉最后一个字母后的编辑距离 f[i][j - 1] + 1
- 删除word1的最后一个字母，此时编辑距离等于1（删除操作） + word1去掉最后一个字母到word2的编辑距离 f[i - 1][j] + 1
- 把word1的最后一个字母替换成word2的最后一个字母，此时编辑距离等于 1（替换操作） + word1和word2去掉最后一个字母的编辑距离。为f[i - 1][j - 1] + 1
三者取最小值即可。
```
###### 代码 ###### 
```
class Solution {
    public int minDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i < m + 1; i++) {
            dp[0][i] = i;
        }
        for (int i = 0; i < n + 1; i++) {
            dp[i][0] = i;
        }
        
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < m + 1; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }else {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]);
                }
            }
        }
        return dp[n][m];
    }
}
```

##### Follow Up #####
#### 描述 ####
```
给定两个字符串 S 和 T, 判断T是否可以通过对S做刚好一次编辑得到。
每次编辑可以选择以下任意一个操作：
 - 在S的任意位置插入一个字符
 - 删除S中的任意一个字符
 - 将S中的任意字符替换成其他字符
```
###### 思路 ###### 
```
模拟：根据题意，我们只能对字符串操作一次， 那么我们根据情况可以分为三种情况：
- 两个字符串的长度之差 大于 1， 直接返回 false
- 长度之差等于 1， 判断长的字符串删掉不一样的字符，剩余的字符串是否相同
- 长度之差等于 1， 判断不相同的字符个数，若超过一个返回 false
```
###### 代码 ###### 
```
public class Solution {
    public boolean isOneEditDistance(String s, String t) {
        if (s.length() > t.length()) {
            return isOneEditDistance(t, s);
        }
        int diff = t.length() - s.length();

        if (diff > 1) {
            return false;
        }
        if (diff == 0) {
            int cnt = 0;
            for (int i = 0; i < s.length(); i++) {
                if (t.charAt(i) != s.charAt(i)) {
                    cnt++;
                }
            }
            return (cnt == 1);
        }
        if (diff == 1) {
            for (int i = 0; i < s.length(); i++) {
                if (t.charAt(i) != s.charAt(i)) {
                    return (s.substring(i).equals(t.substring(i + 1)));
                }
            }
        }
        return true;
    }
}
```

#### 650.2 Keys Keyboard ####
###### 思路 ###### 
```
这是个经典的数学马甲问题， 分解质因数。 如果n是个质数， 那么答案一定是N次;
如果N是个合数： 一个合数肯定可以分解成两个质数的乘积:
n = i * j (i > 1, j > 1) 可以推导 (i - 1)(j - 1) >= 1 得到 i * j >= i + j,
所以两个质数相加的操作次数至少不会超过直接操作 n(n = i * j) 次
那么这个问题就是一个典型的分解质因数的问题。
```
###### 代码 ###### 
```
class Solution {
    public int minSteps(int n) {
        int res = 0;
        for (int i = 2; n > 1; i++) {
            while (n % i == 0) {
                res += i;
                n /= i;
            }
        }
        return res;
    }
}
```

#### 10.Regular Expression Matching ####
###### 思路 ###### 
```
DP:
- 采用字典hash记录各处字符匹配情况，dfs递归进行搜索，记忆化剪枝
- dp使用数组记忆化，实现合理的状态转移
- 当前p串中有 ，就有两种选择，然后 可以不去匹配，直接用p串的下一个匹配当前s串字符，或者重复p串的上一个字符匹配。
- 可以匹配任意字符
```
###### 代码 ###### 
```
[Solution 1]
class Solution {
    public boolean isMatch(String s, String p) {
        char[] ss = s.toCharArray();
        char[] pp = p.toCharArray();
        boolean[][] match = new boolean[s.length() + 1][p.length() + 1];
        
        for (int i = 0; i <= s.length(); i++) {
            for (int j = 0; j <= p.length(); j++) {
                if (i == 0 && j == 0) {
                    match[i][j] = true;
                    continue;
                }
                
                if (j == 0) {
                    match[i][j] = false;
                    continue;
                }
                
                match[i][j] = false;
                if (pp[j - 1] != '*') {
                    if (i > 0 && (ss[i - 1] == pp[j - 1] || pp[j - 1] == '.')) {
                        match[i][j] |= match[i - 1][j - 1];
                    }
                }else {
                    if (j > 1) {
                        match[i][j] |= match[i][j - 2];
                    }
                    if (i > 0 && (pp[j - 2] == '.' || ss[i - 1] == pp[j - 2])) {
                        match[i][j] |= match[i - 1][j];
                    }
                }
            }
        }
        
        return match[s.length()][p.length()];
    }
}

[Solution 2]
class Solution {
    public boolean isMatch(String s, String p) {
        if (p == null || p.length() == 0) {
            return (s == null || s.length() == 0);
        }
        
        boolean firstMatch = (s.length() != 0) && (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.');
        if (p.length() >= 2 && p.charAt(1) == '*') {
            // null | (a*)
            return isMatch(s, p.substring(2)) || (firstMatch && isMatch(s.substring(1), p));
        }else {
            return firstMatch && isMatch(s.substring(1), p.substring(1));
        }
    }
}
```

### Best Time to Buy and Sell Stock ###
#### 309.Best Time to Buy and Sell Stock with Cooldown ####
###### 思路 ###### 
```
设置一个sell[i]数组表示前i天交易，最后一次为卖所获得的最大利润。
设置一个buy[i]数组表示前i天交易，最后一次为买所获得的最大利润。
转移方程为：sell[i] = max(sell[i - 1], buy[i - 1] + price[i])
           buy[i] = nax(buy[i - 1], sell[i - 2] - price[i])
```
###### 代码 ###### 
```
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length < 2) {
            return 0;
        }
        
        int[] sell = new int[prices.length];
        int[] buy = new int[prices.length];
        sell[0] = 0;
        buy[0] = -prices[0];
        
        for (int i = 1; i < prices.length; i++) {
            sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
            buy[i] = Math.max(buy[i - 1], (i > 1 ? sell[i - 2] - prices[i] : - prices[i]));
        }
        return sell[prices.length - 1];
    }
}
```

#### 53.Maximum Subarray ####
###### 思路 ###### 
```
dp[i]表示截止到下标i的元素的最大子数组: 
- 要么是这个数自身
- 要么是这个数加上前面的dp[i-1]
然后打擂台记下最大的dp[i]值即可
```
###### 代码 ###### 
```
class Solution {
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int max = dp[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}
```


##### Follow Up #####
#### Maximum Subarray II ####
###### 描述 ###### 
```
给定一个整数数组，找出两个 不重叠 子数组使得它们的和最大。
每个子数组的数字在数组中的位置应该是连续的。
返回最大的和。
```
###### 思路 ###### 
```
在求最大子段和的基础上，计算出前后缀的最大子段和，就可以枚举分界点来计算结果。
对于前缀的最大子段和，我们可以先求以i位置为结尾的最大子段和的值leftMax[i], 后缀的值也同理进行计算。
```
###### 代码 ###### 
```
public class Solution {
    public int maxTwoSubArrays(List<Integer> nums) {
        int n = nums.size();
        
        // 计算以i位置为结尾的前后缀最大连续和
        List<Integer> leftMax = new ArrayList(nums);
        List<Integer> rightMax = new ArrayList(nums);
        
        for (int i = 1; i < n; i++) {
            leftMax.set(i, Math.max(nums.get(i), nums.get(i) + leftMax.get(i - 1)));
        }
        
        for (int i = n - 2; i >= 0; i--) {
            rightMax.set(i, Math.max(nums.get(i), nums.get(i) + rightMax.get(i + 1)));
        }
        
        // 计算前后缀部分的最大连续和
        List<Integer> prefixMax = new ArrayList(leftMax);
        List<Integer> postfixMax = new ArrayList(rightMax);
        
        for (int i = 1; i < n; i++) {
            prefixMax.set(i, Math.max(prefixMax.get(i), prefixMax.get(i - 1)));
        }
        
        for (int i = n - 2; i >= 0; i--) {
            postfixMax.set(i, Math.max(postfixMax.get(i), postfixMax.get(i + 1)));
        }
        
        int result = Integer.MIN_VALUE;
        for (int i = 0; i < n - 1; i++) {
            result = Math.max(result, prefixMax.get(i) + postfixMax.get(i + 1));
        }
        
        return result;
    }
}
```

#### Maximum Subarray III ####
###### 描述 ###### 
```
给定一个整数数组和一个整数 k，找出 k 个不重叠子数组使得它们的和最大。每个子数组的数字在数组中的位置应该是连续的。
```
###### 思路 ###### 
```
用两个二维数组 local 和 global 分别来记录局部最优解和全局最优解：局部最优解就是必须选取当前元素的最优解，全局最优解就是不一定选取当前元素的最优解; 
local[i][j] 表示整数数组nums的前i个元素被分成j个不重叠子数组时的最大值（必须选取元素nums[i]), global[i][j]表示整数数组nums的前i个元素被分成j个不重叠子数组时的最大值（不一定选取元素nums[i]);

if i == j:
   local[i][j] = local[i - 1][j - 1] + nums[i]
   global[i][j] = global[i - 1][j - 1] + nums[i]
else:
   local[i][j] = max(local[i - 1][j], global[i - 1][j - 1]) + nums[i]
   global[i][j] = max(global[i - 1][j], local[i][j])

==>
if i == j:
   local[j] = local[j - 1] + nums[i]
   global[j] = global[j - 1] + nums[i]
else:
   local[j] = max(local[j], global[j - 1]) + nums[i]
   global[j] = max(global[j], local[j])

// nums[i]表示的是数组nums的第几项，若下标从0开始，则nums[i]表示为nums[i-1]。
```
###### 代码 ###### 
```
public class Solution {
    public int maxSubArray(int[] nums, int k) {
        int n = nums.length;
        if (nums == null || n < k){
            return 0;
        }
        
        int[][] local = new int[n + 1][k + 1];
        int[][] global = new int[n + 1][k + 1];
        
        for (int i = 1; i <= n; i++){
            for (int j = Math.min(i, k); j > 0; j--){
                if (i == j){
                    local[i][j] = local[i - 1][j - 1] + nums[i - 1];
                    global[i][j] = global[i - 1][j - 1] + nums[i - 1];
                }else {
                    local[i][j] = Math.max(local[i - 1][j], global[i - 1][j - 1]) + nums[i - 1];
                    global[i][j] = Math.max(global[i - 1][j], local[i][j]);
                }
            }
        }
        return global[n][k];
    }
}
```

#### 描述 ####
```
给定一个整数数组，找到长度大于或等于 k 的连续子序列使它们的和最大，返回这个最大的和，如果数组中少于k个元素则返回 0
```
###### 思路 ###### 
```
对nums数组求出它的前缀和数组pre
右端点从k到n(nums长度)遍历，对于每一个右端点，计算以此为右端点连续子序列最大的和，记录下最大的那个值即答案
```
###### 代码 ###### 
```
public class Solution {
    public int maxSubarray4(int[] nums, int k) {
        int n = nums.length;
        if (nums == null || n < k){
            return 0;
        }
        
        int rightSum = 0, leftSum = 0, leftMinSum = 0;
        for (int i = 0; i < k; i++){
            rightSum += nums[i];
        }
        
        int result = rightSum;
        for (int i = k; i < n; i++){
            rightSum += nums[i];
            leftSum += nums[i - k];
            leftMinSum = Math.min(leftMinSum, leftSum);
            result = Math.max(result, rightSum - leftMinSum);
        }
        
        return result;
    }
}
```

#### 描述 ####
```
给定一个整数数组，找到长度在 k1 与 k2 之间(包括 k1, k2)的子数组并且使它们的和最大，返回这个最大值，如果数组元素个数小于 k1 则返回 0
```
###### 思路 ###### 
```
滑动窗口: deque维护子数组，每一次窗口滑动则：前缀和 - 队头，最后得出最大值。
```
###### 代码 ###### 
```
public class Solution {
    public int maxSubarray5(int[] nums, int k1, int k2) {
        // Write your code here
        int n = nums.length;
        if (n < k1)
            return 0;

        int result = Integer.MIN_VALUE;

        int[] sum = new int[n + 1];
        sum[0] = 0;
        LinkedList<Integer> queue = new LinkedList<Integer>();

        for (int i = 1; i <= n; ++i) {
            sum[i] = sum[i - 1] + nums[i - 1];

            if (!queue.isEmpty() && queue.getFirst() < i - k2) {
                queue.removeFirst();
            }
            if (i >= k1) {
                while (!queue.isEmpty() && sum[queue.getLast()] > sum[i - k1]) {
                    queue.removeLast();
                }
                queue.add(i - k1);
            }

            // [i - k2, i - k1]
            if (!queue.isEmpty() && sum[i] - sum[queue.getFirst()] > result) {
                result = Math.max(result, sum[i] - sum[queue.getFirst()]);
            }


        }
        return result;
    }
}
```

#### 描述 ####
```
给出一个整数数组，找出异或值最大的子数组。
```
###### 思路 ###### 
```
Using Trie data structure: 
1 saving the xor prefix into Trie
2 for each xor prefix, iterate from the root of Trie to find the current maximum subarray xor value.
```
###### 代码 ###### 
```
public class Solution {
    class TrieNode{
        public int label;
        public TrieNode[] children;
        TrieNode(int label){
            this.label = label;
            this.children = new TrieNode[2];
        }
    }
    public int maxXorSubarray(int[] nums) {
        if(nums == null || nums.length == 0) return 0;
        int max = 0;
        int pre_xor = 0;
        TrieNode root = new TrieNode(0);
        insert(root, 0);
        for(int i=0;i<nums.length;i++){
            pre_xor ^= nums[i];
            insert(root, pre_xor);
            max = Math.max(max, searchAndFindMax(root, pre_xor));
        }
        return max;
    }
    
    private void insert(TrieNode root, int num){
        TrieNode cur = root;
        for(int i=31;i>=0;i--){
            int bit = num >> i & 1;
            if(cur.children[bit] == null){
                cur.children[bit] = new TrieNode(bit);
            }
            cur = cur.children[bit];
        }
    }
    
    private int searchAndFindMax(TrieNode root, int num){
        TrieNode cur = root;
        int res = 0;
        for(int i=31;i>=0;i--){
            int bit = num >> i & 1;
            if(cur.children[1-bit] != null){
                res |= 1 << i;
                cur = cur.children[1-bit];
            }else{
                cur = cur.children[bit];
            }
        }
        return res;
    }
}
```

#### 343.Integer Break ####
###### 思路 ###### 
```
数学问题:方法是尽可能的切割成3.
假设有一个数n, 
case 1: n <= 3 res = n - 1; // (n - 1) 1;
case 2: if n == 4 ; res = 4; // 2 2;
case3: n >= 5: 3(n - 3) > n 
explanation: 3n - 9 > n => 2n > 9 因为n>=5， 所以成立； 对于每个大于等于5的数都可以分解成3 （n -3） > n； 所以分解的所有数字里不是2 就是3（因为4就是两个2）， 而且3很多； 而且2的个数不超过2个 为什么？ 3个2 加起来是6，乘积是8， 2个3也是6，乘积是9， 8 < 9一旦超过两个2 就没有分割成3来的优；
```
###### 代码 ###### 
```
class Solution {
    public int integerBreak(int n) {
        if (n <= 3){
            return n - 1;
        }
        
        int ans = 1;
        if (n % 3 == 1){
            ans *= 4;
            n -= 4;
        }else if (n % 3 == 2){
            ans *= 2;
            n -= 2;
        }
        
        while (n > 0){
            ans *= 3;
            n -= 3;
        }
        
        return ans;
    }
}
```

#### 583.Delete Operation for Two Strings ####
###### 思路 ###### 
```
dp[i] 表示 word1 前 i 个字符和 word2 前 j 个字符相同所需的最少步骤。
若 word1[i - 1] == word2[j - 1] 则 dp[i] = dp[i - 1]， 否则 dp[i] = min(dp[i - 1], dp[i]) + 1。
```
###### 代码 ###### 
```
class Solution {
    public int minDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        int[][] dp = new int[n + 1][m + 1];
        
        for (int i = 1; i <= n; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= m; j++) {
            dp[0][j] = j;
        }
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
        
        return dp[n][m];
    }
}
```

#### 646.Maximum Length of Pair Chain ####
###### 思路 ###### 
```
按照第二个数字大小排序，然后确定最长数对链
```
###### 代码 ###### 
```
class Solution {
    public int findLongestChain(int[][] pairs) {
        if (pairs == null || pairs.length == 0) {
            return 0;
        }
        
        Arrays.sort(pairs, (a, b) -> (a[1] - b[1]));
        int cur = Integer.MIN_VALUE;
        int res = 0;
        for (int i = 0; i < pairs.length; i++) {
            if (pairs[i][0] > cur) {
                res++;
                cur = pairs[i][1];
            }
        }
        return res;
    }
}
```

#### 376.Wiggle Subsequence ####
###### 思路 ###### 
```
贪心 or DP
```
###### 代码 ###### 
```
class Solution {
    public int wiggleMaxLength(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        
        int up = 1;
        int down = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) {
                up = down + 1;
            }else if (nums[i] < nums[i - 1]) {
                down = up + 1;
            }
        }
        return Math.max(up, down);
    }
}
```

#### 494.Target Sum ####
###### 思路 ###### 
```
DFS搜索所有的可能
```
###### 代码 ###### 
```
class Solution {
    private int answer = 0;
    public int findTargetSumWays(int[] nums, int S) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        
        dfs(nums, S, 0);
        return answer;
    }
    
    private void dfs(int[] nums, int target, int startIndex) {
        if (startIndex == nums.length && target == 0) {
            answer++;
            return;
        }
        
        if (startIndex < nums.length) {
            dfs(nums, target - nums[startIndex], startIndex + 1);
            dfs(nums, target + nums[startIndex], startIndex + 1);
        }
    }
}
```

#### 714.Best Time to Buy and Sell Stock with Transaction Fee ####
###### 思路 ###### 
```
- 我们考虑最朴素的方法，对于每一天，如果当前有股票，考虑出售或者保留，如果没股票，考虑购买或者跳过，进行dfs搜索。每天都有两种操作，时间复杂度为O(2^n)
- 如何优化呢？我们用动态规划的思想来解决这个问题，考虑每一天同时维护两种状态：拥有股票(own)状态和已经售出股票(sell) 状态。用own和sell分别保留这两种状态到目前为止所拥有的最大利润。 对于sell，用前一天own状态转移，比较卖出持有股是否能得到更多的利润，即sell = max(sell , own + price - fee)， 而对于own , 我们考虑是否买新的股票更能赚钱(换言之，更优惠），own=max( own, sell-price)
- 初始化我们要把sell设为0表示最初是sell状态且没有profit，把own设为负无穷因为最初不存在该状态，我们不希望从这个状态进行转移
- 因为我们保存的都是最优状态，所以在买卖股票时候取max能保证最优性不变
- 最后直接返回sell即可
```
###### 代码 ###### 
```
class Solution {
    public int maxProfit(int[] prices, int fee) {
        int sell = 0, own = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            sell = Math.max(sell, own + prices[i] - fee);
            own = Math.max(own, sell - prices[i]);
        }
        return sell;
    }
}
```