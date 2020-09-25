## Number of Islands ##
#### 200.Number of Islands ####
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

###### Example ######
```
Example 1
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Example 2
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

###### 分析 ######
```
BFS 
TC: O(n*m) SC: O(n*m)
```
###### 代码 ######
```
class Point {
    int x, y;
    public Point(int x, int y){
        this.x = x;
        this.y = y;
    }
}
class Solution {
    public int numIslands(char[][] grid) {
        int n = grid.length;
        if (grid == null || n == 0){
            return 0;
        }
        int m = grid[0].length;
        if (grid[0] == null || m == 0){
            return 0;
        }
        
        boolean[][] vis = new boolean[n][m];
        int count = 0;
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                if (grid[i][j] == '1' && !vis[i][j]){
                    bfs(grid, i, j, vis);
                    count++;
                }
            }
        }
        
        return count;
    }
    
    private int[] dx = {-1, 0, 1, 0};
    private int[] dy = {0, 1, 0, -1};
    private void bfs(char[][] grid, int x, int y, boolean[][] vis){
        if (grid[x][y] == '0' || vis[x][y]){
            return;
        }
        
        Queue<Point> q = new LinkedList<>();
        q.offer(new Point(x, y));
        vis[x][y] = true;
        
        while (!q.isEmpty()){
            Point curr = q.poll();
            for (int i = 0; i < 4; i++){
                int nx = curr.x + dx[i];
                int ny = curr.y + dy[i];
                if (!isValid(grid, nx, ny)){
                    continue;
                }
                if (vis[nx][ny]){
                    continue;
                }
                q.offer(new Point(nx, ny));
                vis[nx][ny] = true;
            }
        }
    }
    
    private boolean isValid(char[][] grid, int x, int y){
        int n = grid.length;
        int m = grid[0].length;
        if (x < 0 || x >= n || y < 0 || y >= m){
            return false;
        }
        if (grid[x][y] == '0'){
            return false;
        }
        return true;
    }
}
```

#### Number of Islands II ####
Given a n,m which means the row and column of the 2D matrix and an array of pair A( size k). Originally, the 2D matrix is all 0 which means there is only sea in the matrix. The list pair has k operator and each operator has two integer A[i].x, A[i].y means that you can change the grid matrix[A[i].x][A[i].y] from sea to island. Return how many island are there in the matrix after each operator.

###### Example ######
```
Example 1
Input: n = 4, m = 5, A = [[1,1],[0,1],[3,3],[3,4]]
Output: [1,1,2,2]
Explanation:
0.  00000
    00000
    00000
    00000
1.  00000
    01000
    00000
    00000
2.  01000
    01000
    00000
    00000
3.  01000
    01000
    00000
    00010
4.  01000
    01000
    00000
    00011

Example 2
Input: n = 3, m = 3, A = [[0,0],[0,1],[2,2],[2,1]]
Output: [1,1,2,2]
```

###### 分析 ######
```
Union Find 
并查集是指用集合里的一个元素来当这个集合的代表元
如果两个元素所在集合的代表元相同，那么我们就能知道这两个元素在一个集合当中。
如果我们想合并两个集合，只需要把其中一个集合的代表元改为第二个集合的代表元

这道题中，每次将一个海洋i变成一个岛屿i，那么先将岛屿数量加一
再依次查看这个岛屿的四周的四个方格
    如果相邻的方格j也是岛屿，那么先判断i是不是和j在同一个集合里
    如果不是在一个集合里，那么i j所在的两个集合就是连通的，可以合并算为一个集合,然后让岛屿数量-1。
    如果已经是在同一个集合里了，那就不用在进行任何操作
我们只要让i所在集合的代表元改为j所在集合的代表元就完成了合并操作。
注意：数据中有可能多次将一个位置变成岛屿，第一次以后的操作都是无效的操作，跳过就好了

TC: 每次查询代表元均摊是O(α) α代表反阿克曼函数，反阿克曼函数是渐进增长很慢很慢的，我们可以近似的认为每次查询是O(1)的复杂度 我们一共有K次操作，每次操作最多并查集查询4次，并查集合并4次,所以我们最终的时间复杂度是O(K)
SP: O(mn)
```
###### 代码 ######
```
public class Solution {
    private int matrix2Id(int x, int y, int m){
        return x * m + y;
    }
    
    class UnionFind{
        Map<Integer, Integer> father = new HashMap<Integer, Integer>();
        public UnionFind(int n, int m){
            for (int i = 0; i < n; i++){
                for (int j = 0; j < m; j++){
                    int id = matrix2Id(i, j, m);
                    father.put(id, id);
                }
            }
        }
        
        int find(int x){
            if (father.get(x) == x){
                return x;
            }
            father.put(x, find(father.get(x)));
            return father.get(x);
        }
        
        void union(int x, int y){
            int fa_x = find(x);
            int fa_y = find(y);
            if (fa_x != fa_y){
                father.put(fa_x, fa_y);
            }
        }
    }
    
    private int[] dx = {-1, 0, 1, 0};
    private int[] dy = {0, 1, 0, -1};
    public List<Integer> numIslands2(int n, int m, Point[] operators) {
        List<Integer> ans = new ArrayList<Integer>();
        if (operators == null || operators.length == 0){
            return ans;
        }
        
        UnionFind uf = new UnionFind(n, m);
        int[][] island = new int[n][m];
        int count = 0;
        
        for (int i = 0; i < operators.length; i++){
            int x = operators[i].x;
            int y = operators[i].y;
            if (island[x][y] != 1){
                island[x][y] = 1;
                count++;
                int id = matrix2Id(x, y, m);
                int fa = uf.find(id);
                for (int k = 0; k < 4; k++){
                    int nx = x + dx[k];
                    int ny = y + dy[k];
                    if (!isValid(island, nx, ny)){
                        continue;
                    }
                    int newId = matrix2Id(nx, ny, m);
                    int newFa = uf.find(newId);
                    if (fa != newFa){
                        count--;
                        uf.union(newId, id);
                    }
                }
            }
            ans.add(count);
        }
        return ans;
    }
    
    private boolean isValid(int[][] island, int x, int y){
        int n = island.length;
        int m = island[0].length;
        if (x < 0 || x >= n || y < 0 || y >= m){
            return false;
        }
        if (island[x][y] != 1){
            return false;
        }
        return true;
    }
}
```

#### Number of Big Islands ####
Given a boolean 2D matrix, 0 is represented as the sea, 1 is represented as the island. If two 1 is adjacent, we consider them in the same island. We only consider up/down/left/right adjacent.

Find the number of islands that size bigger or equal than K.

###### Example ######
```
Example 1
Input: 
[[1,1,0,0,0],[0,1,0,0,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,1]]
2
Output: 2
Explanation:
the 2D grid is
[
  [1, 1, 0, 0, 0],
  [0, 1, 0, 0, 1],
  [0, 0, 0, 1, 1],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1]
]
there are two island which size is equals to 3.

Example 2
Input: 
[[1,0],[0,1]]
1
Output: 2
```

###### 分析 ######
```
BFS
TC: O(n*m) SC: O(n*m)
```

###### 代码 ######
```
public class Solution {
    class Pair {
        int x, y;
        public Pair(int x, int y){
            this.x = x;
            this.y = y;
        }
    }
    
    public int numsofIsland(boolean[][] grid, int k) {
        int n = grid.length;
        if (grid == null || n == 0){
            return 0;
        }
        int m = grid[0].length;
        if (grid[0] == null || m == 0){
            return 0;
        }
        
        int ans = 0;
        boolean[][] visited = new boolean[n][m];
        
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                if (grid[i][j] && !visited[i][j]){
                    if (bfs(grid, i, j, visited) >= k){
                        ans++;
                    }
                }
            }
        }
        
        return ans;
    }
    
    private int[] dx = {-1, 0, 1, 0};
    private int[] dy = {0, 1, 0, -1};
    private int bfs(boolean[][] grid, int x, int y, boolean[][] visited){
        Queue<Pair> q = new LinkedList<Pair>();
        q.offer(new Pair(x, y));
        visited[x][y] = true;
        int count = 0;
        
        while (!q.isEmpty()){
            Pair cur = q.poll();
            count++;
            for (int i = 0; i < 4; i++){
                int nx = cur.x + dx[i];
                int ny = cur.y + dy[i];
                if (!isValid(grid, nx, ny, visited)){
                    continue;
                }
                q.offer(new Pair(nx, ny));
                visited[nx][ny] = true;
            }
        }
        return count;
    }
    
    private boolean isValid(boolean[][] grid, int x, int y, boolean[][] visited){
        int n = grid.length;
        int m = grid[0].length;
        if (x < 0 || x >= n || y < 0 || y >= m){
            return false;
        }
        if (!grid[x][y]){
            return false;
        }
        if (visited[x][y]){
            return false;
        }
        return true;
    }
}
```

#### Number of Distinct Islands ####
Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical). You may assume all four edges of the grid are surrounded by water.

Count the number of distinct islands. An island is considered to be the same as another if and only if one island has the same shape as another island (and not rotated or reflected).

###### Example ######
```
Example 1
Input: 
  [
    [1,1,0,0,1],
    [1,0,0,0,0],
    [1,1,0,0,1],
    [0,1,0,1,1]
  ]
Output: 3
Explanation:
  11   1    1
  1        11   
  11
   1

Example 2
Input:
  [
    [1,1,0,0,0],
    [1,1,0,0,0],
    [0,0,0,1,1],
    [0,0,0,1,1]
  ]
Output: 1
```

###### 分析 ######
```
我们可以通过BFS/DFS得到每一个岛屿, 然后把每一个岛屿的形状放到 set 里, 最后 set 的大小就是答案.

那么问题的关键在于如何描述一个岛屿的形状.

有以下两个基本思路:
    记录一个岛屿所有点相对于左上角的点的相对位置.
    记录一个岛屿的bfs/dfs轨迹

方法1涉及细节较少, 但是可能复杂度相对较高, 不过 50x50 的数据范围不会超时.
方法1也有多种实现方法, 比如一个岛屿形状可以用set记录, 也可以将所有点的相对坐标排序后转换成字符串.
方法2需要注意一个细节: 不能仅仅储存下来dfs/bfs移动的方向, 因为涉及到回溯等问题, 可以加上一定的间隔符, 或者除方向之外额外记录一个位置信息.

选用方法1
```

###### 代码 ######
```
public class Solution {
    /**
     * @param grid: a list of lists of integers
     * @return: return an integer, denote the number of distinct islands
     */
    public int numberofDistinctIslands(int[][] grid) {
        int n = grid.length;
        if (grid == null || n == 0){
            return 0;
        }
        int m = grid[0].length;
        if (grid[0] == null || m == 0){
            return 0;
        }
        
        Set<String> set = new HashSet<String>();
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                if (grid[i][j] == 1){
                    String path = dfs(grid, i, j);
                    set.add(path);
                }
            }
        }
        
        return set.size();
    }
    
    private int[] dx = {-1, 0, 1, 0};
    private int[] dy = {0, 1, 0, -1};
    private String dfs(int[][] grid, int i, int j){
        String path = "";
        
        for (int k = 0; k < 4; k++){
            int x = i + dx[k];
            int y = j + dy[k];
            if (!isValid(grid, x, y)){
                continue;
            }
            grid[x][y] = 0;
            path += k + dfs(grid, x, y);
        }
        
        return path.length() == 0 ? "#" : path;
    }
    
    private boolean isValid(int[][] grid, int x, int y){
        int n = grid.length;
        int m = grid[0].length;
        if (x < 0 || x >= n || y < 0 || y >= m){
            return false;
        }
        if (grid[x][y] != 1){
            return false;
        }
        return true;
    }
}
```
#### Number of Distinct Islands II ####
Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Count the number of distinct islands. An island is considered to be the same as another if they have the same shape, or have the same shape after rotation (90, 180, or 270 degrees only) or reflection (left/right direction or up/down direction).

###### Example ######
```
Example 1
Input: [[1,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1]]
Output: 1
Explanation:
The island is look like this:
11000
10000
00001
00011

Notice that:
11
1
and
 1
11
are considered same island shapes. Because if we make a 180 degrees clockwise rotation on the first island, then two islands will have the same shapes.

Example 2
Input: [[1,1,1,0,0],[1,0,0,0,1],[0,1,0,0,1],[0,1,1,1,0]]
Output: 2
Explanation:
The island is look like this:
11100
10001
01001
01110

Here are the two distinct islands:
111
1
and
1
1

Notice that:
111
1
and
1
111
are considered same island shapes. Because if we flip the first array in the up/down direction, then they have the same shapes.
```

###### 分析 ######
```
dfs 找出每个岛，旋转比较(getUnique())
```

###### 代码 ######
```
class Point {
    int x, y;
    public Point(int x, int y){
        this.x = x;
        this.y = y;
    }
}
public class Solution {
    /**
     * @param grid: the 2D grid
     * @return: the number of distinct islands
     */
    public int numDistinctIslands2(int[][] grid) {
        int n = grid.length;
        if (grid == null || n == 0){
            return 0;
        }
        int m = grid[0].length;
        if (grid[0] == null || m == 0){
            return 0;
        }
        
        Set<String> res = new HashSet<String>();
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                if (grid[i][j] == 1){
                    List<Point> island = new ArrayList<>();
                    dfs(grid, i, j, island);
                    res.add(getUnique(island));
                }
            }
        }
        
        return res.size();
    }
    
    private int[] dx = {-1, 0, 1, 0};
    private int[] dy = {0, 1, 0, -1};
    private void dfs(int[][] grid, int x, int y, List<Point> island){
        int n = grid.length;
        int m = grid[0].length;
        
        island.add(new Point(x, y));
        grid[x][y] = 0;
        
        for (int k = 0; k < 4; k++){
            int nx = x + dx[k];
            int ny = y + dy[k];
            if (!isValid(grid, x, y)){
                continue;
            }
            dfs(grid, nx, ny, island);
        }
        
    }
    
    private boolean isValid(int[][] grid, int x, int y){
        int n = grid.length;
        int m = grid[0].length;
        if (x < 0 || x >= n || y < 0 || y >= m){
            return false;
        }
        if (grid[x][y] != 1){
            return false;
        }
        return true;
    }
    
    private String getUnique(List<Point> island){
        List<String> sameIslands = new ArrayList<>();
        
        int[][] trans = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        
        for (int k = 0; k < 4; k++){
            List<Point> l1 = new ArrayList<>();
            List<Point> l2 = new ArrayList<>();
            for (Point p :island){
                int x = p.x;
                int y = p.y;
                l1.add(new Point(x * trans[k][0], y * trans[k][1]));
                l2.add(new Point(y * trans[k][0], x * trans[k][1]));
            }
            sameIslands.add(getStr(l1));
            sameIslands.add(getStr(l2));
        }
        
        Collections.sort(sameIslands);
        return sameIslands.get(0);
    }
    
    private String getStr(List<Point> island){
        Collections.sort(island, new Comparator<Point>(){
            public int compare(Point a, Point b){
                if (a.x != b.x){
                    return a.x - b.x;
                }
                return a.y - b.y;
            }
        });
        
        StringBuilder sb = new StringBuilder();
        int x = island.get(0).x;
        int y = island.get(0).y;
        
        for (Point p : island){
            sb.append((p.x - x) + " " + (p.y - y) + " ");
        }
        
        return sb.toString();
    }
}
```