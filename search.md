#### 695.Max Area of Island ####
###### 思路 ###### 
```
通过dfs计算每个岛的大小
```
###### 代码 ###### 
```
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int ans = 0;
        for (int i = 0; i < grid.length; i++){
            for (int j = 0; j < grid[0].length; j++){
                if (grid[i][j] == 1){
                    ans = Math.max(ans, dfs(grid, i, j));
                }
            }
        }
        return ans;
    }
    
    private int dfs(int[][] grid, int x, int y){
        if (isValid(grid, x, y)){
            grid[x][y] = 0;
            return 1 + dfs(grid, x + 1, y) + dfs(grid, x - 1, y) + dfs(grid, x, y + 1) + dfs(grid, x, y - 1);
        }
        return 0;
    }
    
    private boolean isValid(int[][] grid, int x, int y){
        if (x < 0 || x >= grid.length){
            return false;
        }
        if (y < 0 || y >= grid[0].length){
            return false;
        }
        return grid[x][y] == 1;
    }
}
```

#### Friend Circles ####
###### 描述 ###### 
```
一个班中有N 个学生。他们中的一些是朋友，一些不是。他们的关系传递。例如，如果A是B的一个直接朋友，而B是C的一个直接朋友，那么A是C的一个间接朋友。我们定义朋友圈为一组直接和间接朋友。
给出一个N*N, 矩阵M表示一个班中学生的关系。如果m[i][j]＝1，那么第i个学生和第j个学生是直接朋友，否则不是。你要输出朋友圈的数量。
```
###### 思路 ###### 
```
BFS
DFS
Union Find
```
###### 代码 ###### 
```
public class Solution {

    public int find(int x, int []fa) {
        if(x == fa[x]) {
            return x;
        } else {
            return fa[x] = find(fa[x], fa);
        }
    }
    public int beginset(int [][] M) {
        //人数
        int n = M.length;
        //答案
        int ans = n;
        //标记是否访问的数组
        int []fa = new int[n];

        for(int i = 0; i < n; i++) {
            fa[i] = i;
        }
        //遍历每个人，进行并查集合并
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {

                //找到朋友
                if(M[i][j] == 1 && i != j) {

                    //这两个朋友在不同的集合里 就把这两个集合合并
                    //合并两个集合，集合数量减少1个
                    if(find(i, fa) != find(j, fa)) {
                        fa[find(i, fa)] = find(j, fa);
                        ans -= 1;
                    }
                }
            }
        }
        return ans;
    }
    
    public int findCircleNum(int[][] M) {
        int ansset = beginset(M);
        return ansset;
    }
}
```

#### 417.Pacific Atlantic Water Flow ####
###### 思路 ###### 
```
从四周往内部dfs，最后判断哪些点两个大洋都可达。
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> pacificAtlantic(int[][] matrix) {
        List<List<Integer>> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0){
            return res;
        }
        
        int n = matrix.length;
        int m = matrix[0].length;
        boolean[][] Pacific = new boolean[n][m];
        boolean[][] Atlantic = new boolean[n][m];
        
        for (int i = 0; i < n; i++){
            dfs(matrix, Pacific, Integer.MIN_VALUE, i, 0);
            dfs(matrix, Atlantic, Integer.MIN_VALUE, i, m - 1);
        }
        for (int i = 0; i < m; i++){
            dfs(matrix, Pacific, Integer.MIN_VALUE, 0, i);
            dfs(matrix, Atlantic, Integer.MIN_VALUE, n - 1, i);
        }
        
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                if (Pacific[i][j] && Atlantic[i][j]){
                    List<Integer> tmp = new ArrayList<>();
                    tmp.add(i);
                    tmp.add(j);
                    res.add(tmp);
                }
            }
        }
        return res;
    }
    
    private void dfs(int[][] matrix, boolean[][] visited, int height, int x, int y){
        int n = matrix.length;
        int m = matrix[0].length;
        if (x < 0 || x >= n || y < 0 || y >= m || matrix[x][y] < height || visited[x][y]){
            return;
        }
        visited[x][y] = true;
        
        int[] dx = {0, -1, 0, 1};
        int[] dy = {1, 0, -1, 0};
        for (int k = 0; k < 4; k++){
            dfs(matrix, visited, matrix[x][y], x + dx[k], y + dy[k]);
        }
    }
}
```


#### 46.Permutations ####
###### 思路 ###### 
```
DFS: 使用 visited 数组记录某个数是否被放到 permutation 里了。
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums == null || nums.length == 0){
            return results;
        }
        
        dfs(nums, new boolean[nums.length], new ArrayList<Integer>(), results);
        return results;
    }
    
    private void dfs(int[] nums, boolean[] visited, List<Integer> permutation, List<List<Integer>> results){
        if (nums.length == permutation.size()){
            results.add(new ArrayList<Integer>(permutation));
            return;
        }
        
        for (int i = 0; i < nums.length; i++){
            if (visited[i]){
                continue;
            }
            permutation.add(nums[i]);
            visited[i] = true;
            dfs(nums, visited, permutation, results);
            visited[i] = false;
            permutation.remove(permutation.size() - 1);
        }
    }
}
```
##### Follow Up #####
#### 47.Permutations II ####
###### 思路 ###### 
```
有重复元素和没有重复元素的 Permutation 一题相比， 只加了两句话：
- Arrays.sort(nums) // 排序这样所有重复的数
- if (i > 0 && numsi == numsi - 1 && !visitedi - 1) { continue; } // 跳过会造成重复的情况
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums == null || nums.length == 0){
            return results;
        }
        Arrays.sort(nums);
        dfs(nums, new boolean[nums.length], new ArrayList<Integer>(), results);
        return results;
    }
    
    private void dfs(int[] nums, boolean[] visited, List<Integer> permutation, List<List<Integer>> results){
        if (nums.length == permutation.size()){
            results.add(new ArrayList<Integer>(permutation));
            return;
        }
        
        for (int i = 0; i < nums.length; i++){
            if (visited[i]){
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]){
                continue;
            }
            
            permutation.add(nums[i]);
            visited[i] = true;
            dfs(nums, visited, permutation, results);
            visited[i] = false;
            permutation.remove(permutation.size() - 1);
        }
    }
}
```

#### 77.Combinations ####
###### 思路 ###### 
```
对于递归函数helper，步骤如下：
- 如果当前组合已满k个数，将这个组合加入最终结果，退出。
- 如果已经访问过所有的数，退出。
- 可行性剪枝处理：如果接下来的数都放入当前的状态数组，也不足k个，退出。
- 假设当前的数pos加入数组，进行下一步递归，结束后，回溯，弹出数组末端的数。
- 假设当前的数pos不加入数组，直接进行下一步递归。
递归结束。
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> combinations = new ArrayList<>();
        if (k == 0 || n <= 0){
            return combinations;
        }
        dfs(1, new ArrayList<Integer>(), n, k, combinations);
        return combinations;
    }
    
    private void dfs(int pos, List<Integer> combination, int n, int k, List<List<Integer>> combinations){
        if (combination.size() == k){
            combinations.add(new ArrayList<Integer>(combination));
            return;
        }
        if (pos == n + 1){
            return;
        }
        if (combination.size() + n - pos + 1 < k){
            return;
        }
        
        combination.add(pos);
        dfs(pos + 1, combination, n, k, combinations);
        combination.remove(combination.size() - 1);
        dfs(pos + 1, combination, n, k, combinations);
    }
}
```

#### 79.Word Search ####
###### 思路 ###### 
```
DFS: 遍历board每个点，看它是否和word开头字母相同，如果相同就就进入dfs过程
```
###### 代码 ###### 
```
class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0){
            return false;
        }
        if (word == null || word.length() == 0){
            return true;
        }
        
        for (int i = 0; i < board.length; i++){
            for (int j = 0; j < board[0].length; j++){
                if (board[i][j] == word.charAt(0)){
                    boolean res = find(board, i, j, word, 0);
                    if (res){
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    private int[] dx = {-1, 0, 1, 0};
    private int[] dy = {0, 1, 0, -1};
    private boolean find(char[][] board, int i, int j, String word, int start){
        if (start == word.length()){
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(start)){
            return false;
}
        
        board[i][j] = '#';
        boolean res = false;
        for (int k = 0; k < 4; k++){
            res = res || find(board, i + dx[k], j + dy[k], word, start + 1);
        }
        board[i][j] = word.charAt(start);
        return res;
    }
}
```

##### Follow Up #####
#### 212.Word Search II####
###### 思路 ###### 
```
Tire树，一种树形结构，是一种哈希树的变种。典型应用是用于统计，排序和保存大量的字符串（但不仅限于字符串），所以经常被搜索引擎系统用于文本词频统计。

- 首先建立字典树，字典树从root开始，每个节点利用hashmap动态开点，利用字母的公共前缀建树
- 遍历字母矩阵，将字母矩阵的每个字母，从root开始dfs搜索，搜索到底部时，将字符串存入答案返回即可
```
###### 代码 ###### 
```
class TrieNode {
    String word;
    HashMap<Character, TrieNode> children;
    public TrieNode() {
        word = null;
        children = new HashMap<Character, TrieNode>();
    }
}

class TrieTree {
    TrieNode root;
    
    public TrieTree(TrieNode t) {
        root = t;
    }
    
    public void insert(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            if (!node.children.containsKey(word.charAt(i))) {
                node.children.put(word.charAt(i), new TrieNode());
            }
            node = node.children.get(word.charAt(i));
        }
        node.word = word;
    }
}

class Solution {
    public int[] dx = {1, 0, -1, 0};
    public int[] dy = {0, -1, 0, 1};
    
    public List<String> findWords(char[][] board, String[] words) {
        List<String> results = new ArrayList<>();
        
        TrieTree tree = new TrieTree(new TrieNode());
        for (String word : words){
            tree.insert(word);
        }
        
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                search(board, i, j, tree.root, results);
            }
        }
        
        return results;
    }
    
    private void search(char[][] board, 
                        int x, 
                        int y, 
                        TrieNode root, 
                        List<String> results) {
        if (!root.children.containsKey(board[x][y])) {
            return;
        }
        
        TrieNode child = root.children.get(board[x][y]);
        
        if (child.word != null) {
            if (!results.contains(child.word)) {
                results.add(child.word);
            }
        }
        
        char tmp = board[x][y];
        board[x][y] = '#';
        
        for (int k = 0; k < 4; k++){
            if (!isValid(board, x + dx[k], y + dy[k])) {
                continue;
            }
            search(board, x + dx[k], y + dy[k], child, results);
        }
        
        board[x][y] = tmp;
    }
    
    private boolean isValid(char[][] board, int x, int y) {
        if (x < 0 || x >= board.length || y < 0 || y >= board[0].length) {
            return false;
        }
        return board[x][y] != '#';
    }
}
```

#### Follow Up####
###### 描述 ###### 
```
给出一个由小写字母组成的矩阵和一个字典。找出最多同时可以在矩阵中找到的字典中出现的单词。一个单词可以从矩阵中的任意位置开始，可以向左/右/上/下四个相邻方向移动。一个字母在整个矩阵中只能被使用一次。且字典中不存在重复单词。
```
###### 思路 ###### 
```
word search II整体类似，只是最后要求的东西改变了，改为要求同时在整个棋盘上可以圈出的单词数量

首先使用trie去预处理dict，用于表示前缀。然后从棋盘的起始点开始dfs，每当我们找到一个单词以后，我们继续在当前棋盘继续进行dfs，直到搜不到单词为止。

优化：我们可以在棋盘搜索的时候进行优化，当一个单词找到以后，下一个单词我们可以从上一个单词的起点的后面开始搜索。因此我们在dfs的过程中可以记录一下单词的起始点，这样用于下一次开始的枚举。
```
###### 代码 ###### 
```
class TrieNode {            //定义字典树的节点
    String word;
    HashMap<Character, TrieNode> children;   //使用HashMap动态开节点
    public TrieNode() {
        word = null;
        children = new HashMap<Character, TrieNode>();
    }
};


class TrieTree{
    TrieNode root;
    
    public TrieTree(TrieNode TrieNode) {
        root = TrieNode;
    }
    
    public void insert(String word) {       //字典树插入单词
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {       
            if (!node.children.containsKey(word.charAt(i))) {
                node.children.put(word.charAt(i), new TrieNode());
            }
            node = node.children.get(word.charAt(i));
        }
        node.word = word;
    }
};

public class Solution {
    
    public int[] dx = {1, 0, -1, 0};   //搜索方向
    public int[] dy = {0, 1, 0, -1};
    
    public void search(char[][] board,          //在字典树上dfs查找
                       int x,
                       int y,
                       int start_x,
                       int start_y,
                       TrieNode cur,
                       TrieNode root,
                       List<String> results,
                       int[] ans) {
        if (!cur.children.containsKey(board[x][y])) {
            return;
        }
        
        TrieNode child = cur.children.get(board[x][y]);
        char tmp = board[x][y];
        board[x][y] = 0;  // mark board[x][y] as used
        if (child.word != null) {      //如果访问到字典树叶子，将字符串压入result即可
            String tmpstr = child.word;
            results.add(tmpstr);
            child.word = null;
            ans[0] = Math.max(ans[0], results.size());
            for (int i = start_x; i < board.length; i++) {
                int startj = 0;
                if (i == start_x) {
                    startj = start_y + 1;
                }
                for (int j = startj; j < board[0].length; j++) {
                    if (board[i][j] != 0) {
                        search(board, i, j, i, j, root, root, results, ans);
                    }
                }
            }
            results.remove(results.size() - 1);
            child.word = tmpstr;
        }
        
        
        for (int i = 0; i < 4; i++) {      //向四个方向dfs搜索
            if (!isValid(board, x + dx[i], y + dy[i])) {
                continue;
            }
            search(board, x + dx[i], y + dy[i], start_x, start_y, child, root, results, ans);
        }
        
        board[x][y] = tmp;  // revert the mark
    }
    
    private boolean isValid(char[][] board, int x, int y) {     //检测搜索位置合法
        if (x < 0 || x >= board.length || y < 0 || y >= board[0].length) {
            return false;
        }
        
        return board[x][y] != 0;
    }
    
    public int wordSearchIII(char[][] board, List<String> words) {
        List<String> results = new ArrayList<String>();
        int[] ans = new int[1];
        ans[0] = 0;
        TrieTree tree = new TrieTree(new TrieNode());
        for (String word : words){
            tree.insert(word);
        }
        
        for (int i = 0; i < board.length; i++) {                //遍历字母矩阵，将每个字母作为单词首字母开始搜索
            for (int j = 0; j < board[i].length; j++) {
                search(board, i, j, i, j, tree.root, tree.root, results, ans);
            }
        }
        
        return ans[0];
    }
}
```

#### 51.N-Queens ####
###### 思路 ###### 
```
DFS:
- 按行摆放
- 合法性判断
O(N!)
```
###### 代码 ###### 
```
class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> results = new ArrayList<>();
        if (n <= 0){
            return results;
        }
        
        search(results, new ArrayList<Integer>(), n);
        return results;
    }
    
    private void search(List<List<String>> results, List<Integer> cols, int n){
        if (cols.size() == n){
            results.add(drawBoard(cols));
            return;
        }
        
        for (int colIndex = 0; colIndex < n; colIndex++) {
            if (!isValid(cols, colIndex)) {
                continue;
            }
            cols.add(colIndex);
            search(results, cols, n);
            cols.remove(cols.size() - 1);
        }
    }
    
    private boolean isValid(List<Integer> cols, int col) {
        int row = cols.size();
        for (int rowIndex = 0; rowIndex < cols.size(); rowIndex++) {
            if (cols.get(rowIndex) == col) {
                return false;
            }
            if (row + col == rowIndex + cols.get(rowIndex)) {
                return false;
            }
            if (row - col == rowIndex - cols.get(rowIndex)) {
                return false;
            }
        }
        return true;
    }
    
    private List<String> drawBoard(List<Integer> cols){
        List<String> result = new ArrayList<>();
        for (int i = 0; i < cols.size(); i++){
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < cols.size(); j++){
                sb.append(j == cols.get(i) ? 'Q' : '.');
            }
            result.add(sb.toString());
        }
        return result;
    }
}
```

##### Follow Up #####
#### 52.N-Queens II ####
###### 思路 ###### 
```
dfs 暴力搜索
```
###### 代码 ###### 
```
class Solution {
    public int totalNQueens(int n) {
        if (n <= 0){
            return 0;
        }
        return search(new ArrayList<Integer>(), n);
    }
    
    private int search(List<Integer> cols, int n) {
        int rowIndex = cols.size();
        
        if (rowIndex == n){
            return 1;
        }
        
        int sum = 0;
        for (int colIndex = 0; colIndex < n; colIndex++) {
            if (!isValid(cols, colIndex)) {
                continue;
            }
            cols.add(colIndex);
            sum += search(cols, n);
            cols.remove(cols.size() - 1);
        }
        return sum;
    }
    
    private boolean isValid(List<Integer> cols, int colIndex) {
        int rowIndex = cols.size();
        for (int row = 0; row < cols.size(); row++) {
            int col = cols.get(row);
            if (col == colIndex) {
                return false;
            }
            if (row + col == rowIndex + colIndex) {
                return false;
            }
            if (row - col == rowIndex - colIndex) {
                return false;
            }
        }
        return true;
    }
}
```

#### 934.Shortest Bridge ####
###### 思路 ###### 
```
BFS + DFS
题目中有两个岛屿，所以先遇到1，dfs找出该岛屿的范围，标记为2，存入队列。然后Bfs，一旦标记为2的岛屿遇到标记为1的岛屿，返回答案即可。
```
###### 代码 ###### 
```
class Node {
    int x, y;
    public Node (int x, int y) {
        this.x = x;
        this.y = y;
    }
}

class Solution {
    private int[] dr = {1, 0, -1, 0};
    private int[] dc = {0, -1, 0, 1}; 
    public int shortestBridge(int[][] A) {
        int row = A.length;
        int col = A[0].length;
        Queue<Node> q = new LinkedList<Node>();
        boolean flag = false;
        
        for (int r = 0; r < row && !flag; r++) {
            for (int c = 0; c < col; c++) {
                if (A[r][c] == 1) {
                    dfs(q, A, r, c, row, col);
                    flag = true;
                    break;
                }
            }
        }
        
        int ans = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                Node tmp = q.poll();
                for (int k = 0; k < 4; k++) {
                    int nr = tmp.x + dr[k];
                    int nc = tmp.y + dc[k];
                    if (valid(nr, nc, row, col) && A[nr][nc] != 2) {
                        if (A[nr][nc] == 1) {
                            return ans;
                        }else {
                            q.offer(new Node(nr, nc));
                            A[nr][nc] = 2;
                        }
                    }
                }
            }
            ans++;
        }
        return ans;
    }
    
    private void dfs(Queue<Node> q, int[][] A, int r, int c, int row, int col) {
        if (!valid(r, c, row, col) || A[r][c] != 1) {
            return;
        }
        A[r][c] = 2;
        q.offer(new Node(r, c));
        for (int k = 0; k < 4; k++) {
            dfs(q, A, r + dr[k], c + dc[k], row, col);
        }
    }
    
    private boolean valid(int r, int c, int row, int col) {
        if (r < 0 || c < 0 || r >= row || c >= col) {
            return false;
        }
        return true;
    }
}
```

#### 126.Word Ladder II ####
###### 思路 ###### 
```
BFS + DFS (end -> start : bfs start -> end : dfs)
- 首先使用bfs，调用getnext()方法寻找当前单词的下一步单词，如果单词在dict中，就存入ret。
- 然后使用dfs，枚举当前字符串的下一步方案，方案存入path,然后枚举搜索，搜索完成后，再将其删除。
```
###### 代码 ###### 
```
class Solution {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> ladders = new ArrayList<List<String>>();
        Set<String> dict = new HashSet<String>(wordList);
        
        if (!dict.contains(endWord)) {
            return ladders;
        }
        
        Map<String, List<String>> map = new HashMap<>();
        Map<String, Integer> distance = new HashMap<String, Integer>();
        
        dict.add(beginWord);
        dict.add(endWord);
        
        bfs(map, distance, endWord, beginWord, dict);
        
        List<String> path = new ArrayList<String>();
        
        dfs(ladders, path, beginWord, endWord, distance, map);
        
        return ladders;
    }
    
    private void dfs(List<List<String>> ladders, List<String> path, String cur, String end, Map<String, Integer> distance, Map<String, List<String>> map) {
        path.add(cur);
        if (cur.equals(end)) {
            ladders.add(new ArrayList<String>(path));
        }else if (map.containsKey(cur)){
                for (String next : map.get(cur)) {
                if (distance.containsKey(next) && distance.get(cur) == distance.get(next) + 1) {
                    dfs(ladders, path, next, end, distance, map);
                }
            }
        }
        path.remove(path.size() - 1);
    }
    
    private void bfs(Map<String, List<String>> map, Map<String, Integer> distance, String start, String end, Set<String> wordList) {
        Queue<String> q = new LinkedList<String>();
        q.offer(start);
        distance.put(start, 0);
        
        for (String str : wordList) {
            map.put(str, new ArrayList<String>());
        }
        
        while (!q.isEmpty()) {
            String cur = q.poll();
            List<String> nextList = expand(cur, wordList);
            for (String next : nextList) {
                map.get(next).add(cur);
                if (!distance.containsKey(next)) {
                    distance.put(next, distance.get(cur) + 1);
                    q.offer(next);
                }
            }
        }
    }
    
    private List<String> expand(String cur, Set<String> wordList) {
        List<String> expansion = new ArrayList<String>();
        
        for (int i = 0; i < cur.length(); i++) {
            for (char c = 'a'; c <= 'z'; c++) {
                if (cur.charAt(i) == c) {
                    continue;
                }
                String word = cur.substring(0, i) + c + cur.substring(i + 1);
                if (wordList.contains(word)) {
                    expansion.add(word);
                }
            }
        }
        
        return expansion;
    }
}
```

##### Follow Up #####
#### 127.Word Ladder ####
###### 思路 ###### 
```
BFS:
- 新建队列，将当前队列中所有节点遍历并取出，将这些节点能走到的所有节点均推入队列，当遍历到end节点时退出bfs，否则将路径长度+1，然后继续遍历直到队列为空
- 当判断当前节点的可到达节点时，可以循环本节点单词的所有字符，用'a' - 'z'中与原来不相等的字符替换后判断dict内是否存在该字符，若存在，则可到达该节点
```
###### 代码 ###### 
```
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> dict = new HashSet<String>(wordList);
        if (beginWord.equals(endWord)) {
            return 1;
        }
        
        Set<String> hash = new HashSet<String>();
        Queue<String> queue = new LinkedList<String>();
        hash.add(beginWord);
        queue.offer(beginWord);
        
        int len = 1;
        while (!queue.isEmpty()) {
            len++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String word = queue.poll();
                for (String nextWord : getNextWords(word, dict)) {
                    if (hash.contains(nextWord)) {
                        continue;
                    }
                    if (nextWord.equals(endWord)) {
                        return len;
                    }
                    hash.add(nextWord);
                    queue.offer(nextWord);
                }
            }
        }
        return 0;
    }
    
    private List<String> getNextWords(String word, Set<String> dict) {
        List<String> nextWords = new ArrayList<String>();
        for (int i = 0; i < word.length(); i++) {
            for (char c = 'a'; c <= 'z'; c++) {
                if (c == word.charAt(i)) {
                    continue;
                }
                String nextWord = word.substring(0, i) + c + word.substring(i + 1);
                if (dict.contains(nextWord)) {
                    nextWords.add(nextWord);
                }
            }
        }
        return nextWords;
    }
}
```

#### 130.Surrounded Regions ####
###### 思路 ###### 
```
从每个边界的 'O' 开始遍历, 只访问 'O', 先都暂时设置为 'T' 或其他字符. 遍历结束之后, 将剩下的 'O' 替换为 'X' 然后再将 'T' 还原即可.
```
###### 代码 ###### 
```
class Solution {
    public void solve(char[][] board) {
        int n = board.length;
        if (board == null || n == 0) {
            return;
        }
        int m = board[0].length;
        if (board[0] == null || m == 0) {
            return;
        }
        
        for (int i = 0; i < n; i++) {
            bfs(board, i, 0);
            bfs(board, i, m - 1);
        }
        for (int j = 0; j < m; j++) {
            bfs(board, 0, j);
            bfs(board, n - 1, j);
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (board[i][j] == 'T') {
                    board[i][j] = 'O';
                }else {
                    board[i][j] = 'X';
                }
            }
        }
    }
    
    private void bfs(char[][] board, int x, int y) {
        if (board[x][y] != 'O') {
            return;
        }
        
        int n = board.length, m = board[0].length;
        int[] dx = {1, 0, -1, 0};
        int[] dy = {0, -1, 0, 1};
        
        Queue<Integer> qx = new LinkedList<Integer>();
        Queue<Integer> qy = new LinkedList<Integer>();
        qx.offer(x);
        qy.offer(y);
        
        board[x][y] = 'T';
        
        while (!qx.isEmpty()) {
            int cx = qx.poll();
            int cy = qy.poll();
            for (int k = 0; k < 4; k++) {
                int nx = cx + dx[k];
                int ny = cy + dy[k];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m || board[nx][ny] != 'O') {
                    continue;
                }
                board[nx][ny] = 'T';
                qx.offer(nx);
                qy.offer(ny);
            }
        }
    }
}
```

#### 257.Binary Tree Paths ####
###### 思路 ###### 
```
- Divide Conquer
- Traverse
```
###### 代码 ###### 
```
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> paths = new ArrayList<>();
        if (root == null) {
            return paths;
        }
        
        List<String> leftPaths = binaryTreePaths(root.left);
        List<String> rightPaths = binaryTreePaths(root.right);
        for (String path : leftPaths) {
            paths.add(root.val + "->" + path);
        }
        for (String path : rightPaths) {
            paths.add(root.val + "->" + path);
        }
        
        if (paths.size() == 0) {
            paths.add(root.val + "");
        }
        
        return paths;
    }
}

class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<String>();
        if (root == null) {
            return result;
        }
        helper(root, String.valueOf(root.val), result);
        return result;
    }
    
    private void helper(TreeNode root, String path, List<String> paths) {
        if (root == null) {
            return;
        }
        
        if (root.left == null && root.right == null) {
            paths.add(path);
            return;
        }
        
        if (root.left != null) {
            helper(root.left, path + "->" + String.valueOf(root.left.val), paths);
        }
        if (root.right != null) {
            helper(root.right, path + "->" + String.valueOf(root.right.val), paths);
        }
    }
}
```

#### 47.Permutations II ####
###### 思路 ###### 
```
排列式深度优先搜索算法:
和没有重复元素的 Permutation 一题相比，只加了两句话：
- Arrays.sort(nums) // 排序这样所有重复的数
- if (i > 0 && numsi == numsi - 1 && !visitedi - 1) { continue; } // 跳过会造成重复的情况
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return results;
        }
        
        Arrays.sort(nums);
        dfs(nums, new boolean[nums.length], new ArrayList<Integer>(), results);
        
        return results;
    }
    
    private void dfs(int[] nums,
                    boolean[] visited,
                    List<Integer> permutation,
                    List<List<Integer>> results) {
        if (nums.length == permutation.size()) {
            results.add(new ArrayList<Integer>(permutation));
            return;
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) {
                continue;
            }
            
            permutation.add(nums[i]);
            visited[i] = true;
            dfs(nums, visited, permutation, results);
            visited[i] = false;
            permutation.remove(permutation.size() - 1);
        }
    }
}
```

##### Follow Up #####
#### 46.Permutations ####
###### 思路 ###### 
```
DFS: 使用 visited 数组记录某个数是否被放到 permutation 里了。
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return results;
        }
        dfs(nums, new boolean[nums.length], new ArrayList<Integer>(), results);
        return results;
    }
    
    private void dfs(int[] nums,
                     boolean[] visited,
                     List<Integer> permutation,
                     List<List<Integer>> results) {
        if (permutation.size() == nums.length) {
            results.add(new ArrayList<Integer>(permutation));
            return;
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            permutation.add(nums[i]);
            visited[i] = true;
            dfs(nums, visited, permutation, results);
            visited[i] = false;
            permutation.remove(permutation.size() - 1);
        }
    }
}
```

#### 40.Combination Sum II ####
###### 思路 ###### 
```
dfs
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> results = new ArrayList<>();
        if (candidates == null || candidates.length == 0) {
            return results;
        }
        
        Arrays.sort(candidates);
        List<Integer> combination = new ArrayList<Integer>();
        helper(candidates, 0, combination, target, results);
        return results;
    }
    
    private void helper(int[] candidates, 
                        int startIndex, 
                        List<Integer> combination,
                        int target,
                        List<List<Integer>> results) {
        if (target == 0) {
            results.add(new ArrayList<Integer>(combination));
            return;
        }
        
        for (int i = startIndex; i < candidates.length; i++) {
            if (i > startIndex && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (target < candidates[i]) {
                break;
            }
            combination.add(candidates[i]);
            helper(candidates, i + 1, combination, target - candidates[i], results);
            combination.remove(combination.size() - 1);
        }
    }
}
```

##### Follow Up #####
#### 39.Combination Sum ####
###### 思路 ###### 
```
dfs
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> results = new ArrayList<>();
        if (candidates == null || candidates.length == 0) {
            return results;
        }
        
        Arrays.sort(candidates);
        List<Integer> combination = new ArrayList<Integer>();
        helper(candidates, 0, combination, target, results);
        return results;
    }
    
    private void helper(int[] candidates, 
                        int startIndex, 
                        List<Integer> combination,
                        int target,
                        List<List<Integer>> results) {
        if (target == 0) {
            results.add(new ArrayList<Integer>(combination));
            return;
        }
        
        for (int i = startIndex; i < candidates.length; i++) {
            if (i > startIndex && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (target < candidates[i]) {
                break;
            }
            combination.add(candidates[i]);
            helper(candidates, i, combination, target - candidates[i], results);
            combination.remove(combination.size() - 1);
        }
    }
}
```

#### 216.Combination Sum III ####
###### 思路 ###### 
```
DFS
```
###### 代码 ###### 
```
class Solution {
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        dfs(result, new ArrayList<Integer>(), k, 1, n);
        return result;
    }
    
    private void dfs(List<List<Integer>> result,
                     List<Integer> combination,
                     int k,
                     int start,
                     int n) {
        if (combination.size() == k && n == 0) {
            result.add(new ArrayList<Integer>(combination));
            return;
        }
        
        for (int i = start; i <= 9; i++) {
            combination.add(i);
            dfs(result, combination, k, i + 1, n - i);
            combination.remove(combination.size() - 1);
        }
    }
}
```

#### 377.Combination Sum IV ####
###### 思路 ###### 
```
DP
```
###### 代码 ###### 
```
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int[] sum = new int[target + 1];
        sum[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int num : nums) {
                if (i < num) {
                    continue;
                }
                sum[i] += sum[i - num];
            }
        }
        return sum[target];
    }
}
```

##### Follow Up #####
###### 描述 ###### 
```
Float Combination Sum
给出一个小数数组A，一个非负整数target。对A中的每个小数进行向上取整或者向下取整的操作，最后得到一个整数数组，要求整数数组的所有数字和等于target，并且要求对小数数组的调整和最小。
例如ceil(1.2)，则调整数为0.8；floor(1.2)，则调整数为0.2。返回该整数数组。
```
###### 思路 ###### 
```
DP: 类比于分组背包问题，每个数字可以看成是包含两个互斥的物品放入即可。对于每个小数，看作是向上取整和向下取整的两个物品，必须选择一个，考虑分组背包。在第二层循环即背包容量的循环中同时考虑两个物品，则可保证选择具有互斥性。
```
###### 代码 ###### 
```
class Solution {
    public int[] getArray(double[] A, int target) {
        double [] dp = new double [15500];
        int [][] path = new int [15500][500];
        int n = A.length;
        int [] res = new int[n];
        for(int i = 1;i <= target;i++)
            dp[i] = 100000;
        dp[0] = 0;
        for(int i = 0;i < n;i++) {
            for(int j = target;;j--) {
                int x = (int)Math.floor(A[i]);
                int y = (int)Math.ceil(A[i]);
                if(j < x && j < y)
                    break;
                if(j >= x && j >= y) {      //两个物品均可以放入，必选其一
                    if(dp[j - x] + A[i] - x < dp[j - y] + y - A[i]) {
                        dp[j] = dp[j - x] + A[i] - x;
                        path[j][i] = 1;
                    }
                    else {
                        dp[j] = dp[j - y] + y - A[i];
                        path[j][i] = 2;
                    }
                    
                }
                else if(j >= x) {       //只能放入向下取整整数，直接放入
                        dp[j] = dp[j - x] + A[i] - x;
                        path[j][i] = 1;
                }
                else if(j >= y) {       //只能放入向上取整整数，直接放入
                        dp[j] = dp[j - y] + y - A[i];
                        path[j][i] = 2;
                }
            }
        }
        if(dp[target] >= 10000)
            return res;
        else {
            int sum = target;
            for(int i = n-1;i >= 0;i--) {       //答案的记录此处通过对背包状态回溯完成还原(同样可以参考背包路径问题)。
                if(path[sum][i] == 1) {
                    res[i] = (int)Math.floor(A[i]);
                    sum- = (int)Math.floor(A[i]);
                }
                else if(path[sum][i] == 2) {
                    res[i] = (int)Math.ceil(A[i]);
                    sum -= (int)Math.ceil(A[i]);
                }

            }
            return res;
        }
        
    }
}
```

#### 37.Sudoku Solver ####
###### 思路 ###### 
```
DFS + backtracking
```
###### 代码 ###### 
```
class Solution {
    public void solveSudoku(char[][] board) {
        solve(board);
    }
    
    private boolean solve(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    continue;
                }
                for (int k = 1; k <= 9; k++) {
                    board[i][j] = (char)(k + '0');
                    if (isValid(board, i, j) && solve(board)) {
                        return true;
                    }
                    board[i][j] = '.';
                }
                return false;
            }
        }
        return true;
    }
    
    private boolean isValid(char[][] board, int a, int b){
        Set<Character> contained = new HashSet<Character>();
        for(int j=0;j<9;j++){
            if(contained.contains(board[a][j])) return false;
            if(board[a][j]>'0' && board[a][j]<='9')
                contained.add(board[a][j]);
        }
            
        
    
        contained = new HashSet<Character>();
        for(int j=0;j<9;j++){
            if (contained.contains(board[j][b])) {
                return false;
            }
            if (board[j][b]>'0' && board[j][b]<='9') {
                contained.add(board[j][b]);
            }
        }
        
    
        contained = new HashSet<Character>();
        for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++){
                int x = a / 3 * 3 + m, y = b / 3 * 3 + n;
                if (contained.contains(board[x][y])) {
                    return false;
                }
                if (board[x][y] > '0' && board[x][y] <= '9') {
                        contained.add(board[x][y]);
                }
            } 
        }
    
        return true;
    }
}
```

##### Follow Up #####
#### 36.Valid Sudoku ####
###### 思路 ###### 
```
利用一个set来记录已经被用过的数，每次遍历一行，一列，一块，将访问过的元素加入set，冲突的话，返回false退出。最后返回true。
```
###### 代码 ###### 
```
class Solution {
    public boolean isValidSudoku(char[][] board) {
        Set<Character> used = new HashSet<>();
        // check rows
        for (int row = 0; row < 9; row++) {
            used.clear();
            for (int col = 0; col < 9; col++) {
                if (!isValid(board[row][col], used)) {
                    return false;
                }
            }
        }
        
        // check cols
        for (int col = 0; col < 9; col++) {
            used.clear();
            for (int row = 0; row < 9; row++) {
                if (!isValid(board[row][col], used)) {
                    return false;
                }
            }
        }
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                used.clear();
                for (int row = i * 3; row < i * 3 + 3; row++) {
                    for (int col = j * 3; col < j * 3 + 3; col++) {
                        if (!isValid(board[row][col], used)) {
                            return false;
                        }
                    }
                }
            }
        }
        
        return true;
    }
    
    private boolean isValid(char c, Set<Character> used) {
        if (c == '.') {
            return true;
        }
        if (used.contains(c)) {
            return false;
        }
        used.add(c);
        return true;
    }
}
```

#### 310.Minimum Height Trees ####
###### 思路 ###### 
```
拓扑排序：虽然不是有序图，但是当节点入度为1时，必然为叶子节点。 由此可见当我们用BFS从叶子节点一层一层搜索下去时，最后一组的节点就是最短高度根节点。 我们的目标是找到那个根节点。
- 创建图
- 统计每个节点的入度
- 把所有入度为1的节点放入queue （叶子节点）
- 从每个叶子节点找到他们各自的根节点，把他们的入度减一
- 当根节点入度为1时， 放入queue
最后的一组queue就是最后的根节点，也就是答案
```
###### 代码 ###### 
```
class Solution {
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) {
            return Arrays.asList(0);
        }
        Map<Integer, List<Integer>> graph = new HashMap<>();
        Map<Integer, Integer> indegree = new HashMap<>();
        
        for (int[] edge : edges) {
            int n1 = edge[0];
            int n2 = edge[1];
            graph.putIfAbsent(n1, new ArrayList<Integer>());
            graph.putIfAbsent(n2, new ArrayList<Integer>());
            
            graph.get(n1).add(n2);
            graph.get(n2).add(n1);
            indegree.put(n1, indegree.getOrDefault(n1, 0) + 1);
            indegree.put(n2, indegree.getOrDefault(n2, 0) + 1);
        }
        
        List<Integer> result = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (indegree.get(i) == 1) {
                queue.offer(i);
            }
        }
        
        while (!queue.isEmpty()) {
            result = new ArrayList<Integer>();
            int size = queue.size();
            for (int j = 0; j < size; j++) {
                int cur = queue.poll();
                result.add(cur);
                for (int next : graph.get(cur)) {
                    indegree.put(next, indegree.get(next) - 1);
                    if (indegree.get(next) == 1) {
                        queue.offer(next);
                    }
                }
            }
        }
        
        return result;
    }
}
```