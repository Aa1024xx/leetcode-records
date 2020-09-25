## LRU Cache & LFU Cache ##
#### 146. LRU Cache ####
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.

###### Follow Up ###### 
Could you do get and put in O(1) time complexity?

###### Example ######
```
Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
```

###### 分析 ######
```
Hashmap + DoubleLinkedList
TC: O(1) SC: O(capacity)
```

###### 代码 ######
```
class Node {
    int key, val;
    Node next, prev;
    public Node(int key, int val){
        this.key = key;
        this.val = val;
        this.next = null;
        this.prev = null;
    }
}
class LRUCache {
    private Map<Integer, Node> mp;
    private List<Node> list;
    private Node head, tail;
    private int capacity;

    public LRUCache(int capacity) {
        mp = new HashMap<Integer, Node>();
        list = new ArrayList<Node>();
        head = new Node(-1, -1);
        tail = new Node(-1, -1);
        head.next = tail;
        tail.prev = head;
        this.capacity = capacity;
    }
    
    public int get(int key) {
        if (!mp.containsKey(key)){
            return -1;
        }
        
        Node curr = mp.get(key);
        curr.prev.next = curr.next;
        curr.next.prev = curr.prev;
        moveToTail(curr);
        return curr.val;
    }
    
    private void moveToTail(Node curr){
        tail.prev.next = curr;
        curr.prev = tail.prev;
        curr.next = tail;
        tail.prev = curr;
    }
    
    public void put(int key, int value) {
        if (get(key) != -1){
            mp.get(key).val = value;
            return;
        }
        
        if (mp.size() == capacity){
            Node node = head.next;
            head.next = head.next.next;
            head.next.prev = head;
            mp.remove(node.key);
        }
        
        Node curr = new Node(key, value);
        mp.put(key, curr);
        moveToTail(curr);
        return;
    }
}
```
#### 460. LFU Cache ####
Design and implement a data structure for Least Frequently Used (LFU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reaches its capacity, it should invalidate the least frequently used item before inserting a new item. For the purpose of this problem, when there is a tie (i.e., two or more keys that have the same frequency), the least recently used key would be evicted.

Note that the number of times an item is used is the number of calls to the get and put functions for that item since it was inserted. This number is set to zero when the item is removed.

###### Follow Up ###### 
Could you do both operations in O(1) time complexity?

###### Example ######
```
LFUCache cache = new LFUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.get(3);       // returns 3.
cache.put(4, 4);    // evicts key 1.
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

###### 分析 ######
```
Hashmap + DoubleLinkedList
LRU 基础上加了 freq
TC: O(1) SC: O(capacity)
```

###### 代码 ######
```
class Node {
    int key, val, freq;
    Node next, prev;
    public Node (int key, int val, int freq){
        this.key = key;
        this.val = val;
        this.freq = freq;
        this.next = null;
        this.prev = null;
    }
}
class LFUCache {
    private int capacity;
    private Map<Integer, Node> hm = new HashMap<>();
    private Node head = new Node(-1, -1, 0);
    private Node tail = new Node(-1, -1, 0);

    public LFUCache(int capacity) {
        this.capacity = capacity;
        head.next = tail;
        tail.prev = head;
        hm = new HashMap<Integer, Node>();
    }
    
    public int get(int key) {
        if (!hm.containsKey(key)){
            return -1;
        }
        
        Node curr = hm.get(key);
        curr.freq++;
        curr.prev.next = curr.next;
        curr.next.prev = curr.prev;
        move2Position(curr);
        
        return curr.val;
    }
    
    private void move2Position(Node curr){
        Node last = tail.prev;
        while (last.prev != null){
            if (last.freq <= curr.freq){
                break;
            }
            last = last.prev;
        }
        curr.next = last.next;
        last.next.prev = curr;
        curr.prev = last;
        last.next = curr;
    }
    
    public void put(int key, int value) {
        if (get(key) != -1){
            hm.get(key).val = value;
            hm.get(key).freq++;
            hm.get(key).prev.next = hm.get(key).next;
            hm.get(key).next.prev = hm.get(key).prev;
            move2Position(hm.get(key));
            return;
        }
        
        if (capacity == 0){
            return;
        }
        
        if (capacity == hm.size()){
            hm.remove(head.next.key);
            head.next = head.next.next;
            head.next.prev = head;
        }
        
        Node newNode = new Node(key, value, 1);
        hm.put(key, newNode);
        move2Position(newNode);
    }
}
```