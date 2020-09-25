## Add Two Numbers ##
#### 2. Add Two Numbers ####
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

###### Example ######
```
Example 1
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

###### 分析 ######
```
TC: O(max(n, m)) SC: O(max(n, m))
注意 carry
```

###### 代码 ######
```
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode last = dummy;
        
        int carry = 0;
        
        while (l1 != null || l2 != null){
            int n1 = l1 == null ? 0 : l1.val;
            int n2 = l2 == null ? 0 : l2.val;
            last.next = new ListNode((n1 + n2 + carry) % 10);
            
            l1 = l1 == null ? l1 : l1.next;
            l2 = l2 == null ? l2 : l2.next;
            last = last.next;
            carry = (n1 + n2 + carry) / 10;
        }
        
        if (carry != 0){
            last.next = new ListNode(carry);
        }
        
        return dummy.next;
    }
}
```
#### 445. Add Two Numbers II ####
You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

###### 分析 ######
```
将两个链表l1和l2反转，这样就能从低位到高位依次相加得到ans，最后将ans反转就是答案了。注意最后ans反转前需要判断是否有前导0的存在，需要删去这个0
TC: O(max(n, m)) SC: O(max(n, m))

注意corner case: [0] [0] -> [0]
```

###### 代码 ######
```
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        l1 = reverse(l1);
        l2 = reverse(l2);
        
        int carry = 0;
        ListNode dummy = new ListNode(0);
        ListNode last = dummy;
        while (l1 != null || l2 != null){
            int v1 = l1 == null ? 0 : l1.val;
            int v2 = l2 == null ? 0 : l2.val;
            last.next = new ListNode((v1 + v2 + carry) % 10);
            last = last.next;
            carry = (v1 + v2 + carry) / 10;
            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
        }
        if (carry != 0){
            last.next = new ListNode(carry);
        }
        ListNode res = reverse(dummy.next);
        // 此处注意corner case: [0] [0] -> [0]
        while (res.val == 0 && res.next != null){
            res = res.next;
        }
        return res;
    }
    
    private ListNode reverse(ListNode head){
        ListNode prev = null;
        while (head != null){
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;
            head = tmp;
        }
        return prev;
    }
}
```
###### Follow Up ######
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.

###### 分析 ######
```
用2个Stack把l1和l2装住然后计算。最后把答案加起来。每次加到List Head.
```
###### 代码 ######
```
class Solution {
    public ListNode addLists2(ListNode l1, ListNode l2) {
        Stack <Integer> num1 = new Stack<>();
        Stack <Integer> num2 = new Stack<>();
        
        ListNode dummy = l1;
        
        while(dummy != null)
        {
            num1.push(dummy.val);
            dummy = dummy.next;
        }
        
        dummy = l2;
        while(dummy != null)
        {
            num2.push(dummy.val);
            dummy = dummy.next;
        }
        
        int cur1 = num1.pop();
        int cur2 = num2.pop();
        int cur = cur1 + cur2;
        int carry = cur/10;
        dummy = new ListNode(cur % 10);
        
        while(!num1.empty() || !num2.empty() || carry > 0)
        {
            cur1 = num1.empty() ? 0 : num1.pop();
            cur2 = num2.empty() ? 0 : num2.pop();
            cur = cur1 + cur2 + carry;
            carry = cur/10;
            
            ListNode old = dummy;
            dummy = new ListNode(cur % 10);            
            dummy.next = old;
        }
     
        return dummy;        
    }
}
```
