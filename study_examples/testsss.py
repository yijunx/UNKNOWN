class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        return str(self.data)

    def print_node_and_following(self):
        to_print = str(self.data)
        node = self
        while node.next:
            to_print = f'{to_print} -> {node.next}'
            node = node.next
        to_print = to_print + ' -> None'
        return to_print

class MyLinkedList:
    def __init__(self, nodes_data=None):
        self.head = None
        if nodes_data:
            self.head = Node(nodes_data.pop(0))
            prev_node = self.head
            for element in nodes_data:

                node = Node(element)
                prev_node.next = node
                prev_node = node

    def __repr__(self):
        return self.head.print_node_and_following()
x = MyLinkedList([1,2,3,4])
y = MyLinkedList([7,8,2,3,4])


node_a_1 = Node(1)
node_a_2 = Node(2)
node_a_3 = Node(3)
node_a_4 = Node(4)
node_a_5 = Node(5)
node_b_1 = Node(6)
node_b_2 = Node(7)
node_b_3 = Node(8)
node_b_4 = Node(9)
node_a_1.next = node_a_2
node_a_2.next = node_a_3
node_a_3.next = node_a_4
node_a_4.next = node_a_5
node_b_1.next = node_b_2
node_b_2.next = node_b_3
node_b_3.next = node_b_4#  node_a_3  #
print(node_b_1.print_node_and_following())
print(node_a_1.print_node_and_following())

def getIntersectionNode(head_a, head_b):

    # travers a
    print(head_a.print_node_and_following())
    print(head_b.print_node_and_following())

    node_a = head_a
    node_b = head_b

    # a_extended = False
    # b_extended = False
    # count = 0
    while True:  # and count < 20:
        print(f'a is {node_a}')
        print(f'b is {node_b}')
        # count += 1

        if node_a is node_b:
            print('found it')
            return node_a

        node_a = node_a.next
        node_b = node_b.next

        if node_a is None and node_b is None:
            print('both none.. exit and return nothing')
            return None

        if node_a is None:
            print(f' >>> a is {node_a}, last one, appending b')
            node_a = head_b
            # a_extended = True

        if node_b is None:
            print(f' >>> b is {node_b}, last one, appending a')
            node_b = head_a
            # b_extended = True




getIntersectionNode(node_a_1, node_b_1)



# a: 1 2     3 4 5   -> 6 7 4
# b: 6 7 4




node_a_1 = Node(1)
node_a_2 = Node(2)
node_a_3 = Node(3)
node_a_4 = Node(4)
node_a_5 = Node(5)
node_a_1.next = node_a_2
node_a_2.next = node_a_3
node_a_3.next = node_a_4
node_a_4.next = node_a_5
node_a_5.next = node_a_3

def detectCycle(head):
    node = head
    node_pool = set()
    count = 0
    while node not in node_pool and count < 20:
        count += 1
        node_pool.add(node)
        node = node.next
        if node is None:
            print('no cycle')
            return None

    return node

detectCycle(node_a_1)







x = Node(1)
x.next = Node(2)
x.next.next = Node(3)
x.next.next = Node(4)
x.next = Node(5)
x.next = Node(6)
