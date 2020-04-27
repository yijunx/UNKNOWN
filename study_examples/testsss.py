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


def nextLargerNodes(input_number_list):

    # form the linked list to take the input
    head = Node(input_number_list.pop(0))
    node = head
    for element in input_number_list:
        node.next = Node(element)
        node = node.next

    print(head.print_node_and_following())

    output = []
    node = head
    loc = 0
    stack = []

    while node:

        output.append(0)

        while stack and stack[-1][1] < node.data:
            idx, number = stack.pop()
            print(f'popped {idx}, {number}')
            output[idx] = node.data
            print(output)

        stack.append((loc, node.data))
        print(f'stack after append is {stack}')
        node = node.next
        loc = loc + 1

    return output

nextLargerNodes([1,7,5,1,9,2,5,1])


# 2 7 4 3 5
# 0 1 2 3 4


# 7 5 4 3 2

# 7 0 5 5 0

# 2 7 5 4 3
# 7 5 4 3 2
#-7 0 0 0 0


def facto(x: int) -> int:
    if x >= 1:
        return x * facto(x - 1)
    else:
        return 1

facto(4)



def oddEvenList(head):

    first_even_node = head.next
    even_node = head.next

    node = head

    print(node, even_node)

    while True:

        node.next = node.next.next
        node = node.next
        even_node.next = node.next
        even_node = node.next

        # stopping criteria
        if node.next is None or node.next.next is None:
            break

        print(node, even_node)

    node.next = first_even_node
    return head

node_a_1 = Node(1)
node_a_2 = Node(2)
node_a_3 = Node(3)
node_a_4 = Node(4)
node_a_5 = Node(5)
node_a_6 = Node(6)
node_a_1.next = node_a_2
node_a_2.next = node_a_3
node_a_3.next = node_a_4
node_a_4.next = node_a_5
node_a_5.next = node_a_6
# x = oddEvenList(node_a_1)
# x.print_node_and_following()
g = [3, 2, 1, 5]

def numComponents(head, G) -> int:

    numCom = 0

    status_list = []

    node = head
    while node:


        if node.data in G:
            print(f'{node.data} is in {g}')
            G.remove(node.data)

            status_list.append(1)
        else:
            status_list.append(0)

        node = node.next

    # now need to count how many groups of 1
    print(status_list)
    previous_item = -1
    for item in status_list:
        if item == 1:
            if previous_item != 1:
                numCom += 1
        previous_item = item

    return numCom
numComponents(node_a_1, g)



















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


def detectCycle2(head):

    fast_mover = normal_mover = another_normal_mover = head
    looped = False
    while fast_mover or fast_mover.next:
        fast_mover = fast_mover.next.next
        normal_mover = normal_mover.next
        if fast_mover is normal_mover:
            looped = True
            break
    if not looped:
        return None

    while normal_mover != another_normal_mover:
        normal_mover = normal_mover.next
        another_normal_mover = another_normal_mover.next

    return normal_mover



detectCycle2(node_a_1)

# 1 2 3 4 5 3 4 5 3 4 5 3 4 5
# 1 3 5 4 3 4 4 3 4

# slow : 3 + 2 = 2 outloop 1 inloop 2 inloop
# head : 0 + 2 = out_loop
# fast : 6 = length + in_loop






x = Node(1)
x.next = Node(2)
x.next.next = Node(3)
x.next.next = Node(4)
x.next = Node(5)
x.next = Node(6)


# odd even linked list

