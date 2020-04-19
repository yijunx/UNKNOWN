class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.next = None

    def __repr__(self):
        return str(self.val)

x = Node(0)
x.left = Node(1)
x.left.left = Node(3)
x.left.right = Node(4)
x.right = Node(2)
x.right.left = Node(5)


def connect(node):
    if not node or (not node.left and not node.right): return
    node.left.next = node.right
    if node.next:
        node.right.next = node.next.left

    connect(node.left)
    connect(node.right)
    return node

x = connect(x)





