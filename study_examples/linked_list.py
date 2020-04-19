# linked list
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        return str(self.data)


class LinkedList:
    def __init__(self, nodes=None):
        self.head = None
        if nodes is not None:
            # get the head node done
            node = Node(data=nodes.pop(0))
            self.head = node
            for elem in nodes:
                # for the rest of elements....
                # define the next node
                node.next = Node(data=elem)
                # make the next node as current node
                node = node.next

    def __repr__(self):

        nodes = []
        node = self.head
        while node is not None:
            nodes.append(node)
            node = node.next
        nodes.append(None)
        return " -> ".join([str(x) for x in nodes])

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def add_first(self, node):
        node.next = self.head
        self.head = node

    def add_last(self, node):
        if self.head is None:
            self.head = node
        else:
            for a_node in self:  # here make use of __iter__
                pass
            a_node.next = node

    def add_after(self, target_node_data, new_node):
        if not self.head:
            raise Exception('this is an empty list')

        for node in self:  # find the target node here
            if node.data == target_node_data:
                # carry whats behind
                new_node.next = node.next
                # become the after of the target node
                node.next = new_node
                return

        raise Exception(f'node with data {target_node_data} not found')

ll = LinkedList()


# binary tree








