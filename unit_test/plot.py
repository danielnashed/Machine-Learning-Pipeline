class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def print_tree(node, level=0):
    if node is not None:
        print_tree(node.right, level + 1)
        print('  ' * level + str(node.value))
        print_tree(node.left, level + 1)

# Example usage:
root = Node(1, Node(2, Node(4), Node(5)), Node(3))
print_tree(root)