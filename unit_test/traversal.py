import unit_test.node_void as TreeNode
import copy
root = TreeNode.Node(data='root', feature=None, threshold=None, children=[])


# Create child nodes
child1 = TreeNode.Node(data='child1', feature=None, threshold=None, children=[])
child2 = TreeNode.Node(data='child2', feature=None, threshold=None, children=[])
child1.depth = 1
child2.depth = 1
# Add the child nodes to the root node's children list
root.children.append(child1)
root.children.append(child2)

# Add a child to child1
child1_1 = TreeNode.Node(data='child1_1', feature=None, threshold=None, children=[])
child1_2 = TreeNode.Node(data='child1_2', feature=None, threshold=None, children=[])
child1_1.depth = 2
child1_2.depth = 2
child1.children.append(child1_1)
child1.children.append(child1_2)


# Add a child to child2
child2_1 = TreeNode.Node(data='child2_1', feature=None, threshold=None, children=[])
child2_2 = TreeNode.Node(data='child2_2', feature=None, threshold=None, children=[])
child2_1.depth = 2
child2_2.depth = 2
child2.children.append(child2_1)
child2.children.append(child2_2)


# Print the tree
def print_tree(node, level=0):
    print('    ' * level + str(node.data))
    if node.children:
        for child in node.children:
            print_tree(child, level + 1)

print_tree(root)

# # function to traverse from root to a node at a given depth
# def traverse(node, depth, parent=None):
#     if node.depth == depth:
#         node.parent = parent
#         node.visited = True
#         new_tree = copy.deepcopy(tree)

#         return node
#     if node.children:
#         for child in node.children:
#             return traverse(child, depth, node)
#     return None

# function to traverse from root to a node at a given depth
def traverse(A, node, depth):
    if node.depth == depth:
        node.visited = True
        A.append(node.id)
        print(node.id)
    elif node.children:
        for child in node.children:
            return traverse(A, child, depth)
    return None

# # function to find all nodes at a given depth and store their id in a list
# def find_nodes_at_depth(node, depth):
#     if node.depth == depth:
#         return [node.id]
#     if node.children:
#         for child in node.children:
#             return find_nodes_at_depth(child, depth)
#     return None

print('\n')
A = []
# Traverse to a node at depth 2
traverse(A, root, 2)
# find_nodes_at_depth(root, 2)
# node = traverse(A, root, 2)
# print(node.data)